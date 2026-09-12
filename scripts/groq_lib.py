"""
Shared Groq client + intent-classification call, used by both the
standalone classifier script and the golden-set candidate builder.
Pulled out to one place so there's a single source of truth for the
prompt, retry behavior, and model choice.

Swapped in for gemini_lib.py: Gemini's free tier (20 req/day on
gemini-3.6-flash, then a 500 req/day cap even on gemini-flash-lite-latest)
was too tight for a golden set + reply-drafting + eval harness pipeline
that easily adds up to 1,000-3,000+ calls. Groq's free tier is far more
generous on requests/day, which is what actually matters here.

TAXONOMY v2 (2026-09-10): the response schema grew from a flat 3-field
{intent, confidence, reason} (v1, 8 intents) to include secondary_intent,
turn_type, and a set of flags -- see taxonomy.md for the full rationale.
The core finding: v1 put topic, conversation-turn, and risk signals all
into one intent enum (e.g. followup_ticket_status was a turn type wearing
an intent's clothes), which meant every mid-thread tweet lost either its
topic or its turn-type. v2 keeps intent as pure topic and moves
everything else into its own field, at zero extra golden-set budget
(flags don't need their own 15-example minimum the way an intent does).
"""
import json
import os

from dotenv import load_dotenv
from groq import Groq
from groq import APIError, APIConnectionError, APITimeoutError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

load_dotenv()

# Model catalog on Groq changes over time -- llama-3.1-8b-instant/
# llama-3.3-70b-versatile (originally planned) are no longer available
# on this key as of 2026-09-10; `client.models.list()` was used to
# check what's actually live rather than guessing. openai/gpt-oss-20b
# is a mid-size OSS model with solid JSON-mode support, good enough
# for picking 1 of 13 categories plus a handful of flags. Swap this one
# constant if that changes again or accuracy on the golden set says
# otherwise.
MODEL = "openai/gpt-oss-20b"

# Bumped whenever intents.json / turn_types.json / the flag set changes
# in a way that would make an old cached result mean something
# different than a new one. Used as a namespace prefix on cache keys
# (see cache.py) so re-running the pipeline after a taxonomy change
# never silently serves a stale v1 label as if it were a v2 one --
# it just re-classifies, which is the whole point of versioning this.
TAXONOMY_VERSION = "v2"

# Bumped whenever SYSTEM_PROMPT / USER_PROMPT_TEMPLATE change in a way
# that could change an answer -- wording, examples, field order, decoding
# params. TAXONOMY_VERSION alone was not enough: it only moves when the
# label SET changes, so two different prompts over the same 13 intents
# would share a cache namespace and the second experiment would silently
# read back the first one's predictions.
#
# "p1" is the prompt exactly as it stood when the 189-row golden set was
# classified -- i.e. the baseline every later variant is measured against.
# Do not reuse a retired value.
PROMPT_VERSION = "p1"

# p2 WAS TESTED AND REJECTED (2026-09-12). Kept here because the negative
# result is the useful part.
#
# p2 = p1 + PRECEDENCE_RULES (taxonomy.md Part 11) in the system prompt.
# The hypothesis was sound: those 9 rules decided 53 of the golden set's
# relabels and the classifier could not read any of them.
#
# Dev split (86 scorable rows), same model, same schema:
#     p1  70/86  81.4%
#     p2  68/86  79.1%      7 rows fixed, 9 broken
#
# Not a regression worth calling one at that n -- but no gain either, at
# 2.6x the prompt tokens per call (1035 vs ~400) against a 200k/day cap.
# Rejected on cost, not on accuracy.
#
# The breakdown is what matters for p3. The MECHANICAL rules worked:
#   R5 device-disk      2028354, 2976667 fixed
#   R7b authenticated   2222820 fixed
#   R8a symptom-governs 1851629 fixed
# The JUDGMENT rules backfired -- the model applies them bluntly:
#   R2 grievance/feedback  2124237, 847924 went feature_request -> complaint
#   R4 "prefer specific"   2509185, 1299266 went TO how_to_usage, the
#                          opposite of what the rule says
#
# p3 should therefore keep the mechanical rules and replace the judgment
# rules with the FEW_SHOT examples below -- a small model imitates an
# example more reliably than it applies an abstract preference.
#
# The rules stay exported: 08_label_golden_set.py shows them to the human
# labeler, where judgment rules are an asset rather than a liability.
# That half of the change is live and unaffected by this revert.


def cache_namespace(model: str | None = None, prompt_version: str | None = None) -> str:
    """Cache key prefix. Every axis that can change an answer for the same
    tweet_id belongs here: the label set, the prompt, and the model.

    Model is in the namespace because comparing two models means running
    both over the SAME tweet_ids -- without it the second model reads the
    first one's cached answers and the comparison silently returns
    'identical', which is the most convincing wrong result available.
    """
    return f"{TAXONOMY_VERSION}:{prompt_version or PROMPT_VERSION}:{model or MODEL}"

with open("intents.json") as f:
    INTENTS = json.load(f)

with open("turn_types.json") as f:
    TURN_TYPES = json.load(f)

INTENT_NAMES = [i["name"] for i in INTENTS]
TURN_TYPE_NAMES = [t["name"] for t in TURN_TYPES]

INTENT_LIST_TEXT = "\n".join(f"- {i['name']}: {i['description']}" for i in INTENTS)
TURN_TYPE_LIST_TEXT = "\n".join(f"- {t['name']}: {t['description']}" for t in TURN_TYPES)

SENTIMENT_VALUES = ["positive", "neutral", "negative", "angry"]

# Every non-intent, non-turn_type field in the response, with its type
# and allowed values -- kept as one dict so _validate() and the prompt
# builder both read from a single source of truth instead of drifting.
# Kept terse deliberately -- this text is resent on every single API call
# (see "Prompt cost" note below), so verbosity here has a real dollar/quota
# cost. Full rationale for each flag lives in taxonomy.md, not here.
FLAG_SPEC = {
    "wants_human": {"type": "bool", "doc": "explicitly asks for a phone number, live chat, or a person."},
    "legal_sensitive": {"type": "bool", "doc": "GDPR/CCPA erasure, account closure, or legal/ToS dispute."},
    "churn_threat": {"type": "bool", "doc": "threatens to cancel or switch to a competitor."},
    "abusive_content": {"type": "bool", "doc": "profanity/abuse directed at the brand or staff."},
    "needs_human_triage": {"type": "bool", "doc": "genuinely unsure which intent fits; explicit abstain."},
    "sentiment": {"type": "enum", "values": SENTIMENT_VALUES, "doc": f"one of: {', '.join(SENTIMENT_VALUES)}."},
    "language": {"type": "str", "doc": "lowercase ISO 639-1 code, e.g. \"en\", \"es\", \"fr\"."},
}

_FLAG_DOC_LIST = "\n".join(f"- {name}: {spec['doc']}" for name, spec in FLAG_SPEC.items())

# Groq's structured-output support (response_format=json_schema) is
# model-dependent and newer than Gemini's; json_object mode (guaranteed
# valid JSON, but not guaranteed to match a schema) is supported
# everywhere on Groq, so the schema is enforced via the prompt instead
# and validated after parsing -- see _validate below.
#
# PROMPT COST (found the hard way, 2026-09-10): this whole prompt is
# resent on EVERY call, and Groq's real constraint turned out to be a
# 200,000 tokens/DAY cap per key -- separate from, and much tighter
# than, the 8,000 tokens/minute cap rate_limiter.py was built around.
# The first version of this v2 prompt (documentation-length intent/
# flag descriptions, matching taxonomy.md's prose) cost ~1,317 prompt
# tokens/call -- a ~130 calls/day ceiling, nowhere near enough to build
# a 195+ candidate golden-set pool. Trimmed to short phrases here
# (v1's intents.json was already this terse; v2's first draft wasn't).
# Full rationale for every intent/turn_type/flag lives in taxonomy.md
# for humans -- keep additions here to what the MODEL needs to decide,
# not why. Re-measure token cost (see scripts/00_test_groq.py-style
# usage check) after any edit to intents.json/turn_types.json/FLAG_SPEC.
# taxonomy.md Part 11, compressed to what the model needs to DECIDE. The
# reasoning behind each rule stays in taxonomy.md for humans. Order is
# load-bearing: first match wins, and R2/R3 collide unless R1 is applied
# first (a proposed product change is feature_request even when the object
# is a shared folder -- see 1874363).
PRECEDENCE_RULES = """Rules, in order -- first match wins:
1. Asks something answerable or requests an action -> TOPIC intent, never complaint_dissatisfaction. Anger goes in sentiment.
2. complaint_dissatisfaction only if NO request AND aimed at the service/company/support/cost. Criticising a feature or design, or proposing a change -> feature_request.
3. Asserts how the product works in a way support could correct -> that topic, not complaint.
4. how_to_usage needs nothing broken; "how do I fix X" where X is broken -> X's intent. Prefer a specific intent over how_to_usage.
5. Shared link/folder/team folder -> sharing_permissions (incl. how-tos), UNLESS the failure is quota -> storage_quota_plan_limits, or load/save/sync/crash -> sync_app_bug.
6. Deleted files but storage unchanged -> storage_quota_plan_limits. Laptop/device disk -> NOT storage_quota_plan_limits.
7. Answer is a price/refund/purchase -> billing_subscription; answer is how storage is counted -> storage_quota_plan_limits.
8. account_access only if they cannot log in or reach the account; authenticated + wants an operation -> how_to_usage.
9. Personal symptom + "is it down?" -> the symptom's intent, service_outage as secondary. service_outage alone only with no personal symptom or wide-scope evidence."""

# NOT IN THE p2 PROMPT -- staged for a p3 experiment, kept here so the
# examples are picked once and reviewed.
#
# Held back for two reasons. It costs ~194 prompt tokens on EVERY call
# against a 200k/day cap, and it overlaps heavily with PRECEDENCE_RULES --
# shipping both at once would mean a p2 result that cannot say which of
# the two did the work. p2 changes exactly one thing: the rules become
# reachable by the model.
#
# Every example is drawn from the DEV split (data/golden_split.csv), never
# from holdout, which would make the held-out number self-fulfilling.
FEW_SHOT = """Examples:
"don't like the new tray icon" -> feature_request
"so expensive, you ought to review this price" -> complaint_dissatisfaction
"one shared folder uses up my entire quota is not cool" -> storage_quota_plan_limits
"changes reverted, how do I fix this?" -> sync_app_bug
"how can I eliminate a Team folder?" -> sharing_permissions
"can't open a shared link, just keeps loading" -> sync_app_bug
"permanently deleted items, still says out of space" -> storage_quota_plan_limits
"not enough space on my laptop, will an external drive help?" -> how_to_usage
"need more space, won't pay for 5 users, what are my options?" -> billing_subscription
"how can I unlink my work account from my home computer?" -> how_to_usage
"Is Dropbox down? Mac Client does not sync" -> sync_app_bug"""

SYSTEM_PROMPT = """Classify this customer support tweet sent to Dropbox's support account.

Intent (pick ONE, the customer's primary need):
{intent_list}

secondary_intent: a second clearly-present intent, else null. Must differ from intent.

turn_type (conversational position, separate from topic):
{turn_type_list}

Flags:
{flag_list}

Respond with ONLY this JSON shape, no other text:
{{"intent": "<name>", "secondary_intent": "<name or null>", "turn_type": "<name>", "confidence": <0-1>, "reason": "<one sentence>", "wants_human": <bool>, "legal_sensitive": <bool>, "churn_threat": <bool>, "abusive_content": <bool>, "needs_human_triage": <bool>, "sentiment": "<{sentiment_values}>", "language": "<ISO 639-1 code>"}}"""

USER_PROMPT_TEMPLATE = 'Customer tweet:\n"""{text}"""'

RETRYABLE_NETWORK_ERRORS = (APIConnectionError, APITimeoutError)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIError):
        status = getattr(exc, "status_code", None)
        return status in (429, 500, 502, 503, 504)
    return isinstance(exc, RETRYABLE_NETWORK_ERRORS)


class InvalidClassification(ValueError):
    """Model returned JSON that doesn't match our schema (bad intent name,
    missing field, etc.) -- distinct from a transport/API failure."""


def _validate(result: dict) -> dict:
    if result.get("intent") not in INTENT_NAMES:
        raise InvalidClassification(f"intent {result.get('intent')!r} not in {INTENT_NAMES}")

    secondary = result.get("secondary_intent")
    if secondary is not None and secondary not in INTENT_NAMES:
        raise InvalidClassification(f"secondary_intent {secondary!r} not in {INTENT_NAMES}")
    if secondary is not None and secondary == result["intent"]:
        raise InvalidClassification("secondary_intent must differ from intent")

    if result.get("turn_type") not in TURN_TYPE_NAMES:
        raise InvalidClassification(f"turn_type {result.get('turn_type')!r} not in {TURN_TYPE_NAMES}")

    if "confidence" not in result or "reason" not in result:
        raise InvalidClassification(f"missing required field(s) in {result}")

    if result.get("sentiment") not in SENTIMENT_VALUES:
        raise InvalidClassification(f"sentiment {result.get('sentiment')!r} not in {SENTIMENT_VALUES}")

    for flag_name, spec in FLAG_SPEC.items():
        if spec["type"] != "bool":
            continue
        if not isinstance(result.get(flag_name), bool):
            raise InvalidClassification(f"{flag_name} must be true/false, got {result.get(flag_name)!r}")

    if not isinstance(result.get("language"), str) or not result["language"]:
        raise InvalidClassification(f"language must be a non-empty string, got {result.get('language')!r}")

    return result


def make_client() -> Groq:
    return Groq(api_key=os.environ["GROQ_API_KEY"], timeout=30.0)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception(_is_retryable),
    reraise=True,  # raise the ORIGINAL exception on final failure, not
                   # tenacity's own RetryError wrapper -- otherwise
                   # `except SKIPPABLE_ERRORS` at call sites never
                   # matches and the whole batch crashes anyway
)
def classify_message(client: Groq, text: str, model: str | None = None) -> dict:
    """`model` overrides the default for A/B runs. Everything else --
    prompt, temperature, reasoning_effort, schema -- stays fixed, so a
    model comparison changes exactly one variable."""
    response = client.chat.completions.create(
        model=model or MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    intent_list=INTENT_LIST_TEXT,
                    turn_type_list=TURN_TYPE_LIST_TEXT,
                    flag_list=_FLAG_DOC_LIST,
                    sentiment_values=", ".join(SENTIMENT_VALUES),
                ),
            },
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        reasoning_effort="low",  # gpt-oss-20b spends hidden "reasoning" tokens
        # by default (111 tokens on a test call) for a task that's really just
        # a constrained pick-one -- "low" cut that to 6 tokens with no change
        # in the answer on the same test call. Small saving next to the
        # prompt-size fix above, but free, so keep it.
    )
    result = json.loads(response.choices[0].message.content)
    # Observed in testing: the model sometimes returns the STRING "null"
    # (or "none"/"") for secondary_intent instead of the JSON literal null,
    # which would otherwise fail validation as an unrecognized intent name.
    # Normalize before validating rather than trusting the model's JSON
    # literal discipline.
    if isinstance(result.get("secondary_intent"), str) and result["secondary_intent"].strip().lower() in ("null", "none", ""):
        result["secondary_intent"] = None
    return _validate(result)


# APIError, the network errors above, and InvalidClassification (bad
# JSON shape that survived retries) are all worth catching the same way
# at the call site: skip this example rather than crash the whole batch.
SKIPPABLE_ERRORS = (APIError, *RETRYABLE_NETWORK_ERRORS, InvalidClassification, json.JSONDecodeError)
