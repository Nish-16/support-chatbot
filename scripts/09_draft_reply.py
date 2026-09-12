"""
Grounded reply drafting: retrieve similar past DropboxSupport threads
(retrieval.py) and prompt Groq to draft a reply consistent with how the
brand actually responded before, not the model's own generic instincts.

Policy enforced in the prompt (README.md, "Policy: what 'good' means
here (the DM-deflection problem)"): a naive retrieval-grounded agent
would learn to mimic the dataset's extremely common "sorry, please DM
us" deflection for EVERYTHING, since that's what a lot of the retrieved
grounding examples literally say. Explicit split, passed to the model
as an instruction rather than left for it to infer from the examples:
  - informational intents (how_to_usage, feature_request): must give
    the real answer/doc pointer -- deflecting to DM is a failure here.
  - account-specific/PII intents (data_loss_recovery,
    billing_subscription, account_access, security_account_compromise):
    DM is expected, but the reply must say exactly what to send
    (ticket number, account email), not a bare "please DM us".

Independent of the golden-set hand-labeling -- doesn't read
golden_labels.csv, only dropbox_paired.csv (retrieval corpus) and
whatever intent the classifier or the caller supplies. Safe to run
before labeling finishes.

Usage:
  ./.venv/Scripts/python.exe scripts/09_draft_reply.py --tweet-id 722441
  ./.venv/Scripts/python.exe scripts/09_draft_reply.py --text "..." --intent account_access
"""
import argparse
import re
import sys

import pandas as pd

from retrieval import ReplyRetriever
from reply_guard import validate, sanitize, regeneration_note
from groq_lib import make_client, MODEL
from groq import APIError, APIConnectionError, APITimeoutError, RateLimitError

# Same split logic as escalation.py's INTENT_DEFAULTS, but this is a
# drafting-behavior split, not an auto-handle/escalate split -- kept
# separate because they answer different questions (how do I answer vs.
# do I answer at all) and could diverge later (e.g. an escalated ticket
# still gets an acknowledgment reply drafted for the human to review).
INFORMATIONAL_INTENTS = {"how_to_usage", "feature_request"}
ACCOUNT_SPECIFIC_INTENTS = {
    "data_loss_recovery", "billing_subscription", "account_access",
    "security_account_compromise", "storage_quota_plan_limits",
}

DRAFT_SYSTEM_PROMPT = """You are drafting a public Twitter reply for @DropboxSupport to a customer.

You are given examples of how DropboxSupport has actually replied to similar past customer messages. Use them to match the brand's tone and typical resolution pattern -- do not invent a generic corporate-support voice.

Policy (read carefully -- the examples below over-represent one failure mode):
{policy_line}

General rules:
- Stay under 280 characters (it's a tweet).
- Do not invent specific facts (account details, ticket numbers, dates) not present in the customer's message.
- Write the finished reply text only. NEVER emit a placeholder or template slot of any kind -- no "[Name]", "[Your Name]", "@username", "@123456", "XYZ", or a made-up link. If you don't know something, leave it out and write a sentence that reads correctly without it.
- Do NOT begin the reply with an @handle. The posting client adds the recipient. Start with the greeting or the substance.
- Do NOT address the customer by name. Their name is not reliably known, and the examples' first names belong to the original customers, not to you. "Hi there," or no greeting at all is correct.

Respond with ONLY this JSON shape, no other text:
{{"reply": "<the drafted reply text>", "note": "<one sentence on what you grounded it in or any info still needed from the customer>"}}"""

POLICY_INFORMATIONAL = (
    "This is an informational/how-to request. Give the actual answer or point to the specific "
    "feature/setting, even if the examples below mostly show a DM deflection -- deflecting to DM "
    "here would be a failure, not a valid reply."
)
POLICY_ACCOUNT_SPECIFIC = (
    "This is an account-specific or PII-adjacent request. Asking the customer to DM is correct "
    "(their identity/account details can't be verified in a public reply), but say EXACTLY what "
    "to send (e.g. \"DM us your ticket number and the email on the account\") -- a bare \"please DM us\" "
    "is not acceptable."
)
POLICY_DEFAULT = (
    "Give a direct, helpful reply grounded in the examples below. Only ask the customer to DM if "
    "the examples show that's genuinely how this kind of issue gets resolved, and say what to send."
)


def _policy_line(intent: str | None) -> str:
    if intent in INFORMATIONAL_INTENTS:
        return POLICY_INFORMATIONAL
    if intent in ACCOUNT_SPECIFIC_INTENTS:
        return POLICY_ACCOUNT_SPECIFIC
    return POLICY_DEFAULT


_LEADING_HANDLES_RE = re.compile(r"^(?:\s*@\w+)+\s*")


def _strip_leading_handles(text: str) -> str:
    """Drop the @handle prefix from a grounding example.

    Every brand_text in the dataset starts with the (anonymized, numeric)
    handle it was replying to -- "@549821 Hey Gustaf, ...". Shown raw, the
    model imitates the SHAPE of that prefix and invents one, which is where
    the "@123456" in 12% of drafted replies came from. The prefix carries
    no information the drafter needs: the posting client addresses the
    recipient, not the text."""
    return _LEADING_HANDLES_RE.sub("", str(text)).strip()


def build_user_prompt(customer_text: str, neighbors: pd.DataFrame) -> str:
    examples = "\n".join(
        f'{i+1}. Customer: "{_strip_leading_handles(row.customer_text)}"\n'
        f'   DropboxSupport: "{_strip_leading_handles(row.brand_text)}"'
        for i, row in enumerate(neighbors.itertuples())
    ) if len(neighbors) else "(no sufficiently similar past examples found -- draft from general Dropbox support knowledge)"
    return f'Past similar exchanges:\n{examples}\n\nCustomer message to reply to now:\n"""{_strip_leading_handles(customer_text)}"""'


def draft_reply(client, customer_text: str, intent: str | None, retriever: ReplyRetriever,
                 exclude_tweet_id=None, k: int = 3) -> dict:
    import json
    neighbors = retriever.top_k(customer_text, k=k, exclude_tweet_id=exclude_tweet_id)
    system = DRAFT_SYSTEM_PROMPT.format(policy_line=_policy_line(intent))
    user = build_user_prompt(customer_text, neighbors)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.3,  # a little more than the classifier's temperature=0 -- drafting benefits
        # from some variation, unlike picking 1-of-13 categories, but 0.3 keeps it from
        # rambling off the grounding examples.
    )
    result = json.loads(response.choices[0].message.content)

    # Deterministic placeholder guard. DRAFT_SYSTEM_PROMPT already asks for
    # no placeholders; this is the check that does not depend on the model
    # having complied, because "@123456" reached 12% of drafts while that
    # instruction was already in place.
    #
    # Regenerate ONCE with the offending text quoted back, then fall back to
    # removal. Never substitute -- inventing a name or ticket number to fill
    # a slot is a worse failure than the slot.
    reply = result.get("reply", "")
    ok, hits = validate(reply)
    if not ok:
        retry = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": json.dumps(result)},
                {"role": "user", "content": regeneration_note(hits)},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        result = json.loads(retry.choices[0].message.content)
        result["_guard_regenerated"] = True
        result["_guard_first_hits"] = [h["match"] for h in hits]

        reply = result.get("reply", "")
        ok, hits = validate(reply)
        if not ok:
            cleaned, ok = sanitize(reply)
            result["reply"] = cleaned
            result["_guard_sanitized"] = True
            result["_guard_hits"] = [h["match"] for h in hits]

    # False means the draft still is not safe to send. The caller must check
    # it rather than assume the guard fixed things -- some placeholders
    # cannot be removed without destroying the sentence.
    result["_guard_ok"] = ok
    result["_neighbors"] = neighbors
    return result


def main():
    parser = argparse.ArgumentParser(description="Draft a grounded reply for a customer tweet.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--tweet-id", type=int, help="customer_tweet_id from dropbox_paired.csv")
    src.add_argument("--text", type=str, help="arbitrary customer text")
    parser.add_argument("--intent", type=str, default=None, help="intent name (see intents.json); affects DM-deflection policy")
    parser.add_argument("-k", type=int, default=3, help="number of retrieved grounding examples")
    args = parser.parse_args()

    retriever = ReplyRetriever()
    exclude_id = None
    if args.tweet_id is not None:
        row = retriever.df[retriever.df["customer_tweet_id"] == args.tweet_id]
        if row.empty:
            print(f"tweet_id {args.tweet_id} not found in {retriever.__class__.__module__}'s data", file=sys.stderr)
            sys.exit(1)
        customer_text = row.iloc[0]["customer_text"]
        exclude_id = args.tweet_id
    else:
        customer_text = args.text

    client = make_client()
    try:
        result = draft_reply(client, customer_text, args.intent, retriever, exclude_tweet_id=exclude_id, k=args.k)
    except (APIError, APIConnectionError, APITimeoutError, RateLimitError) as e:
        print(f"Groq call failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Customer: {customer_text}")
    print(f"Intent (given): {args.intent or '(none given)'}\n")
    print("Grounded on:")
    for _, n in result["_neighbors"].iterrows():
        print(f"  sim={n['similarity']:.3f}  \"{n['brand_text']}\"")
    print(f"\nDraft reply: {result['reply']}")
    print(f"Note: {result['note']}")


if __name__ == "__main__":
    main()
