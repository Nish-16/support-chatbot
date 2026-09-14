"""
The v2.1 boundary rules (taxonomy.md Part 11 and its addendum), in the one form
both consumers of the taxonomy read:

  groq_lib.build_system_prompt("p3")   the classifier
  08_label_golden_set.py               the human labeler

Part 11 said it plainly: "a rule that lives only here is a rule that does not
exist." Until 2026-09-13 the labeler saw a compressed ladder
(groq_lib.PRECEDENCE_RULES) and the production classifier saw no rules at all,
so a human and the model applied different procedures to the same tweet. This
module is the single source, and tests/test_taxonomy_rules.py fails if either
consumer stops showing it verbatim.

What this is NOT: a new taxonomy. The 13 intents are unchanged, no rule is new,
and turn_type and flags stay separate fields. Every line compresses a decision
already recorded in taxonomy.md and carries its codes, so the compression can
be checked against the source. Two things are deliberately absent: R9b
(on-topic but unroutable has no intent to assign, so a model cannot act on it;
the labeler footer covers it) and a tie-break for C3 (R3a vs R3b when a client
is named), which taxonomy.md leaves undecided.

Wording follows the p2 lesson recorded in groq_lib.py: mechanical rules
transferred to the model, abstract preferences ("prefer a specific intent over
how_to_usage") were applied bluntly and backfired. So every judgment rule names
its concrete cases, and every hard boundary has examples on both sides.

Examples come from the DEV split only, quote the tweet verbatim, and carry the
adjudicated v3 label -- the tests check all three against the data. The two
without a tweet_id are the illustrative cases from the 2026-09-13 task brief.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    codes: tuple[str, ...]  # the taxonomy.md Part 11 codes this line compresses
    text: str               # shown verbatim to the classifier and the labeler


@dataclass(frozen=True)
class Example:
    boundary: str           # key into BOUNDARIES
    text: str               # verbatim excerpt of the customer tweet
    intent: str             # the adjudicated (v3) label -- the side that wins
    not_intent: str         # the nearby intent that loses
    why: str
    rule: str               # the Part 11 code that decides it
    tweet_id: str | None = None  # None only for illustrative examples


PREAMBLE = (
    "Boundary rules. Decide on the customer's goal -- the work support must do -- "
    "never on keywords or tone. Anger belongs in sentiment. turn_type and flags are "
    "separate fields and never change the intent."
)

STEP_1 = "Step 1 -- is there a topic under the tone? First match wins."
STEP_1_RULES = [
    Rule(("R1a", "C2"),
         "An answerable question or a request support can act on -> the intent for its "
         "subject, however angry the wording. how_to_usage only when no other intent covers "
         "that subject: a sharing how-to is sharing_permissions, cancelling is "
         "billing_subscription, reporting phishing is phishing_abuse_report."),
    Rule(("R1b",),
         "No request, but proposes or judges a specific feature, design, capability or "
         "packaging change -> feature_request."),
    Rule(("R1d", "R1e"),
         "No request, but asserts how the product or an account works in a way support could "
         "correct -> that topic. Does not apply when the message talks to other users rather "
         "than to Dropbox."),
    Rule(("R1c",),
         "No request and no proposed change; criticises the company, service, support, "
         "reliability or price -> complaint_dissatisfaction."),
]

STEP_2 = "Step 2 -- which topic."
STEP_2_RULES = [
    Rule(("R2",),
         "how_to_usage only if nothing is malfunctioning. \"How do I fix X\" where X is "
         "broken -> the intent for X."),
    Rule(("R3a", "R3b"),
         "Shared link, shared or team folder, membership or permission, including how-tos -> "
         "sharing_permissions. But when the failure on a shared object is quota -> "
         "storage_quota_plan_limits, and when it is load/save/sync/crash -> sync_app_bug."),
    Rule(("R4", "R5"),
         "Files deleted but the storage reading will not drop -> storage_quota_plan_limits, "
         "never sync_app_bug. Laptop or phone disk space -> never storage_quota_plan_limits: "
         "usually how_to_usage (selective sync), or sync_app_bug if something fails."),
    Rule(("R6a", "R6b"),
         "Storage and money together: the answer is a price, invoice, refund, purchase or "
         "upgrade -> billing_subscription; the answer is how storage is counted, why a quota "
         "is full, or a plan's limit -> storage_quota_plan_limits."),
    Rule(("R7a", "R7b"),
         "Cannot log in, reset a password, pass 2FA or prove ownership -> account_access. "
         "Signed in and wants to perform an operation, even on an account -> how_to_usage."),
    Rule(("R8a", "R8b"),
         "A device- or account-specific symptom -> sync_app_bug, even if they ask whether "
         "Dropbox is down (put service_outage in secondary_intent). service_outage only with "
         "no personal symptom, or evidence of wide scope such as the status page or a whole "
         "team affected. \"Anyone else?\" is not evidence."),
    Rule(("R9a", "C1"),
         "no_action_needed only if there is no support work even with the whole thread: "
         "thanks, praise, spam, jobs, off-topic, \"all good now\"."),
]
RULES = STEP_1_RULES + STEP_2_RULES

# code -> (heading, side A, side B). Each boundary must have a winning example
# on both sides; the tests enforce it.
BOUNDARIES: dict[str, tuple[str, str, str]] = {
    "A": ("Complaint vs. an actionable topic",
          "complaint_dissatisfaction", "how_to_usage"),
    "B": ("Feature request vs. complaint",
          "feature_request", "complaint_dissatisfaction"),
    "C": ("Account quota vs. sync and device disk",
          "storage_quota_plan_limits", "sync_app_bug"),
    "D": ("Storage vs. billing",
          "storage_quota_plan_limits", "billing_subscription"),
    "E": ("Sharing vs. how-to and client failures",
          "sharing_permissions", "how_to_usage"),
    "F": ("Account access vs. how-to",
          "account_access", "how_to_usage"),
    "G": ("Outage vs. device bug",
          "service_outage", "sync_app_bug"),
}

EXAMPLES = [
    # A -- complaint vs. an actionable topic
    Example("A", "This is ridiculous, how do I fix syncing?", "sync_app_bug", "complaint_dissatisfaction",
            "A request sits under the anger; anger goes in sentiment.", "R1a"),
    Example("A", "Dropbox is absolutely useless.", "complaint_dissatisfaction", "how_to_usage",
            "Nothing asked, nothing proposed, aimed at the service.", "R1c"),
    Example("A", "why did you remove the green check that signs everything is alright",
            "how_to_usage", "complaint_dissatisfaction",
            "Grievance phrasing, but it asks about a design change support can explain.", "R1a", "615215"),
    Example("A", "Has Dropbox Support always been this bad? Still no intelligent answer.",
            "complaint_dissatisfaction", "how_to_usage",
            "The question is rhetorical and about the support experience; there is nothing to answer.",
            "R1c", "810000"),
    Example("A", "Dropbox is a RIPOFF! They charge customers for TRIALS before they are OVER!!",
            "billing_subscription", "complaint_dissatisfaction",
            "Asserts how billing works; support can confirm or correct it.", "R1d", "2551167"),
    Example("A", "What am I supposed to do with this hidden menu? I can't access my shared folder!",
            "sharing_permissions", "complaint_dissatisfaction",
            "Opens with WTF, but a real access failure is the work item.", "R3a", "37940"),

    # B -- feature request vs. complaint
    Example("B", "I don't like the new @118189 tray icon, every time I see it I think there's something wrong with it",
            "feature_request", "complaint_dissatisfaction",
            "Judges a specific design element -- backlog input, not a retention signal.", "R1b", "1960138"),
    Example("B", "It would be great if they offered a little bit more space. I would pay more if it did.",
            "feature_request", "complaint_dissatisfaction",
            "Proposes a packaging change.", "R1b", "53189"),
    Example("B", "In Brazil is so expensive this service. You ougth to review this price.",
            "complaint_dissatisfaction", "feature_request",
            "Objects to price without proposing a product or packaging change; support cannot act on "
            "\"review this price\".", "R1c", "847909"),
    Example("B", "If you want even more fun, download the app for your computer and it syncs everything!",
            "complaint_dissatisfaction", "sync_app_bug",
            "Sarcasm aimed at another user, no change proposed; peer commentary is not a claim to correct.",
            "R1e", "2179658"),

    # C -- account quota vs. sync and device disk
    Example("C", "I have permanently deleted my deleted items, but still have \"ran out of Space\" notice. How do we fix it?",
            "storage_quota_plan_limits", "sync_app_bug",
            "The files are gone and the quota reading is not: quota accounting, not sync.", "R4", "1999298"),
    Example("C", "I moved a shared folder to my desktop, edited files, then synced to a different acnt. changes reverted, how do I fix this?",
            "sync_app_bug", "storage_quota_plan_limits",
            "Changes reverted after a sync -- a malfunction, and no space or quota is involved.", "R2", "988433"),
    Example("C", "I don't seem to have enough space on my laptop to connect my work Dropbox.",
            "how_to_usage", "storage_quota_plan_limits",
            "The full disk is the laptop's; the answer is selective sync, not the plan.", "R5", "2976667"),

    # D -- storage vs. billing
    Example("D", "the cost of the Plan that would give me unlimited storage",
            "billing_subscription", "storage_quota_plan_limits",
            "The answer is a price.", "R6a", "1992403"),
    Example("D", "I need more space and refuse to pay for 5 users on Business. What are my options?",
            "billing_subscription", "storage_quota_plan_limits",
            "The answer is which plan to buy -- a purchase decision.", "R6a", "1495052"),
    Example("D", "one shared folder uses up my entire quota is not cool.",
            "storage_quota_plan_limits", "billing_subscription",
            "The answer is how storage is counted; nothing is being bought.", "R1d", "622045"),

    # E -- sharing vs. how-to and client failures
    Example("E", "how can I eliminate a Team folder?",
            "sharing_permissions", "how_to_usage",
            "A how-to, but its subject is a team folder, and sharing claims its own how-tos.", "R3a", "259327"),
    Example("E", "can a read only version be shared with clients ?",
            "sharing_permissions", "how_to_usage",
            "A permissions question about sharing.", "R3a", "1972451"),
    Example("E", "i can't open a shared link. Just keeps loading the files but never shows them",
            "sync_app_bug", "sharing_permissions",
            "The object is shared, but the failure is loading, not permission.", "R3b", "1600556"),
    Example("E", "I don't seem to have enough space on my laptop to connect my work Dropbox.",
            "how_to_usage", "sharing_permissions",
            "Mentions a work Dropbox, but the subject is laptop disk space.", "R5", "2976667"),

    # F -- account access vs. how-to
    Example("F", "I accidentally unlinked my accounts and now I can't get into that account again.",
            "account_access", "how_to_usage",
            "Unlinking locked them out: they cannot reach the account.", "R7a", "843989"),
    Example("F", "Why is the web browser displaying 'too many login attempts' even on my first login? I can't access dropbox website at all.",
            "account_access", "how_to_usage",
            "Authentication is failing.", "R7a", "2119014"),
    Example("F", "I am getting notifications from my work account on my home computer. How can I unlink their account from home?",
            "how_to_usage", "account_access",
            "Signed in throughout; wants to perform an operation on an account.", "R7b", "2222820"),

    # G -- outage vs. device bug
    Example("G", "is dropbox down? we cant access it",
            "service_outage", "sync_app_bug",
            "No device or account detail; \"can't access it\" restates the outage question.", "R8b", "799566"),
    Example("G", "This is not the moment to stop working",
            "service_outage", "sync_app_bug",
            "Asserts the service stopped, with no personal symptom.", "R8b", "1383896"),
    Example("G", "Anyone else having sync issues with #Dropbox .. client is making changes to shared sheet, I can't see them",
            "sync_app_bug", "service_outage",
            "A specific, recurring personal symptom; polling strangers is not evidence of scope.", "R8a", "512851"),
]

LABELER_FOOTER = (
    "Labeler only: an on-topic tweet you cannot route without the earlier message is NOT "
    "no_action_needed (taxonomy.md R9b). Pick the closest topic and write insufficient_context "
    "in the note, so it can be excluded from scoring the way golden_labels_v3.csv's v3_status does.\n"
    "Full reasoning, counterexamples and open cases: taxonomy.md Part 11."
)


def render_rules() -> str:
    out, n = [PREAMBLE], 0
    for heading, rules in ((STEP_1, STEP_1_RULES), (STEP_2, STEP_2_RULES)):
        out.append(heading)
        for rule in rules:
            n += 1
            out.append(f"{n}. {rule.text}")
    return "\n".join(out)


def render_examples(with_why: bool) -> str:
    out = ["Boundary examples (the winning intent, then the nearby intent it beats):"]
    for key, (heading, _, _) in BOUNDARIES.items():
        out.append(f"{heading}:")
        for ex in (e for e in EXAMPLES if e.boundary == key):
            out.append(f'- "{ex.text}" -> {ex.intent}, not {ex.not_intent}')
            if with_why:
                source = f"tweet {ex.tweet_id}" if ex.tweet_id else "illustrative"
                out.append(f"    why: {ex.why} [{ex.rule}, {source}]")
    return "\n".join(out)


RULES_TEXT = render_rules()
# What the classifier gets (prompt p3) ...
PROMPT_BLOCK = f"{RULES_TEXT}\n\n{render_examples(with_why=False)}"
# ... and what the labeler gets: the same rules and examples, plus each reason.
LABELER_BLOCK = f"{RULES_TEXT}\n\n{render_examples(with_why=True)}\n\n{LABELER_FOOTER}"
