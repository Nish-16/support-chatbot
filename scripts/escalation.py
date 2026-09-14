"""
Auto-handle vs. escalate decision logic.

README.md's escalation matrix (section "Escalation matrix (default
rules, v1)") was written against the v1 8-intent taxonomy and never
updated when taxonomy v2 (13 intents + turn_type + flags, see
taxonomy.md) was adopted for the actual classifier/cache/golden set.
This module is the v2 version of that table -- same philosophy
(intent-based defaults, not open-ended LLM discretion, every decision
gets a stated reason), remapped onto the 13 v2 intent names and
extended to use the flags/turn_type v2 actually produces instead of
just intent+confidence.

Two-stage decision, checked in this order:
  1. Flag/turn_type overrides -- these fire regardless of intent,
     because they signal something the intent-only default table can't
     see (the model explicitly abstained, the customer already said
     the last answer didn't work, confidence is too low to trust any
     default). Checked first and short-circuit, so a message that
     matches multiple triggers reports the single most decision-
     relevant reason rather than an arbitrary one.
  2. Intent default -- the v2 remap of README's matrix, for anything
     that didn't hit an override.

Confidence threshold (0.5) is a placeholder, not a tuned value --
there's no eval harness yet to tune it against. Revisit once
data/golden_labels.csv is complete and the eval harness can measure
false-escalation / false-auto-handle rates at different thresholds.
"""
from dataclasses import dataclass

LOW_CONFIDENCE_THRESHOLD = 0.5

ESCALATE = "escalate"
AUTO_HANDLE = "auto_handle"
NO_ACTION = "no_action"  # not a support issue at all -- neither handled nor escalated


@dataclass
class Decision:
    action: str  # ESCALATE, AUTO_HANDLE, or NO_ACTION
    reason: str  # one sentence, for the same auditability reason groq_lib's `reason` field exists


# Intent default table -- v2 remap of README.md's v1 matrix. Each entry
# is (default_action, reason). `how_to_usage`/`feature_request` stay
# auto-handle per the README's explicit DM-deflection policy: these are
# exactly the intents where a deflected "please DM us" reply would be a
# failure, not a valid auto-handle, so the agent must give a real answer.
INTENT_DEFAULTS: dict[str, tuple[str, str]] = {
    "data_loss_recovery": (ESCALATE, "Permanent data-loss risk; needs manual account/backup inspection."),
    "billing_subscription": (ESCALATE, "Financial transaction / refund authorization needs human verification."),
    "security_account_compromise": (ESCALATE, "Active account compromise; needs human security review, not a scripted reply."),
    "phishing_abuse_report": (ESCALATE, "Abuse/phishing reports route to the trust & safety team, not support."),
    # Was AUTO_HANDLE until calibrated against the golden set (2026-09-12):
    # you escalate 89% of account_access rows, the auto-handle default got
    # 11% of them right. README's v1 table always said "escalate if
    # 2FA/compromise, auto-handle if plain password reset" -- collapsing
    # that to auto-handle was the error, and the labels say the
    # can't-get-in cases in this dataset skew overwhelmingly toward the
    # half that needs a human.
    "account_access": (ESCALATE, "Locked-out accounts need identity verification a public reply can't do."),
    "sync_app_bug": (AUTO_HANDLE, "Standard troubleshooting steps cover most sync/upload bugs."),
    "service_outage": (AUTO_HANDLE, "Status-page link and standard outage acknowledgment; not one user's account issue."),
    "sharing_permissions": (AUTO_HANDLE, "Standard sharing/permissions guidance."),
    "storage_quota_plan_limits": (AUTO_HANDLE, "Standard quota/upgrade guidance."),
    "how_to_usage": (AUTO_HANDLE, "Safe, deterministic product guidance -- must give the real answer, not a DM deflection."),
    "feature_request": (AUTO_HANDLE, "Acknowledge and log; no risk -- must give a real acknowledgment, not a DM deflection."),
    "complaint_dissatisfaction": (AUTO_HANDLE, "Empathetic acknowledgment; escalated separately if churn/abuse flags fire."),
    "no_action_needed": (NO_ACTION, "Praise, thanks, spam, or off-topic -- not a support issue."),
}


# Experimental overrides from groq_lib.ESCALATION_SIGNALS, tested in
# scripts/27_escalation_signals.py. OFF unless a caller passes them in
# `signal_triggers` -- the default policy, and every number 12_evaluate.py
# reports, is unchanged. Checked in this order, after the v1 overrides.
SIGNAL_REASONS: dict[str, str] = {
    "urgent": "Customer signals urgency or a deadline -- a scripted reply risks a costly delay.",
    "wide_impact": "Problem reaches a team, clients or many users -- impact beyond one account.",
    "repeated_failure": "Issue is long-running or already failed once -- the standard playbook has not worked.",
}


def decide(result: dict, signal_triggers: frozenset[str] = frozenset()) -> Decision:
    """result is a groq_lib.classify_message()-shaped dict: intent,
    secondary_intent, turn_type, confidence, reason, plus the FLAG_SPEC
    fields (wants_human, legal_sensitive, churn_threat, abusive_content,
    needs_human_triage, sentiment, language).

    signal_triggers names which SIGNAL_REASONS signals may escalate; empty
    means the current policy, exactly."""
    unknown = set(signal_triggers) - set(SIGNAL_REASONS)
    if unknown:
        raise ValueError(f"unknown signal trigger(s) {sorted(unknown)}; known: {list(SIGNAL_REASONS)}")
    intent = result["intent"]
    confidence = result.get("confidence", 0.0)
    turn_type = result.get("turn_type")

    # -- 1. Overrides (checked in priority order; first match wins) --
    if result.get("needs_human_triage"):
        return Decision(ESCALATE, "Model explicitly abstained on intent -- needs human triage.")
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return Decision(ESCALATE, f"Classifier confidence {confidence:.2f} below {LOW_CONFIDENCE_THRESHOLD} threshold.")
    if turn_type == "disputing_prior_answer":
        return Decision(ESCALATE, "Customer says the prior answer was wrong or didn't help; standard playbook already failed once.")
    if result.get("legal_sensitive"):
        return Decision(ESCALATE, "GDPR/CCPA, account closure, or legal/ToS dispute needs human sign-off.")
    if result.get("wants_human"):
        return Decision(ESCALATE, "Customer explicitly asked for a phone number, live chat, or a person.")
    if result.get("churn_threat"):
        return Decision(ESCALATE, "Customer threatened to cancel/switch -- retention risk needs human judgment.")
    if result.get("abusive_content"):
        return Decision(ESCALATE, "Abusive content directed at the brand/staff -- needs human de-escalation, not a scripted reply.")
    for signal, reason in SIGNAL_REASONS.items():
        if signal in signal_triggers and result.get(signal):
            return Decision(ESCALATE, reason)

    # -- 2. Intent default --
    if intent not in INTENT_DEFAULTS:
        return Decision(ESCALATE, f"Unrecognized intent {intent!r} -- escalate rather than guess.")
    action, reason = INTENT_DEFAULTS[intent]
    return Decision(action, reason)


if __name__ == "__main__":
    # Quick self-check against a few hand-built examples -- not a real
    # test suite, just enough to eyeball the policy before it's wired
    # into anything that spends API budget.
    examples = [
        {"intent": "data_loss_recovery", "confidence": 0.9, "turn_type": "first_contact",
         "needs_human_triage": False, "legal_sensitive": False, "wants_human": False,
         "churn_threat": False, "abusive_content": False},
        {"intent": "how_to_usage", "confidence": 0.95, "turn_type": "first_contact",
         "needs_human_triage": False, "legal_sensitive": False, "wants_human": False,
         "churn_threat": False, "abusive_content": False},
        {"intent": "sync_app_bug", "confidence": 0.88, "turn_type": "disputing_prior_answer",
         "needs_human_triage": False, "legal_sensitive": False, "wants_human": False,
         "churn_threat": False, "abusive_content": False},
        {"intent": "how_to_usage", "confidence": 0.4, "turn_type": "first_contact",
         "needs_human_triage": False, "legal_sensitive": False, "wants_human": False,
         "churn_threat": False, "abusive_content": False},
        {"intent": "complaint_dissatisfaction", "confidence": 0.9, "turn_type": "first_contact",
         "needs_human_triage": False, "legal_sensitive": False, "wants_human": False,
         "churn_threat": True, "abusive_content": False},
    ]
    for ex in examples:
        d = decide(ex)
        print(f"{ex['intent']:30s} -> {d.action:12s} ({d.reason})")
