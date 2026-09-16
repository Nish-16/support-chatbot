"""
Interactive demo: type a customer message, see what the agent does with it.

This is the only place the three stages run end to end on arbitrary input.
09_draft_reply.py drafts a reply but must be HANDED an intent (--intent) and
never consults escalation.py, so before this script there was no single command
that answered "what would the agent actually do with this message?".

Nothing here is new logic. It calls the same production functions the
evaluation scores, so what you see is what 12_evaluate.py measured:

  classify_message()  -> intent + turn_type + flags + confidence   (groq_lib)
  decide()            -> escalate / auto_handle / no_action + why  (escalation)
  draft_reply()       -> grounded reply + placeholder guard        (09_draft_reply)

Costs API calls (unlike 12_evaluate.py): 1 classify + 1 draft per message, plus
one extra draft call if the placeholder guard has to regenerate. Needs
GROQ_API_KEY in .env.

Usage:
  ./.venv/Scripts/python.exe scripts/30_demo.py                     # interactive
  ./.venv/Scripts/python.exe scripts/30_demo.py --text "..."        # one-shot
  ./.venv/Scripts/python.exe scripts/30_demo.py --text "..." --json # machine-readable
  ./.venv/Scripts/python.exe scripts/30_demo.py --no-reply          # classify+route only, 1 call
"""
import argparse
import importlib.util
import json
import os
import re
import sys

from groq import APIError, APIConnectionError, APITimeoutError, RateLimitError

from escalation import AUTO_HANDLE, ESCALATE, NO_ACTION, decide
from groq_lib import MODEL, classify_message, make_client
from vector_retrieval import get_retriever

# 09_draft_reply.py starts with a digit, so it cannot be imported by name.
# Same loader 29_p4_eval.py uses for 28_p3_eval.py.
_spec = importlib.util.spec_from_file_location(
    "draft_reply_mod", os.path.join(os.path.dirname(os.path.abspath(__file__)), "09_draft_reply.py"))
_draft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_draft)

ACTION_LABEL = {
    ESCALATE: "ESCALATE to a human",
    AUTO_HANDLE: "AUTO-HANDLE",
    NO_ACTION: "NO ACTION (not a support issue)",
}

# -- Courtesy reply for bare greetings ---------------------------------------
#
# "hi" classifies as no_action_needed, which is CORRECT: intents.json defines
# that intent as "praise, thanks, spam, sales/jobs, off-topic, or too
# fragmentary to route". The policy then drafts nothing, which is right for
# spam and job ads and looks broken for a greeting.
#
# This is deliberately NOT a fix in escalation.py or draft_reply():
#   * escalation.decide() still returns NO_ACTION, so the escalation metric in
#     12_evaluate.py (which scores `action == ESCALATE`, a boolean) is byte for
#     byte unaffected.
#   * draft_reply() is untouched, so 13_reply_eval.py -- which stratifies by
#     human intent and therefore samples no_action rows -- keeps drafting for
#     exactly the rows it did before, and the judged 3.60 cannot move.
#
# It is presentation, scoped to the demo, and reported as such. The real fix is
# splitting no_action_needed so greetings stop sharing a label with spam; that
# needs relabelling and belongs in the next-steps list, not here.
#
# Deterministic on purpose: a canned string costs no API call and cannot
# hallucinate, and the whole thing is testable offline (tests/test_demo_greeting.py).
COURTESY_REPLY = "Hi there! What can we help you with today?"

_GREETING_WORDS = {
    "hi", "hii", "hiii", "hello", "helo", "hey", "heyy", "heya", "hiya", "yo",
    "howdy", "sup", "greetings", "morning", "afternoon", "evening", "gm",
}
# Words that may accompany a greeting without adding a request.
_GREETING_FILLER = {
    "there", "all", "team", "guys", "folks", "everyone", "good", "dropbox",
    "dropboxsupport", "please", "pls", "a", "you",
}
_HANDLE_OR_URL = re.compile(r"(?:@\w+|https?://\S+)")
_WORD = re.compile(r"[a-z']+")
# 4 words covers "hi there dropbox team"; beyond that it is carrying content.
_MAX_GREETING_WORDS = 4


def is_bare_greeting(text: str) -> bool:
    """True only for a message that is a greeting and nothing else.

    Must never fire on a greeting that also carries a request -- "hi, my files
    won't sync" is a sync_app_bug and has to reach the real drafter."""
    stripped = _HANDLE_OR_URL.sub(" ", str(text).lower())
    words = _WORD.findall(stripped)
    if not words or len(words) > _MAX_GREETING_WORDS:
        return False
    if not any(w in _GREETING_WORDS for w in words):
        return False
    return all(w in _GREETING_WORDS or w in _GREETING_FILLER for w in words)


def run_once(client, text: str, retriever, want_reply: bool = True, k: int = 3,
             courtesy: bool = True) -> dict:
    """Classify -> route -> (optionally) draft. Returns everything, prints nothing.

    `courtesy` answers a bare greeting with a fixed string instead of silence;
    see COURTESY_REPLY. It changes nothing the evaluation measures."""
    # Pre-filter, BEFORE the classifier sees it. A bare greeting carries no
    # intent to classify, and asking anyway produces a low-confidence guess that
    # trips escalation.decide()'s `confidence < 0.5` override -- which escalates
    # "hey" to a human. Short-circuiting here costs no API call and cannot
    # escalate. Nothing the evaluation scores reaches this path: no golden row
    # is a bare greeting (tests/test_demo_greeting.py pins that).
    if courtesy and is_bare_greeting(text):
        return {
            "customer_text": text,
            "prefiltered": "bare_greeting",
            "intent": None,
            "confidence": None,
            "turn_type": None,
            "flags": {},
            "action": NO_ACTION,
            "action_reason": "Bare greeting -- answered by a deterministic pre-filter, no classifier call.",
            "reply": COURTESY_REPLY,
            "reply_safe": True,
            "courtesy_reply": True,
        }

    result = classify_message(client, text)
    decision = decide(result)

    out = {
        "customer_text": text,
        "intent": result["intent"],
        "secondary_intent": result.get("secondary_intent"),
        "turn_type": result.get("turn_type"),
        "confidence": result.get("confidence"),
        "intent_reason": result.get("reason"),
        "flags": {f: result.get(f) for f in
                  ("wants_human", "legal_sensitive", "churn_threat", "abusive_content",
                   "needs_human_triage") if result.get(f)},
        "sentiment": result.get("sentiment"),
        "language": result.get("language"),
        "action": decision.action,
        "action_reason": decision.reason,
    }

    # A no_action message has nothing to reply to, and drafting one would spend
    # a call to produce a reply the policy says not to send.
    if want_reply and decision.action != NO_ACTION:
        draft = _draft.draft_reply(client, text, result["intent"], retriever, k=k)
        out["reply"] = draft.get("reply")
        out["reply_note"] = draft.get("note")
        out["reply_safe"] = draft.get("_guard_ok", True)
        out["grounded_on"] = [
            {"similarity": round(float(n["similarity"]), 3), "past_reply": str(n["brand_text"])}
            for _, n in draft["_neighbors"].iterrows()
        ]
        if draft.get("_guard_regenerated"):
            out["guard"] = f"regenerated once (found {draft.get('_guard_first_hits')})"
        if draft.get("_guard_sanitized"):
            out["guard"] = f"sanitised (still had {draft.get('_guard_hits')})"
    return out


def print_result(r: dict, show_grounding: bool = True) -> None:
    if r.get("prefiltered"):
        print(f"\n  pre-filter  {r['prefiltered']} -- classifier not called")
        print(f"\n  -> {ACTION_LABEL.get(r['action'], r['action'])}")
        print(f"     {r['action_reason']}")
        print(f"\n  courtesy reply:\n    {r['reply']}")
        print("  (fixed greeting response -- outside the evaluated policy, no API call)\n")
        return

    conf = r.get("confidence")
    conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf)
    print(f"\n  intent      {r['intent']}  (confidence {conf_s})")
    if r.get("secondary_intent"):
        print(f"  secondary   {r['secondary_intent']}")
    print(f"  turn_type   {r.get('turn_type')}")
    if r.get("flags"):
        print(f"  flags       {', '.join(k for k in r['flags'])}")
    if r.get("intent_reason"):
        print(f"  why         {r['intent_reason']}")

    print(f"\n  -> {ACTION_LABEL.get(r['action'], r['action'])}")
    print(f"     {r['action_reason']}")

    if "reply" in r:
        if show_grounding and r.get("grounded_on"):
            print("\n  grounded on past replies:")
            for n in r["grounded_on"]:
                snippet = " ".join(str(n["past_reply"]).split())[:100]
                print(f"    sim={n['similarity']:.3f}  \"{snippet}\"")
        label = "courtesy reply" if r.get("courtesy_reply") else "draft reply"
        print(f"\n  {label}:\n    {r['reply']}")
        if r.get("courtesy_reply"):
            print("  (fixed greeting response -- outside the evaluated policy, no API call)")
        if r.get("reply_note"):
            print(f"  note: {r['reply_note']}")
        if r.get("guard"):
            print(f"  guard: {r['guard']}")
        if not r.get("reply_safe", True):
            print("  WARNING: the placeholder guard could not make this draft safe to send.")
    elif r["action"] == NO_ACTION:
        print("\n  (no reply drafted -- policy says this needs none)")
    print()


BANNER = """DropboxSupport agent demo -- type a customer message, see what the agent does.
Each message costs API calls. Ctrl-C or an empty line to quit.
"""


def main():
    p = argparse.ArgumentParser(description="Interactive end-to-end demo of the support agent.")
    p.add_argument("--text", help="classify and answer one message, then exit")
    p.add_argument("--no-reply", action="store_true", help="classify and route only (1 API call, no drafting)")
    p.add_argument("--no-courtesy", action="store_true",
                   help="stay silent on a bare greeting, matching the evaluated policy exactly")
    p.add_argument("--json", action="store_true", help="emit JSON instead of the formatted view")
    p.add_argument("-k", type=int, default=3, help="grounding examples to retrieve (default 3)")
    p.add_argument("--retriever", choices=["tfidf", "vector"], default="tfidf",
                   help="tfidf (default, no downloads) or vector (needs vector_retrieval.py --build)")
    args = p.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if "GROQ_API_KEY" not in os.environ or not os.environ["GROQ_API_KEY"]:
        print("GROQ_API_KEY is not set. Put it in .env (see .env.example).\n"
              "Evaluation needs no key -- only this demo and the other live calls do.",
              file=sys.stderr)
        sys.exit(1)

    client = make_client()
    # TF-IDF fits its matrix at construction, so build the retriever once and
    # reuse it across turns rather than per message.
    retriever = None if args.no_reply else get_retriever(args.retriever)

    def handle(text: str) -> None:
        try:
            r = run_once(client, text, retriever, want_reply=not args.no_reply, k=args.k,
                         courtesy=not args.no_courtesy)
        except (APIError, APIConnectionError, APITimeoutError, RateLimitError) as e:
            print(f"Groq call failed: {e}", file=sys.stderr)
            return
        if args.json:
            print(json.dumps(r, indent=2, ensure_ascii=False))
        else:
            print_result(r)

    if args.text:
        handle(args.text)
        return

    print(BANNER)
    print(f"model: {MODEL}   retriever: {args.retriever if retriever else '(none, --no-reply)'}\n")
    while True:
        try:
            text = input("customer> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            return
        handle(text)


if __name__ == "__main__":
    main()
