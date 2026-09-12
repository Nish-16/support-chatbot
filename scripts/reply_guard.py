"""
Deterministic placeholder guard for drafted replies.

DRAFT_SYSTEM_PROMPT already tells the model not to emit placeholders. That
instruction is necessary and not sufficient: the "@123456" failure reached
12% of drafts precisely because a prompt instruction is a request, not a
guarantee. Anything that must never reach a customer needs a check that
does not depend on the model complying.

Two-stage, in this order:

  1. validate()  -- find placeholders. If any, the caller regenerates once
                    with the offending text quoted back.
  2. sanitize()  -- last resort if regeneration still fails. Removes the
                    placeholder and repairs the surrounding whitespace and
                    punctuation. It NEVER invents a replacement: a reply
                    that says "DM us your ticket number" is correct, while
                    one that says "DM us your ticket number, Sarah" when we
                    do not know the name is a fabrication.

A placeholder that cannot be removed without breaking the sentence makes
the draft unusable -- ok=False, and the caller must not send it.
"""
import re

# Each pattern is (name, regex, why it matters). Ordered roughly by how
# often it showed up in drafted replies.
PLACEHOLDER_PATTERNS = [
    ("numeric_handle", re.compile(r"@\d{4,}"),
     "The dataset anonymizes handles to numbers (@118189). A drafted @123456 "
     "is a hallucinated mention that would tag a real, unrelated account."),
    ("username_token", re.compile(r"@(?:username|user|customer|name|handle)\b", re.I),
     "Literal template slot."),
    ("square_slot", re.compile(r"\[[^\]\n]{1,40}\]"),
     "[Name], [Your Name], [ticket number], [link]."),
    ("curly_slot", re.compile(r"\{\{?[^}\n]{1,40}\}?\}"),
     "{customer_name}, {{ticket}} -- template engine syntax."),
    ("angle_slot", re.compile(r"<[a-z_][a-z0-9_ ]{1,30}>", re.I),
     "<name>, <ticket id>."),
    ("anon_token", re.compile(r"__[a-z]+__"),
     "The dataset's own anonymization tokens (__email__) leaking into a reply."),
    ("xyz_filler", re.compile(r"\b(?:XYZ|ABC|TODO|TBD|FIXME|LOREM)\b", re.I),
     "Filler standing in for a real value."),
    # NARROW ON PURPOSE. A first draft of this matched
    # r"\b(insert|add|enter|your)\s+(your\s+)?(name|link|ticket|email|details)\b",
    # which flagged "DM us your ticket number and the email on the account"
    # -- the exact phrasing README.md's DM policy REQUIRES for
    # account-specific intents. A guard that fires on the target behaviour
    # is worse than no guard, so this now only catches an imperative aimed
    # at the writer ("insert name here"), never a request aimed at the
    # customer ("send us your ticket number").
    ("insert_instruction", re.compile(r"\b(?:insert|fill in|replace|enter|type|provide)\b"
                                      r"[^.!?\n]{0,30}\b(?:here|below|above)\b", re.I),
     "An instruction to the writer that was left in the output."),
]

# Checked only inside a URL: the drafter inventing a plausible-looking doc
# link is worse than omitting one, because it looks authoritative.
_URL_RE = re.compile(r"https?://[^\s)]+", re.I)
_ALLOWED_URL_HOSTS = ("dropbox.com", "help.dropbox.com", "status.dropbox.com")


def find_placeholders(text: str) -> list[dict]:
    """Every placeholder-like span in `text`, as {name, match, span, why}."""
    if not text:
        return []
    hits = []
    for name, pattern, why in PLACEHOLDER_PATTERNS:
        for m in pattern.finditer(text):
            hits.append({"name": name, "match": m.group(0), "span": m.span(), "why": why})

    for m in _URL_RE.finditer(text):
        url = m.group(0)
        host = re.sub(r"^https?://", "", url).split("/")[0].lower().lstrip("www.")
        if not any(host == h or host.endswith("." + h) for h in _ALLOWED_URL_HOSTS):
            hits.append({
                "name": "unverified_url", "match": url, "span": m.span(),
                "why": f"Link to {host!r}, which is not a known Dropbox domain -- "
                       f"the drafter inventing a doc URL is worse than omitting one.",
            })

    return sorted(hits, key=lambda h: h["span"])


def validate(text: str) -> tuple[bool, list[dict]]:
    """(ok, hits). ok=True means the text is safe to send as-is."""
    hits = find_placeholders(text)
    return (not hits), hits


def sanitize(text: str) -> tuple[str, bool]:
    """Remove placeholders without inventing anything. Returns (text, ok).

    ok=False means the result is still not sendable -- either a placeholder
    survived, or removing it left a sentence too damaged to send.
    """
    if not text:
        return text, False

    out = text
    for _ in range(len(PLACEHOLDER_PATTERNS) + 2):  # bounded; removals can expose new matches
        hits = find_placeholders(out)
        if not hits:
            break
        h = hits[0]
        start, end = h["span"]
        out = out[:start] + out[end:]

    # Repair the damage removal leaves behind: doubled spaces, a space
    # before punctuation, a dangling comma, an orphaned "Hi ," opener.
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\s+([,.!?;:])", r"\1", out)
    out = re.sub(r"([,;:])\s*([.!?])", r"\2", out)
    out = re.sub(r"^\s*[,;:]\s*", "", out)
    out = re.sub(r"\b(Hi|Hey|Hello)\s*[,!]\s*", r"\1 there, ", out)
    out = re.sub(r"\(\s*\)", "", out)
    out = re.sub(r"\s{2,}", " ", out).strip()

    still = find_placeholders(out)
    # A reply reduced to almost nothing is a failure even if it is "clean".
    too_short = len(out.split()) < 6
    # Cutting a noun out of "send it to {name}" leaves "send it to." --
    # clean by every pattern above and obviously unsendable. A dangling
    # preposition or article before punctuation or end-of-text is the
    # reliable tell, and it is why sanitize() reports ok separately rather
    # than just returning a string the caller trusts.
    # Two shapes, both produced by cutting a noun out of the middle:
    #   "...send it to."      preposition/article against punctuation or end
    #   "...your note at and" preposition immediately followed by a conjunction
    _DANGLERS = r"at|to|for|with|from|on|in|by|of|and|or|the|a|an"
    damaged = bool(
        re.search(rf"\b(?:{_DANGLERS})\s*(?=[.,!?;:]|$)", out, re.I)
        # "Please and we'll sort it out." -- a bare verb/adverb left
        # stranded against a conjunction. Same damage, different word class.
        or re.search(rf"\b(?:{_DANGLERS}|please|send|share|tell|give)\s+(?:and|or|but)\b",
                     out, re.I)
    )
    return out, (not still and not too_short and not damaged)


def regeneration_note(hits: list[dict]) -> str:
    """Corrective instruction for the retry, quoting what was wrong."""
    bad = ", ".join(sorted({repr(h["match"]) for h in hits}))
    return (
        "Your previous draft contained placeholder or unverifiable text: "
        f"{bad}. Rewrite it so the reply reads correctly with that text simply "
        "absent. Do not substitute a different placeholder, do not invent a "
        "name, ticket number, or link. If you do not know a value, write the "
        "sentence without it."
    )


if __name__ == "__main__":
    CASES = [
        ("Hi [Name], sorry about that! Please DM us your ticket number.", False),
        ("@123456 Sorry to hear that, please DM us the email on the account.", False),
        ("Hi there, DM us your ticket number and the email on the account and "
         "we'll take a look.", True),
        ("Send your details to {customer_name} and we will help.", False),
        ("Check https://help.dropbox.com/sync for the steps.", True),
        ("Check https://dropbox-help-center.example.com/fix for the steps.", False),
        ("We got your note at __email__ and will follow up.", False),
        ("Please enter your name below and we'll sort it out.", False),
    ]
    print(f"\n{'ok':>5s}  {'expected':>8s}  text")
    failures = 0
    for text, expected in CASES:
        ok, hits = validate(text)
        flag = "OK" if ok == expected else "MISMATCH"
        if ok != expected:
            failures += 1
        print(f"{str(ok):>5s}  {str(expected):>8s}  {flag:8s} {text[:58]}")
        if hits:
            print(f"          -> {', '.join(h['name'] + '=' + h['match'] for h in hits)}")

    print(f"\n-- sanitize --")
    for text, expected in CASES:
        if expected:
            continue
        cleaned, ok = sanitize(text)
        print(f"  sendable={str(ok):5s}  {cleaned!r}")

    print(f"\n{len(CASES) - failures}/{len(CASES)} validation cases pass\n")
