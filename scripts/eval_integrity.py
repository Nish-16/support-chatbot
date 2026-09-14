"""
Frozen fingerprints of the evaluation state, and every tweet id that state has
already touched. Shared by 28_p3_eval.py, which refuses to run on a changed
state, and by tests/, which fail on one.

Pinned 2026-09-14, before the p1-vs-p3 evaluation set was built. Hashes are of
file bytes with CRLF folded to LF, so a git checkout that rewrites line endings
does not count as a change. The classification cache is append-only, so only
the lines that existed at pin time are fingerprinted -- new classifications
may be appended after them, but nothing before them may move.
"""
import csv
import hashlib
import json
import os
import re

import pandas as pd

# sha256 of groq_lib.build_system_prompt("p1"): the production prompt, taken
# before p3 existed.
P1_PROMPT_SHA256 = "7c7f00c655bc727970b1aa12b4e56514b87eb3122828044539b5e9643f0d98bf"

FROZEN_FILES = {
    "data/golden_labels.csv": "300f6cf5df292a09cee4f343fcbc88b1ccd0dc8172eccb71da05d556822fbc65",
    "data/golden_labels_v2.csv": "0f7753975e222aff0f94e64211f42e17e7c6cc4432ffc9cebda47e29f2335bc6",
    "data/golden_labels_v3.csv": "39222c297d566c51555d433614e723f3e11db9cf272fe17a328e40a2c9441960",
    "data/golden_labels_v3_audit.csv": "a06823abdbc71793172babac011c17cf3ffcc8a1f403a1b6cc48de692af025c1",
    "data/golden_split.csv": "ac5d8151ff2786e4c413039eabc1cad0ac366cb47467f41d833b3f169b6e6ca5",
    "data/golden_candidates.csv": "d08572021552e02e8c392aaff0a0462abe40c80f1d25dda53f0844684d2fd88e",
    "data/adjudication_v1.csv": "0cf3ae57fa75fc368699a7d6512b8035b4be05cf2f33b959eff3bc19c656b20d",
    "data/adjudication_v2.csv": "413b74f03112c73daaf64dffbb9c2c92676fc990b00fbdd16f88f2606336f93b",
    "data/model_ab_openai_gpt-oss-120b.csv": "2c39ada03069542f7dc0241cf2e2df00248fe8cd5d6fa8408c93b2901d089d1c",
    "data/prompt_ab_p2_dev.csv": "ab40ab07110e33e660bd3325e5c4378dfc6f0da17e0eed8a5ce35f18dac4fef9",
    "data/reply_evals.csv": "4b6617760ba9b58e802196137213c84f16e9a7ce4da0ae51cacd71700c86c4e0",
    "data/escalation_uncertainty_holdout_s3.csv": "aa936bf16d463d91ce49dced5d67003599e59502265776c5a53415174593bc50",
    "data/escalation_signals_dev.csv": "ffbf478a8992c3aa49c0fa888c5c24fa94457cc13565548d7070f564334ec495",
    "data/escalation_signals_holdout_A+repeated_failure.csv": "778b50d338a3c10921ca39f6c489dc40856591567e516e2319d00e2f47751b7a",
    "data/golden_context.csv": "1fb337389cfb9f43d57d096afd2f636873b3ee1a132a97f124e9c286443f7e04",
    "data/judge_agreement.csv": "1dabc60a16d37345aa7e1c5d68c8423654440921a28f78ee086a814458473244",
}

CACHE_PATH = "data/classification_cache.jsonl"
CACHE_LINES_AT_PIN = 1943
CACHE_PREFIX_SHA256 = "1c239cea439cdf6cea64a7e45e905635d7f932063b9118bb30e3c54f65b709de"

# Two adjudicated dev rows whose recorded rule contradicts their label
# (taxonomy.md Part 12). Documented exceptions: never silently relabelled.
KNOWN_R7A_EXCEPTIONS = {"1000504": "security_account_compromise", "1057223": "security_account_compromise"}

PAIRED_PATH = "data/dropbox_paired.csv"
READING_SAMPLE_PATH = "data/reading_sample.txt"
# taxonomy.md Part 8: the coverage samples read while designing v2, drawn from
# the unique customer tweets as (random_state, n).
COVERAGE_SAMPLES = [(999, 30), (20260910, 40)]

# Every data file that names tweets someone has read, labelled, classified or
# scored. A tweet id in any of them is not "new".
USED_ID_SOURCES = [
    "data/golden_labels.csv", "data/golden_labels_v1_backup.csv", "data/golden_labels_v2.csv",
    "data/golden_labels_v2_audit.csv", "data/golden_labels_v3.csv", "data/golden_labels_v3_audit.csv",
    "data/golden_split.csv", "data/golden_candidates.csv", "data/golden_candidates_251_backup.csv",
    "data/golden_candidates_v1_backup.csv", "data/classified_sample.csv",
    "data/classified_sample_gemini_backup.csv", "data/sample.csv", "data/reply_evals.csv",
    "data/reply_evals_claude.csv", "data/reply_evals_pre_retriever_backup.csv",
    "data/model_ab_openai_gpt-oss-120b.csv", "data/prompt_ab_p2_dev.csv", "data/adjudication_v1.csv",
    "data/adjudication_v2.csv", "data/judge_agreement.csv", "data/escalation_uncertainty_holdout_s3.csv",
    "data/escalation_signals_dev.csv", "data/escalation_signals_holdout_A+repeated_failure.csv",
    "data/golden_context.csv",
]
ID_COLUMNS = ("customer_tweet_id", "tweet_id", "parent_tweet_id")

# Documents whose quoted tweets informed the taxonomy and its rules.
DOC_SOURCES = [
    "taxonomy.md", "ADJUDICATION.md", "README.md", "README_SIMPLE.md", "WHAT_WENT_WRONG.md",
    "intents.md", "golden_set.md", "REPORT.md", "ISSUES_AND_DECISIONS.md", "PROGRESS.md", "SCRIPTS.md",
]
SHINGLE_CHARS = 40


def _normalized_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read().replace(b"\r\n", b"\n")


def file_sha256(path: str) -> str:
    return hashlib.sha256(_normalized_bytes(path)).hexdigest()


def cache_lines(path: str = CACHE_PATH) -> list[bytes]:
    return [line for line in _normalized_bytes(path).split(b"\n") if line.strip()]


def cache_prefix_sha256(n_lines: int, path: str = CACHE_PATH) -> str:
    return hashlib.sha256(b"\n".join(cache_lines(path)[:n_lines])).hexdigest()


def frozen_state_problems() -> list[str]:
    problems = []
    for path, expected in FROZEN_FILES.items():
        if not os.path.exists(path):
            problems.append(f"{path} is missing")
        elif file_sha256(path) != expected:
            problems.append(f"{path} changed since it was pinned")
    if len(cache_lines()) < CACHE_LINES_AT_PIN:
        problems.append(f"{CACHE_PATH} lost lines (had {CACHE_LINES_AT_PIN})")
    elif cache_prefix_sha256(CACHE_LINES_AT_PIN) != CACHE_PREFIX_SHA256:
        problems.append(f"{CACHE_PATH}: a line that existed at pin time was changed")
    return problems


def unique_customer_tweets() -> pd.DataFrame:
    """One row per customer tweet -- the same collapse taxonomy.md Part 8 sampled from."""
    paired = pd.read_csv(PAIRED_PATH, dtype={"customer_tweet_id": str})
    return paired.drop_duplicates("customer_tweet_id").reset_index(drop=True)


def used_tweet_ids(n_cache_lines: int) -> dict[str, set[str]]:
    """source -> tweet ids it touched. Only the first n_cache_lines of the cache
    count, so predictions made for the new set later do not mark it as used."""
    used: dict[str, set[str]] = {}
    for path in USED_ID_SOURCES:
        if not os.path.exists(path):
            continue
        # csv, not pandas: reply_evals_pre_retriever_backup.csv has 21 ragged
        # rows, which pandas refuses and on_bad_lines="skip" would silently
        # drop -- letting their tweets leak into a "new" set. The id columns
        # come first, so every row still yields them.
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            positions = [header.index(c) for c in ID_COLUMNS if c in header]
            if not positions:
                raise ValueError(f"{path} has no tweet id column ({ID_COLUMNS}) -- update USED_ID_SOURCES")
            used[path] = {row[i] for row in reader for i in positions if i < len(row) and row[i]}
    used[CACHE_PATH] = {str(json.loads(line)["tweet_id"]) for line in cache_lines()[:n_cache_lines]}
    with open(READING_SAMPLE_PATH, encoding="utf-8") as f:
        used[READING_SAMPLE_PATH] = set(re.findall(r"customer_tweet_id=(\d+)", f.read()))
    unique = unique_customer_tweets()
    for seed, n in COVERAGE_SAMPLES:
        used[f"taxonomy.md Part 8 sample (random_state={seed})"] = set(
            unique.sample(n=n, random_state=seed).customer_tweet_id)
    return used


def normalize_for_match(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", str(text).lower())
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def shingles(text: str, k: int = SHINGLE_CHARS) -> set[str]:
    norm = normalize_for_match(text)
    if len(norm) <= k:
        return {norm} if norm else set()
    return {norm[i:i + k] for i in range(len(norm) - k + 1)}


def reference_shingles() -> set[str]:
    """Every 40-char window of the project documents and the taxonomy_rules
    examples. A tweet sharing one was quoted somewhere and is not new."""
    import taxonomy_rules
    out: set[str] = set()
    for path in DOC_SOURCES:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                out |= shingles(f.read())
    for example in taxonomy_rules.EXAMPLES:
        out |= shingles(example.text)
    return out
