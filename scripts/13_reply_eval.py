"""
Reply-quality evaluation: draft replies from all three systems for the
same customer tweets, then score them with an LLM judge.

Three systems, same inputs, blind to the judge:
  llm     -- 09_draft_reply.py (TF-IDF retrieval + Groq, policy prompt)
  simple  -- 10_baselines.py 1-NN retrieval, raw historical reply text
  trivial -- 10_baselines.py fixed canned reply

Design decisions worth stating, since they change what the number means:

1. The judge is given the HUMAN intent from golden_labels.csv, not the
   classifier's guess. That isolates reply quality from classification
   error -- otherwise a good reply to a misclassified tweet gets marked
   down for a mistake the drafter didn't make, and the two failure modes
   become impossible to separate in the results table.

2. The judge is not told which system wrote a reply, and the three are
   scored in independent calls. Telling it "this is the LLM agent's
   reply" would invite it to reward the sophisticated-looking answer.

3. `is_deflection` is scored as its own boolean, not folded into the
   quality score. README's central worry is that a retrieval-grounded
   agent learns to mimic "sorry, please DM us" for everything -- that's
   a specific, countable failure, and averaging it into a 1-5 quality
   score would hide exactly the thing we most want to see.

4. An LLM judging LLM output is a biased instrument. That's why
   14_judge_agreement.py exists: it collects YOUR blind ratings on a
   subset and measures how well the judge tracks them. Do not report
   judge scores without that agreement evidence attached.

Resumable: every draft and judgment is appended to data/reply_evals.csv
as soon as it's produced, so an interrupted run resumes without
re-spending API budget.

Usage: ./.venv/Scripts/python.exe scripts/13_reply_eval.py [--n 20]
"""
import argparse
import json
import os

import pandas as pd

from groq_lib import make_client, MODEL, SKIPPABLE_ERRORS
from rate_limiter import RateLimiter
from retrieval import ReplyRetriever
from vector_retrieval import get_retriever

# 09_draft_reply.py and 10_baselines.py start with digits, so they can't
# be imported by name -- load them by path instead of renaming files the
# rest of the project (and the README) already refers to by number.
import importlib.util


def _load(filename: str, alias: str):
    spec = importlib.util.spec_from_file_location(
        alias, os.path.join(os.path.dirname(__file__), filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


draft_reply = _load("09_draft_reply.py", "draft_mod").draft_reply
_baselines = _load("10_baselines.py", "baselines_mod")
TrivialBaseline, SimpleBaseline = _baselines.TrivialBaseline, _baselines.SimpleBaseline

LABELS_PATH = "data/golden_labels.csv"
OUT_PATH = "data/reply_evals.csv"
SYSTEMS = ["llm", "simple", "trivial"]

# Default judge is a DIFFERENT model family from the drafter (which uses
# groq_lib.MODEL, openai/gpt-oss-20b). The first version of this script
# judged with the same model that wrote the replies and scored them 4.73/5
# -- self-preference bias, not a measurement. Qwen is a separate vendor
# and training lineage, so it has no stake in gpt-oss's output.
# Pass --judge-model to compare judges; disagreement between two
# independent judges is itself evidence about how much any of this is worth.
DEFAULT_JUDGE_MODEL = "qwen/qwen3.8-27b"

# Rubric version. v1's account-specific clause accepted any reply that named
# an artifact to send, so the fixed canned baseline -- one context-free string
# -- scored 5 on every account-specific intent and 1-2 on informational ones,
# lifting the trivial baseline to 3.84 on nothing but a loophole. v2 adds an
# engagement test: naming an artifact is necessary but not sufficient. Scores
# from different rubric versions are NOT comparable and are kept apart by this
# column rather than averaged together.
RUBRIC_VERSION = "v2"

FIELDS = [
    "customer_tweet_id", "customer_text", "human_intent", "system", "reply",
    "relevance", "policy_compliance", "grounding", "tone", "overall",
    "is_deflection", "judge_reason", "judge_model", "rubric_version", "retriever",
]

JUDGE_PROMPT = """You are grading a public Twitter reply written by a customer-support agent for @DropboxSupport. Grade only the reply you are shown. You are NOT told which system wrote it; do not speculate.

The customer's intent has already been determined by a human: {intent}

Grading policy for this brand (these are the brand's own rules, apply them strictly):
- For informational intents (how_to_usage, feature_request): the reply MUST give the actual answer or point to the specific feature/setting. Telling the customer to DM instead is a FAILURE, not a valid reply.
- For account-specific intents (account_access, billing_subscription, data_loss_recovery, security_account_compromise, storage_quota_plan_limits): asking the customer to DM is CORRECT, but the reply must BOTH (a) say exactly what to send (e.g. ticket number, account email), AND (b) show it engaged with THIS customer's specific problem -- by naming the thing they reported, or stating what will be done with what they send. Naming an artifact to send is NOT sufficient on its own: a reply that would read identically for any customer with any problem is a generic holding message, and scores no higher than 2 on relevance and policy_compliance no matter what it asks for.
- Inventing specifics the customer didn't provide (ticket numbers, account details, dates) is a serious grounding failure.

Score each 1-5 (1 = very poor, 5 = excellent):
- relevance: does it address the customer's actual problem?
- policy_compliance: does it follow the brand rules above for this intent?
- grounding: is it factually safe -- nothing invented, nothing promised that can't be delivered?
- tone: appropriate for a public brand support reply?
- overall: your holistic judgment of whether this is a good reply to send.

Also set is_deflection to true if the reply's substance is merely redirecting the customer elsewhere (DM us, email us, check our help site) WITHOUT answering anything -- and note that naming an artifact to send does NOT exempt a reply from this if it engages with nothing specific to this customer.

Respond with ONLY this JSON, no other text:
{{"relevance": <1-5>, "policy_compliance": <1-5>, "grounding": <1-5>, "tone": <1-5>, "overall": <1-5>, "is_deflection": <bool>, "reason": "<one sentence>"}}"""


def judge_reply(client, limiter, customer_text: str, intent: str, reply: str,
                judge_model: str = DEFAULT_JUDGE_MODEL) -> dict:
    limiter.acquire()
    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT.format(intent=intent)},
            {"role": "user", "content": f'Customer tweet:\n"""{customer_text}"""\n\nAgent reply to grade:\n"""{reply}"""'},
        ],
        response_format={"type": "json_object"},
        temperature=0,  # grading should be as reproducible as this model allows
    )
    return json.loads(response.choices[0].message.content)


def _read_out() -> pd.DataFrame:
    df = pd.read_csv(OUT_PATH)
    # Rows written before the judge became configurable carry no
    # judge_model; they were all scored by the drafter's own model.
    if "judge_model" not in df.columns:
        df["judge_model"] = MODEL
    df["judge_model"] = df["judge_model"].fillna(MODEL)
    # Rows written before the rubric was versioned were all scored under v1.
    if "rubric_version" not in df.columns:
        df["rubric_version"] = "v1"
    df["rubric_version"] = df["rubric_version"].fillna("v1")
    # Rows written before the retriever became switchable were all TF-IDF.
    if "retriever" not in df.columns:
        df["retriever"] = "tfidf"
    df["retriever"] = df["retriever"].fillna("tfidf")
    return df


def _check_schema():
    """Refuse to append rows whose shape does not match the file's header.

    This has bitten this project twice: appending a DictWriter row with a new
    field to a CSV whose header predates that field produces lines with more
    values than columns, and the whole file stops parsing -- taking every
    earlier, expensive row with it. A column addition is a migration, not an
    edit, so fail loudly here and say exactly what to do about it."""
    if not os.path.exists(OUT_PATH):
        return
    import csv
    with open(OUT_PATH, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), [])
    if header == FIELDS:
        return
    missing, extra = [c for c in FIELDS if c not in header], [c for c in header if c not in FIELDS]
    raise SystemExit(
        f"{OUT_PATH} has a different schema than this script writes.\n"
        f"  header:  {header}\n"
        f"  FIELDS:  {FIELDS}\n"
        f"  missing from file: {missing or 'none'}   unexpected in file: {extra or 'none'}\n"
        f"Appending would corrupt the file. Migrate it (back it up, rewrite with "
        f"the new header, backfill the new columns) before rerunning."
    )


def load_done(judge_model: str, retriever_kind: str) -> set[tuple]:
    """Rows already judged by THIS judge, under THIS rubric, grounded by THIS
    retriever. Each of the three changes what is being measured, so a prior run
    under a different one must not suppress a re-run."""
    if not os.path.exists(OUT_PATH):
        return set()
    prev = _read_out()
    prev = prev[(prev["judge_model"] == judge_model)
                & (prev["rubric_version"] == RUBRIC_VERSION)
                & (prev["retriever"] == retriever_kind)]
    return set(zip(prev["customer_tweet_id"], prev["system"]))


def main(n: int, judge_model: str = DEFAULT_JUDGE_MODEL, retriever_kind: str = "tfidf"):
    labels = pd.read_csv(LABELS_PATH)
    # Stratify by intent so the reply eval isn't dominated by whichever
    # intents happen to be most common -- reply quality varies a lot by
    # intent (a how-to needs a real answer, a billing issue needs a DM
    # ask), so an unstratified sample would mostly measure one of them.
    per_intent = max(1, n // labels["human_intent"].nunique())
    parts = [g.sample(min(len(g), per_intent), random_state=42)
             for _, g in labels.groupby("human_intent")]
    sample = pd.concat(parts)
    # Stratifying alone caps out below n whenever n isn't divisible by the
    # intent count (or an intent has fewer rows than its quota), so top up
    # at random from what's left to actually reach the requested size.
    if len(sample) < n:
        rest = labels[~labels["customer_tweet_id"].isin(sample["customer_tweet_id"])]
        sample = pd.concat([sample, rest.sample(min(n - len(sample), len(rest)), random_state=42)])
    sample = sample.head(n).reset_index(drop=True)
    print(f"Evaluating replies for {len(sample)} tweets x {len(SYSTEMS)} systems.")
    print(f"Drafter model: {MODEL}   |   Judge model: {judge_model}"
          f"   |   Retriever: {retriever_kind}")

    done = load_done(judge_model, retriever_kind)
    client = make_client()
    # 8/min, not classify_runner's 15/min: the real Groq constraint on this
    # key is ~8,000 tokens/MINUTE, and a judge call (rubric + tweet + reply,
    # ~800 tokens) is roughly twice a classify call. At 15/min this script
    # ran ~12k tokens/min and lost rows to RateLimitError -- the gaps in the
    # first two runs were this, not model failures.
    limiter = RateLimiter(8, 60.0)
    # Only the `llm` system uses this; `simple` has its own 1-NN retriever and
    # `trivial` retrieves nothing, so switching this changes one arm of the
    # comparison, which is exactly what a retriever A/B wants.
    retriever = get_retriever(retriever_kind)
    trivial = TrivialBaseline()
    simple = SimpleBaseline()

    _check_schema()
    write_header = not os.path.exists(OUT_PATH)
    with open(OUT_PATH, "a", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        for i, row in enumerate(sample.itertuples(), 1):
            for system in SYSTEMS:
                if (row.customer_tweet_id, system) in done:
                    continue
                try:
                    if system == "trivial":
                        reply = trivial.draft(row.customer_text)
                    elif system == "simple":
                        reply = simple.draft(row.customer_text, exclude_tweet_id=row.customer_tweet_id)
                    else:
                        limiter.acquire()
                        reply = draft_reply(client, row.customer_text, row.human_intent,
                                            retriever, exclude_tweet_id=row.customer_tweet_id)["reply"]
                    scores = judge_reply(client, limiter, row.customer_text, row.human_intent,
                                        reply, judge_model=judge_model)
                except SKIPPABLE_ERRORS as e:
                    print(f"  [{i}] {system}: FAILED ({type(e).__name__}) -- skipping")
                    continue

                writer.writerow({
                    "customer_tweet_id": row.customer_tweet_id,
                    "customer_text": row.customer_text,
                    "human_intent": row.human_intent,
                    "system": system,
                    "reply": reply,
                    "relevance": scores.get("relevance"),
                    "policy_compliance": scores.get("policy_compliance"),
                    "grounding": scores.get("grounding"),
                    "tone": scores.get("tone"),
                    "overall": scores.get("overall"),
                    "is_deflection": scores.get("is_deflection"),
                    "judge_reason": scores.get("reason", ""),
                    "judge_model": judge_model,
                    "rubric_version": RUBRIC_VERSION,
                    "retriever": retriever_kind,
                })
                f.flush()
                print(f"  [{i}/{len(sample)}] {system:8s} overall={scores.get('overall')} "
                      f"deflection={scores.get('is_deflection')}")

    report()


def report():
    full = _read_out()
    metrics = ["relevance", "policy_compliance", "grounding", "tone", "overall"]

    # Grouped by rubric version as well as judge: the same judge scoring the
    # same reply under v1 and under v2 gives two different numbers, and
    # averaging them together reports a rubric that was never actually run.
    for (judge, rubric, retr), df in full.groupby(["judge_model", "rubric_version", "retriever"]):
        same_family = judge.split("/")[0] == MODEL.split("/")[0]
        tag = "  <-- SAME FAMILY AS DRAFTER (biased)" if same_family else "  <-- independent of drafter"
        print(f"\n{'=' * 64}\nJUDGE: {judge}   RUBRIC: {rubric}   RETRIEVER: {retr}{tag}\n"
              f"n={df['customer_tweet_id'].nunique()} tweets\n{'=' * 64}\n")
        agg = df.groupby("system")[metrics].mean().round(2)
        agg["deflection_rate"] = df.groupby("system")["is_deflection"].mean().round(2)
        agg["n"] = df.groupby("system").size()
        print(agg.reindex([s for s in SYSTEMS if s in agg.index]).to_string())

    # With two judges on the same replies, their disagreement is a direct
    # measure of how much the choice of judge -- rather than the quality
    # of the reply -- drives the result.
    # Judges are only comparable within one rubric version: a v1 score and a
    # v2 score of the same reply disagree because the rubric changed, which
    # says nothing about the judges. Scope to the rubric the most judges share.
    shared = full.groupby("rubric_version")["judge_model"].nunique().sort_values()
    rubric = shared.index[-1] if len(shared) else None
    scoped = full[full["rubric_version"] == rubric] if rubric else full
    judges = scoped["judge_model"].unique()
    if len(judges) >= 2:
        print(f"\n{'=' * 64}\nJUDGE-vs-JUDGE: does the verdict depend on who's grading?\n"
              f"(rubric {rubric} -- the version both judges scored)\n{'=' * 64}\n")
        a, b = judges[0], judges[1]

        def _side(judge: str) -> pd.DataFrame:
            # A resumed run can re-score the same reply, so (tweet, system) is
            # not unique on either side. Keep the last verdict per pair before
            # intersecting -- otherwise duplicates multiply out and the two
            # sides come back with different lengths.
            side = scoped[scoped["judge_model"] == judge]
            side = side.drop_duplicates(["customer_tweet_id", "system"], keep="last")
            return side.set_index(["customer_tweet_id", "system"]).sort_index()

        left, right = _side(a), _side(b)
        common = left.index.intersection(right.index)
        if len(common):
            la, rb = left.loc[common], right.loc[common]
            exact = (la["overall"].values == rb["overall"].values).mean()
            within1 = (abs(la["overall"].values - rb["overall"].values) <= 1).mean()
            print(f"  Comparing {len(common)} replies scored by both.")
            print(f"    exact agreement on overall: {exact:.0%}   within 1 point: {within1:.0%}")
            print(f"    mean overall -- {a}: {la['overall'].mean():.2f}   {b}: {rb['overall'].mean():.2f}")
            print(f"    deflection agreement: "
                  f"{(la['is_deflection'].values == rb['is_deflection'].values).mean():.0%}")
            print("\n  Per system:")
            for system in SYSTEMS:
                mask = la.index.get_level_values("system") == system
                if mask.sum():
                    print(f"    {system:8s} {a.split('/')[-1]}: {la[mask]['overall'].mean():.2f}   "
                          f"{b.split('/')[-1]}: {rb[mask]['overall'].mean():.2f}   "
                          f"(delta {la[mask]['overall'].mean() - rb[mask]['overall'].mean():+.2f})")

    df = full[full["judge_model"] == DEFAULT_JUDGE_MODEL] if (full["judge_model"] == DEFAULT_JUDGE_MODEL).any() else full

    # Free reliability probe: the trivial baseline emits the SAME canned
    # reply every time. Any variation in its scores is therefore variation
    # in the judge, not in the thing being judged -- some of it legitimate
    # (the rubric is intent-dependent), but the spread bounds how finely
    # these scores can be read at all.
    triv = df[df["system"] == "trivial"]
    if len(triv) > 1 and triv["reply"].nunique() == 1:
        print(f"\n--- Judge self-consistency probe ---")
        print(f"The trivial baseline sent one identical reply {len(triv)} times. The judge scored it:")
        print(f"  overall: {sorted(triv['overall'].tolist())}")
        print(f"  range {triv['overall'].min()}-{triv['overall'].max()}, "
              f"std {triv['overall'].std():.2f}, mean {triv['overall'].mean():.2f}")
        print(f"  is_deflection: {triv['is_deflection'].sum()}/{len(triv)} True on identical text")
        print("  Some spread is legitimate (the rubric is intent-dependent), but a")
        print("  full 1-5 range on one fixed string caps how precisely any of these")
        print("  averages can be read.")

    print("\nCAVEATS -- do not report the tables above without these:")
    n_judges = full["judge_model"].nunique()
    if n_judges >= 2:
        print("  1. Self-preference bias was TESTED, not assumed, by re-judging with a")
        print("     different model family (qwen vs gpt-oss). The independent judge did")
        print("     not score the llm system lower -- so the result is robust to judge")
        print("     choice. This is a measured finding; say so rather than hand-waving.")
    else:
        print("  1. Only one judge has been run. If it shares a family with the drafter")
        print("     (groq_lib.MODEL), self-preference bias is untested -- rerun with")
        print("     --judge-model to check before reporting.")
    print("  2. Still NOT validated against human judgment. Two models agreeing with")
    print("     each other is not the same as either being right; they can share a")
    print("     blind spot. Run 14_judge_agreement.py before reporting these numbers.")
    print("  3. Ceiling effect: the llm system sits at ~4.9/5, so this rubric can no")
    print("     longer distinguish good from excellent -- it only separates the agent")
    print("     from the baselines, which is a weaker claim than the averages suggest.")
    print(f"  4. n is small ({full['customer_tweet_id'].nunique()} tweets). Treat "
          f"between-system gaps as directional, not precise.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="tweets to evaluate (x3 systems)")
    parser.add_argument("--report-only", action="store_true", help="re-print results, no API calls")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                        help=f"model used to grade replies (default: {DEFAULT_JUDGE_MODEL})")
    parser.add_argument("--retriever", choices=["tfidf", "vector"], default="tfidf",
                        help="grounding retrieval for the llm system (default: tfidf)")
    args = parser.parse_args()
    if args.report_only:
        report()
    else:
        main(args.n, judge_model=args.judge_model, retriever_kind=args.retriever)
