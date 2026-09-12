# Golden Eval Set: Sampling & Labeling Methodology

## Sampling

Two-stage sample, built by `scripts/07_build_golden_candidates.py`:

1. **Main random sample (150 tweets)**: a pure random draw
   (`random_state=123`) from the full 5,938 DropboxSupport
   customer<->reply pairs. This represents real traffic distribution
   with no bias -- whatever fraction of real customer tweets are
   sync bugs vs. billing questions, this sample reflects that.

2. **Rare-intent top-up**: a pure random sample under-represents rare
   intents badly. In an earlier 40-example manual read,
   `data_loss_recovery` and `feature_request` were each only ~5% of
   traffic -- a random 150 would yield roughly 7-8 examples of each,
   too few to say anything statistically meaningful about classifier
   accuracy on those categories specifically. So: after the main
   sample, we check which of the 8 intents fall short of a minimum
   (15 examples), and for those, draw additional random (previously
   unseen) tweets in batches, classify them, and keep any that land
   in a still-short category. This repeats until every intent has at
   least 15 examples, or a size/budget cap is hit (230 total
   examples, to stay within the assignment's 150-250 limit; 900 extra
   API calls, so a pathological run can't burn unlimited quota).

**Important nuance**: the classifier is used here only to *find*
candidates for underrepresented categories -- never to assign the
final label. Every example in the golden set is hand-labeled by a
human (see below) regardless of which stage it came from.

**Known limitation of this design**: because top-up candidates are
selected based on the classifier's own prediction, the golden set is
*not* a perfectly unbiased sample within the rare categories -- it's
biased toward examples the classifier itself was willing to call
`data_loss_recovery` (etc.), which could make classifier accuracy on
those specific categories look slightly better than it would on a
truly random sample of that category. This is a real tradeoff, not
an oversight: the alternative (pure random sampling) would leave us
with too few examples of rare categories to measure anything at all.
Documented here so the report's "what's misleading about my headline
number" section can address it directly.

## Labeling

Done by hand via `scripts/08_label_golden_set.py`, an interactive
CLI you run yourself in your own terminal -- this is not automated
or delegated, since the entire point of a golden set is that it
reflects independent human judgment, not the model's.

**Anti-anchoring design**: the tool asks for your intent judgment and
escalate/auto-handle decision BEFORE revealing what the classifier
predicted for that example. If the model's guess were shown first,
you'd unconsciously agree with it more often than an independent
judgment would -- inflating the reported classifier-accuracy number
in a way that wouldn't hold up under scrutiny. The model's prediction
is revealed immediately after you submit your answer, purely for your
own curiosity/calibration -- it is not used anywhere in the saved
label.

**Fields captured per example** (`data/golden_labels.csv`):
- `human_intent` -- your independent intent judgment (ground truth)
- `human_escalate` -- your judgment on whether this should be
  auto-handled or escalated to a human
- `human_note` -- optional free-text note (e.g. "borderline between
  two intents", "historical reply looks wrong")
- `suggested_intent` / `suggested_confidence` -- the classifier's
  prediction, kept for computing accuracy later, not used during
  labeling
- `agrees_with_suggestion` -- computed at label time, for a quick
  running sense of agreement rate (not the final accuracy number --
  that gets computed properly once labeling is complete, e.g. with a
  confusion matrix per intent)

**Resumability**: every label is appended to
`data/golden_labels.csv` immediately, so labeling can happen across
multiple sessions -- rerunning the script skips tweets already
labeled.

## Status (as of first build attempt)

- [x] Candidate pool mostly built: 234 tweets in `data/golden_candidates.csv`
- [ ] Two intents still short of the 15-minimum: `data_loss_recovery` (9)
  and `followup_ticket_status` (6) -- top-up for these resumes next run
- [ ] Labeling complete (`data/golden_labels.csv`)
- [ ] Final counts + intent distribution written up here

### Note on the Gemini free tier during this build

Building the candidate pool hit two separate free-tier ceilings on
`gemini-flash-lite-latest` (which currently resolves to the
`gemini-3.5-flash-lite` backend): a 15-requests/minute cap, and then
a 500-requests/DAY cap. The daily cap killed the run partway through.

The original version of `scripts/07_build_golden_candidates.py` only
wrote results to disk once, at the very end -- so when the run was
killed, ~150 successful classifications existed only in the terminal
log, not on disk. They were recovered by parsing that log and
rejoining with `dropbox_paired.csv` (no extra API calls needed), but
this was an avoidable loss. The script was rewritten to checkpoint
every single classification to `data/golden_candidates.csv`
immediately and to resume automatically (skip already-classified
tweets) on rerun -- this class of bug can't happen again regardless
of which API tier is in use.

Free tier will likely need another day (or a switch to pay-as-you-go
billing) to finish the remaining top-up + get through labeling +
reply drafting + eval harness calls -- 500/day is tight for a project
this size.
