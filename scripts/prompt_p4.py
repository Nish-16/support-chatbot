"""
Prompt p4 = p3 + P4_BLOCK: boundary refinements from the error analysis of the
250-tweet p1-vs-p3 evaluation (data/p3_eval_results.csv, 2026-09-14).

p3 is left byte-identical (its hash is pinned in data/p3_eval_manifest.json);
this block is inserted after taxonomy_rules.PROMPT_BLOCK. It is NOT shown to the
labeller and NOT in taxonomy_rules.py: the labels already exist, and a rule
added to the labeller now would change what later labels mean.

IN-SAMPLE WARNING. Every rule below was written after reading p1's and p3's
errors on the same 250 tweets p4 is then scored on, and 198 of those labels are
Claude drafts. A p4 gain on that set is optimistic by construction. It can
justify a fresh evaluation, never a production switch on its own.

What each rule targets (eval_index in the frozen set; texts are deliberately
not quoted -- tests/test_p4_prompt.py fails if this block shares a 40-character
window with any eval tweet):

  1 billing needs a transaction        64, 69, 143, 174, 213 went to billing/how-to
  2 availability question != feature  110, 180 went to feature_request under p3
  3 keyword "sync"/"error" != bug     205 (how-to -> bug), 176 (bug -> how-to)
  4 client sign-in failure            65, 140, 231 went to account_access
  5 getting files back                93, 154, 219 routed by cause, not goal
  6 abuse is about someone else       85, 152 went to phishing_abuse_report
  7 open problem != no action         31, 182 went to no_action_needed
  8 shared object that fails to load  97, 247, 248 went to sharing_permissions

Deliberately NOT addressed, because the labels on either side disagree with
each other rather than with the model: feature_request vs complaint on design
opinions (27, 71, 183 vs 77, 145, 226), and disabled links as sharing vs quota
(85 vs 173). A rule fitted to those would learn labelling noise.
"""

P4_BLOCK = """Boundary refinements. Apply after the rules above; where one is more specific, it wins.
1. billing_subscription vs feature_request/complaint_dissatisfaction: billing_subscription needs a transaction or a billing action on the customer's own account -- a charge, refund, invoice, payment method, purchase, upgrade or cancellation step, or how to buy. An opinion about what a plan costs or includes, with no transaction, is not billing: feature_request when it asks for different packaging (a feature in a cheaper plan, a trial, more storage for the price); complaint_dissatisfaction when it only objects or threatens to leave. A rhetorical question such as "only on the top plan?" is an objection, not a how-to.
2. how_to_usage vs feature_request: asking whether a capability already exists or is included in a plan ("is this available", "is there a way", "can I") is how_to_usage, even when it opens with "I would like" or "I wish". feature_request only when the customer proposes something new, states that it is missing, or asks when an unreleased product will ship.
3. how_to_usage vs sync_app_bug: the words sync, error, app or upload do not decide the intent. A question about how to set something up or which setting to use, with no error message and nothing that used to work, is how_to_usage. sync_app_bug needs an observed failure: an error message, something that stopped working, or a problem that keeps happening -- and stays sync_app_bug when phrased as a question. An icon or design that changed in an update is not a failure.
4. account_access vs sync_app_bug: account_access when credentials, a password reset, 2FA or the account's status block sign-in. When the Dropbox desktop or mobile client itself fails while signing in -- an installer error, the client showing offline, or sign-in failing together with files not syncing -- intent is sync_app_bug with account_access as secondary_intent.
5. data_loss_recovery vs sync_app_bug/storage_quota_plan_limits: a request to get files back or keep them from being lost -- restore, backups, an older version, version history, stopping a deletion -- is data_loss_recovery, even when a sync or quota problem caused it.
6. phishing_abuse_report vs the customer's own account: phishing_abuse_report only when the customer reports someone else's email, link or content. Their own links being banned or disabled is sharing_permissions; a virus or malware on their own device or account is security_account_compromise.
7. no_action_needed vs an open problem: a reply is no_action_needed only when the customer says it works now, or thanks with no problem left. A tweet that still names an unresolved problem (still not syncing, draining the battery, asking for a fix) keeps that problem's intent, even with thanks or phrased as a wish.
8. sharing_permissions vs sync_app_bug: sharing_permissions when the problem is who can access something or how sharing, invites or membership are set up. Opening, downloading or displaying a shared file or link that fails (keeps loading, blank window, will not download) is sync_app_bug with sharing_permissions as secondary_intent."""
