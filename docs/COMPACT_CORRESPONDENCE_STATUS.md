# Compact Correspondence Status

Completed on 2026-10-14 at 23:52:16 Asia/Shanghai; local receipt audit
verified all 36 cells. An authenticated check on 2026-10-15 at 00:02:43
found no active analysis jobs. No GPU is needed for the next local tasks.

Started on 2026-10-14 at 23:17:27 Asia/Shanghai, after 188 local regression
tests passed in 337.194 seconds. All 11 compact implementation/receipt
tests also passed remotely before training began at 23:18:06.

The fixed matrix has 30 new fits, six exact-partition cached controls and
36 selected-checkpoint replays. CPU only; GPU is disabled. This round
does not enlarge data, forward excluded/challenge rows, or access original
validation/test. See `COMPACT_CORRESPONDENCE_PLAN.md`.

Code deployment is an isolated overlay, not an update to canonical
remote code, GitHub or the manuscript. The uploaded source archive has
SHA-256 `779184e4d493fee7f1f089c8c014318b036341cb1439d096f0850e489067d5d4`.
See `COMPACT_CORRESPONDENCE_RESULTS.md`: mean inner-selection Macro-F1
is 98.64%, versus 98.60% for removal, 98.17% supervised and 98.12%
BotSSCL mechanism. The module mean gain is only 0.047 points with
one reversed seed mean; these are not final test results.
The weak development gate passes, not a final-model/submission gate.
Follow `SUBMISSION_SPRINT.md` without another open-ended model search.
