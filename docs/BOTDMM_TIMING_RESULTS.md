# Timing-Adapted BotDMM Development Outcomes

All 36 fixed training runs completed on October 13, 2026. All 36 selected
checkpoints replayed their saved challenge probabilities within absolute
tolerance 1e-6. The original account archive, 34 immutable released result
files and working manuscript remained unchanged.

These are previously inspected validation-policy development challenges,
not untouched tests or a full reproduction of the original BotDMM paper.
The backbone uses the local release implementation and adapted input sizes.
The three-class and four-class views are nested, not independent datasets.

## Complete Macro-F1 Table

Values are means and sample standard deviations over seeds 42, 43 and 44.

| View | Held-out policy | BotDMM with timing | BotDMM without timing | ERM MLP |
|---|---|---:|---:|---:|
| Twibot22 | burst_scheduler | 0.5531 +/- 0.0015 | 0.6525 +/- 0.0091 | 0.7869 +/- 0.0258 |
| Twibot22 | cyclic_reposter | 1.0000 +/- 0.0000 | 0.6309 +/- 0.0580 | 0.9967 +/- 0.0033 |
| Twibot22 | uniform_scheduler | 0.9987 +/- 0.0023 | 0.6397 +/- 0.0192 | 0.9770 +/- 0.0125 |
| Quadbot | burst_scheduler | 0.6556 +/- 0.0043 | 0.7329 +/- 0.0208 | 0.8534 +/- 0.0351 |
| Quadbot | cyclic_reposter | 0.9926 +/- 0.0043 | 0.7213 +/- 0.0207 | 0.9730 +/- 0.0026 |
| Quadbot | uniform_scheduler | 0.9940 +/- 0.0030 | 0.7300 +/- 0.0083 | 0.9817 +/- 0.0054 |

With timing, burst-policy Macro-F1 decreased by 9.94 percentage points in
Twibot22 and 7.73 points in Quadbot relative to the no-timing backbone.
Descriptive paired-profile intervals were [-12.23, -7.39] and
[-9.13, -6.18] points. Relative to ERM MLP, decreases were 23.38 and
19.78 points. Other policies improved, but they do not cancel this failure.

The full timing backbone has 1,423,111 parameters in Twibot22 and 1,423,304
in Quadbot, compared with 1,089,667 and 1,089,796 for ERM MLP. The no-timing
backbones have 1,418,631 and 1,418,824 parameters. BotDMM additionally
receives a fold-local feature-kNN neighbor channel; MLP does not. These
architectures and information channels are not capacity-matched.

## Decision

Do not promote the timing-adapted backbone as a unified improvement.
Do not hide the burst-policy failure, change labels, re-split profiles,
or search timing weights against these outer challenge scores.

The next bounded diagnostic removes the existing structure output and the
existing orthogonal processing separately, holding inputs, initialization
seeds, graph context, optimizer and inner selection fixed. These are
diagnostic ablations, not novel modules. All outcomes must be retained.

An independent strategy or generator evaluation is still absent. Descriptive
profile intervals under fixed training seeds cannot remove prior challenge
exposure. No SOTA, external robustness or submission-readiness claim follows
from software completion.
