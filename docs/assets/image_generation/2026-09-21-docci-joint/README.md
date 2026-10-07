# DOCCI connector and diffusion pilot artifacts

The 100-step joint pilot completed on Aurora (job 8847769, exit 0). It trained
100 distinct caption/image pairs, with 100 separate validation pairs, starting
from the earlier 500-step connector and original OmniGen2 transformer. PRISM,
the VAE, and native conditioner remained frozen. Loss changes were small and
sampled caption fidelity remains poor.

- [Execution report](../../../reports/2026-09-21-docci-joint-diffusion-pilot.md)
- [Loss plot](runs/pilot-100-01/plots/connector-diffusion-losses.png) · [PDF](runs/pilot-100-01/plots/connector-diffusion-losses.pdf)
- [Training examples](runs/pilot-100-01/train-comparison.png)
- [Validation examples](runs/pilot-100-01/validation-comparison.png)
- [Raw report](runs/pilot-100-01/report.json) · [optimizer log](runs/pilot-100-01/steps.jsonl) · [evaluation log](runs/pilot-100-01/evaluations.jsonl)
- [Independent checks](provenance/pilot-100-01-acceptance-checks.json) · [PBS status](provenance/pilot-pbs-final.txt)
- [Warm-start and training replay audit](provenance/pilot-warm-start-replay.json)
- [Two-step smoke](runs/smoke-01/report.json)

Generated images use captions alone and 50 diffusion steps, with matched starting
noise. DOCCI target images are included only for visual comparison. Targets and
captions: [Google DOCCI](https://google.github.io/docci/), CC BY 4.0. Full caption,
source, image hashes and attribution are retained in each `gallery-targets.json`.

Full checkpoints remain on Aurora; their paths, sizes, and SHA256 digests are
recorded in the reports and independent checks. The recorded launch-source hashes
refer to executed source. Plots are generated afterwards, with their own source
hashes in `plots/connector-diffusion-loss-series.json`.

The warm-start loss probes match exactly. Separate smoke/pilot backward passes
have small numerical differences whose cause remains unresolved; bitwise real
training replay and a real full-checkpoint resume have not been established.
