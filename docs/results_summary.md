# Results summary and review notes

## What was built (use this framing in the next review)
The Review 1 report described a masked autoencoder. The agreed scope is a **plain autoencoder** whose novelty
is a **continuous, streaming-safe sliding-window loop** on edge hardware. Suggested objectives:

1. Design a lightweight 1D convolutional autoencoder for ECG windows, trained on normal beats only (MIT-BIH).
2. Make preprocessing streaming-safe (causal filter) and implement a continuous sliding-window
   encode, decode, score, alarm loop.
3. Quantize to INT8 and deploy on an ESP32-S3 via TensorFlow Lite for Microcontrollers.
4. Evaluate AUROC, recall, specificity, false alarms per hour, model size, latency, RAM and power,
   and compare against the base paper and other TinyML baselines.

## Ablation / progression
| Step | Change | AUROC |
|---|---|---|
| 1 | Dense bottleneck (16-d latent) | 0.47 (chance) |
| 2 | Fully convolutional, 2-channel bottleneck | 0.75 |
| 3 | 1-channel bottleneck (45 x 1 latent) | 0.860 |
| 4 | Causal filter (deployable) | 0.825 |
| 5 | INT8 post-training quantization | 0.815 |

Notes: a peak-of-window error score gave no gain over the mean (0.857 vs 0.860), so the mean is used.
PTQ cost 0.010 AUROC, so quantization-aware training was not needed.

## Window-level operating points (float, causal, thresholds from validation percentiles)
| Val percentile | Recall | Specificity | False alarms/h |
|---|---|---|---|
| 80 | 60.9% | 85.1% | 538 |
| 90 | 50.6% | 89.1% | 391 |
| 95 | 42.0% | 91.8% | 295 |
| 99 | 22.8% | 95.7% | 154 |

## Deployment-level results (INT8, causal, 10 s segments, k-of-m alarm rule)
| Val percentile | Rule | Segment recall | Specificity | False alarms/h |
|---|---|---|---|---|
| 90 | 1/1 | 71.4% | 72.9% | 97.4 |
| 90 | 2/3 | 57.5% | 85.3% | 52.8 |
| 95 | 3/5 | 44.3% | 93.4% | 23.9 |
| 97 | 3/5 | 38.3% | 95.0% | 18.1 |
| 99 | 3/5 | 23.0% | 97.0% | 11.0 |

## Limitations to state openly
- Not clinical-grade: about 24 false alarms per hour at the balanced preset.
- Strict labelling: any non-N beat (including bundle branch block beats) makes a window anomalous.
- Validation set is only 4 patients, so thresholds transfer imperfectly to held-out patients.
- Poor per-record AUROC (below 0.5) on records 100, 103, 105, 222 and 228, where noisy or variable normal beats
  produce higher error than the anomalies. Pooled AUROC is also helped by records that are 100% anomalous
  (111, 214, 232).
- The 3-of-5 rule responds to sustained abnormal rhythm; it cannot fire on a single isolated beat
  (use the sensitive 2-of-3 preset for that, at about twice the false alarms).
- Offline zero-phase filtering gave AUROC 0.860, but it cannot run on a live device, so the causal 0.825 / 0.815
  figures are the honest headline numbers.
- Desktop Python latency is not meaningful; real latency, RAM and power come from the ESP32-S3 + INA219.
- Literature figures in the Review 1 survey were compiled from search results and must be re-verified against
  the original papers before they are quoted.

## Possible extension
Per-patient threshold calibration (learn the wearer's normal error level during an initial period).
