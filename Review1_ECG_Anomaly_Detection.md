# Self-Supervised ECG Anomaly Detection on Ultra-Low-Power Edge Nodes
## Review 1 Report

---

## 1. Introduction

Cardiovascular disease remains one of the leading causes of death worldwide, and many life-threatening arrhythmias such as Atrial Fibrillation (AFib) occur intermittently, making them easy to miss during short, clinic-based ECG recordings. Continuous, long-term monitoring using wearable ECG devices is therefore essential for early detection. However, most existing wearable solutions transmit raw ECG data to a smartphone or cloud server for analysis, which introduces three major problems: (i) high power consumption from wireless radio transmission, which drains battery life quickly, (ii) latency in detecting critical events, and (iii) privacy risks associated with transmitting sensitive health data.

This project proposes an on-device, self-supervised ECG anomaly detection system designed to run entirely on ultra-low-power microcontroller-class hardware. Instead of relying on large labelled datasets of arrhythmia examples (which are expensive and time-consuming to obtain, since they require expert cardiologist annotation), the system uses a masked autoencoder (MAE) trained in a self-supervised manner: the model learns to reconstruct normal ECG waveforms from partially masked (hidden) segments of the 1D time-series signal. During inference, heartbeats that produce a high reconstruction error are flagged as anomalous, since the model — having only learned the structure of normal rhythms — fails to reconstruct patterns it has not seen before.

To make this feasible on wearable hardware operating within a sub-milliwatt power budget, the trained model is quantized to INT8 precision and deployed using TensorFlow Lite for Microcontrollers (TFLM), a TinyML framework built for resource-constrained edge devices. The model is trained and validated using the PhysioNet MIT-BIH Arrhythmia Database, the most widely used benchmark dataset for ECG arrhythmia research.

The key contributions of this project are:
- A self-supervised masked-autoencoder pretraining approach for 1D ECG time-series, removing the dependency on large labelled arrhythmia datasets.
- INT8 quantization and TFLM deployment for sub-milliwatt, real-time, fully on-device inference — no cloud dependency.
- Evaluation against classical supervised and autoencoder-based baselines on the MIT-BIH dataset in terms of accuracy, model size, memory footprint, and estimated power/energy per inference.

---

## 2. Literature Survey

| # | Study | Approach | Dataset | Key Finding / Relevance |
|---|-------|----------|---------|--------------------------|
| 1 | Zhang et al., "MaeFE: Masked Autoencoders Family of Electrocardiogram for Self-Supervised Pretraining and Transfer Learning," *IEEE Trans. Instrumentation and Measurement*, 2022 | Proposes three masking strategies (temporal, lead-wise, and combined) for self-supervised MAE pretraining on ECG | Multiple public ECG datasets | Establishes that masked-autoencoder pretraining transfers well to downstream ECG classification tasks with limited labels — directly motivates the self-supervised core idea of this project |
| 2 | Busia et al., "A Tiny Transformer for Low-Power Arrhythmia Classification on Microcontrollers," arXiv:2402.10748, 2024 | 6k-parameter transformer, INT8 quantized, deployed on GAP9 ultra-low-power processor | MIT-BIH | Achieves 98.97% accuracy at 0.09 mJ/inference — shows that very small, quantized models can retain high accuracy on MCU-class hardware |
| 3 | ArrythML: "An Autoencoder-Based TinyML Approach for On-Device Arrhythmia Detection on Resource-Constrained Embedded Systems," arXiv:2606.02256, 2026 | INT8 quantized autoencoder, QAT (quantization-aware training), deployed on ESP32-S3 with TFLM | Custom dataset derived from MIT-BIH | Closest existing system to the proposed project — validates reconstruction-error-based anomaly detection on real MCU hardware (9 ms inference latency, ~180 KB model) |
| 4 | "Memory-Efficient CNN Autoencoder for Real-Time ECG Anomaly Detection on TinyML-Enabled Edge Devices," *Future Internet*, 2026 | 1D-CNN autoencoder, INT8 post-training quantization | PTB Diagnostic ECG Database (via Kaggle) | Shows INT8 quantization reduces model size by 75% (5.44 KB → 1.36 KB) with only minor accuracy loss — supports the feasibility of extreme compression |
| 5 | "Real-Time Cardiac Arrhythmia Classification Using TinyML on Ultra-Low-Cost Microcontrollers," *Bioengineering*, 2026 | 1D-CNN, TFLM deployment on Arduino UNO (ATmega328P) | MIT-BIH | Confirms that even 32 KB flash devices can host a functional arrhythmia classifier after INT8 quantization |
| 6 | "Reliable ECG Anomaly Detection on Edge Devices for Internet of Medical Things Applications," *PMC*, 2025 | Pruning + quantization on Raspberry Pi / Arduino | Custom | Reports 0.024 mW power draw with 92.3% accuracy — a useful power-budget reference point |
| 7 | "Privacy-Preserving Federated Autoencoder for ECG Anomaly Detection on Edge Devices," arXiv:2606.11556, 2026 | Federated autoencoder with differential privacy | 12-lead ECG, edge (ARM Cortex-A) | Notes that most prior TinyML anomaly-detection work (pruning/PTQ) does not address federated or privacy-preserving training — identifies a gap this project does not attempt to close but should acknowledge |

**Summary of gap identified:** Most prior TinyML ECG work either (a) uses supervised classifiers that need large labelled datasets, or (b) uses simple autoencoders trained directly on raw signals without a masking-based self-supervised pretext task. Very little existing work combines masked-autoencoder self-supervised pretraining specifically with INT8/TFLM deployment for anomaly (rather than classification) detection — this is the novelty angle of the proposed project.

---

## 3. Comparative Analysis

| Criterion | Supervised CNN Classifier (e.g., Busia et al.) | Plain Autoencoder (ArrythML, CNN-AE) | Proposed: Self-Supervised Masked Autoencoder (this project) |
|---|---|---|---|
| Labelled data requirement | High — needs annotated arrhythmia classes | Low — trained only on normal beats | Low — trained only on normal beats, using masking as the pretext task |
| Generalization to unseen anomaly types | Limited to trained classes | Moderate (any deviation from normal raises error) | Better — masking forces the model to learn structural/temporal dependencies, not just point-wise reconstruction |
| Model size (reported in literature) | ~6k params (~24 KB) | 1.36 KB – 180 KB depending on design | Comparable to autoencoder baselines after INT8 quantization (target: <50 KB) |
| Inference latency (MCU) | ~4.28 ms (GAP9) | ~9 ms (ESP32-S3) | Expected similar order (ms range) — to be benchmarked |
| Power/Energy | 0.09 mJ/inference | Not always reported | Target: sub-milliwatt continuous operation |
| Deployment framework | TFLM / custom | TFLM | TFLM |
| Novelty relative to this project | Baseline for accuracy/efficiency comparison | Baseline for reconstruction-error anomaly detection | Adds masked self-supervised pretraining on top of the autoencoder baseline |

**Positioning:** The proposed approach is benchmarked directly against ArrythML and the CNN-AE approach (same family — reconstruction-error-based anomaly detection) as the primary baselines, and against the tiny transformer classifier as an accuracy/efficiency upper reference. The key differentiator is the masked-reconstruction pretext task, which is expected to improve robustness to noise and to previously unseen anomaly morphologies, since the model must learn temporal context rather than simply compress-and-reconstruct the full signal.

---

## 4. Objectives & Justification

**Objectives:**
1. To design a lightweight 1D masked-autoencoder architecture suitable for self-supervised pretraining on ECG time-series data from the MIT-BIH Arrhythmia Database.
2. To train the model using only normal (non-arrhythmic) heartbeat segments, using a masking-and-reconstruction pretext task, without relying on manually labelled arrhythmia classes for pretraining.
3. To quantize the trained model to INT8 precision and convert it using TensorFlow Lite for Microcontrollers (TFLM) for deployment on a representative low-power MCU/wearable platform.
4. To evaluate the system on accuracy, sensitivity/recall (critical for not missing true arrhythmias), model size, inference latency, and estimated power consumption, benchmarked against existing TinyML autoencoder and classifier baselines from literature.

**Justification:**
- **Clinical relevance:** AFib and other arrhythmias are often paroxysmal (intermittent), so continuous, always-on monitoring materially improves the chance of early detection compared to periodic clinical ECGs.
- **Why self-supervised:** Labelled arrhythmia data is scarce and requires expert cardiologist annotation; a self-supervised approach that only needs normal-rhythm data for training is far more practical to collect and scale.
- **Why edge deployment:** Sending raw ECG continuously over Bluetooth/Wi-Fi is one of the most power-hungry operations on a wearable. On-device inference avoids this, extending battery life from days to potentially weeks, while also keeping sensitive health data on the device (privacy benefit).
- **Why INT8/TFLM specifically:** This is the standard, well-supported path for deploying deep learning models on ARM Cortex-M–class and similar ultra-low-power MCUs commonly used in real wearable products, and is consistent with the tech stack specified for this project.

---

## 5. Dataset

**MIT-BIH Arrhythmia Database (PhysioNet)**
Link: https://www.physionet.org/content/mitdb/1.0.0/

48 half-hour, two-channel ambulatory ECG recordings from 47 subjects, sampled at 360 Hz, with expert beat-by-beat annotations — the standard benchmark for arrhythmia detection research.

---

## 6. Base Paper

**ArrythML: An Autoencoder-Based TinyML Approach for On-Device Arrhythmia Detection on Resource-Constrained Embedded Systems**
Link: https://arxiv.org/abs/2606.02256

This paper is recommended as the base paper because it most closely matches the project's tech stack and goal: an INT8-quantized autoencoder trained on MIT-BIH-derived data, deployed via TensorFlow Lite Micro on a real low-power MCU (ESP32-S3), using reconstruction error for anomaly detection. This project extends it by replacing the plain autoencoder with a **masked** autoencoder (self-supervised pretext task), which is the stated novelty angle.

*Secondary reference for the self-supervised/masking methodology:*
**MaeFE: Masked Autoencoders Family of Electrocardiogram for Self-Supervised Pretraining and Transfer Learning** (Zhang et al., IEEE Trans. Instrumentation and Measurement, 2022)

---

## 7. Domain of the Project

**Primary domain:** Biomedical Signal Processing / Healthcare AI
**Secondary domains:** Embedded Systems & TinyML (Edge AI), Self-Supervised Machine Learning, IoT / Wearable Computing

Single-line domain : **"Edge AI / TinyML for Biomedical Signal Processing (Healthcare)"**
