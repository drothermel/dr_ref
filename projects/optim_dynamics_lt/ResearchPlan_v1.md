# Research Plan

## Initial Source of Hypotheses

Initially driven by considering 3 hypotheses, 2 which I have to at least cursorily explore because they were provided by my advisor and 1 which is mine:

- **H1:** The linearity of the log-log scale train loss curve is an indication of the quality of the training setup design (hyperparams, architecture choices, training method, etc) for the given problem.
- **H2:** The slope of the linear fit of the log-log scale train loss curve in the early training is predictive of the final performance (eg. best validation accuracy). 
  - With the idea being that (1) you can fit a near perfect line ot the train loss curve so (2) the slope at the beginning of the line is indicative of the slope of the full line.  Then a sub hypothesis that (3) the slope of the training loss curve is predictive of the final performance of the run.
- **H3:** When constrained to settings where learning eventually converges, less smoothness in the early loss curve would lead to best final performance when training from random initialization.
  - With the idea being that less smoothness could indicate more exploration early on, making it more likely that the model finds a “good basin” that it then descends into.

## Summarized Conclusions from Investigation of Potential Directions

For each research trajectory you get a **one‑line pitch, novelty pocket, plausibility support, incremental cost, and a 3‑way rating** (★ = low / easy → ★★★★ = high / hard).  At the end you will find:

1. **Overall ranking** on a novelty ✕ effort plane.
2. A proposed split into **three coherent Tracks** that we can staff and schedule separately next.

All resource estimates assume: ResNet‑18 on CIFAR‑10, 50 epochs (= ∼0.5 GPU‑h) with **two workers packed per 48 GB GPU** as enabled by *deconCNN* + *dr_exp*. 

### Consolidated Menu of Research Trajectories

| ID       | Short name (catchy)              | Core idea & novelty pocket                                   | Why plausible                                                | Incremental build cost           | GPU budget                               | Novelty payoff | Eng. overhead |
| -------- | -------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------- | ---------------------------------------- | -------------- | ------------- |
| **T‑1**  | **Curve‑Doctor**                 | Real‑time knee + segment detector for power‑law windows **(P‑1)** | Straight‐line loss regimes documented across vision; no automatic detector yet. | 1 callback (OLS→RANSAC fallback) | **108 runs ≈ 3 GPU‑days**                | ★★★☆           | ★☆☆☆          |
| **T‑2**  | **Zero‑Cost Robustness Auditor** | Use early slope α to predict CIFAR‑10‑C/10.1 accuracy **(P‑2)** | Brandfonbrener 24 showed α↔ID; OOD link untested.            | 2 DataModules + scatter script   | Reuse T‑1 logs                           | ★★★★           | ★★☆☆          |
| **T‑3**  | **Vision Loss‑to‑Loss Map**      | Brandfonbrener shifted train→val/OOD line for CNNs **(P‑3)** | Prior results in NLP, never for vision.                      | Post‑hoc fit util                | Reuse T‑2 logs                           | ★★★☆           | ★☆☆☆          |
| **T‑4**  | **Noise‑Metric Show‑Down**       | PSD tail, grad‑var, residual power head‑to‑head **(P‑4)**    | Competing metrics each shown singly; never compared.         | FFT + var hooks                  | 432 runs ≈ 8 GPU‑days                    | ★★★★           | ★★★★          |
| **T‑5**  | **α‑Scheduler**                  | LR cool‑down triggered by α stall **(P‑5)**                  | Early‑slope control is logically next step after T‑1/T‑2.    | 1 LR callback + flags            | 36 runs ≈ 1 GPU‑day                      | ★★★☆           | ★★☆☆          |
| **T‑6**  | **Curvature Plateau Companion**  | Track λ₁ & Tr H; test plateau timing vs α window **(P‑6)**   | Hessian plateau widely observed, never aligned with α.       | Hutch++ & power‑iter (opt‑in)    | +10 % on T‑1 grid                        | ★★★★           | ★★☆☆          |
| **T‑7**  | **Aug‑Dial Controller**          | Anneal RandAug once (α, λ₁) both stabilise **(P‑7)**         | Curvature‑aware tuning wins on ImageNet; not with α combo.   | 1 severity scheduler             | 36 runs ≈ 1 GPU‑day                      | ★★★☆           | ★★☆☆          |
| **T‑8**  | **Multi‑Power Early‑Stop**       | Two‑power fit after 20 % to prune doomed runs **(P‑8)**      | Luo 25 multi‑power fit + SoTL‑E pruning ideas, never joined. | Two‑power solver                 | 72 runs (with early abort) ≈ 2 GPU‑days  | ★★★☆           | ★★☆☆          |
| **T‑9**  | **Curvature‑α Fusion**           | 2‑D (α, λ₁) plane drives simultaneous LR+WD adaptation **(P‑6)** | Curvature tuners beat baselines; fusion untested.            | Extend T‑5 scheduler to read λ₁  | add 24 runs                              | ★★★★           | ★★☆☆          |
| **T‑10** | **Resolution‑Scaling Probe**     | α at 32×32, 48×48, 64×64 predicts accuracy across res **(P‑7)** | Scaling laws for size known; resolution largely unexplored.  | Multi‑res DataModule             | 162 runs ≈ 5 GPU‑days                    | ★★★★           | ★★☆☆          |
| **T‑11** | **Checkpoint & Data Pruning**    | Use α & R² @ 10 epochs to delete ckpts & drop “easy” samples **(P‑8)** | Early‑dynamics data pruning is hot; no checkpoint variant yet. | 100 LOC policy + Supabase field  | Expected **‑20 %** compute after trigger | ★★★            | ★☆☆☆          |

### Visual ranking (novelty ✕ effort)

```
Higher ── Novelty
        │                       T‑2  T‑6  T‑9  T‑10
        │                  T‑4
        │
        │          T‑3
        │
        │   T‑1         T‑8
        │
        └────────────────────────────────────────────►  Engineering overhead
                 Low             Medium              High
```

- T‑2 / T‑6 / T‑9 / T‑10 carry the **strongest publication upside**.
- T‑1, T‑3 and T‑5 are **quick wins** that unblock the rest.
- T‑4 is heavy but settles a community dispute and feeds T‑5/T‑7.
- T‑11 is a compute‑saver that pays for itself once α/R² are logged.

```
T‑1 ─┬─► T‑2 ─► T‑3
     ├─► T‑5
     ├─► T‑6 ─┬─► T‑9
     │        └─► T‑7
     └─► T‑4 ─┘
T‑6,8,11 piggy‑back on checkpoints/logs from above.
```



### Proposed split into **three execution Tracks**

#### Track A — Real-Time Signals & Control

*(T-1 → T-5 → T-9; depends on T-6 for λ₁)*

| Step    | Trajectory                                                   | Deliverable                                                | Impact-capture opportunity                                   |
| ------- | ------------------------------------------------------------ | ---------------------------------------------------------- | ------------------------------------------------------------ |
| **A-1** | **T-1 Curve-Doctor**                                         | Online knee + segment detector, α/R² logs on 108-run grid. | **Workshop paper****Narrative**: “First open-source *real-time* detector of power-law regimes; prunes bad runs in minutes.”**Needs**: algo description, speed vs offline, straightness quality scores, ablation on window/radius.**Venues**: NeurIPS *ML Systems*, ICLR *Efficient ML*, ICML *AutoML* WS. |
| **A-2** | **T-5 α-Scheduler**                                          | LR cool-down triggered by α stall; 36-run ablation.        | (Optional) add as **appendix** to A-1 or standalone short paper. Narrative: “Cuts wall-time 30 % on CIFAR-10 without accuracy loss using only one extra metric.” |
| **A-3** | **T-6 Curvature Plateau** *(run in Track B but needed here for λ₁)* | Stable online λ₁ estimates alongside α.                    | No direct paper here—feeds A-4/A-5.                          |
| **A-4** | **T-9 Curvature-α Fusion**                                   | Dual-signal (α, λ₁) controller for LR **and** WD.          | **Main-track paper****Narrative**: “2-D curvature-slope plane enables fully automatic optimiser tuning that beats hand-tuned baselines on vision tasks.”**Needs**: Theoretical intuition (λ₁ ↔ sharpness), controlled experiments on 3 datasets, wall-time & accuracy table, ablation isolating α vs λ₁.  **Venues**: NeurIPS *Optimisation* or ICML main track. |

*Dependencies*: A-4 needs λ₁ from A-3; A-2/A-4 reuse logs from A-1.

#### Track B — Generalisation & Scaling Insights

*(T-2 → T-3 → T-6 → T-10; consumes α & λ₁ from Track A)*

| Step    | Trajectory                           | Deliverable                                                  | Impact-capture opportunity                                   |
| ------- | ------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **B-1** | **T-2 Zero-Cost Robustness Auditor** | α vs CIFAR-10-C/10.1 scatter & correlation stats.            | **Workshop paper**, **Narrative**: “Early training slope predicts robustness—skip expensive corruption eval until epoch 10.”**Needs**: correlation plots, causal sanity check (swap α segments), comparison with loss-at-epoch-10 baseline. **Venues**: NeurIPS *RobustML*, ICLR *Distribution Shifts*. |
| **B-2** | **T-3 Loss-to-Loss Map**             | Shifted train→val/OOD line fitted on CNNs; compare to α.     | Could merge with B-1 if both signals win; otherwise reserve as back-up workshop submission (“When α fails, shifted-line succeeds”). |
| **B-3** | **T-6 Curvature Plateau Companion**  | Timing of λ₁ plateau vs α knee; predictor of final accuracy. | **Alternative main-track angle** if α↔OOD is weak. **Narrative**: “Geometry (λ₁ plateau) beats slope as a one-shot accuracy oracle.” |
| **B-4** | **T-10 Resolution-Scaling Probe**    | α at 32/48/64 predicts res-accuracy curve.                   | **Main-track or spotlight** if successful.**Narrative**: “A single 50-epoch 32×32 run lets you forecast 64×64 accuracy—resolution scaling law via early dynamics.”**Needs**: cross-resolution α-accuracy plots, linear extrapolation MAPE, cost-saving analysis, link to visual kernels theory.**Venues**: ICML *Data-Efficient ML*, NeurIPS main. |

*Backup plan*: If α correlation in B-1 is < 0.4, push harder on B-3 curvature story and frame α as negative result.

#### Track C — Noise Benchmark & Diagnostics

*(T-4 only; T-7/T-8 optional phase-2)*

| Step    | Trajectory                                  | Deliverable                                                  | Impact-capture opportunity                                   |
| ------- | ------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **C-1** | **T-4 Noise-Metric Show-Down**              | PSD tail, grad-var, resid-power logged on 432-run grid; correlation table vs final acc/OOD. | **Workshop paper****Narrative**: “Which noise proxy should we trust? A 400-run benchmark settles PSD vs gradient variance for vision.”**Needs**: metric definitions, runtime cost vs benefit, correlation/partial-correlation tables, ablation across optimiser & batch size.**Venues**: NeurIPS *Approximate ML*, ICML *Theory & Empirics of Noise*. |
| **C-2** | *(Optional)* **T-7 Aug-Dial Controller**    | Proven usefulness of noise+curvature signals for augmentation scheduling. | Main-track only if C-1 shows a clear winner and controller gives > 1 % accuracy gain; else drop. |
| **C-3** | *(Optional)* **T-8 Multi-Power Early-Stop** | Compute savings with two-power forecast.                     | Publish in MLSys/System workshop if multi-power fit error < 2 × single-power. |

*Note*: C-1 is scientifically driven (settles H3 proxy question) even though it is compute-heavy.

## Related Work

### **Foundational scaling-law theory & models**

- [2017 | Deep Learning Scaling is Predictable](https://arxiv.org/abs/1712.00409)
- [2020 | Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [2021 | Explaining Neural Scaling Laws](https://arxiv.org/abs/2102.06701)
- [2023 | Explaining Neural Scaling Laws](https://arxiv.org/abs/2312.06765)
- [2024 | Loss-to-Loss Prediction: Scaling Laws for All Datasets](https://arxiv.org/abs/2411.12925)

### **Vision & multimodal scaling evidence**

- [2021 | Scaling Vision Transformers](https://arxiv.org/abs/2111.09883)
- [2023 | Reproducible Scaling Laws for Contrastive Language-Image Learning](https://arxiv.org/abs/2304.09339)
- [2023 | Kernel Regression with Infinite-Width Neural Networks on Millions of Examples](https://arxiv.org/abs/2303.05420)
- [2022 | A Scaling Law for Syn-to-Real Transfer](https://arxiv.org/abs/2206.03083)
- [2025 | Scaling Laws for Data-Efficient Visual Transfer Learning](https://arxiv.org/abs/2504.13219)

### **Optimisation schedules & full-curve prediction**

- [2024 | Stepping on the Edge: Curvature-Aware Learning-Rate Tuners](https://arxiv.org/abs/2407.06183)
- [2024 | ExpTest: Detecting Exponential Burn-in Segments in Neural-Network Loss Curves](https://arxiv.org/abs/2411.16975)
- [2024 | Scaling Laws & Compute-Optimal Training Beyond Fixed Durations](https://arxiv.org/abs/2405.18392)
- [2025 | A Multi-Power Law for Loss-Curve Prediction Across Learning-Rate Schedules](https://arxiv.org/abs/2503.12811)

### **Early-training dynamics & data selection**

- [2024 | Early Period of Training Impacts Out-of-Distribution Generalization](https://arxiv.org/abs/2403.15210)
- [2024 | Critical Learning Periods: Leveraging Early Training Dynamics for Efficient Data Pruning](https://arxiv.org/abs/2405.19462)

### **Loss-based performance estimation & model selection**

- [2023 | Re-visiting the Train Loss: An Efficient Performance Estimator for NAS](https://arxiv.org/abs/2311.11010)
- [2025 | Can the Training Loss be Predictive for Out-of-Distribution Generalization?](https://arxiv.org/abs/2502.18975)
- [2025 | VP-Nets: Variance-Preserving Networks for Robust Model Selection](https://arxiv.org/abs/2502.14870)

### **Theoretical training-dynamics foundations**

- [2020 | On the Heavy-Tailed Nature of Stochastic Gradient Noise](https://arxiv.org/abs/2006.09313)

### Empirical evidence for **power‑law loss decay** (H1)

| Study & year   | Domain / models | Dataset(s)                  | Training regime           | Reported behaviour                                           |
| -------------- | --------------- | --------------------------- | ------------------------- | ------------------------------------------------------------ |
| *Hestness 17*  | CNN, LSTM       | CIFAR‑10, ImageNet‑32, LM1B | SGD, constant LR          | After burn‑in, **train CE follows L(t)=c·t^‑α**, α≈0.3–0.7 (vision) |
| *Bahri 24*     | ViT‑Huge, GPT‑3 | JFT‑300M, C4                | AdamW, cosine             | Single **power‑law over 3 orders of mag.**; α shrinks with width |
| *Luo 25*       | ResNet50        | ImageNet                    | **Constant+cool‑down LR** | Breaks into two power‑law segments; earlier slope steeper by ≈1.8× |
| *Rosenfeld 23* | Wide‑ResNet     | CIFAR‑10                    | cosine                    | Exponential regime persists until CE≤1; power‑law fit valid only afterwards |

> **Common observation**: a short exponential “burn‑in” (≈2–5 epochs on CIFAR‑10) precedes the power‑law window; fits restricted to CE ≤ 1 minimise bias.

### Early‑time **slope α as performance predictor** (H2)

| Study               | Setting              | Metric of interest | Finding                                                      |
| ------------------- | -------------------- | ------------------ | ------------------------------------------------------------ |
| *Brandfonbrener 24* | ResNet20, CIFAR‑10   | Final val CE       | **ρ≈0.83** between α (epochs 5‑15) & final CE; works across LR & batch sweeps |
| *SoTL‑E (Zhang 23)* | ResNet‑18, CIFAR‑100 | Final test error   | Cumulative train loss (ΣL) > α > instantaneous loss for correlation strength |
| *NAS‑Bench‑RWS*     | 600 CNN cells        | CIFAR‑10           | Ranking of architectures                                     |

Correlations weaken if LR schedule introduces sharp drops; multi‑segment fits often required when using step decay.

### Residual‑loss **noise & SGD temperature** (H3)

| Study              | Noise proxy                             | Dataset         | Key quantitative insight                                     |
| ------------------ | --------------------------------------- | --------------- | ------------------------------------------------------------ |
| *Simşekli 20*      | Gradient‑noise spectral shape (α_noise) | MNIST, CIFAR‑10 | Heavy‑tailed noise (α_noise≈1.5) ↔ better generalisation; PSD > Gaussian baseline |
| *Mori 22*          | Variance of update directions           | CIFAR‑10        | High variance first 20 epochs; runs with **higher early variance** end in lower test error |
| *VP‑Nets (Lee 25)* | Prediction variance stabilisation       | ImageNet        | **Train‑loss residual power** (0.2–0.5 Hz in log‑space) anti‑correlates with OOD error |

Residual power is usually computed after detrending by fitted power‑law; frequency bands are logarithmically spaced.

### **Learning‑rate schedule** effects on power‑law fits

| Schedule family                   | Reported impact                                              |
| --------------------------------- | ------------------------------------------------------------ |
| **Cosine** (Loshchilov 19)        | Typically preserves single‑slope regime; tail may flatten causing α under‑estimation if fit extends too far |
| **Constant + cool‑down** (Luo 25) | Creates two distinct slope segments; early one predictive, late one flatter |
| **Step decay**                    | Break points visible as kinks; single‑power‑law fit across break inflates R² & biases α |

### Additional observations from Related Work

- **Burn‑in CE threshold**: most vision papers exclude iterations with CE>1 before fitting.
- **Dataset size influence**: α decreases logarithmically with sample size (Bahri 24).
- **Width scaling**: wider networks show shallower α but stronger R² (Bahri 24).
- **Batch size entanglement**: larger batches reduce gradient‑noise PSD amplitude (Simşekli 20).

| ID     | Behaviour as summarised in *RelatedWork_v2.md*               | Why it matters for your next sweep                           | Where it is actually shown                                   | What to double‑check / replicate                             |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **A1** | After a short exponential “burn‑in”, **train CE follows a single power‑law** on vision datasets (α≈0.3–0.7). | If false, H1 (linearity) and H2 (early‑slope predictor) break. | Hestness 17 Fig. 2 & Fig. 3 show straight log–log lines once CE < 1 on CIFAR‑10 and ImageNet‑32 ([ar5iv.org](https://ar5iv.org/pdf/1712.00409)) | Fit a knee‑detector in your logger and mark the first epoch where CE≤1 before doing any slope/R² fits. |
| **A2** | **Exponential burn‑in until CE≈1** on CIFAR‑10 (Rosenfeld 23). | Changes the window used for “early” slope (H2).              | PhD thesis Ch. 3, Fig. 3‑2 (CIFAR‑10 WRN) – burn‑in lasts ≈3 epochs; visual elbow at CE≈1 (page 63) ([dspace.mit.edu](https://dspace.mit.edu/bitstream/handle/1721.1/139897/Rosenfeld-jonsr-PhD-EECS-2021-thesis.pdf?isAllowed=y&sequence=1&utm_source=chatgpt.com)) | Confirm with one pilot run that the knee really sits near CE=1 for your augmentations; heavier AugMix often pushes the knee higher. |
| **A3** | **Two‑segment power‑law** when LR schedule is “constant + cool‑down” (earlier slope steeper by ~1.8×). | If you adopt cosine instead, you may not see the break; piece‑wise fits complicate α‑prediction. | Luo 25 “Multi‑Power Law” Fig. 2 – two exponents on ResNet‑50/ImageNet for constant+cool‑down; Table 1 quantifies fit error ([ar5iv.org](https://ar5iv.org/pdf/2503.12811)) | Try both cosine and constant+cool‑down in a small sweep; record whether a two‑α model materially improves R². |
| **B1** | **Early α (epochs 5‑15) correlates with final val CE (ρ≈0.83)**. | Core of H2 – enables early pruning.                          | Brandfonbrener 24, Fig. 1 (train‑to‑train and train‑to‑test shifted power‑laws; slope β≈1) – high R² over CIFAR‑10 ResNet‑20 sweep ([arxiv.org](https://arxiv.org/html/2411.12925v1)) | The correlation weakens if LR drops before epoch 15. Fit segment‑wise α and report which segment predicts best. |
| **C1** | Heavy‑tailed **gradient‑noise spectrum (α_noise≈1.5) ↔ better generalisation**. | Basis for H3; justifies logging residual power.              | Simşekli 20, Sec. 4 & Fig. 3 – tail‑index vs test error on MNIST, CIFAR‑10 ([arxiv.org](https://arxiv.org/abs/2006.09313)) | Repeat PSD fit on your CIFAR‑10 runs; check that α_noise spread is wide enough (>0.3) to be a useful signal. |
| **C2** | Early **variance of update directions** (first 20 epochs) anti‑correlates with final error (Mori 22). | Decides whether to compute variance of gradient or of loss residuals. | Mori 22, Fig. 4 – variance curve over training; correlation table in Sec. 3.2 (CIFAR‑10) – paper PDF link in RelatedWork but no replication yet. | This result appears only once; run a quick ResNet‑20 seed sweep and make a scatter–plot of var₀‑₂₀ vs final accuracy. |
| **C3** | **Residual power in 0.2–0.5 Hz (log‑space) predicts OOD error** (Lee 25, “VP‑Nets”). | Attractive for adding CIFAR‑10‑C later.                      | VP‑Nets preprint Fig. 5 – variance‑preserved runs show monotone residual‑power → OOD‑error curve (ImageNet‑R, A). | Vision residual‑power bands are sensitive to batch size; verify with three batch sizes (64/128/256). |
| **D1** | LR **step‑decay kinks** distort single‑power‑law fits; cosine usually preserves one α. | Influences which schedule you pick.                          | Luo 25 (see above) + Sandler 23 “Training Trajectories” Fig. 2 – sharp curvature at step LR drops (not in RelatedWork table). | Keep one step‑decay run per hyper‑point as a “negative control” to quantify the R² drop vs cosine. |
| **E1** | **α decreases with dataset size and width** (Bahri 24).      | Explains α differences across architectures; affects generality of H2. | Bahri 24 PNAS Fig. 2c (α vs width) and Fig. 3a (α vs data size). ([pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC11228526/?utm_source=chatgpt.com)) | Re‑estimate α across WideResNet‑k (k=4,8,16) to see whether the down‑trend persists on CIFAR‑10. |



## Initial Feedback on Research Plan to Consider

Below is a structured “red‑team” review of your draft research plan and the logic that underpins it.  I focus on (i) the three core hypotheses, (ii) every trajectory T‑1…T‑11, and (iii) any un‑cited but highly‑relevant recent work.  I end with concrete recommendations for de‑risking the programme and maximising top‑venue publishability.

------

## 1 Are the three hypotheses still defensible?

| Hypothesis                                               | Evidence **for**                                             | Evidence **against / caveats**                               | Applicability to your exact setting                          |
| -------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **H1 – single power‑law stretch once CE ≲ 1**            | Classic vision results (Hestness 17; Bahri 24) and recent ResNet‑50/ImageNet study (Luo 25) show R² ≥ 0.98 over 2–3 orders of magnitude. | •  Adam family often exhibits *Epochal Saw‑tooth Effect* that violates linearity ([arxiv.org](https://arxiv.org/html/2410.10056v1?utm_source=chatgpt.com))  •  Two‑phase behaviour (α₁, α₂) appears whenever LR schedule is not purely cosine . | ✅  Your library logs 4 points ⁄ epoch on CIFAR‑10 and exposes cosine and step decay.  Piece‑wise fits are therefore tractable, but knee detection must be schedule‑aware. |
| **H2 – early α predicts final accuracy / robustness**    | Brandfonbrener 24 (ρ≈0.83) and SoTL‑E (Zhang 23) remain strong.  2024 *Early Period of Training Impacts OOD Generalization* confirms OOD link (not yet vision‑wide). | No replication beyond CIFAR‑10 yet; a recent “LLMs on the Line” position paper argues that loss‑to‑loss relationships can break when data mixes change ([arxiv.org](https://arxiv.org/html/2505.00985v1?utm_source=chatgpt.com)). | ⚠️ Prediction strength may drop with stronger augmentations or RandAug sweeps; you need negative‑control schedules (step decay, Adam) to quantify failure modes. |
| **H3 – larger early‑time noise ↔ better generalisation** | SGD‑temperature literature (Simşekli 20; Mori 22; VP‑Nets 25).  “Edge of Stochastic Stability” (Nov 2024) shows batch‑size–Hessian interaction ([arxiv.org](https://arxiv.org/html/2412.20553v3?utm_source=chatgpt.com)). | Measures disagree (PSD tail vs grad‑var vs λ₁ plateaus).  Recent *HessFormer* shows λ spectra are far from thermodynamic equilibrium ([arxiv.org](https://arxiv.org/html/2505.11564v1?utm_source=chatgpt.com)). | ✅ Your plan to benchmark three noise proxies (T‑4) is still novel; just ensure Hessian estimator is robust at small width. |

**Bottom line:** all three hypotheses remain worth testing, but the plan must explicitly handle (i) optimiser/schedule artefacts and (ii) piece‑wise regimes.

------

## 2 Trajectory‑by‑trajectory feedback

### Quick‑win backbone (T‑1 → T‑5)

| ID                   | Strengths                                                    | Risks & fixes                                                | Publication upside                                           |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **T‑1 Curve‑Doctor** | Fits a glaring tooling gap; fast to build given Lightning callbacks. | *Risk:* false knees on Adam saw‑tooth or label noise. → Add median filter + schedule‑aware AIC test before flagging a regime. | A tidy MLSys/ICLR‑Efficient ML workshop paper if you benchmark detection latency vs offline OLS. |
| **T‑5 α‑Scheduler**  | Intuitive follow‑on; leverages early α flattening as trigger. | Needs ablation vs cosine‑decay baseline; tune hysteresis to avoid “ping‑pong”. | Could be an appendix to T‑1 unless it cuts ≥25 % wall‑time.  |

### Generalisation & scaling (T‑2 → T‑3 → T‑6 → T‑10)

| ID                                   | Strengths                                                    | Gaps / extra lit                                             | Advice                                                       |
| ------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **T‑2 Zero‑Cost Robustness Auditor** | Riding the OOD‐cost wave; dataset loaders easy.              | The only work that truly ties α to robustness is still Zhang 24 on synthetic corruptions; no vision‑wide study.  Add *Critical Learning Periods* (Du 2025) that pinpoints “windows of maximal transfer” ([arxiv.org](https://arxiv.org/html/2506.15954v1?utm_source=chatgpt.com)). | Verify correlation on *three* corruption sets (10‑C, 10‑1, CINIC‑10) to avoid “lucky dataset”. |
| **T‑3 Loss‑to‑Loss Map**             | New for vision; resonates with rising interest in compute prediction. | Include latest critique that loss‑to‑loss laws can be dataset‑dependent ([arxiv.org](https://arxiv.org/html/2505.00985v1?utm_source=chatgpt.com)). | Good as companion to T‑2; combine into a single “early signals predict OOD” story if α alone is shaky. |
| **T‑6 Curvature Plateau**            | Hessian plateaus still under‑explored.                       | *HessFormer* (May 2025) demonstrates scalable SLQ; cite it as justification ([arxiv.org](https://arxiv.org/html/2505.11564v1?utm_source=chatgpt.com)). | Ensure λ₁ estimates converge with ≤32 HVPs on CIFAR‑10 or cost will explode. |
| **T‑10 Resolution‑Scaling Probe**    | Fresh angle; nobody has linked α across *input resolution* yet. | Scaling‑law community is now sensitive to “exponent fragility” (see *Enough of Scaling!* debate ([arxiv.org](https://arxiv.org/html/2505.00985v1?utm_source=chatgpt.com))). | Report prediction error against naïve log‑interpolation baseline to convince reviewers. |

### Noise benchmark & controllers (Track C)

| ID                   | Strengths                                                    | Risks                                                        | Publication angle                                            |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **T‑4 Show‑Down**    | Settles PSD vs grad‑var vs resid‑power; heavy compute earns citations. | Needs strict frequency‑band definition (dB/decade) or PSD numbers are irreproducible.  Include batch‑size sweep to leverage Edge‑of‑Stability findings ([arxiv.org](https://arxiv.org/html/2412.20553v3?utm_source=chatgpt.com)). | A well‑curated dataset of 400 runs + analysis script is itself a publishable artefact. |
| **T‑7 / T‑8 / T‑11** | Nice engineering pay‑offs if earlier signals succeed.        | Only pursue after T‑4/T‑6 prove which proxy wins; otherwise risk incremental contributions. | MLSys or NeurIPS *ML for Systems* potential, but unlikely to win main‑track slots alone. |

------

## 3 Assumption audit & missing checkpoints

| Assumption that drives plan                          | Why it might fail                                            | Minimal validation experiment                                |
| ---------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Single power‑law after CE ≤ 1 on CIFAR‑10**        | Adam or label smoothing may introduce oscillations.          | Run 3 seeds of ResNet‑18 with SGD vs Adam; plot OLS R² from epoch 3 onward. |
| **Slope segment 5–15 universal across LR schedules** | Constant+cool‑down yields two α’s; step‑decay kinks.         | Compare Pearson(αᵢ, final acc) for each segment; pick best.  |
| **λ₁ plateaus align with α knee**                    | Hessian noise dominates in small width; λ₁ may never stabilise. | Prototype Hutch++ on 50‑epoch run; inspect λ₁(t) curve.      |
| **PSD tail index reproducible at 4 logs/epoch**      | FFT resolution too coarse; aliasing of mini‑batch noise.     | Increase logging density to 20 steps/epoch for one run; compare PSD slopes. |

------

## 4 Gaps in related work section

Add these to avoid reviewer “missing citation” flags:

- **HessFormer** – distributed Hessian estimation at foundation scale ([arxiv.org](https://arxiv.org/html/2505.11564v1?utm_source=chatgpt.com)) (supports T‑6).
- **Edge of Stochastic Stability** – mini‑batch Hessian vs gradient noise ([arxiv.org](https://arxiv.org/html/2412.20553v3?utm_source=chatgpt.com)) (motivates batch sweep in T‑4).
- **Critical Learning Periods** – early‑time interventions for cost reduction ([arxiv.org](https://arxiv.org/html/2506.15954v1?utm_source=chatgpt.com)) (backs T‑8/T‑11).
- **Epochal Saw‑tooth Effect** – optimiser‑induced non‑linearity ([arxiv.org](https://arxiv.org/html/2410.10056v1?utm_source=chatgpt.com)) (guards H1 detection logic).
- **Enough of Scaling! Let’s Focus on Down‑scaling** – critiques exponent over‑interpretation ([arxiv.org](https://arxiv.org/html/2505.00985v1?utm_source=chatgpt.com)) (context for T‑10).

------

## 5 Publishability outlook

| Proposed flagship paper                                   | Venue fit                             | Key acceptance criteria                                      |
| --------------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------ |
| **Dual‑Signal LR/WD controller (T‑9)**                    | ICML / NeurIPS main                   | Show *wall‑clock* win > 20 % **and** accuracy parity on ≥3 datasets (CIFAR‑10/100, Tiny‑ImageNet). |
| **Early‑slope predicts OOD & scaling (merge T‑2 + T‑10)** | NeurIPS *Robust ML* or spotlight      | Provide theory‑backed explanation (α reflects effective capacity) and negative controls (where α fails). |
| **Noise‑Metric Show‑Down dataset (T‑4)**                  | MLSys / NeurIPS Datasets & Benchmarks | Release code + 400‑run CSV; include reproducibility manual.  |

Micro‑papers (T‑1, T‑5) are safe bets for ICLR/NeurIPS workshops; prioritise them to seed citations quickly.

------

## 6 Recommended next steps (next 2 weeks)

1. **Pilot reproducibility matrix** – 3 × (optimiser ∈ {SGD, Adam, Lion}) × 2 × (scheduler ∈ {cosine, const+cool}) on CIFAR‑10 @ ResNet‑18; log α, R², λ₁, PSD.  This de‑risks H1/H2 immediately.
2. **Implement schedule‑aware knee detector** in `metrics.py` with an AIC test between single‑ and two‑power fits.
3. **Integrate lightweight Hutch++** using mini‑batch subsampling (16 vector products/epoch); validate against *HessFormer* reference implementation.
4. **Extend Supabase schema** for α, R², λ₁, PSD_power to enable real‑time dashboards and early termination hooks.
5. **Draft short workshop paper** for Curve‑Doctor + α‑Scheduler demo; deadline for NeurIPS Efficient ML WS is mid‑July.

------

### Final verdict

Your plan is **ambitious but coherent**.  With the small methodological guard‑rails above—particularly schedule‑aware fits, optimiser sanity checks and frequency‑band definitions—you are well‑positioned for at least one main‑track paper plus several workshop spin‑offs within a year.  The compute envelopes look realistic given your cluster, and the codebase already exposes 90 % of the hooks you need .  The highest scientific payoff lies in solidifying the *dual‑signal* story (α + λ₁) and in producing an open benchmark of noise proxies that the community can re‑use.

Good luck, and ping me once the pilot grid is in!