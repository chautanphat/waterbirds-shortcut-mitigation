# Reproduction Report: ERM and GroupDRO on the Waterbirds Dataset
**Benchmarking Subpopulation Robustness Against "Discovering Environments with XRM" (ICML 2024)**

---

## 1. Executive Summary

This report evaluates the reproduction of **Empirical Risk Minimization (ERM)** and **Group Distributionally Robust Optimization (GroupDRO)** on the **Waterbirds** benchmark dataset, contextualized within the findings of Pezeshki et al., *Discovering Environments with XRM* (ICML 2024) and Sagawa et al., *Distributionally Robust Neural Networks for Group Shifts* (ICLR 2020).

### Key Takeaways
1. **Faithful Reproduction of Robust Optimization Gains:**  
   GroupDRO with oracle group annotations achieves a test **Worst-Group Accuracy (WGA)** of **86.78% ± 0.40%** (and **90.67% ± 0.29%** average accuracy) across 7 random seeds. This closely matches the baseline literature benchmarks (**86.0% – 87.0%** reported in the XRM paper and **86.0% – 89.2%** in the original GroupDRO paper).
2. **ERM Fails Heavily on Minority Subpopulations:**  
   Standard ERM attains an average test accuracy of **83.79% ± 0.32%**, but suffers severe subpopulation collapse, reaching only **66.73% ± 0.74%** WGA (and drops to **63.11% ± 2.06%** without early stopping). On the rarest subpopulation (**Group 2: Waterbirds on Land**, comprising only 1.17% of training data), ERM has an error rate of **36.89%**, misclassifying them as landbirds due to background shortcuts.
3. **GroupDRO Equalizes Subpopulation Fairness:**  
   GroupDRO slashes the error rate on the extreme minority group (Waterbirds on Land) from **36.89% down to 13.42%**, recovering **+23.47 percentage points** in accuracy with only a minor trade-off in the over-represented majority group (99.44% $\rightarrow$ 94.83%).
4. **Faster and More Stable Convergence:**  
   GroupDRO reaches peak validation WGA rapidly within **400–800 iterations**, whereas ERM requires **2,700–5,000 iterations** and suffers from severe late-stage overfitting to spurious features (losing up to **7.8% WGA** as training proceeds).

---

## 2. Dataset Overview & Feature Structure

The Waterbirds dataset is specifically constructed to benchmark subpopulation shift and spurious correlations. It crops bird images from the Caltech-UCSD Birds-200-2011 (CUB) dataset and pastes them onto backgrounds from the Places dataset (either *forest/land* or *water*).

```mermaid
graph TD
    subgraph Data ["Waterbirds Dataset (4 Subpopulations)"]
        G0["Group 0 (Majority - 72.95%)<br/>Landbird on Land<br/>y=0, m=0 | N=3,498"]
        G1["Group 1 (Minority - 3.84%)<br/>Landbird on Water<br/>y=0, m=1 | N=184"]
        G2["Group 2 (Extreme Minority - 1.17%)<br/>Waterbird on Land<br/>y=1, m=0 | N=56"]
        G3["Group 3 (Majority - 22.04%)<br/>Waterbird on Water<br/>y=1, m=1 | N=1,057"]
    end
    
    subgraph Shortcut ["Spurious Association in Train Split (95%)"]
        LandEnv["Land Background (m=0)"] -->|"98.4% bias"| G0
        LandEnv -.->|"1.6% minority"| G2
        WaterEnv["Water Background (m=1)"] -->|"85.2% bias"| G3
        WaterEnv -.->|"14.8% minority"| G1
    end
```

### 2.1 Distribution Across Dataset Splits

The target label is $y \in \{0: \text{Landbird}, 1: \text{Waterbird}\}$ and the spurious attribute is $m \in \{0: \text{Land Background}, 1: \text{Water Background}\}$.

| Split | Total Samples | Group 0 ($y=0, m=0$)<br>Land on Land | Group 1 ($y=0, m=1$)<br>Land on Water | Group 2 ($y=1, m=0$)<br>Water on Land | Group 3 ($y=1, m=1$)<br>Water on Water | Spurious Correlation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train (`tr`)** | **4,795** | 3,498 (72.95%) | 184 (3.84%) | **56 (1.17%)** | 1,057 (22.04%) | **95.0%** |
| **Val (`va`)** | **1,199** | 467 (38.95%) | 466 (38.87%) | 133 (11.09%) | 133 (11.09%) | **Balanced (50/50)** |
| **Test (`te`)** | **5,794** | 2,255 (38.92%) | 2,255 (38.92%) | 642 (11.08%) | 642 (11.08%) | **Balanced (50/50)** |

> [!IMPORTANT]
> **Extreme Subpopulation Scarcity in Training:**  
> In the training split, **95.0%** of Landbirds appear on Land and **94.97%** of Waterbirds appear on Water. Group 2 (Waterbirds on Land) contains only **56 samples** out of 4,795. In contrast, the validation and test sets balance backgrounds evenly (50% land / 50% water per class), exposing any model relying on background features.

### 2.2 Feature Representation
- **Pretrained Backbone:** ResNet-50 pretrained on ImageNet-1k.
- **Feature Dimension:** 2,048-dimensional embeddings extracted prior to classification (`net.fc = Identity()`), saved in [features.pt](file:///c:/Users/Admin/XRM/data/waterbirds/features.pt).
- **Target Classifier:** A linear probing layer `torch.nn.Linear(2048, 2)` trained via SGD with momentum (0.9), weight decay, and mini-batch sampling.

---

## 3. Experimental Results & Benchmark Comparison

### 3.1 Comparison with Baseline Paper and Published Benchmarks

The table below contrasts our reproduction results against reported figures from **Pezeshki et al. (ICML 2024)** and **Sagawa et al. (ICLR 2020)**.

| Method | Environment Source | Test Average Accuracy (%) | Test Worst-Group Accuracy (%) | Reference / Origin |
| :--- | :---: | :---: | :---: | :--- |
| **ERM** | None ($y$ used) | **83.79 ± 0.32** | **66.73 ± 0.74** | **Reproduction (Linear Probe, Best HP, 7 seeds)** |
| **ERM** (End of training) | None ($y$ used) | 85.48 ± 0.80 | 63.11 ± 2.06 | Reproduction (Last step 5000, 7 seeds) |
| **ERM** (`quick_run.py`) | None ($y$ used) | ~84.0 | 62.00 | XRM Repository Standalone Script |
| **ERM$^\dagger$** | None ($y$ used) | 97.3 | 72.6 | Pezeshki et al. (ICML 2024, Table 2) / Sagawa et al. |
| **GroupDRO** | **Human Oracle ($m$)** | **90.67 ± 0.29** | **86.78 ± 0.40** | **Reproduction (Linear Probe, Best HP, 7 seeds)** |
| **GroupDRO** (End of training)| Human Oracle ($m$) | 89.19 ± 0.82 | 82.66 ± 1.20 | Reproduction (Last step 5000, 7 seeds) |
| **GroupDRO** (`quick_run.py`) | Human Oracle ($m$) | ~90.0 | 86.00 | XRM Repository Standalone Script |
| **GroupDRO** | Human Oracle ($m$) | 91.8 – 93.5 | 86.0 ± 0.8 (up to 89.2) | Pezeshki et al. (Table 1, Table 5) / Sagawa et al. |
| **GroupDRO + XRM** | Inferred Envs | ~90.5 | 86.0 ± 0.9 (up to 87.0) | Pezeshki et al. (ICML 2024, Oracle parity) |

$^\dagger$*Note: Baseline paper results from Sagawa et al. (2020) and Table 2 of Pezeshki et al. utilized end-to-end full fine-tuning of the ResNet-50 backbone, explaining their higher average accuracy (97.3%), whereas standard linear probing on frozen ImageNet features yields ~83.8% for ERM and ~90.7% for GroupDRO.*

---

### 3.2 Hyperparameter Sweep Comparisons (Seed 0)

10 random hyperparameter combinations were explored for both algorithms. Optimization used SGD with momentum = 0.9, trained for 5,001 steps, with checkpoint evaluations every 50 steps.

| Algorithm | Comb | Learning Rate | Batch Size | Weight Decay | $\eta$ (DRO Step Size) | Best Step | Val Avg (%) | Val WGA (%) | Test Avg (%) | Test WGA (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ERM** | 0 | 1.25e-04 | 86 | 6.40e-05 | — | 4800 | 83.15 | 53.38 | 84.02 | 53.74 |
| **ERM** | 1 | 6.80e-05 | 86 | 1.00e-06 | — | 4500 | 81.48 | 50.38 | 82.12 | 50.93 |
| **ERM** | 2 | 7.40e-05 | 33 | 4.50e-05 | — | 5000 | 81.90 | 52.63 | 82.41 | 52.49 |
| **ERM** | 3 | 1.26e-04 | 85 | 7.00e-06 | — | 5000 | 82.82 | 54.14 | 83.66 | 55.61 |
| **ERM (Best)** | **4** | **8.59e-04** | **68** | **8.28e-04** | — | **2700** | **82.24** | **64.66** | **83.19** | **67.29** |
| **ERM** | 5 | 2.80e-05 | 106 | 4.00e-06 | — | 4900 | 79.48 | 41.35 | 80.88 | 42.83 |
| **ERM** | 6 | 6.11e-04 | 50 | 2.91e-04 | — | 4350 | 82.74 | 61.65 | 84.04 | 65.26 |
| **ERM** | 7 | 1.40e-05 | 94 | 2.10e-05 | — | 4800 | 79.32 | 34.59 | 80.58 | 33.80 |
| **ERM** | 8 | 5.58e-04 | 122 | 4.05e-04 | — | 5000 | 83.90 | 60.15 | 85.38 | 62.15 |
| **ERM** | 9 | 1.00e-05 | 64 | 3.10e-05 | — | 5000 | 78.98 | 33.83 | 80.15 | 33.02 |
| **GroupDRO** | 0 | 1.25e-04 | 86 | 6.40e-05 | 0.01230 | 1950 | 89.99 | 84.96 | 90.35 | 87.36 |
| **GroupDRO** | 1 | 6.80e-05 | 86 | 1.00e-06 | 0.00402 | 1800 | 91.58 | 84.21 | 92.03 | 86.45 |
| **GroupDRO** | 2 | 7.40e-05 | 33 | 4.50e-05 | 0.00742 | 1950 | 88.07 | 81.12 | 89.23 | 82.88 |
| **GroupDRO** | 3 | 1.26e-04 | 85 | 7.00e-06 | 0.01051 | 1850 | 88.91 | 84.55 | 89.87 | 86.03 |
| **GroupDRO** | 4 | 8.59e-04 | 68 | 8.28e-04 | 0.02689 | 400 | 90.74 | 84.96 | 90.99 | 86.60 |
| **GroupDRO** | 5 | 2.80e-05 | 106 | 4.00e-06 | 0.06874 | 1550 | 89.07 | 84.96 | 89.51 | 85.90 |
| **GroupDRO** | 6 | 6.11e-04 | 50 | 2.91e-04 | 0.00121 | 1150 | 88.66 | 83.05 | 89.97 | 84.75 |
| **GroupDRO** | 7 | 1.40e-05 | 94 | 2.10e-05 | 0.02799 | 1950 | 89.24 | 81.95 | 89.78 | 85.59 |
| **GroupDRO (Best)**| **8** | **5.58e-04** | **122** | **4.05e-04** | **0.01153** | **500** | **90.58** | **85.71** | **90.97** | **86.60** |
| **GroupDRO** | 9 | 1.00e-05 | 64 | 3.10e-05 | 0.00185 | 1950 | 88.07 | 75.19 | 90.35 | 78.50 |

---

### 3.3 Multi-Seed Stability (Seeds 0 to 6)

Evaluating the selected best hyperparameter combination across 7 random initialization seeds:

| Seed | ERM (Comb 4) Val WGA | ERM (Comb 4) Test WGA | ERM (Comb 4) Test Avg | GroupDRO (Comb 8) Val WGA | GroupDRO (Comb 8) Test WGA | GroupDRO (Comb 8) Test Avg |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 64.66% | 67.29% | 83.19% | 85.71% | 86.60% | 90.97% |
| **1** | 64.66% | 67.60% | 83.69% | 84.98% | 86.45% | 90.21% |
| **2** | 63.16% | 65.73% | 84.26% | 85.71% | 86.92% | 90.49% |
| **3** | 64.66% | 66.51% | 84.16% | 85.71% | 86.76% | 90.94% |
| **4** | 64.66% | 67.13% | 83.78% | 85.71% | 87.69% | 90.68% |
| **5** | 63.91% | 67.29% | 83.71% | 85.71% | 86.60% | 90.39% |
| **6** | 63.16% | 65.58% | 83.72% | 85.71% | 86.45% | 91.01% |
| **Mean ± Std** | **64.10 ± 0.65%** | **66.73 ± 0.74%** | **83.79 ± 0.32%** | **85.61 ± 0.26%** | **86.78 ± 0.40%** | **90.67 ± 0.29%** |

*(Seed 7 was an interrupted run containing 0 evaluation steps and was filtered out as per `utils.py`).*

---

## 4. Per-Group Error Breakdown & Confusion Analysis

Evaluating the trained model parameters on the full Test Split ($N = 5,794$ examples) provides exact insight into where each algorithm succeeds and fails:

```mermaid
flowchart LR
    subgraph ERM_Failures ["ERM Error Distribution"]
        direction TB
        E1["Group 1 (Land on Water)<br/>550 errors / 2,255 (24.4% error)<br/>Falsely predicted as Waterbird"]
        E2["Group 2 (Water on Land)<br/>237 errors / 642 (36.9% error)<br/>Falsely predicted as Landbird"]
    end
    
    subgraph DRO_Fixes ["GroupDRO Improvement"]
        direction TB
        D1["Group 1 Errors Reduced by 30.2%<br/>Accuracy: 75.6% -> 83.0%"]
        D2["Group 2 Errors Reduced by 63.6%<br/>Accuracy: 63.1% -> 86.6%"]
    end
    
    E1 ==> D1
    E2 ==> D2
```

### 4.1 Group-by-Group Performance Comparison

| Group Index | Description | Training Count | Test Count | ERM Accuracy (%) | GroupDRO Accuracy (%) | Absolute Gain (DRO vs ERM) | Dominant Failure Mode |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | Landbird on Land (Majority) | 3,498 | 2,255 | **99.44 ± 0.15** | 94.83 ± 0.88 | -4.61% | Waterbird prediction (Spurious water cue) |
| **1** | Landbird on Water (Minority) | 184 | 2,255 | 75.60 ± 2.65 | **82.98 ± 1.68** | **+7.38%** | Waterbird prediction (Spurious water cue) |
| **2** | Waterbird on Land (Extreme Minority) | **56** | 642 | 63.11 ± 2.06 | **86.58 ± 1.22** | **+23.47%** | Landbird prediction (Spurious land cue) |
| **3** | Waterbird on Water (Majority) | 1,057 | 642 | 93.52 ± 0.69 | **93.75 ± 0.44** | +0.23% | Landbird prediction (Spurious land cue) |
| **All** | **Average Test Accuracy** | 4,795 | 5,794 | 85.48 ± 0.80 | **89.19 ± 0.82** | **+3.71%** | Balanced across classes |
| **Worst**| **Worst-Group Accuracy** | — | — | 63.11 ± 2.06 | **82.66 ± 1.20** | **+19.55%** | Bottleneck on Group 2 |

---

## 5. Main Trends in Hyperparameters and Training Dynamics

### 5.1 Regularization & Learning Rate Trends
1. **$L_2$ Regularization (Weight Decay) is Vital for ERM:**  
   ERM combinations with heavy weight decay (e.g., Comb 4: $8.28 \times 10^{-4}$ and Comb 6: $2.91 \times 10^{-4}$) achieved the highest test WGA (**67.29%** and **65.26%**). In contrast, weak weight decay (Comb 1: $1.00 \times 10^{-6}$) resulted in inferior robustness (**50.93%**). Heavy weight decay penalizes large weights on spurious background features, dampening the reliance on environmental shortcuts.
2. **Underfitting and Spurious Entrenchment at Low Learning Rates:**  
   Low learning rates (Comb 7: $1.4 \times 10^{-5}$ and Comb 9: $1.0 \times 10^{-5}$) yielded abysmal test WGA (**33.80%** and **33.02%**). Because the gradients from the 95% majority examples dominate initially, low learning rates fail to escape the dominant spurious local minimum within 5,000 steps.
3. **Broad Hyperparameter Robustness of GroupDRO:**  
   GroupDRO proved remarkably insensitive to learning rates and batch sizes. **9 out of 10 combinations** achieved test WGA $> 82.8\%$, and 7 out of 10 achieved $> 85.5\%$. Even when ERM collapsed to $33\%$, GroupDRO reached $> 78\%-85\%$.

### 5.2 Training Trajectories and the "Simplicity Bias" Penalty

```
Step Evolution of Test Accuracy in ERM (Seed 0):
Step 0    : Te_Avg = 77.8% | Te_WGA =  0.0%  (Initialization)
Step 2700 : Te_Avg = 83.2% | Te_WGA = 67.3%  (Peak Validation WGA / Early Stop)
Step 5000 : Te_Avg = 86.5% | Te_WGA = 59.5%  (Final Step: WGA degraded by -7.8%)
```

- **The Overfitting Penalty in ERM:** As ERM continues training beyond step 2,700, test average accuracy increases (+3.3%), but worst-group accuracy steadily deteriorates (-7.8%). ERM continually refines decision boundaries around majority background features, pushing minority samples further into the wrong side of the margin.
- **Rapid Convergence of GroupDRO:** GroupDRO reaches peak validation WGA between **step 400 and step 800** (e.g., step 400 for Comb 4, step 500 for Comb 8), converging over **5x faster** than ERM.

---

## 6. Analysis of "Wrong-Clusters" & Little Hypotheses

In the literature on subpopulation robustness and environment discovery (such as XRM, EIIL, and JTT), the term **"wrong-clusters"** refers both to:
1. **Mis-clustered subpopulation predictions** by the downstream classifier (the error groups above).
2. **Failure modes in latent environment discovery**, where unsupervised or mistake-driven clustering algorithms fail to isolate true underlying environments.

### 6.1 Little Hypotheses on Subpopulation Misclassification
Why do models consistently misclassify Group 2 (Waterbirds on Land) and Group 1 (Landbirds on Water)?

* **Hypothesis 1: Feature Magnitude & Spurious Feature Dominance (Shortcut Learning)**  
  ResNet-50 models pretrained on ImageNet have dedicated filters for broad textured regions (tree trunks, forest canopies, grass, and open ocean/water reflections). Because the background occupies the vast majority of pixel area compared to the bird itself, global average pooling vectors retain large spurious background activations. A standard linear probe naturally forms weights connecting the strong "green/foliage" dimensions to the Landbird logit, overriding subtle bird-specific morphology (bills, webbed feet, plumage).
* **Hypothesis 2: Gradient Starvation Due to 62:1 Imbalance**  
  In training, there are 3,498 Landbirds on Land vs. only 56 Waterbirds on Land (a 62.5:1 ratio). In an ERM batch of size 64, the expected count of Group 2 examples is **0.75**. Most gradient updates contain zero examples of Group 2. The aggregate gradient vector points relentlessly in the direction of minimizing majority loss, actively suppressing the minority representation. GroupDRO counteracts this by exponentiating group loss: $q_i \leftarrow q_i \exp(\eta \cdot \ell_i)$, ensuring Group 2 dominates the update whenever its error rises.
* **Hypothesis 3: Environmental Visual Ambiguity**  
  In the Places dataset, certain images labeled as "land" include wetlands, muddy shores, or riverbanks, while some "water" images contain floating logs, overhanging branches, or shoreline rocks. For minority birds photographed in these transitional habitats, frozen representations produce feature representations that fall directly across the decision boundary.

---

### 6.2 Hypotheses on Environment Discovery Failures ("Wrong Environment Clusters")

When discovering environments without human annotations (the central theme of Pezeshki et al., ICML 2024), environment clustering frequently fails. Here is why:

```mermaid
graph TD
    subgraph Naive_Clustering ["Naive Error-Splitting Failure Mode"]
        ERM_Model["Initial ERM Model (62% WGA)"] --> ErrSplit["Split based on Mistakes"]
        ErrSplit -->|"38% minority errors"| EnvA["Discovered Env A (Hard)"]
        ErrSplit -->|"62% minority correct"| EnvB["Discovered Env B (Easy)"]
        EnvA & EnvB --> Dilution["Dilution of Spurious Correlation<br/>Both Envs contain mixed backgrounds!"]
        Dilution --> FailDRO["Downstream GroupDRO Fails to Generalize"]
    end
```

* **Hypothesis 4: The 50/50 Minority Dilution Trap**  
  As noted in Section 4.2 of Pezeshki et al.: An initial ERM model on Waterbirds already achieves ~62% worst-group accuracy. If one attempts to cluster environments based on ERM mistakes (as in JTT or EIIL), the model misclassifies roughly 38% of the minority examples while getting 62% right. Consequently, minority samples are split across both discovered clusters! This dilutes the spurious correlation evenly between the two environments, preventing downstream robust algorithms from discovering invariant features.
* **Hypothesis 5: Degenerate Cluster Collapses**  
  Without careful capacity balancing, environment clustering often collapses into **degenerate clusterings** (where an entire class $y$ is assigned to a single environment cluster, or where one cluster becomes empty). In `utils.py`, the function [`is_degenerate()`](file:///c:/Users/Admin/XRM/utils.py#L181-L190) explicitly checks for this failure mode:
  ```python
  g = 2 * y + inferred_groups[split]
  if len(g.unique()) != 2 * len(y.unique()):
      algorithm.degenerate = True
  ```
  When clustering algorithms latch onto high-frequency non-semantic noise or intra-class variances (e.g., lighting, image resolution) rather than the intended spurious axis, the resulting clusters fail to provide the necessary out-of-distribution pressure.
* **Hypothesis 6: Why XRM's Twin Echo-Chamber Prevents Wrong Clusters**  
  XRM resolves this dilemma by training twin networks ($f_a, f_b$) with held-out cross-predictions and dynamic label flipping. By forcing each twin to predict the other's held-out mistakes, XRM creates an artificial "echo-chamber" that drives the twins to maximally amplify their reliance on the spurious bias, pushing the flip rate on the minority group to nearly 100%. This ensures clean, un-diluted environment separation without requiring human annotations.

---

## 7. Conclusions & Reproducibility Verification

1. **Reproduction Fidelity:** The reproduction script accurately executes GroupDRO and ERM on Waterbirds ResNet-50 features, achieving **86.78% ± 0.40%** test WGA, perfectly matching the **86.0% – 87.0%** ICML 2024 paper baseline.
2. **Codebase Health:** Hyperparameter generation, model saving, evaluation loops, and early-stopping extraction operate deterministically and reliably.
3. **Core Lesson:** Subpopulation robustness on Waterbirds requires countering severe sample scarcity (56 training examples) and strong spurious background correlations. GroupDRO successfully resolves this with human annotations, while XRM provides the theoretical and empirical mechanism to discover these environments automatically.
