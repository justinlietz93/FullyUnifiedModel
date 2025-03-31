### B. Phase 2: Tandem Complexity Scaling (Refinement and Competence)

#### B.1. Objective
Refine the initial graph structure, strengthen domain-specific pathways, build robust cross-domain associations, and achieve baseline competence (>85% accuracy target) on more complex tasks using a curated curriculum (target: up to 300 total inputs).

#### B.2. Cellular Components & Mechanisms
*   **Data Curriculum:** Sequentially introduce batches of data with increasing complexity.
*   **Training Loop (Enhanced):** Iterate through data batches.
    *   **For each batch/input item:**
        *   Execute core **Simulation Loop**, **STDP Calc**, **Trace Update** as in Phase 1.
        *   **Targeted SIE Feedback (Crucial):**
            *   **Decoder Module:** Generate task-specific output.
            *   **Evaluation:** Compare output rigorously against target -> Reward `r`.
            *   **Advanced Reward Calculation (MI100):** Compute `total_reward = TD_error + novelty - habituation + self_benefit`. TD error becomes more significant as `V_states` learns.
        *   **SIE-Modulated STDP Update (7900 XTX):** Apply weight update using `eta_effective = eta * (1 + mod_factor)` where `mod_factor` is derived from `total_reward`.
        *   **Intrinsic Plasticity Update (7900 XTX).**
        *   **Knowledge Graph Monitoring:** Periodically analyze `w` (strength, sparsity, centrality).
        *   **Performance Tracking:** Log SIE rewards per domain/cluster.
        *   **Adaptive Clustering (MI100):** Run every 1000 steps to update clusters (using dynamic `k`) and `V_states` mapping.
        *   **Reward-Driven Structural Plasticity (Initiation):**
            *   **Trigger:** If `avg_reward[c] < 0.5` over 1000 steps.
            *   **Mechanism:** Activate Growth algorithm (Sec 4.C.2) for cluster `c`.

#### B.3. Mathematical Formulations
1. **STDP Learning Rule (Excitatory/Inhibitory):** As defined in Sec 2.B.
2. **Eligibility Trace:** `e_ij(t) = 0.95 * e_ij(t-1) + Δw_ij(t)`.
3. **SIE Modulation:** `eta_effective = 0.01 * (1 + (2 * sigmoid(total_reward) - 1))`.
4. **TD Learning:** `TD_error = r + 0.9 * V(next_state) - V(current_state)`; `V(state) += 0.1 * TD_error`.
5. **Cluster Coherence Metric (Silhouette Score):** Used to determine `k` for k-means.

#### B.4. Expected Outcome
Knowledge graph significantly refined, strong intra-domain pathways (`w[i,j] ≈ 0.8`), emerging inter-domain connections. Baseline competence (>85% accuracy) achieved. Minor structural growth may have occurred.

---

