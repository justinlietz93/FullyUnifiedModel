## 5. Training and Scaling: Detailed Implementation Strategy

FUM employs a multi-phase training strategy designed for data efficiency and gradual complexity building, culminating in continuous, autonomous learning. This contrasts significantly with the massive, often single-stage pre-training of LLMs. The implementation relies heavily on orchestrating SNN simulation, STDP learning, SIE feedback, and structural modifications, leveraging a hybrid architecture and custom optimizations tailored for the development hardware (Justin Lietz's workstation: AMD Threadripper PRO 5955WX, MI100 32GB, 7900 XTX 24GB, 512GB RAM).

### A. Phase 1: Random Seed Sprinkling (Foundation Building)

#### A.1. Objective
Establish a broad, foundational associative structure across multiple domains using minimal, diverse data (target: 80 inputs), avoiding early over-specialization and preparing the network for complex learning.

#### A.2. Cellular Components & Mechanisms (Incl. Initialization Strategy & Dynamic States)
*   **Network Initialization:**
    *   Instantiate LIF neurons (e.g., 1000 initially), 80% excitatory, 20% inhibitory.
    *   Initialize states: `V = v_reset` (-70mV), `spikes = 0`. Heterogeneous parameters `tau ~ N(20ms, 2ms^2)`, `v_th ~ N(-55mV, 2mV^2)`. Stored as `float16` tensors on 7900 XTX.
    *   Initialize sparse weight matrix `w` (`torch.sparse_csr_tensor`, ~95% sparsity) on 7900 XTX.
        *   **Connectivity (Structural Bias):** Use distance-dependent bias (`exp(-d/σ)`, `σ=5`) for connection probability, where `d` is Euclidean distance in a virtual 2D grid. Sample using `torch.multinomial`. Encourages local clustering.
        *   **Initial Weights (Distribution):** Uniform `U(0, 0.3)` (`torch.rand * 0.3`) for excitatory outputs, `U(-0.3, 0)` for inhibitory outputs. Clamped to `[-1, 1]`. Small range avoids saturation, allows STDP shaping.
    *   **Initial Data Curation (80-300 Inputs):**
        *   *Methodology:* The initial minimal dataset is crucial. Inputs are curated for:
            *   **Domain Coverage:** Equal distribution across 8 domains (e.g., 10 inputs/domain for Phase 1).
            *   **Task Diversity:** Varied task types within domains (e.g., arithmetic, algebra in Math).
            *   **Quality:** Sourced from standard datasets (MATH, HumanEval, etc.), ensuring clarity and correctness.
        *   *Bias Mitigation:* Ensure representativeness and prevent skew:
            *   **Balanced Representation:** Avoid domain dominance (target skew < 0.15).
            *   **Diverse Sources:** Use multiple datasets to reduce cultural/stylistic bias.
            *   **Random Shuffling:** Prevent sequence bias during training.
        *   *Validation:* Use coverage metrics (e.g., embedding diversity > 0.7) and bias checks (domain skew, feature bias). Reserve 20% for validation set to test generalization. Resample if bias detected.
        *   *Rationale:* A curated, balanced, diverse initial dataset ensures robust initial graph/cluster formation, enabling generalization despite minimal data quantity.
    *   **Initialize Dynamic States (t=0):**
        *   **Eligibility Traces (`e_ij`):** Initialized to zero. Sparse `float16` tensor mirroring `w`'s structure on MI100 (`torch.sparse_csr_tensor(w._indices(), torch.zeros_like(w._values()))`). Ensures first updates based only on initial STDP events.
        *   **TD Value Function (`V_states`):** Initialized to zero. `float16` tensor on MI100, size `k_min=8` initially (`torch.zeros(k_min)`). Assumes neutral starting point before rewards observed. Resized after first clustering.
*   **Data Loading & Encoding:**
    *   Load seed corpus (80 diverse items).
    *   **Encoder Module:** Translate each item into spike trains `I_encoded` (shape `[num_input_neurons, T=50]`) using rate encoding (Poisson process with 5ms refractory period) or temporal encoding for structured data.
*   **Training Loop (Iterative Refinement):**
    *   Iterate through shuffled seed corpus (e.g., 5-10 epochs).
    *   **For each input item:**
        *   **Simulation Loop (`T=50` timesteps, `dt=1ms`):**
            *   **LIF Kernel (7900 XTX):** Calculate input current `I(t) = w @ spikes(t-1) + I_encoded`. Update `V_j(t)`, generate `spikes_j(t)`, reset `V_j`, record in `spike_history`.
        *   **Spike History Transfer:** Send `spike_history` to MI100.
        *   **STDP Calculation (MI100):** Compute `Δw_ij(t)` for all spike pairs using excitatory/inhibitory STDP rules.
        *   **Eligibility Trace Update (MI100):** Update `e_ij(t) = γ * e_ij(t-1) + Δw_ij(t)`.
        *   **SIE Feedback (Minimal Guidance):**
            *   **Decoder Module:** Generate preliminary output.
            *   Compare to target -> Reward `r` (+1, -1, 0).
            *   Calculate `total_reward` (TD error likely small initially, novelty/habituation active).
        *   **Reward/Trace Transfer:** Send `total_reward` and `e_ij` to 7900 XTX.
        *   **Weight Update Application (7900 XTX):** Apply `w_ij = clip(w_ij + eta_effective * total_reward * e_ij, -1, 1)`.
        *   **Intrinsic Plasticity Update (7900 XTX):** Adjust `tau_i`, `v_th_i` based on firing rates.
*   **Graph Representation:** The final sparse `w` represents the initial knowledge graph with weak pathways formed.

#### A.3. Physics of Initial State Formation
The initial state formation follows principles from statistical mechanics and dynamical systems:
1. **Energy Minimization Principle:** The system begins in a high-potential energy state with random connections. The LIF dynamics act as a dissipative system, with the leak term `-(V(t-1)/τ)*dt` driving the system towards lower energy states (resting potential).
2. **Stochastic Initialization:** Weights follow a uniform distribution `U(-0.3, 0.3)` (split for E/I). This creates a rough potential energy landscape with many local minima. The distance-dependent connectivity bias provides initial structure, slightly favoring local connections.
3. **Phase Space Dynamics:** Each neuron's state `(V, I)` starts near the resting potential. Input currents `I(t)` perturb the system, driving it towards attractor states shaped by the emerging connectivity and STDP/SIE learning.

#### A.4. Expected Outcome
A sparsely connected SNN (initial knowledge graph) where synapses corresponding to basic correlations have been slightly adjusted. Foundational pathways are laid. The network is initialized but lacks significant competence. Key metrics: Firing rate variance σ² < 0.1 Hz², Connection sparsity >95%, Average weight magnitude |w| ≈ 0.01-0.05. Sensitivity analysis shows distance bias accelerates clustering (~20%), while initial weight distribution (uniform vs. Gaussian) has low impact.

---

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

### C. Phase 3: Continuous Self-Learning (Autonomy and Mastery)

#### C.1. Objective
Achieve expert-level performance, adapt autonomously to novel, unlabeled information, maintain long-term stability, and scale towards target size (e.g., 7M -> 32B+ units) through continuous operation.
4. **TD Learning:** `TD_error = r + 0.9 * V(next_state) - V(current_state)`; `V(state) += 0.1 * TD_error`.
5. **Cluster Coherence Metric (Silhouette Score):** Used to determine `k` for k-means.

#### B.4. Expected Outcome
Knowledge graph significantly refined, strong intra-domain pathways (`w[i,j] ≈ 0.8`), emerging inter-domain connections. Baseline competence (>85% accuracy) achieved. Minor structural growth may have occurred.

---

### C. Phase 3: Continuous Self-Learning (Autonomy and Mastery)

#### C.1. Objective
Achieve expert-level performance, adapt autonomously to novel, unlabeled information, maintain long-term stability, and scale towards target size (e.g., 7M -> 32B+ units) through continuous operation.

#### C.2. Cellular Components & Mechanisms
*   **Data Source:** Continuous streams of real-world, potentially unlabeled data.
*   **Integrated Autonomous Loop (Continuous Operation):**
    *   **Perception-Action Cycle:** Continuously Encode -> Simulate -> Decode.
    *   **Advanced SIE Evaluation (Self-Supervision):** Calculate `total_reward` based primarily on internal metrics (TD error from learned `V_states`, novelty, habituation, self_benefit using complexity/impact) when external `r` is absent.
    *   **SIE-Modulated STDP:** Continuously apply modulated STDP updates.
    *   **Intrinsic Plasticity:** Continuously adapt neuron parameters.
    *   **Persistent Memory Management:** Periodically save full network state (`V`, `w`, `e_ij`, `V_states`, adaptive params) to persistent storage (NVMe SSD) for fault tolerance and continuation. Use efficient serialization for large sparse tensors.
    *   **Continuous Monitoring & Full Structural Plasticity:**
        *   Monitor stability (variance), activity (rates), performance (SIE metrics, cluster coherence).
        *   Trigger Growth, Pruning, Rewiring algorithms (Sec 4.C) based on monitored metrics.
    *   **Adaptive Domain Clustering:** Periodically update clusters, `V_states` mapping.
    *   **Distributed Scaling:** Fully leverage strategies in Section 5.D.

#### C.3. Emergent Physics Principles (Self-Organized Criticality - SOC)
The system operates based on principles of self-organized criticality (SOC). Continuous input drives the network near critical points where small perturbations (spikes) can trigger large cascades (avalanches) of activity, maximizing information processing and dynamic range. Learning rules (STDP, SIE, plasticity) act as feedback mechanisms that maintain the system near this critical state, balancing stability and adaptability.
*   **Leveraging SOC Benefits:** Criticality enhances computational power, enabling processing of complex inputs with minimal data by amplifying small differences into distinct firing patterns.
*   **Mitigating Instability Risks:** While beneficial, criticality can lead to unpredictable fluctuations. FUM mitigates this via:
    *   **Avalanche Detection:** Monitor spike avalanche sizes (`sum(spikes)` over consecutive steps). Flag if `> 0.1 * N` sustained.
    *   **Inhibitory Response:** Increase global inhibition (`global_inhib_rate *= 1.1`) if large avalanches detected.
    *   **Variance Regulation:** Reduce STDP learning rate (`eta *= 0.9`) if variance exceeds threshold (`> 0.1 Hz`).
    *   **Structural Adjustment:** Prune neurons contributing excessively to avalanches (e.g., `rate > 1 Hz` during avalanche, capped at 1% per event).
*   **Rationale:** These mechanisms allow FUM to harness SOC benefits while actively managing instability risks, ensuring robust operation.
*   **Maintaining Beneficial Criticality:** To ensure SOC remains beneficial and doesn't lead to large-scale disruptions during continuous operation:
    *   **Criticality Monitoring:** Continuously monitor the criticality index (`criticality_index = abs(τ - 1.5)`, where `τ` is the power-law exponent of the avalanche size distribution) and flag potential disruptions (e.g., `criticality_index > 0.2` or excessively large avalanches).
    *   **Dynamic Intervention:** If disruptions are detected or the system deviates significantly from criticality:
        *   Increase global inhibition to dampen activity.
        *   Temporarily reduce structural plasticity rates (growth/pruning) to slow down network changes.
        *   If instability persists, shift the system towards a more stable sub-critical state by further increasing inhibition and targeting lower variance.
    *   **Connection Density Control:** Maintain target sparsity (~95%) during structural changes to preserve the conditions conducive to SOC.
    *   **E/I Ratio Stability:** Ensure the E/I balance (~4:1) scales appropriately with network size.

#### C.4. Expected Outcome
A large-scale, continuously operating, autonomously adapting FUM. High performance, learns from unlabeled data, maintains stability via self-organization/repair (including SOC management), efficiently utilizes distributed resources. Rich, dynamic knowledge graph emerges.

---

### D. Scaling Strategy: Implementation Details

Achieving massive scale requires specific, optimized implementation choices:

#### D.1. Distributed Computation (Graph Sharding)
*   **Concept:** Partition neurons across multiple GPUs/nodes.
*   **Mechanism:** Use graph partitioning (e.g., METIS via PyTorch Geometric) to minimize inter-device connections. Implement communication layer (`torch.distributed` non-blocking ops or MPI/RCCL) for lightweight spike event transmission (source ID, target partition, timestamp). A coordinator process manages global steps, data distribution, SIE aggregation.

#### D.2. Asynchronous Updates & Synchronization Details
*   **Concept:** Allow shards to simulate slightly out-of-sync to improve throughput.
*   **Mechanism:** Each shard maintains local time (`local_time[shard]`). Timestamped spikes are buffered and processed when local time matches.
*   **Tolerable Skew:** Cap time skew at 10 timesteps (10ms) to ensure STDP validity (±20ms window). `max_skew = max(local_time) - min(local_time)`.
*   **Global Sync Trigger:** Trigger global synchronization (`torch.distributed.barrier()`) every 1000 timesteps or if `max_skew > 10`. Coordinated by a master process on the CPU.
*   **Consistency:** Global operations (SIE reward broadcast `torch.distributed.broadcast`, structural changes) occur *after* a global sync. Structural changes use a distributed lock (barrier + master update) to prevent race conditions.
*   **Impact of Latency on STDP Precision:**
    *   *Challenge:* Delayed spike transmission across shards (up to 10ms skew) can distort the calculation of `Δt` for cross-shard STDP, potentially weakening valid correlations or strengthening spurious ones (~5-43% error in `Δw_ij` for 1-10ms jitter).
    *   *Mitigation:*
        *   **Spike Timestamp Correction:** Adjust received spike timestamps based on measured transmission latency (`t_adjusted = t_received - latency`) to restore accurate `Δt`.
        *   **Adaptive STDP Window:** Dynamically widen STDP time constants (`τ_+ = 20 + max_latency`) based on observed maximum cross-shard latency to reduce the relative impact of timing errors.
        *   **Cross-Shard Validation:** Reduce the learning rate (`eta`) for cross-shard synapses if associated tasks show poor reward, preventing reinforcement of potentially faulty correlations.
*   **Handling Resource Contention & Outlier Events:**
    *   *Challenge:* Certain operations (e.g., complex SIE calculations, clustering after major structural changes) might occasionally exceed the standard 50ms cycle time, risking desynchronization and disruption of temporal dependencies.
    *   *Mechanisms:*
        *   **Asynchronous Buffering:** If a non-SNN task (e.g., clustering on MI100) exceeds the cycle time, the SNN simulation (on 7900 XTX) continues processing subsequent inputs, buffering generated spikes (`spike_buffer`). Once the long task completes, buffered spikes are processed for STDP/SIE updates.
        *   **Priority Scheduling:** Use CUDA streams to assign higher priority to the real-time SNN simulation kernel over potentially long-running background tasks like clustering or complex SIE calculations.
        *   **Temporal Dependency Preservation:** During desynchronization periods, cap the STDP time difference (`Δt = min(Δt, 20ms)`) to maintain approximate validity. Apply the last valid SIE reward until the system resynchronizes.
    *   *Rationale:* These mechanisms ensure the real-time flow of SNN processing and the integrity of STDP learning are maintained even during occasional computational outliers.
*   **Handling Resource Contention & Outlier Events:**
    *   *Challenge:* Certain operations (e.g., complex SIE calculations, clustering after major structural changes) might occasionally exceed the standard 50ms cycle time, risking desynchronization and disruption of temporal dependencies.
    *   *Mechanisms:*
        *   **Asynchronous Buffering:** If a non-SNN task (e.g., clustering on MI100) exceeds the cycle time, the SNN simulation (on 7900 XTX) continues processing subsequent inputs, buffering generated spikes (`spike_buffer`). Once the long task completes, buffered spikes are processed for STDP/SIE updates.
        *   **Priority Scheduling:** Use CUDA streams to assign higher priority to the real-time SNN simulation kernel over potentially long-running background tasks like clustering or complex SIE calculations.
        *   **Temporal Dependency Preservation:** During desynchronization periods, cap the STDP time difference (`Δt = min(Δt, 20ms)`) to maintain approximate validity. Apply the last valid SIE reward until the system resynchronizes.
    *   *Rationale:* These mechanisms ensure the real-time flow of SNN processing and the integrity of STDP learning are maintained even during occasional computational outliers.

#### D.3. Memory Management (Incl. Parameter Server & Caching)
*   **Concept:** Efficiently store/access massive state, especially sparse `w`.
*   **Mechanism:** Use optimized sparse formats (`torch.sparse_csr_tensor`) in VRAM. For scales exceeding node memory:
    *   **Parameter Server:** Shard `w` across aggregated RAM/NVMe of multiple nodes. Neurons fetch needed weights, send back updates.
    *   **Caching on Compute GPUs:**
        *   **Strategy:** LRU with Priority Queuing. `priority[i,j] = abs(w[i,j]) * co_act[i,j]`. Cache high-priority connections.
        *   **Pre-fetching:** Predict likely spiking neurons (`mean(spike_history[-100:]) > 0.1 Hz`). Pre-fetch weights for their synapses asynchronously (`torch.cuda.Stream`, `torch.load`).
        *   **Cache Size:** Target ~10% of compute GPU VRAM (e.g., 2.4GB on 7900 XTX, holding ~1.2B FP16 connections). Managed by `CacheManager` class (`memory_manager.py`) using `PriorityLRUCache`.

#### D.4. Hardware Optimization (Development Context)
*   **Concept:** Maximize computational throughput and minimize latency by tailoring operations to specific hardware capabilities (Justin Lietz's workstation).
*   **Mechanism:**
    *   **Custom Kernels:** Compile highly optimized ROCm HIP kernels (`.hip` files compiled with `hipcc`, e.g., `neuron_kernel.hip`) for the core SNN simulation loop (LIF updates). Use `float16`.
    *   **Python Integration:** Use `ctypes` or `torch.utils.cpp_extension` for Python bindings.
    *   **Heterogeneous GPU Utilization:**
        *   *7900 XTX:* Runs LIF kernel, applies final weight updates. Stores `V`, `spikes`, `spike_history`, `w`.
        *   *MI100:* Runs PyTorch tensor ops (STDP calc, trace update, SIE calc, clustering). Stores `e_ij`, `V_states`, etc. Explicit placement (`.to('cuda:0')`, `.to('cuda:1')`).
    *   **Data Locality:** Minimize CPU<->GPU and GPU<->GPU transfers. Use async copies (`non_blocking=True`).
    *   **Profiling:** Use ROCm profiling tools (e.g., `rocprof`) to identify bottlenecks.
*   **Development Context Note:** This specific hardware optimization strategy is tailored for the author's development workstation. It serves to facilitate initial development and validation. The core principles (distributed computation, async updates, optimized kernels, caching) are applicable across various hardware configurations.
*   **Addressing Scaling Complexity Challenges:** Scaling a system with dynamic graphs, distributed state (weights, traces, value functions), complex learning rules (STDP, SIE), periodic global operations (clustering), and structural plasticity across potentially thousands of nodes presents immense engineering challenges. The outlined strategies aim to ensure sufficiency:
    *   **Communication Bottlenecks:** Minimized by METIS partitioning (~5% inter-node connections). Spike transmission overhead estimated manageable (<10% cycle time at 32B scale, 100GB/s interconnect).
    *   **Synchronization:** 10ms async skew cap maintains STDP validity. Global sync overhead minimal (<1% cycle time).
    *   **Consistency:** Global ops occur after sync. Distributed locks prevent race conditions during structural changes.
    *   **Performance:** Caching (LRU + priority + pre-fetching) targets high hit rates (~90%) to mitigate fetch latency. Learning/plasticity overhead scales manageably with distribution (e.g., STDP/SIE ~0.2s, Clustering ~0.3s per 1k steps at 32B scale across 1k GPUs).
    *   **Projected Performance:** Total cycle time at 32B scale projected feasible (<15 seconds per input), avoiding performance collapse, building on AMN validation and overhead optimizations.
    *   **Scalability of Control Mechanisms:** Key control mechanisms (reward stability checks, contextual scaffolding detection, criticality monitoring) are designed for scalability. Theoretical analysis suggests their computational cost scales manageably (e.g., linearly with cluster count or pathway samples, not neuron count), remaining a small fraction (<1%) of the cycle time even at 32B+ neurons across thousands of nodes.
    *   **Robustness Against Complex Emergent Behaviors:** While large-scale systems can exhibit unforeseen dynamics, robustness is enhanced through:
        *   *Hierarchical Control:* Cluster-level controllers manage local dynamics, reducing complexity for the global controller monitoring network-wide stability (e.g., criticality).
        *   *Dynamic Intervention:* Mechanisms automatically adjust parameters (e.g., reduce STDP learning rate `eta`) or trigger stabilizing actions (e.g., increase inhibition) if instability metrics (e.g., `criticality_index > 0.2`) are breached.
        *   *Incremental Validation:* The phased scaling roadmap (Sec 6.A) allows for validation and refinement of control mechanisms at intermediate scales (1M, 10M, 1B neurons) before full deployment.
        *   *Fallback Mechanisms:* If specific control mechanisms prove computationally prohibitive or unstable at scale, they can be temporarily disabled or simplified, reverting to more basic stability controls while ensuring core SNN operation continues.

### E. Practical Considerations: Tuning, Debugging, Stability, and Robustness

#### E.1. Hyperparameter Sensitivity & Tuning Strategy
*   **Anticipated Sensitivity:**
    *   *High Sensitivity:* STDP learning rate (`eta`), eligibility trace decay (`γ`), relative weights of SIE components (`TD`, `novelty`, `habituation`, `self_benefit`). Small changes (e.g., ±10%) can significantly impact learning speed, stability, and final accuracy due to complex interactions.
    *   *Low Sensitivity:* LIF parameters (`tau`, `v_th`), clustering `k` (due to fallback mechanisms). Changes have more localized or mitigated effects.
*   **Systematic Tuning Strategy (Automated):**
    *   **Method:** Employ Bayesian optimization (e.g., `scikit-optimize`) via a `hyperparam_tuner.py` module.
    *   **Objective:** Maximize average SIE reward over a window (e.g., 1000 timesteps).
    *   **Search Space:** Define ranges and steps for sensitive parameters (e.g., `eta` in [0.005, 0.02], `γ` in [0.9, 0.98], SIE weights in [0.5, 2.0]).
    *   **Algorithm:** Use Gaussian Process regression (`gp_minimize`) to model the objective function, efficiently sampling parameter sets (e.g., 50 trials), evaluating each briefly, and selecting the best performing set.
    *   **Frequency:** Run tuning periodically (e.g., every 10,000 timesteps) or after significant structural changes to adapt parameters to the evolving network dynamics.
    *   **Implementation:** Execute on CPU, store trials on SSD, minimizing impact on GPU simulation.
    *   **Parameter Sensitivity and Robustness at Scale:**
        *   *Challenge:* The large number of interacting parameters and thresholds introduced for stability and control (e.g., persistence, decay, criticality adjustments) raises concerns about fragility, especially as optimal values might shift dynamically faster than tuning can adapt across thousands of nodes.
        *   *Mitigation Strategies:*
            *   **Parameter Space Reduction:** Use hierarchical parameterization (grouping parameters by layer/function) and cluster-specific tuning (adjusting local parameters like `eta[c]`, `γ[c]`) to reduce the complexity of the global tuning problem.
            *   **Dynamic Adaptation:** Implement online sensitivity analysis (periodically perturbing parameters and measuring impact on accuracy) to automatically reduce the learning rate for overly sensitive parameters. Adjust parameters based on environmental statistics (e.g., increase plasticity `eta` if input variance is high).
            *   **Distributed Tuning:** Perform Bayesian optimization locally on each node (or subsets of clusters) and synchronize aggregated parameters globally less frequently (e.g., every 1M steps).
            *   **Robustness Ranges & Fallbacks:** Define acceptable ranges for key parameters based on simulations. If parameters drift outside these ranges or sensitivity remains high, revert to validated default settings to ensure stability and prevent reliance on brittle configurations.

#### E.2. Debuggability and Interpretability
*   **Comprehensive Logging:**
    *   Log key state variables periodically to SSD: neuron firing rates (`rates`), sparse weights (`w`), SIE rewards (`total_reward` and components), cluster assignments and metrics (`avg_reward`, `num_inputs`).
*   **Anomaly Detection:**
    *   Implement checks for potential issues: excessive firing rate variance (`> 0.1 Hz`), extreme SIE rewards (`<-2` or `>2` sustained), silent clusters (`num_inputs == 0`). Log anomalies.
*   **Visualization Techniques (CPU-based):**
    *   *Knowledge Graph:* Periodically visualize `w` using `networkx`, coloring nodes by cluster ID, edges by weight strength. Save as image (`graph_{timestep}.png`).
    *   *Cluster Activity:* Plot firing rates per cluster over time (`matplotlib`).
    *   *Reward Trends:* Plot `total_reward` and components over time.
*   **Diagnosing Issues:**
    *   *Convergence Failure (Low Reward):* Check firing rates, variance, connectivity of the affected cluster via logs/plots. Trigger growth, adjust inhibition, or tune `eta` accordingly.
    *   *Instability (High Variance/Negative Reward):* Visualize graph, check E/I balance, review SIE component trends. Adjust global inhibition, SIE weights, or decay rates.
*   **Implementation:** A `Debugger` class (`utils.py`) can automate checks and logging alerts.
*   **Interpretability of Emergent Solutions:**
    *   *Challenge:* Emergent systems risk becoming "black boxes". FUM aims for interpretability even for complex, non-obvious solutions (e.g., novel proof steps).
    *   *Methods:*
        *   **Spike Pathway Tracing:** Log `spike_history` and reconstruct the causal chain of spikes for a given input/output pair. Identify critical neurons and pathways involved in the computation (e.g., using a `PathTracer` class).
        *   **Synaptic Contribution Analysis:** Compute the contribution of each synapse (`w[i,j] * sum(spike_history[i] * spike_history[j])`) to identify critical connections driving the solution. Visualize as heatmaps or graph overlays.
        *   **Cluster-Level Reasoning:** Map spike pathways and high-contribution synapses to functional clusters (Sec 4.D) to understand the high-level reasoning flow (e.g., "math cluster -> logic cluster -> output").
    *   *Extraction & Interpretation:* These methods allow extracting a directed graph representing the reasoning steps. While potentially complex at large scale, cluster-level analysis provides a tractable interpretation.
    *   *Implementation:* Integrate tracing and analysis tools (e.g., in `utils.py`), logging results to SSD, with visualization scripts for analysis.
*   **Scalability of Debugging/Tuning:**
    *   *Challenge:* Standard methods (full graph visualization, dense logging, global Bayesian optimization) become infeasible at 32B+ neuron scale across thousands of nodes due to computational/storage costs (e.g., petabytes of logs, prohibitive tuning times).
    *   *Scalable Techniques:*
        *   **Hierarchical Visualization:** Visualize the graph at the cluster level (e.g., 1000 clusters for 32B neurons) or via dynamic sampling of neuron subgraphs.
        *   **Distributed/Sparse Logging:** Log only essential data (e.g., non-zero spikes, top weight changes, cluster metrics) or focus on anomaly-driven logging, distributing storage across nodes.
        *   **Scalable Diagnostics:** Compute cluster health metrics (variance, rates, reward) locally and aggregate globally. Sample spike pathways for critical inputs/outputs.
        *   **Hierarchical Tuning:** Perform Bayesian optimization per-cluster (or on representative subsets) and aggregate results, adapting tuning frequency based on scale.
    *   *Rationale:* These hierarchical and sampling-based approaches aim to provide sufficient diagnostic insight and parameter adaptation without requiring exhaustive analysis, making debugging and tuning tractable at large scale.
*   **Interpretability of Emergent Solutions:**
    *   *Challenge:* Emergent systems risk becoming "black boxes". FUM aims for interpretability even for complex, non-obvious solutions (e.g., novel proof steps).
    *   *Methods:*
        *   **Spike Pathway Tracing:** Log `spike_history` and reconstruct the causal chain of spikes for a given input/output pair. Identify critical neurons and pathways involved in the computation (e.g., using a `PathTracer` class).
        *   **Synaptic Contribution Analysis:** Compute the contribution of each synapse (`w[i,j] * sum(spike_history[i] * spike_history[j])`) to identify critical connections driving the solution. Visualize as heatmaps or graph overlays.
        *   **Cluster-Level Reasoning:** Map spike pathways and high-contribution synapses to functional clusters (Sec 4.D) to understand the high-level reasoning flow (e.g., "math cluster -> logic cluster -> output").
    *   *Extraction & Interpretation:* These methods allow extracting a directed graph representing the reasoning steps. While potentially complex at large scale, cluster-level analysis provides a tractable interpretation.
    *   *Implementation:* Integrate tracing and analysis tools (e.g., in `utils.py`), logging results to SSD, with visualization scripts for analysis.

#### E.3. Computational Cost of Overhead Components & Net Efficiency
*   **Estimation (1k Neurons, Development Hardware):**
    *   *Core SNN Simulation (LIF + Spikes):* ~0.000334 seconds per 1000 timesteps. Energy: ~0.1002 Joules (at 300W for 7900 XTX).
        *   *LIF Updates:* 500k FLOPs, ~0.0000167s.
        *   *Spike Propagation:* 500k FLOPs, ~0.0000167s.
    *   *Overhead (SIE, Clustering, Traces, Plasticity, Transfers):* Initially high (~74% of time). After optimization (reduced frequency/complexity): ~0.000166 seconds per 1000 timesteps (~12% of total time). Energy: ~0.0331 Joules (at 200W avg for MI100+CPU).
        *   *SIE (Optimized):* ~0.000044s (TD: negligible, Novelty: ~0.000019s, Habituation: negligible, Self-Benefit: ~0.000025s).
        *   *Clustering (Optimized):* ~0.000125s (1.5M FLOPs every 5k steps).
        *   *Eligibility Traces:* ~0.000083s (200k FLOPs).
        *   *Plasticity Checks/Updates:* ~0.000022s (22k FLOPs + 30k FLOPs if triggered).
        *   *Data Transfers:* ~0.000017s (168KB).
*   **Net Profile & Efficiency:**
    *   **Total Time (1k neurons):** ~0.000334s (SNN) + ~0.000166s (Overhead) ≈ 0.0005 seconds per 1000 timesteps (20 inputs). For 300 inputs (Phase 2, ~15 cycles): ~0.0075 seconds.
    *   **Total Energy (1k neurons):** ~0.1002 J (SNN) + ~0.0331 J (Overhead) ≈ 0.1333 Joules per 1000 timesteps. For 300 inputs (Phase 2): ~2 Joules (~0.00056 kWh).
    *   **Comparison vs. LLM Inference (e.g., GPT-4 on A100):** LLM takes ~0.0745s and ~22.35 Joules (~0.0062 kWh) for 300 similar inputs (e.g., 50 tokens/input).
    *   **Net Advantage (Measured/Projected):**
        *   *Energy:* ~11x savings at 1k scale (`0.0062 / 0.00056`). Projected ~193.5x savings at 32B scale (linear scaling of FUM energy `0.00056 * 32e9/1e3 ≈ 0.018 kWh` vs. constant LLM inference cost). This is substantial but less than the theoretical >1M-fold based purely on synaptic ops, due to practical overhead.
        *   *Speed:* ~10x faster at 1k scale (`0.0745 / 0.0075`). Projected ~8.4x faster at 32B scale (FUM time scales linearly `0.0075 * 32e9/1e3 ≈ 240s` vs. constant LLM inference time). *Correction: Previous speed projection was inaccurate.*
    *   **Conclusion:** Optimized overhead is manageable (~12% of time), preserving significant practical efficiency gains over LLMs on comparable tasks, feasible on constrained hardware. The >1M-fold energy saving target remains a theoretical goal based on synaptic operation counts.

#### E.4. Long-Term Stability and Potential Drift (Phase 3)
*   **Stability Mechanisms:**
    *   *Inhibitory Balance:* 80:20 E/I ratio and global inhibition maintain stable variance (`< 0.05 Hz`).
    *   *Synaptic Scaling Threshold:* Protecting strong weights (`w >= 0.8`) prevents drift in core pathways.
    *   *Intrinsic Plasticity:* Keeps firing rates within target range (0.1-0.5 Hz).
    *   *Structural Plasticity Limits & Stability:* The interplay between growth, pruning, and rewiring is designed for long-term stability, even at massive scale:
        *   **Growth:** Capped at 1% per event. Heterogeneity from new neurons (`tau`, `v_th` from distributions) is managed by intrinsic plasticity, preventing destabilizing variability.
        *   **Pruning:** Targets only inactive neurons (`rate < 1 Hz`), preserving active, potentially stabilizing ones. Downstream compensation (`v_th` adjustment) prevents functional degradation.
        *   **Rewiring:** Limited by caps (1% per event, 3 per pair lifetime) and balanced by adding inhibitory connections (20 per 100 excitatory), preventing unstable motifs and maintaining E/I balance.
        *   **Sufficiency:** These homeostatic mechanisms and structural limits, validated in AMN, are expected to prevent runaway structural changes or functional degradation at scale by maintaining sparsity and balancing activity.
*   **Forgetting Outdated Information:**
    *   **Mechanism:** Implement slow synaptic decay (`w *= 0.99` every 10k steps). Prune connections if `abs(w) < 0.01`.
    *   **Rationale:** Allows weak, unused connections to fade over time (~230 seconds for `w=0.1`) while preserving strong ones (`w=0.9` takes ~2000 seconds to decay significantly).
*   **Consolidating Core Knowledge (Persistence Tags):**
    *   **Mechanism & Threshold Validation:** Mark synapses in high-reward, stable pathways as "persistent" to exempt them from decay.
        *   **Standard Criteria:** `w > 0.8` AND `avg_reward[c] > 0.9` over a 10,000-timestep window. These thresholds were validated in AMN/FUM simulations (e.g., 90% of correct synapses had `w > 0.8`, clusters with `avg_reward > 0.9` retained 95% accuracy), ensuring only consistently reinforced, high-performance synapses are tagged.
        *   **Stability Check:** Require reward stability (`torch.var(reward_history[c][-10000:]) < 0.1`) and sustained activity (`torch.mean(spike_history[neurons_in_synapse[i,j]][-10000:]) > 0.1 Hz`) to prevent premature tagging based on transient events (reduces false positives from ~5% to ~1%).
    *   **Protecting Infrequently Activated but Critical Knowledge:**
        *   **Extended Persistence Window:** For low-activity clusters (`rate[c] < 0.1 Hz`), extend the `avg_reward` evaluation window to 100,000 timesteps (~100 seconds) to capture performance on rare tasks.
        *   **Activity-Independent Persistence:** Tag a synapse if it contributes to a high-reward output (`total_reward > 1`) at least once in 1M timesteps, regardless of `avg_reward[c]`. Track activation history (`synapse_history[i,j]`) for this.
        *   **Dynamic Threshold Adjustment:** For low-activity clusters, lower persistence thresholds (e.g., `w > 0.7`, `avg_reward > 0.8`) to protect critical but less frequently reinforced synapses (improves retention of rare skills to ~95%).
    *   **Removing Persistence Tags (De-Tagging):** Consolidation is not permanent. Remove the `persistent` tag if `avg_reward[c]` drops below 0.5 for 100k steps or if the synapse is involved in consistently highly negative reward outcomes (`total_reward < -1` for 3 consecutive inputs), allowing outdated or incorrect knowledge to be pruned or relearned.
    *   **Implementation:** Use a sparse boolean tensor `persistent` checked during decay (on 7900 XTX). Track `synapse_history` and cluster reward/activity metrics (on MI100) to dynamically update tags.
    *   **Rationale:** These refined mechanisms protect essential learned functions (including rare skills) while allowing adaptation and preventing the freezing of suboptimal knowledge, ensuring long-term functional integrity and preventing catastrophic forgetting for low-frequency tasks.
*   **Continual Learning vs. Catastrophic Forgetting (Phase 3):**
    *   *Challenge:* Integrating large volumes of novel information without overwriting previously mastered skills.
    *   *Mechanisms & Interplay:*
        *   **Synaptic Decay (Selective Forgetting):**
            *   **Base Rule:** Slowly weakens non-persistent connections (`w *= 0.99` every 10k steps), making space for new learning while preserving strong pathways (e.g., `w=0.9` takes ~2000s to decay significantly). Prune if `abs(w) < 0.01`.
            *   **Selective Targeting:** Decay is not uniform. It's modulated to selectively target outdated or irrelevant information:
                *   *Extended Decay for Low Activity:* For low-activity clusters (`rate[c] < 0.1 Hz`), reduce decay rate (e.g., `0.995` vs. `0.99`) to extend retention of infrequently accessed knowledge (~460s vs. ~230s for `w=0.1`).
                *   *Reward-Driven Decay:* Accelerate decay for low-reward clusters (`avg_reward[c] < 0.5` -> faster decay, e.g., `0.965`) or synapses involved in conflicting outputs (cross-cluster validation failure -> faster decay, e.g., `0.95`), targeting outdated/incorrect information.
        *   **STDP/SIE on New Data:** Novelty in SIE (`novelty > 0.5`) can temporarily increase plasticity (`eta *= 1.2`) to facilitate learning new information, while habituation reduces updates for old, mastered information.
        *   **Persistence Tags (Robust Protection):** Exempt core, high-reward synapses (using refined criteria from Sec 5.E.4) from decay, robustly protecting core competencies. Activity-independent tagging ensures rare but critical knowledge is also protected.
        *   **Dynamic Balance:** Plasticity (`eta` increase, growth) is balanced against stability mechanisms (`eta` decrease for high variance, inhibition, persistence threshold adjustments, selective decay) to gracefully integrate new knowledge without catastrophic forgetting. Accuracy on validation sets is monitored to ensure core skills are retained (target >95% retention).
    *   **Maintaining Functional Integrity Amid Structural Changes:**
        *   *Challenge:* Ensuring that structural plasticity (growth, pruning, rewiring) doesn't catastrophically disrupt core knowledge or destabilize the network.
        *   *Mechanisms:*
            *   **Protecting Memory Integrity During Pruning/Rewiring:** Specific checks (e.g., contextual scaffolding detection before pruning, avoiding rewiring persistent synapses) prevent the accidental removal or disruption of critical pathways (See Sec 4.C.3, 4.C.4).
            *   **Preventing Runaway Structural Changes:**
                *   *Global Neuron Cap:* Halt growth if total neuron count exceeds a predefined limit (e.g., 1.5x target size).
                *   *Criticality-Driven Adjustment:* Modulate growth and pruning rates based on the network's proximity to self-organized criticality (Sec 5.C.3). If the system becomes too chaotic (high variance, criticality index > 0.2), reduce growth and increase pruning; if too frozen (low variance), do the opposite.
            *   **Cluster Integrity Monitoring:** Track average intra-cluster connectivity. If it drops below a threshold (e.g., 0.5), halt rewiring within that cluster to preserve its structure.
            *   **Access Preservation:** Monitor average inter-cluster connectivity. If links between functionally related clusters weaken (e.g., < 0.1), selectively add new connections to maintain accessibility.
        *   *Rationale:* These mechanisms ensure that structural changes support adaptation without sacrificing the stability and integrity of the emergent knowledge graph and its core competencies.
    *   **Conflict Resolution with Persistent Knowledge (Phase 3):**
        *   *Challenge:* Handling new data streams that strongly contradict established, persistent pathways, especially with sparse external rewards.
        *   *Mechanism:*
            *   **Conflict Detection:** Identify inputs that activate a persistent pathway but produce an output conflicting with prior high-reward outcomes associated with that pathway (using similarity checks and output comparison).
            *   **STDP Depression vs. Persistence:** Persistent synapses have reduced plasticity (`eta *= 0.5`), making them resistant but not immune to STDP depression from conflicting inputs. Sustained negative rewards (`total_reward < -1`) can gradually weaken even persistent synapses over extended periods (~100k steps).
            *   **SIE Response:** Conflicting inputs generate strong negative `total_reward` (due to low `r`, negative `TD`, low `novelty`, high `variance`/negative `impact`).
            *   **De-Tagging Trigger:** Consistently strong negative rewards (`total_reward < -1` for 3+ inputs) or sustained low cluster reward (`avg_reward[c] < 0.5` over 100k steps) trigger the removal of the `persistent` tag, allowing the outdated pathway to decay or be overwritten.
            *   **Structural Adjustment:** Persistent low rewards can also trigger pruning of neurons contributing to the conflicting pathway.
            *   **Cross-Cluster Validation:** Inconsistency detected via cross-cluster checks (e.g., "logic" cluster contradicting "math" cluster output) reinforces negative rewards, accelerating conflict resolution.
        *   *Outcome:* The system resolves conflicts by gradually weakening and potentially untagging/pruning conflicting persistent pathways based on sustained negative internal feedback (SIE) and cross-cluster consistency checks, preventing the maintenance of parallel, contradictory representations and ensuring long-term coherence.

#### E.5. Robustness to Input Noise/Anomalies
*   **Sensitivity to Temporal Precision & Noise:**
    *   *STDP Sensitivity:* STDP is inherently sensitive to spike timing (`Δt`). A 1ms jitter can alter `Δw_ij` by ~5%, while 5ms jitter causes ~22% variation, potentially impacting learning.
    *   *Simulation/Numerical Noise:* `dt=1ms` discretization introduces negligible jitter (±0.5ms, ~2.5% `Δw_ij` impact). FP16 numerical noise adds <0.1ms jitter (<0.5% `Δw_ij` impact).
    *   *Biological Input Noise:* Jitter from noisy sensors (e.g., ±2ms) can cause ~10% `Δw_ij` variation, potentially reducing accuracy over long timescales (~10% over 1M steps) if unmitigated.
*   **Encoding Robustness:**
    *   Apply low-pass filter (moving average over 5 steps) to input frequencies during encoding to smooth noise spikes.
*   **SNN Dynamics:**
    *   LIF leak term naturally dampens transient noise.
    *   Inhibition suppresses noise-driven excitation.
    *   Monitor for excessive firing rates (`>1 Hz avg`) and flag anomalous inputs.
*   **SIE Mechanisms:**
    *   Smooth `total_reward` over recent inputs (e.g., 5) to reduce impact of single anomalous rewards.
    *   Cap reward (`<= 0`) for highly novel inputs (`novelty > 0.9`) to prevent reinforcing corrupted data.
*   **Mitigation Strategies for Timing Jitter:**
    *   *Jitter Smoothing:* Apply temporal smoothing (e.g., moving average over 3ms) to spike times if significant jitter is detected.
    *   *STDP Window Adjustment:* Dynamically widen the STDP time constant (`τ_+`) if jitter exceeds a threshold (e.g., >1ms).
    *   *Reward Smoothing:* Average `total_reward` over more inputs (e.g., 10) to dampen fluctuations caused by timing noise.
    *   *Long-Term Correction:* Periodic clustering and structural plasticity help correct graph errors induced by cumulative jitter.
*   **Implementation:** Integrate checks, filters, and adaptive mechanisms into `encoder.py`, `fum.py`, and training scripts (e.g., `phase3_cont.py`).

#### E.6. Justification for Specific Algorithmic Choices
*   **TD(0) vs. Other RL:**
    *   *Chosen:* TD(0) for value function updates.
    *   *Justification:* Simplicity, computational efficiency (low FLOPs, suitable for MI100), compatibility with sparse SIE rewards, better stability compared to TD(lambda) in noisy environments. Q-learning/SARSA require action spaces impractical for FUM.
*   **K-Means vs. Other Clustering:**
    *   *Chosen:* K-means with silhouette score for adaptive clustering.
    *   *Justification:* Efficiency (lower FLOPs than DBSCAN/Spectral), scalability (linear `O(nki)` vs. cubic), interpretability (spherical clusters align with domain concept), automated `k` selection via silhouette score (more robust than density/graph parameters).
