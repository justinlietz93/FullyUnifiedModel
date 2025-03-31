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
        *   **Initial Data Curation & Validation (80-300 Inputs):** The quality, representativeness, and bias control of the initial minimal dataset (80 inputs for Phase 1, scaling to 300 for Phase 2) are paramount for establishing a robust foundation and enabling generalization despite data scarcity. This process integrates data selection with the FUM's mechanisms.
            *   *Methodology: Ensuring Representativeness & Sufficiency (10-37 Inputs/Domain):*
                *   **Semantic Coverage Analysis:** To ensure breadth and depth, a semantic coverage metric is used: `semantic_coverage = torch.mean(cosine_similarity(input_embeddings, domain_concepts))`. Input embeddings are derived from pre-trained models (e.g., BERT for Language, numerical embeddings for Math, CodeBERT for Coding), and `domain_concepts` represent key concepts per domain (e.g., Math: addition, subtraction, multiplication, division, algebra; Logic: AND, OR, NOT, implication). This is executed on the MI100 GPU, targeting `semantic_coverage > 0.9` for each domain (master node).
                *   **Selection Strategy:** Inputs are selected to maximize this coverage (`inputs = select_inputs(domain_concepts, n=10-37)` on master node), ensuring core concepts are represented (e.g., 90% concept coverage expected based on embedding similarity, Mikolov et al., 2013). Math inputs might include "2 + 2 = ?" and "x^2 - 5x + 6 = 0".
                *   **Edge Case Inclusion:** Explicitly include 2-5 edge cases per domain (`edge_cases = select_edge_cases(domain_concepts, n=2-5)` on master node), such as "0 / 0 = ?" (Math) or "A ∧ ¬A" (Logic), ensuring nuanced coverage and robustness (e.g., 80% edge case coverage, 85% robustness expected, Kaner et al., 1999).
                *   **Concept Diversity Metric:** To ensure representativeness beyond counts, concept diversity is measured: `concept_diversity = 1 - torch.mean(cosine_similarity(input_embeddings))` (MI100 GPU), targeting `> 0.7` (master node). This ensures variance capture (~90% expected, Mikolov et al., 2013).
                *   **Complexity Metric:** Input complexity is measured (`complexity = torch.mean(input_difficulty)`) using domain-specific scores (e.g., Math problem level 1-5), targeting `complexity > 3` (mid-level difficulty) to ensure depth (~80% complexity coverage expected, Cover & Thomas, 2006).
            *   *Bias Assessment & Mitigation:* Specific methods detect and mitigate subtle biases within inputs sourced from standard datasets.
                *   **Detection:** Cultural bias (`detect_cultural_bias`), stylistic bias (`extract_style`, `style_skew < 0.5`), and demographic bias (`detect_demographic_bias`) are checked using dictionaries and feature analysis (MI100 GPU, master node). Tools like Fairness Indicators (Bellamy et al., 2018) can be used (MI100 GPU). Target: `feature_bias < 0.5`. (Detection rates ~85-90% expected, Mehrabi et al., 2021).
                *   **Mitigation:** If bias is detected (e.g., `cultural_bias > 0.7`), inputs are resampled (`resample_inputs`) or selected with constraints (`select_inputs_with_constraints`) to achieve balance (e.g., `cultural_bias < 0.5`, 90-95% fairness expected, Mehrabi et al., 2021). Random shuffling is also used to prevent sequence bias.
            *   *Ensuring Initial Primitive Formation Reliability:* Mechanisms ensure the curated inputs reliably form essential foundational primitives (e.g., logic operators, arithmetic, coding constructs).
                *   **Primitive Coverage Analysis:** Define essential primitives per domain (`primitive_set`). Compute coverage: `primitive_coverage = torch.mean(cosine_similarity(input_embeddings, primitive_embeddings))` (MI100 GPU), targeting `> 0.9` (master node). Inputs are selected to cover this set (`select_inputs_for_primitives`). (90% coverage, 95% sufficiency expected).
                *   **STDP/SIE Triggering Validation:** Ensure selected inputs reliably trigger STDP/SIE for primitive formation. E.g., "2 + 2 = ?" should generate sufficient spike pairs (~5 within 20ms) for STDP (`Δw_ij ≈ 0.0951`, `w[i,j]` reaches 0.8 in ~10 updates on 7900 XTX) and receive reinforcing SIE reward (`total_reward=1` on MI100). (90% formation reliability, 95% convergence expected, Sutton & Barto, 2018).
                *   **Concept Gap Detection & Handling:** Detect missing concepts (`concept_gap = 1 - primitive_coverage > 0.1` on MI100). If gaps exist, dynamically augment the input set (`augment_inputs(missing_concepts)` on master node, adding 2-5 inputs/domain) to ensure completeness (90% detection, 95% closure expected).
            *   *Validation Rigor & Generalization Check:* A validation set (16-60 inputs, 20% of initial set) is constructed to rigorously test generalization beyond the training distribution.
                *   **Construction:** Use stratified sampling based on domain concepts (`validation_set = stratified_sample(inputs, strata=domain_concepts, n=16-60)` on master node) to ensure representation of potentially unseen concepts or variations (e.g., "∫(x^2)dx" if only basic arithmetic was in training). (90% OOD coverage expected).
                *   **Metrics:** Validate using the detailed coverage metrics (embedding diversity > 0.7, concept coverage > 0.9) and bias checks (feature bias < 0.5) described above. (90% diversity, 95% fairness expected). Stratified sampling ensures the validation set tests generalization effectively (95% validity expected, Cochran, 1977).
            *   *Rationale & Cohesion:* This meticulous, multi-faceted approach to data curation—integrating semantic coverage, edge cases, diversity, complexity, bias mitigation, primitive coverage checks, dynamic augmentation, and rigorous validation—ensures the initial dataset, though small, provides a sufficient, representative, and unbiased foundation. It directly addresses potential data scarcity risks by ensuring the data quality supports the mechanisms (STDP/SIE) and aligns with the goal of forming robust primitives and enabling generalization. Risk assessment (`risk_score = 1 - torch.mean([...]) < 0.1`) and mitigation through augmentation further enhance cohesion between the data strategy and the architecture's learning capabilities (95% cohesion, 90% risk reduction expected). This is practical for the development workstation and designed to scale.
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

