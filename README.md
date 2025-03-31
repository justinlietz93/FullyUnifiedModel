# How the Fully Unified Model (FUM) Works

**Table of Contents**
*   [1. High-Level Concept: Brain-Inspired Efficient Superintelligence](#1-high-level-concept-brain-inspired-efficient-superintelligence)
    *   [A. Goal](#a-goal)
    *   [B. Core Philosophy](#b-core-philosophy)
    *   [C. Key Differentiators vs. Broader Machine Learning Landscape](#c-key-differentiators-vs-broader-machine-learning-landscape)
*   [2. Core Architecture Components](#2-core-architecture-components)
    *   [A. Spiking Neurons: Leaky Integrate-and-Fire (LIF) with Heterogeneity and Intrinsic Plasticity](#a-spiking-neurons-leaky-integrate-and-fire-lif-with-heterogeneity-and-intrinsic-plasticity)
        *   [A.1. Model & Rationale](#a1-model--rationale)
        *   [A.2. Contrast with ANNs](#a2-contrast-with-anns)
        *   [A.3. Equation & Simulation Timestep](#a3-equation--simulation-timestep)
        *   [A.4. Firing Mechanism & Reset](#a4-firing-mechanism--reset)
        *   [A.5. Heterogeneity](#a5-heterogeneity)
        *   [A.6. Intrinsic Plasticity (Adaptivity)](#a6-intrinsic-plasticity-adaptivity)
        *   [A.7. Implementation (Kernel Scope & Responsibility)](#a7-implementation-kernel-scope--responsibility)
    *   [B. Neural Plasticity: Spike Timing-Dependent Plasticity (STDP) with Inhibition](#b-neural-plasticity-spike-timing-dependent-plasticity-stdp-with-inhibition)
        *   [B.1. Purpose & Contrast with Backpropagation](#b1-purpose--contrast-with-backpropagation)
        *   [B.2. Excitatory STDP Rule](#b2-excitatory-stdp-rule)
        *   [B.3. Inhibitory STDP Rule & Neuron Types](#b3-inhibitory-stdp-rule--neuron-types)
        *   [B.4. Parameters & Weight Range](#b4-parameters--weight-range)
        *   [B.5. Eligibility Traces for Temporal Credit Assignment](#b5-eligibility-traces-for-temporal-credit-assignment)
        *   [B.6. STDP Calculation Location & Final Weight Update](#b6-stdp-calculation-location--final-weight-update)
        *   [B.7. Role & Stability Mechanisms (Incl. Synaptic Scaling)](#b7-role--stability-mechanisms-incl-synaptic-scaling)
    *   [C. Continuous Reinforcement Learning: Self-Improvement Engine (SIE) with TD Learning](#c-continuous-reinforcement-learning-self-improvement-engine-sie-with-td-learning)
        *   [C.1. Purpose & Contrast with Supervised Learning](#c1-purpose--contrast-with-supervised-learning)
        *   [C.2. Reward Signal (`total_reward`) & Component Calculation](#c2-reward-signal-total_reward--component-calculation)
        *   [C.3. TD Learning Specifics (TD(0), Value Function)](#c3-td-learning-specifics-td0-value-function)
        *   [C.4. Novelty Calculation](#c4-novelty-calculation)
        *   [C.5. Habituation Calculation](#c5-habituation-calculation)
        *   [C.6. Self-Benefit Calculation (Complexity & Impact Metrics)](#c6-self-benefit-calculation-complexity--impact-metrics)
        *   [C.7. Influence on Learning (Modulation)](#c7-influence-on-learning-modulation)
        *   [C.8. Goal & Alignment Concerns](#c8-goal--alignment-concerns)
    *   [D. Unified Knowledge Graph (Emergent)](#d-unified-knowledge-graph-emergent)
        *   [D.1. Concept & Contrast with ANNs/GNNs](#d1-concept--contrast-with-annsgnns)
        *   [D.2. Structure](#d2-structure)
        *   [D.3. Formation & Evolution](#d3-formation--evolution)
        *   [D.4. Self-Coordination and Routing](#d4-self-coordination-and-routing)
    *   [E. Tensor-Based Computation and Hybrid Interface](#e-tensor-based-computation-and-hybrid-interface)
        *   [E.1. Hybrid Approach Rationale](#e1-hybrid-approach-rationale)
        *   [E.2. Frameworks & Hardware Roles (Development Context)](#e2-frameworks--hardware-roles-development-context)
        *   [E.3. Interface: Data Flow & Synchronization](#e3-interface-data-flow--synchronization)
*   [3. Multimodal Input/Output Processing](#3-multimodal-inputoutput-processing)
    *   [A. Encoder Mechanism: From Raw Data to Spike Trains](#a-encoder-mechanism-from-raw-data-to-spike-trains)
        *   [A.1. Purpose & Contrast with LLM Input](#a1-purpose--contrast-with-llm-input)
        *   [A.2. Encoding Methods (Rate & Temporal)](#a2-encoding-methods-rate--temporal)
        *   [A.3. Poisson Spike Generation Details](#a3-poisson-spike-generation-details)
        *   [A.4. Output & Extensibility](#a4-output--extensibility)
    *   [B. Decoder Mechanism: From Spike Trains to Structured Output](#b-decoder-mechanism-from-spike-trains-to-structured-output)
        *   [B.1. Purpose](#b1-purpose)
        *   [B.2. Decoding Methods (Rate & Temporal)](#b2-decoding-methods-rate--temporal)
        *   [B.3. Emergent Formation](#b3-emergent-formation)
        *   [B.4. Implementation](#b4-implementation)
*   [4. Emergent Behaviors and Self-Organization](#4-emergent-behaviors-and-self-organization)
    *   [A. Emergent Energy Landscape](#a-emergent-energy-landscape)
    *   [B. Knowledge Graph Evolution (Detailed)](#b-knowledge-graph-evolution-detailed)
    *   [C. Self-Modification (Structural Plasticity - Detailed Algorithms)](#c-self-modification-structural-plasticity---detailed-algorithms)
        *   [C.1. Rationale & Triggers](#c1-rationale--triggers)
        *   [C.2. Growth Algorithm](#c2-growth-algorithm)
        *   [C.3. Pruning Algorithm](#c3-pruning-algorithm)
        *   [C.4. Rewiring Algorithm & Limits](#c4-rewiring-algorithm--limits)
    *   [D. Adaptive Domain Clustering (Dynamic k and Edge Cases)](#d-adaptive-domain-clustering-dynamic-k-and-edge-cases)
        *   [D.1. Purpose & Mechanism](#d1-purpose--mechanism)
        *   [D.2. Determining Number of Clusters (k)](#d2-determining-number-of-clusters-k)
        *   [D.3. Cluster Assignment & Reward Attribution (Domain Identification)](#d3-cluster-assignment--reward-attribution-domain-identification)
        *   [D.4. Edge Case Handling (Small k, Empty Clusters)](#d4-edge-case-handling-small-k-empty-clusters)
        *   [D.5. Adaptation](#d5-adaptation)
*   [5. Training and Scaling: Detailed Implementation Strategy](#5-training-and-scaling-detailed-implementation-strategy)
    *   [A. Phase 1: Random Seed Sprinkling (Foundation Building)](#a-phase-1-random-seed-sprinkling-foundation-building)
        *   [A.1. Objective](#a1-objective)
        *   [A.2. Cellular Components & Mechanisms (Incl. Initialization Strategy & Dynamic States)](#a2-cellular-components--mechanisms-incl-initialization-strategy--dynamic-states)
        *   [A.3. Physics of Initial State Formation](#a3-physics-of-initial-state-formation)
        *   [A.4. Expected Outcome](#a4-expected-outcome)
    *   [B. Phase 2: Tandem Complexity Scaling (Refinement and Competence)](#b-phase-2-tandem-complexity-scaling-refinement-and-competence)
        *   [B.1. Objective](#b1-objective)
        *   [B.2. Cellular Components & Mechanisms](#b2-cellular-components--mechanisms)
        *   [B.3. Mathematical Formulations](#b3-mathematical-formulations)
        *   [B.4. Expected Outcome](#b4-expected-outcome)
    *   [C. Phase 3: Continuous Self-Learning (Autonomy and Mastery)](#c-phase-3-continuous-self-learning-autonomy-and-mastery)
        *   [C.1. Objective](#c1-objective)
        *   [C.2. Cellular Components & Mechanisms](#c2-cellular-components--mechanisms)
        *   [C.3. Emergent Physics Principles (Self-Organized Criticality - SOC)](#c3-emergent-physics-principles-self-organized-criticality---soc)
        *   [C.4. Expected Outcome](#c4-expected-outcome)
    *   [D. Scaling Strategy: Implementation Details](#d-scaling-strategy-implementation-details)
        *   [D.1. Distributed Computation (Graph Sharding)](#d1-distributed-computation-graph-sharding)
        *   [D.2. Asynchronous Updates & Synchronization Details](#d2-asynchronous-updates--synchronization-details)
        *   [D.3. Memory Management (Incl. Parameter Server & Caching)](#d3-memory-management-incl-parameter-server--caching)
        *   [D.4. Hardware Optimization (Development Context)](#d4-hardware-optimization-development-context)
    *   [E. Practical Considerations: Tuning, Debugging, Stability, and Robustness](#e-practical-considerations-tuning-debugging-stability-and-robustness)
        *   [E.1. Hyperparameter Sensitivity & Tuning Strategy](#e1-hyperparameter-sensitivity--tuning-strategy)
        *   [E.2. Debuggability and Interpretability](#e2-debuggability-and-interpretability)
        *   [E.3. Computational Cost of Overhead Components & Net Efficiency](#e3-computational-cost-of-overhead-components--net-efficiency)
        *   [E.4. Long-Term Stability and Potential Drift (Phase 3)](#e4-long-term-stability-and-potential-drift-phase-3)
        *   [E.5. Robustness to Input Noise/Anomalies](#e5-robustness-to-input-noiseanomalies)
        *   [E.6. Justification for Specific Algorithmic Choices](#e6-justification-for-specific-algorithmic-choices)
*   [6. Feasibility and Rationale Summary](#6-feasibility-and-rationale-summary)
    *   [A. Why is FUM considered feasible despite its ambitious goals?](#a-why-is-fum-considered-feasible-despite-its-ambitious-goals)
    *   [B. Strategic Foundation: Balancing Initialization and Learning](#b-strategic-foundation-balancing-initialization-and-learning)

---

This document explains the intended design, architecture, operational mechanics, and underlying rationale of the Fully Unified Model (FUM), based on its design specifications, highlighting its key differences from conventional AI approaches.

## 1. High-Level Concept: Brain-Inspired Efficient Superintelligence

### A. Goal

Achieve autonomous, expert-level mastery across diverse domains (e.g., Mathematics, Logic, Coding, Language, Visual Perception, Introspection) using **minimal training data** (target: 80-300 inputs). The aim is to outperform large-scale models (like 700B parameter LLMs) in accuracy and speed, while operating **efficiently on constrained hardware**.

*   **Extreme Data Efficiency Explained:** The claim of achieving broad mastery from only 80-300 inputs, which seems orders of magnitude beyond current AI, relies on several core mechanisms:
    *   **Sparse, Temporal Learning (SNN/STDP):** Unlike ANNs averaging over vast datasets, FUM's SNNs with STDP (Section 2.B) learn efficiently from temporal correlations in sparse spike patterns. STDP focuses on causality (`Δt > 0`) within short windows (`±20ms`), allowing meaningful weight updates (`Δw_ij`) from relatively few events. For 80 inputs (50 timesteps each), ~4000 timesteps generate enough spike pairs (~200k for 1k neurons) to constrain the sparse connections (~5k weights), leveraging the temporal richness of spikes.
    *   **Emergent Generalization (Knowledge Graph):** The dynamic graph (Section 2.D) forms cross-domain links via STDP/SIE, generalizing beyond specific inputs. Learning "2+2=4" and "3+3=6" strengthens a "math" cluster, enabling inference on "5+5=?" by activating similar pathways.
    *   **SIE Reward Shaping & Anti-Overfitting:** The SIE reward (`total_reward`, Section 2.C) actively prevents overfitting. High `novelty` for unseen inputs amplifies learning (`eta_effective` increases), while `habituation` penalizes repeated inputs, discouraging memorization. `Sparsity` (95%) inherently limits memorization capacity, forcing generalization. `Structural Plasticity` (Section 4.C) adds resources (neurons/connections) if performance stagnates (low reward), preventing over-specialization on the small dataset. `Cross-domain validation` during training ensures learned associations are robust.
    *   **Rationale:** This combination, validated by the AMN predecessor (82% accuracy with 3 examples), allows FUM to extract robust patterns from minimal data, contrasting sharply with the data hunger of LLMs.
*   **Defining "Expert-Level Mastery":** Mastery is defined by specific, measurable benchmarks achieved after training on the minimal dataset:
    *   **Phase 1 (80 Inputs - Foundational Mastery):** Target >50% accuracy on 20 unseen validation inputs across 8 domains (simple arithmetic, logic evaluation, code snippets, basic Q&A).
    *   **Phase 2 (300 Inputs - Expert-Level Mastery):** Target >85% accuracy on 60 unseen validation inputs, with increased complexity (e.g., quadratic equations, logical deduction, function writing, text summarization). Accuracy uses exact match or BLEU score (>0.8) as appropriate.
    *   **Comparison to SOTA & Specific Benchmarks:**
        *   *Target Benchmarks:* FUM's mastery will be rigorously validated against specific subsets of standard benchmarks:
            *   **Math:** MATH dataset (Levels 1-5 Algebra subset, target >85%).
            *   **Logic:** GPQA dataset (Levels 1-3 subset, target >85%).
            *   **Coding:** HumanEval subset (target >80% pass@1).
            *   **Language:** CNN/DailyMail summarization subset (target BLEU > 0.8).
            *   **Physics:** Custom simulation problems (target >80%).
        *   *SOTA Models for Comparison (as of Q1 2025):* Performance compared against GPT-4 (~700B params), LLaMA-2-70B, and Grok (~100B params).
        *   *Validation Goal:* Demonstrate comparable or superior accuracy on these targeted complex reasoning tasks with 67M-fold fewer inputs (300 vs. ~20B) and significant energy savings (~11x-194x projected) compared to LLM inference costs. FUM prioritizes data/energy efficiency and reasoning depth over encyclopedic knowledge breadth initially.

*   **Hardware Context (Development & Validation):** The specific hardware configurations mentioned throughout this document (Linux workstation with AMD Threadripper PRO 5955WX, MI100 32GB VRAM, 7900 XTX 24GB VRAM, 512GB RAM, 6TB SSD) represent the author's (Justin Lietz) test environment. These are **not rigid requirements** for FUM deployment but serve as the platform where the model's theoretical foundations are validated. Notably, the predecessor model, AMN (Adaptive Modular Network), has already been successfully validated up to a 10-unit model size on this hardware, demonstrating the feasibility of the core concepts.
*   **Why Minimal Data?** Unlike LLMs requiring terabytes of data and vast pre-training, FUM aims for human-like learning efficiency, inferring complex patterns from sparse examples. This reduces reliance on massive datasets and computational resources, making advanced AI potentially achievable within the constraints of the development hardware. The design philosophy balances a minimal seeded structure during initialization with knowledge purely learned from these minimal examples (see Section 6.B for details).

### B. Core Philosophy

Mimic the efficiency (human brain ~20W) and adaptability of biological brains by employing a **hybrid architecture**. This contrasts with monolithic architectures like Transformers used in most LLMs. The design prioritizes **functional equivalence** over strict biological analogy, using biologically inspired mechanisms simplified for computational tractability. The claimed efficiency and learning capability rely on these functional algorithms (e.g., LIF dynamics, STDP temporal correlation, SIE reward modulation) rather than precise replication of biological details (like ion channels or dopamine pathways). While omitting certain biological details (e.g., synaptic tagging) might slightly reduce long-term retention (~10-15%), the core efficiency (>1M-fold theoretical energy savings from sparse, event-driven SNNs) and minimal-data learning capabilities (validated by AMN) are expected to hold, as they stem from the computational properties of the chosen abstractions.

1.  **Sparse Spiking Neural Networks (SNNs):**
    *   Chosen for inherent **temporal processing** (information encoded in spike timing, not just rate), potential for massive **energy efficiency** (neurons only compute when they spike, targeting >1M-fold savings vs. LLMs theoretically, though practical overhead reduces this - see Sec 5.E.3), and **biological plausibility**. High sparsity (target: 95%) drastically reduces the number of active connections, further saving computation and memory compared to dense ANNs/Transformers. Includes both excitatory and inhibitory neurons (typically 80:20 ratio) for stability and balanced dynamics.
    *   **Practical SNN Performance:** While theoretically efficient, practical SNNs face challenges. FUM addresses this via optimized kernels and a hybrid approach, but acknowledges the computational cost of overhead components (SIE, plasticity, etc.). Net system-level efficiency is estimated at ~11x savings vs. LLM inference at 1k scale, projecting to ~193.5x at 32B scale, significantly less than the theoretical 1M-fold but still substantial. Speed advantage is estimated at ~4x at 1k scale, ~8.4x at 32B scale. (See Sec 5.E.3 for detailed cost breakdown).
2.  **Emergent Knowledge Graph:**
    *   A dynamic graph structure replaces fixed layers or a predefined coordinator network. **Why?** This allows relationships between concepts and domains to emerge organically from neuron interactions and learning feedback, fostering adaptability and cross-domain knowledge transfer without manual design. This differs significantly from the fixed, layered structure of most deep learning models.
    *   **Advantages over LLMs:** The emergent graph enables dynamic cross-domain associations and flexible reasoning potentially superior to static Transformer attention for certain tasks. SNN temporal processing naturally handles sequential dependencies and multi-step reasoning. The SIE allows autonomous learning from sparse rewards, unlike supervised LLMs. (See Section 6.A for arguments on outperforming LLMs).
3.  **Tensor-based Computation:**
    *   Leverages frameworks like PyTorch for efficient batch processing of certain operations (e.g., graph analysis, SIE calculations, clustering) and seamless integration with GPU acceleration (ROCm), complementing the SNN's event-driven nature via a carefully managed hybrid interface.

### C. Key Differentiators vs. Broader Machine Learning Landscape

FUM's design choices distinguish it not only from LLMs but also from various other ML paradigms:

*   **vs. Deep Learning (ANNs, CNNs, RNNs, Transformers):**
    *   **Neuron Model:** Uses spiking (LIF) neurons processing information temporally, unlike rate-based ANUs (ReLU, sigmoid, etc.). Incorporates heterogeneity and intrinsic plasticity.
    *   **Learning Rule:** Primarily uses local, biologically plausible STDP (for both excitatory and inhibitory synapses) modulated by reinforcement (SIE) via eligibility traces, not global backpropagation.
    *   **Architecture:** Dynamic, emergent graph structure vs. fixed, layered architectures. Includes structural plasticity.
    *   **Data/Energy:** Aims for significantly higher data and energy efficiency.
    *   **Adaptability:** Built-in structural plasticity vs. generally static architectures requiring retraining.
*   **vs. Traditional ML (SVMs, Decision Trees, k-NN, etc.):**
    *   **Representation:** Learns distributed, dynamic representations in a neural graph, unlike the explicit feature engineering or fixed decision boundaries common in traditional ML.
    *   **Learning:** Learns online and continuously via STDP/SIE, unlike batch training on fixed datasets typical for many traditional models.
    *   **Complexity Handling:** Designed to handle complex, high-dimensional, temporal data patterns where traditional models might struggle without extensive feature engineering.
*   **vs. Symbolic AI / Expert Systems:**
    *   **Knowledge Representation:** Knowledge emerges in the graph's connection weights (both positive and negative), unlike the explicit, human-defined rules and symbols of symbolic AI.
    *   **Learning:** Learns from data and feedback, unlike primarily relying on pre-programmed knowledge bases.
    *   **Robustness:** Aims for robustness to noisy data, whereas symbolic systems can be brittle. FUM integrates symbolic-like reasoning capabilities (Logic domain) within its neural framework.
*   **vs. Standard Reinforcement Learning (Q-Learning, Policy Gradients):**
    *   **Core Mechanism:** Uses STDP as the primary synaptic learning rule, modulated by the SIE's reinforcement signal (incorporating TD(0) learning). Standard RL typically learns value functions or policies directly via algorithms like Q-learning or policy gradients, often requiring many environment interactions.
    *   **Representation:** Learns within the SNN/graph structure, using cluster-based state representations for the TD value function, not typically relying on explicit state-action tables or separate policy/value networks in the same way as standard RL.
*   **vs. Evolutionary Algorithms (Genetic Algorithms, Neuroevolution):**
    *   **Learning Timescale:** Learns within the "lifetime" of the model via STDP/SIE. Evolutionary approaches typically operate over generations, selecting or modifying entire networks based on fitness, which can be slower for online adaptation.
    *   **Mechanism:** Relies on synaptic plasticity (STDP, structural plasticity) and reinforcement (SIE), not population-based selection and genetic operators (mutation, crossover), although FUM's self-modification has conceptual parallels to structural evolution.

## 2. Core Architecture Components

### A. Spiking Neurons: Leaky Integrate-and-Fire (LIF) with Heterogeneity and Intrinsic Plasticity

#### A.1. Model & Rationale
*   Employs the standard Leaky Integrate-and-Fire (LIF) model. **Why LIF?** It offers a good balance between biological realism and computational tractability, capturing essential integrate-and-fire dynamics without the complexity of models like Hodgkin-Huxley. This efficiency is crucial for large-scale simulation.

#### A.2. Contrast with ANNs
*   Unlike Artificial Neuron Units (ANUs) in standard ANNs (like ReLUs, Sigmoids) which compute a static output based on summed weighted inputs in one pass, LIF neurons integrate inputs *over time* and communicate via discrete *spikes* (events), enabling richer temporal coding.

#### A.3. Equation & Simulation Timestep
*   The membrane potential `V` of a neuron `i` at time `t` is updated based on the previous potential `V_i(t-1)`, the input current `I_i(t)` (sum of weighted spikes from connected neurons), and a leak term determined by the neuron's specific membrane time constant `tau_i`:
    `V_i(t) = V_i(t-1) + I_i(t) - (V_i(t-1) / tau_i) * dt`
    (where `dt` is the simulation timestep). This equation models how a neuron accumulates charge and naturally loses it over time if input is insufficient.
*   **Simulation Timestep (dt):** Fixed at `1ms`. **Rationale:** This value balances simulation fidelity (sufficient to capture STDP dynamics with `tau_` parameters around 20ms, as the STDP window is 20 timesteps) and computational cost (avoiding the 100x cost increase of a 0.01ms step). On the development hardware (Justin’s 7900 XTX GPU), `dt=1ms` ensures reasonable training times (e.g., ~2–3 hours for Phase 1).

#### A.4. Firing Mechanism & Reset
*   A neuron generates an output spike (a discrete event, `spikes_i(t) = 1`) when its membrane potential `V_i(t)` crosses its specific defined threshold `v_th_i`. This event-driven nature is key to SNN efficiency.
*   After firing, the neuron's potential is reset to a fixed resting value `v_reset` (-70mV), preventing immediate re-firing and mimicking a biological refractory period.

#### A.5. Heterogeneity
*   Neuron parameters are **not uniform** but are drawn from distributions at initialization to mimic biological variability and enhance network dynamics:
    *   `tau_i`: Drawn from a Normal distribution `N(20ms, 2ms^2)` (`torch.normal(mean=20.0, std=2.0)`).
    *   `v_th_i`: Drawn from a Normal distribution `N(-55mV, 2mV^2)` (`torch.normal(mean=-55.0, std=2.0)`).
    *   `v_reset`: Fixed at -70mV for all neurons.
*   **Rationale:** Heterogeneity ensures diverse temporal dynamics, preventing overly synchronized firing and enhancing network robustness.

#### A.6. Intrinsic Plasticity (Adaptivity)
*   Neuron parameters (`tau_i`, `v_th_i`) adapt over time based on their firing rate to maintain activity within a target range, preventing silent or hyperactive neurons:
    *   **Target Rate:** 0.1–0.5 Hz (5–25 spikes over a 50-timestep window).
    *   **Adjustment Rule:**
        *   If `rate_i > 0.5 Hz`, increase `v_th_i` by 0.1mV (`v_th += 0.1`) and decrease `tau_i` by 0.1ms (`tau -= 0.1`), reducing excitability.
        *   If `rate_i < 0.1 Hz`, decrease `v_th_i` by 0.1mV (`v_th -= 0.1`) and increase `tau_i` by 0.1ms (`tau += 0.1`), increasing excitability.
    *   **Bounds:** `v_th_i` is clamped to [-60mV, -50mV], `tau_i` to [15ms, 25ms].
    *   **Timing & Implementation:** Applied every 50 timesteps after STDP updates, computed on the 7900 XTX GPU, updating `v_th` and `tau` tensors in-place.

#### A.7. Implementation (Kernel Scope & Responsibility)
*   The core LIF update loop (integration, thresholding, reset) is executed via a custom ROCm HIP kernel (`neuron_kernel.hip`, specifically `pulse_kernel`) for massive parallelism on the designated GPU (AMD Radeon 7900 XTX), operating on `float16` tensors.
*   **Kernel Responsibility:** This kernel computes `V_i(t)`, generates `spikes_i(t)`, and records spike times in a `spike_history` buffer (shape `(num_neurons, T)`, e.g., `1000x50`, stored as `uint8` on 7900 XTX). It **does not** compute STDP changes (`Δw_ij`) or update eligibility traces (`e_ij`) within the kernel itself. These are handled separately in PyTorch (see Sec 2.B, 2.E).

### B. Neural Plasticity: Spike Timing-Dependent Plasticity (STDP) with Inhibition

#### B.1. Purpose & Contrast with Backpropagation
*   Enables the network to learn by adjusting the strength (weight `w_ij`) of connections between neurons based on the *precise relative timing* of their spikes. It's a biologically plausible mechanism for Hebbian learning ("neurons that fire together, wire together") that leverages the temporal information inherent in SNNs.
*   This is fundamentally different from backpropagation used in most ANNs/LLMs. STDP is a *local* learning rule – weight changes depend only on the activity of the pre- and post-synaptic neurons. Backpropagation requires a *global* error signal calculated at the output layer and propagated backward through all layers, demanding differentiability and often large amounts of labeled data. STDP allows unsupervised or reinforcement-based learning directly from spike patterns, making it more biologically plausible and potentially more efficient for certain learning tasks.

#### B.2. Excitatory STDP Rule
*   For connections originating from an excitatory neuron (`i`), the change in synaptic weight (`Δw_ij`) depends exponentially on the time difference (`Δt = t_post - t_pre`) between post-synaptic and pre-synaptic spikes:
    *   **Potentiation (Strengthening):** If the pre-synaptic neuron fires shortly *before* the post-synaptic neuron (`Δt > 0`), the connection is strengthened: `Δw_ij = A_+ * exp(-Δt / τ_+)`.
    *   **Depression (Weakening):** If the pre-synaptic neuron fires shortly *after* the post-synaptic neuron (`Δt < 0`), the connection is weakened: `Δw_ij = -A_- * exp(Δt / τ_-)`.
    *   If `Δt = 0`, `Δw_ij = 0`.

#### B.3. Inhibitory STDP Rule & Neuron Types
*   FUM incorporates inhibitory connections (typically 20% of neurons, e.g., indices 800-999 for 1000 neurons) for stability.
*   For connections originating from an inhibitory neuron (`i`), the STDP rule is modified to promote stability:
    *   **Weakening Inhibition:** If `Δt > 0` (pre before post), the inhibitory connection is weakened (made less negative): `Δw_ij = -A_+ * exp(-Δt / τ_+)`.
    *   **Strengthening Inhibition:** If `Δt < 0` (post before pre), the inhibitory connection is strengthened (made more negative): `Δw_ij = A_- * exp(Δt / τ_-)`.
*   **Implementation:** During STDP calculation, check the pre-synaptic neuron type (`is_inhibitory[i]`) and apply the appropriate rule.

#### B.4. Parameters & Weight Range
*   Key parameters: `A_+ = 0.1`, `A_- = 0.12`, `τ_+ = 20ms`, `τ_- = 20ms`.
*   Weights `w_ij` can be positive (excitatory) or negative (inhibitory) and are clamped to the range `[-1, 1]` (`w.clamp_(-1, 1)`).

#### B.5. Eligibility Traces for Temporal Credit Assignment
*   To bridge the temporal gap between local STDP events and potentially delayed global SIE rewards, each synapse maintains an eligibility trace `e_ij`.
*   **Update Rule:** `e_ij(t) = γ * e_ij(t-1) + Δw_ij(t)`, where `γ = 0.95` (decay factor, ~200ms time constant for `dt=1ms`) and `Δw_ij(t)` is the STDP weight change calculated based on spike pairs occurring at timestep `t`.
*   **Physics/Math:** The trace `e_ij(t) = Σ (γ^(t-k) * Δw_ij(k))` sums past STDP events, weighted by their temporal relevance. An event at `t=0` contributes `~0.0951` initially, decaying to `~0.0004` after 200ms.
*   **Storage:** `e_ij` is a sparse tensor mirroring `w`'s structure (shape `(num_nonzero_connections,)`), stored in FP16 on the MI100 GPU (e.g., 10KB for 5k connections). Initialized to zero at `t=0`.
*   **Update Location:** Updated using PyTorch on the MI100 GPU after STDP `Δw_ij` calculation.

#### B.6. STDP Calculation Location & Final Weight Update
*   **STDP Calculation:** The calculation of `Δw_ij(t)` based on spike pairs from `spike_history` (recorded by the LIF kernel on the 7900 XTX) is performed **outside** the LIF kernel.
    *   **Sequence:** After 50 timesteps, transfer `spike_history` to MI100. Identify spike pairs within ±20ms window, compute `Δt`, apply STDP rules (excitatory/inhibitory), sum `Δw_ij` per synapse. Executed using PyTorch tensor operations on MI100.
*   **Final Weight Update:** The actual weight update `w_ij = clip(w_ij + eta_effective * total_reward * e_ij(T), -1, 1)` occurs after the SIE reward (`total_reward`) is calculated (on MI100) and transferred (along with `e_ij`) back to the 7900 XTX GPU. (`eta_effective` is the modulated learning rate, see Sec 2.C).

#### B.7. Role & Stability Mechanisms (Incl. Synaptic Scaling)
*   STDP is the fundamental mechanism for associative learning. The inclusion of inhibitory neurons and inhibitory STDP is crucial for managing network stability and preventing runaway excitation.
*   **Additional Stability Mechanisms:**
    *   **Inhibitory Feedback:** Inhibitory neurons provide negative input `sum(w[i,j] * spikes(t-1)[i])` where `w[i,j] < 0`, counteracting excitation.
    *   **Global Inhibition:** A subset of inhibitory neurons fire proportionally to the network's average rate, providing broad dampening.
    *   **Intrinsic Plasticity:** Adapts neuron excitability (Sec 2.A.6).
    *   **Synaptic Scaling:** Normalizes total excitatory input to prevent saturation.
        *   **Mechanism:** Every 1000 timesteps, compute `total_exc[j] = sum(w[i,j] for i in excitatory and w[i,j] > 0)`. If `total_exc[j] > 1`, calculate `scale_factor = 1 / total_exc[j]`.
        *   **Interaction & Timing:** Applied *after* STDP/SIE updates within the 1000-step cycle. To protect learned pathways, only scale weaker connections: `w[i,j] *= scale_factor` only if `w[i,j] < 0.8`. Executed on 7900 XTX.

### C. Continuous Reinforcement Learning: Self-Improvement Engine (SIE) with TD Learning

#### C.1. Purpose & Contrast with Supervised Learning
*   Provides a sparse, global feedback signal (`total_reward`) to guide the local STDP learning process towards desired high-level outcomes (task success), enabling the network to learn from trial-and-error even with minimal explicit supervision.
*   Unlike supervised learning which requires detailed labels for every input, the SIE uses a potentially complex reward signal derived from task success, internal consistency, and novelty. **Why?** This allows learning complex tasks where detailed labels are unavailable or impractical to obtain, mimicking how biological systems learn goal-directed behaviors.

#### C.2. Reward Signal (`total_reward`) & Component Calculation
*   Calculated after each simulation window (e.g., 50 timesteps) on the MI100 GPU.
*   **Formula:** `total_reward = TD_error + novelty - habituation + self_benefit`

#### C.3. TD Learning Specifics (TD(0), Value Function)
*   **Algorithm:** Uses TD(0) for simplicity: `TD_error = r + γ * V(next_state) - V(current_state)`.
    *   `r`: Immediate external reward (+1 correct, -1 incorrect, 0 neutral/unknown) if available, else 0.
    *   `γ`: Discount factor (0.9).
*   **Value Function `V(state)`:**
    *   **Predicted Value:** Predicts expected future cumulative reward.
    *   **Representation:** Tensor `V_states` (shape: `num_states`), stored on MI100 GPU. Initialized to zero.
    *   **State Definition:** States correspond to clusters identified by adaptive clustering (Sec 4.D). `num_states` determined by `k`.
    *   **Update:** After identifying `current_state_idx` and `next_state_idx` via clustering, update `V_states[current_state_idx] += α * TD_error` (where `α=0.1`, learning rate).

#### C.4. Novelty Calculation
*   **Storage:** Maintain history of recent input patterns (`recent_inputs` buffer, shape `(history_size, num_input_neurons, T)` on MI100).
*   **Comparison:** Compute cosine similarity between current `I_encoded` and `recent_inputs`.
*   **Metric:** `novelty = 1 - max(similarity)`. Ranges [0, 1].

#### C.5. Habituation Calculation
*   **Storage:** Maintain `habituation_counter[i]` for each pattern in `recent_inputs` on MI100.
*   **Update:** If `max(similarity) > 0.9`, increment `habituation_counter[matched_input] += 0.1` (capped at 1).
*   **Decay:** Periodically decay counters (`*= 0.95`).
*   **Metric:** `habituation = habituation_counter[matched_input]`. Ranges [0, 1].

#### C.6. Self-Benefit Calculation (Complexity & Impact Metrics)
*   Internal measure of computation quality: `self_benefit = complexity * impact`.
*   **Complexity:**
    *   **Definition:** Average spikes per neuron per timestep: `complexity = torch.sum(spike_history) / (num_neurons * T)`. Calculated on 7900 XTX, transferred to MI100.
    *   **Granularity:** Can be calculated per cluster (`complexity[c]`) for more targeted feedback, reflecting domain-specific computational effort. If used, `self_benefit` becomes a weighted average of `complexity[c] * impact[c]`.
*   **Impact:**
    *   **Definition:** Reduction in firing rate variance: `impact = (variance_before - variance_after) / max(variance_baseline, 0.01)`. `variance_before` is avg over last 1k steps, `variance_after` is current, `variance_baseline` is avg over 10k steps. Calculated on 7900 XTX, transferred to MI100.
    *   **Sensitivity & Safeguards:** Sensitive to input shifts/exploration. Normalized by `variance_baseline`. Penalty reduced during exploration (`impact_adjusted = impact * (1 - novelty)`). Clamped to `[-1, 1]`.

#### C.7. Influence on Learning (Modulation)
*   The calculated `total_reward` modulates the base STDP learning rate (`eta = 0.01`).
*   **Mapping:** `total_reward` (potentially unbounded) is mapped to a modulation factor `mod_factor` in [-1, 1] using a sigmoid: `mod_factor = 2 * torch.sigmoid(total_reward) - 1`.
*   **Effective Learning Rate:** `eta_effective = eta * (1 + mod_factor)`. Positive rewards amplify learning, negative rewards suppress it.
*   **Application:** The final weight update uses this modulated rate and the reward itself: `Δw_ij(T) = eta_effective * total_reward * e_ij(T)` (applied on 7900 XTX). This quadratic scaling emphasizes significant outcomes.

#### C.8. Goal & Alignment Concerns
*   Drives the network's self-organization process (STDP, structural plasticity) to find internal configurations (synaptic weights `w_ij` and network structure) that maximize the cumulative `total_reward` signal over time, thereby improving performance on target tasks and promoting stable, efficient, and novel computation.
*   **Reliability and Goal Alignment:** The complex `total_reward` function aims to reliably guide the system towards accuracy, efficiency, and adaptability.
    *   **Component Alignment:** External `r` drives accuracy, `TD` promotes long-term success, `novelty` ensures adaptability, `habituation` prevents overfitting, and `self_benefit` rewards efficient/stable computation.
    *   **Safeguards:** Normalization (`sigmoid` mapping to `mod_factor`), exploration adjustments (scaling `impact` by `1 - novelty`), and reward smoothing (averaging over recent inputs) prevent misleading internal metrics or undesirable loops.
    *   **Sensitivity & Tuning:** The relative weighting of components is sensitive (e.g., doubling novelty weight reduces accuracy ~15%). Bayesian optimization (Sec 5.E.1) is used to tune weights, maximizing average cluster rewards and ensuring balanced goal alignment.
    *   **Preventing "Gaming" Internal Metrics (Phase 3):** In autonomous operation with sparse external rewards, specific mechanisms prevent the system from optimizing internal SIE metrics at the expense of useful outputs:
        *   *Novelty Gaming:* Capping novelty's contribution (`min(novelty, 0.5)`) and using habituation prevent loops of random, meaningless outputs.
        *   *Complexity/Impact Gaming:* Normalizing complexity/impact metrics (`clamp(metric / baseline, 0, 1)`) and enforcing firing rate limits via intrinsic plasticity (`rate <= 0.5 Hz`) prevent artificial inflation of these values.
        *   *TD_Error Gaming:* Regularizing `V_states` updates (`TD - λ * V_states`) prevents manipulation of predicted values.
    *   **Ensuring Long-Term Alignment with External Reality:**
        *   *Periodic Ground Truth:* Injecting labeled validation inputs periodically (e.g., every 100k steps) provides external reward `r` to anchor `total_reward` and recalibrate `V_states`.
        *   *Metric Recalibration:* Resetting novelty history (`recent_inputs`) and regularizing SIE weights towards defaults prevents long-term drift.
        *   *Stability Constraints:* Enforcing stable dynamics (variance `< 0.05 Hz`) inherently links internal optimization to effective external interaction.
        *   *External Validation:* Periodically testing against validation sets and resetting SIE weights if accuracy drops below a threshold (e.g., 80%) ensures continued alignment.

### D. Unified Knowledge Graph (Emergent)

#### D.1. Concept & Contrast with ANNs/GNNs
*   FUM avoids predefined layers or a fixed coordinator module. Instead, it relies on a knowledge graph structure that **emerges dynamically** from the learned connections (both excitatory and inhibitory) between neurons. **Why?** This allows for maximum flexibility and adaptability. The network itself discovers and represents relationships between concepts and across different domains based on the input data and learning feedback. It acts as a distributed, associative memory and reasoning substrate.
*   This differs significantly from the fixed, layered topology of most ANNs/CNNs/Transformers, and also from Graph Neural Networks (GNNs) which typically operate on *predefined* graph structures. FUM *builds* its own graph as it learns, more akin to biological network formation than applying convolutions or message passing on a static graph. It also differs from Symbolic AI knowledge graphs which are typically human-curated.

#### D.2. Structure
*   Nodes in the graph conceptually represent individual neurons (LIF with specific parameters). Edges represent the synaptic connections (`w_ij` in range [-1, 1]) whose strengths are learned via STDP and modulated by SIE. Sparsity is maintained around 95%.

#### D.3. Formation & Evolution
*   Edges are not predefined but emerge and evolve. An effective connection (edge) strengthens between neurons `i` and `j` if they consistently fire with a timing relationship (`Δt`) that correlates with positive SIE rewards (`total_reward > 0`). Connections irrelevant to success or associated with errors (`total_reward < 0`) are weakened by STDP or potentially pruned by self-modification (Sec 4.C). The graph continuously evolves as learning progresses.

#### D.4. Self-Coordination and Routing
*   There is no central module directing information flow. Instead, processing and reasoning occur via the propagation of spiking activity across the strongest pathways (edges with large `abs(w_ij)`) in the emergent graph.
*   **Reliable Routing:** For specific computations (e.g., "2+2=?"), input spike patterns activate corresponding input neurons. These spikes propagate through pathways strengthened by previous STDP/SIE reinforcement for similar tasks (e.g., `w[i,j]` increased for neurons co-firing during "2 + 2 = 4" training). Inhibitory connections and sparse connectivity help filter out irrelevant associations (weak or non-existent pathways, `w[i,j] < 0.1`), ensuring spikes reliably reach functionally relevant clusters (e.g., "math cluster" identified via adaptive clustering) and ultimately the correct output neurons (e.g., neuron representing '4').
*   **Functional Circuits:** Specific circuits (e.g., for arithmetic) emerge through the interplay of STDP (forming connections between co-active neurons), SIE reward shaping (reinforcing correct outputs for specific tasks, e.g., `r=1` for "4"), adaptive clustering (identifying functional groups like "math"), and structural plasticity (allocating resources, pruning irrelevant connections).
*   **Task Representation & Context Switching:**
    *   *Abstract Goal Representation:* Task goals (e.g., "solve math problem") are represented by the sustained activity patterns within specific emergent clusters (e.g., "math" cluster). Temporal encoding of inputs (Sec 3.A.2) activates these clusters, and SIE rewards reinforce goal-directed activity until task completion.
    *   *Handling Multi-Domain Inputs:* For inputs spanning domains (e.g., a math word problem), the system relies on:
        *   **Temporal Encoding:** Separates components (e.g., language parsing vs. math calculation) into different time windows during input encoding.
        *   **Cluster Activation:** Temporally distinct spike patterns activate the relevant clusters sequentially (e.g., "language" cluster then "math" cluster).
        *   **Inhibitory Suppression:** Active clusters trigger inhibitory neurons that suppress activity in irrelevant clusters, preventing interference. Sparsity also limits cross-talk.
    *   *Dynamic Context:* Context is maintained implicitly by the sustained activity within the currently relevant cluster(s), guided by the emergent graph structure and inhibitory dynamics, without needing an explicit context-setting module.
*   **Controllability of Emergence:** Ensuring the emergent graph consistently forms correct representations and avoids counter-productive structures relies on several mechanisms:
    *   **SIE Guidance:** Rewarding task success (`r=1`) and stability (`impact`) strengthens correct pathways and prunes incorrect ones.
    *   **Adaptive Clustering:** Identifies functional domains, guiding reward attribution and growth. Incorrect representations trigger corrective growth (`avg_reward < 0.5`).
    *   **Cross-Domain Validation:** Tests ensure pathways generalize.
    *   **Stability Mechanisms:** Sparsity constraints (~95%), inhibitory balancing (20% inhibitory neurons, inhibitory STDP, global inhibition), and structural plasticity limits (caps on growth/rewiring, pruning inactive neurons) prevent unstable structures or dynamics during autonomous operation (Phase 3). Continuous monitoring flags anomalies.

### E. Tensor-Based Computation and Hybrid Interface

#### E.1. Hybrid Approach Rationale
*   While SNNs excel at temporal processing, certain operations like analyzing graph properties, calculating complex SIE rewards, managing large state vectors (like eligibility traces `e_ij` or value function `V_states`), or performing clustering are often more efficiently handled using optimized tensor libraries. FUM adopts a hybrid approach, leveraging the strengths of both SNN simulation and tensor computation.

#### E.2. Frameworks & Hardware Roles (Development Context)
*   Utilizes PyTorch for tensor manipulation.
*   **AMD Radeon 7900 XTX (24GB VRAM):** Primarily runs the custom ROCm HIP kernel (`neuron_kernel.hip`) for high-frequency, parallel LIF updates and spike generation. Also handles the final STDP weight updates (`w += ...`). Stores `V`, `spikes`, `spike_history`, `w`.
*   **AMD Instinct MI100 (32GB VRAM):** Primarily runs PyTorch tensor operations for tasks like STDP `Δw_ij` calculation, eligibility trace (`e_ij`) updates, SIE component calculations (novelty, habituation, complexity, impact, TD error), value function (`V_states`) updates, and k-means clustering. Stores `e_ij`, `V_states`, `recent_inputs`, `habituation_counter`, etc.
*   **CPU (AMD Threadripper PRO 5955WX):** Manages overall orchestration, data loading, potentially graph partitioning (METIS), parameter server logic (if scaling beyond node memory), and decoding outputs.

#### E.3. Interface: Data Flow & Synchronization
*   **Frequency:** Interaction occurs primarily after each 50-timestep simulation window. Global operations like clustering or scaling occur less frequently (e.g., every 1000 timesteps).
*   **Data Flow (SNN -> Tensor):**
    1.  `spike_history` (uint8, ~6KB for 1k neurons) recorded on 7900 XTX by LIF kernel.
    2.  After 50 timesteps, transfer `spike_history` to MI100 (`spike_history_mi100 = spike_history.to('cuda:0')`).
    3.  MI100 computes `Δw_ij`, updates `e_ij`, calculates `rates`, computes SIE components (`novelty`, `habituation`, `complexity`, `impact`, `TD_error`), updates `V_states`.
*   **Data Flow (Tensor -> SNN):**
    1.  `total_reward` (float16 scalar) calculated on MI100.
    2.  `e_ij` (sparse float16, ~10KB) updated on MI100.
    3.  Transfer `total_reward` and `e_ij` to 7900 XTX (`total_reward.to('cuda:1')`, `e_ij.to('cuda:1')`).
    4.  7900 XTX applies final weight update to `w` using `total_reward` and `e_ij`.
*   **Synchronization:** Use `torch.cuda.synchronize()` or CUDA events to ensure data transfers are complete before dependent computations begin. Buffering mechanisms (e.g., `rate_buffer` on MI100, appending `rates.to('cuda:0')` every 50 steps) handle aggregation for less frequent operations like k-means (processed when buffer full, e.g., 1000 steps). Timing mismatches are managed by the fixed interaction frequency (every 50 timesteps).

## 3. Multimodal Input/Output Processing

### A. Encoder Mechanism: From Raw Data to Spike Trains

#### A.1. Purpose & Contrast with LLM Input
*   To act as the sensory interface, translating diverse raw input data from various modalities (text, images, video, potentially audio, touch, etc.) into a **universal spike-based format** that the SNN core can process uniformly. **Why?** This allows the core network to be modality-agnostic, simplifying the architecture and enabling seamless integration of new sensor types.
*   This differs markedly from LLMs which typically use tokenization (breaking text into sub-words) followed by embedding layers to convert input into dense vectors. FUM uses temporal spike patterns.

#### A.2. Encoding Methods (Rate & Temporal)
*   **Rate Encoding (Primary):** Maps features to firing frequencies `f` over a window `T` (e.g., 50 timesteps).
    *   *Text:* ASCII value `c` -> `f = (ord(c) % 50) Hz`.
    *   *Images:* Pixel intensity `p` (0-255) -> `f = (p / 2.55) Hz`.
*   **Temporal Encoding (Structured Inputs):** For complex inputs like code syntax trees or logical propositions, use hierarchical temporal encoding:
    *   *Code Syntax Trees:* Parse tree (e.g., using `ast`). Encode node type/value using frequency bands (e.g., Call: 10-20Hz) modulated by value (e.g., `print`: 15Hz). Encode hierarchy over sequential time windows (e.g., 50 timesteps per level: Root -> Children -> Grandchildren).
    *   *Logical Propositions:* Encode variables (A: 10Hz) and operators (∧: 30Hz, →: 35Hz) as frequencies in a temporal sequence (e.g., [A, ∧, B, →, C] over 250 timesteps).

#### A.3. Poisson Spike Generation Details
*   **Formula:** Spikes are generated using a Poisson process based on target frequency `f` and timestep `dt=1ms`.
    *   Probability of spike per timestep: `p = f * dt`. (e.g., `f=50Hz` -> `p=0.05`).
    *   Algorithm: For each timestep `t`, if `torch.rand(1) < p`, emit spike `spike[t] = 1`.
*   **Refractory Period:** A 5ms refractory period (5 timesteps) is imposed on input neurons after spiking.
    *   **Implementation:** Maintain `refractory[i]` counter. If `spike[t]=1`, set `refractory[i]=5`. Only generate spike if `refractory[i]==0`. Decrement counter each step.
    *   **Rationale:** Prevents unrealistically high firing rates (caps at 200 Hz), aligning with biological limits.

#### A.4. Output & Extensibility
*   Output is a tensor `I_encoded` (shape: `[num_input_neurons, T_total]`) containing spike trains (0s and 1s) fed into the SNN core.
*   Adding a new sensor only requires designing a new encoder module mapping its data to spike trains.

### B. Decoder Mechanism: From Spike Trains to Structured Output

#### B.1. Purpose
*   To translate the internal spiking activity patterns of designated output neurons back into a human-understandable format (e.g., text, classification label, numerical value, code, logical steps), relevant to the task performed.

#### B.2. Decoding Methods (Rate & Temporal)
*   **Rate Decoding (Simple Outputs):** Average firing rates of output neurons over a window `T` (e.g., 50 timesteps) are mapped to symbols.
    *   *Classification:* Highest firing rate indicates the class.
    *   *Numerical:* `symbol = int(rate * 2)` (e.g., `rate = torch.sum(spike_history[output_neuron]) / 50`, so 2 Hz -> '4').
*   **Temporal Decoding (Structured Outputs):** Generate sequences by interpreting firing rates of output neurons over sequential time windows.
    *   *Code Generation:* `print(2+2)` -> Window 1: Neuron 'print' fires at 10Hz; Window 2: Neuron '(' fires at 11Hz; Window 3: Neuron '2' fires at 12Hz, etc. Map rates (`rate = torch.sum(...) / 50`) to tokens using a lookup table (`token = lookup[rate]`).
    *   *Logical Deduction:* Output steps ("Given A=1", "A ∧ B = 1", "C = 1") sequentially, mapping tokens to firing rates in successive windows.

#### B.3. Emergent Formation
*   STDP and SIE reinforce connections from internal processing clusters to the appropriate output neurons, ensuring they fire at the correct rates/times to produce the desired output, guided by rewards (`r=1`) for successful task completion.

#### B.4. Implementation
*   Decoding typically occurs on the CPU after retrieving spike history or firing rates from the GPU, logging outputs to SSD (`torch.save(outputs, 'outputs.pt')`).

## 4. Emergent Behaviors and Self-Organization

### A. Emergent Energy Landscape

1.  **Concept & Novelty:**
    *   FUM aims for network stability (analogous to a low-energy state) to **emerge naturally** from the interaction of local learning rules (STDP for excitatory/inhibitory synapses, intrinsic plasticity) and global feedback (SIE), rather than being imposed by a predefined mathematical energy function (like Hopfield networks). **Why is this novel/useful?** It allows the network to find its own stable configurations best suited to the data and tasks, potentially leading to more flexible and robust solutions.
2.  **Mechanism:**
    *   STDP reinforces consistent, reliable pathways. Inhibitory STDP and connections actively balance excitation. Intrinsic plasticity adapts neuron excitability. Synaptic scaling normalizes inputs. SIE feedback guides this process towards rewarded, stable states. The network effectively "settles" into configurations where rewarded patterns are produced with minimal extraneous activity (low variance).
3.  **Stability Metric:**
    *   Firing rate variance (e.g., standard deviation < 0.05 Hz across relevant neuron populations over ~1000 timesteps) is used as a practical, measurable proxy for this emergent stability. High variance indicates inefficient processing and can trigger corrective actions like structural plasticity or SIE penalty (via the 'impact' metric).

### B. Knowledge Graph Evolution (Detailed)

1.  **Process:**
    *   The graph's structure *is* the pattern of learned synaptic weights `w_ij`. It starts sparsely connected with a distance bias (Phase 1). As the network processes input (Phase 2 & 3) and receives SIE feedback, STDP strengthens connections between neurons firing with appropriate timing (`Δt`) for rewarded computations (`total_reward > 0`). Connections irrelevant or detrimental (`total_reward < 0`) are weakened or pruned.
2.  **Outcome:**
    *   Continuous evolution results in a self-organized graph where edge weights implicitly represent learned relationships. Strong paths emerge connecting related concepts (e.g., "calculus" to "algebra") and spanning domains (e.g., visual "square" to mathematical "four sides"). Inhibitory connections shape dynamics and prevent runaway loops.

### C. Self-Modification (Structural Plasticity - Detailed Algorithms)

#### C.1. Rationale & Triggers
*   Mimics biological structural plasticity (synaptogenesis, pruning) for **autonomy and long-term adaptation**. Allows the network to allocate resources and change its structure in response to performance or new demands without external intervention.
*   **Triggers:**
    *   Sustained low average cluster reward (`avg_reward[c] < 0.5` over 1000 steps).
    *   Persistently high firing variance (`std dev > 0.05 Hz` over 1000 steps in a cluster).
    *   Persistently low neuron activity (`rate < 1 Hz` over 10,000 steps).

#### C.2. Growth Algorithm
*   **Trigger:** Low cluster reward (`avg_reward[c] < 0.5`).
*   **Allocation:** Add `num_new_neurons` (e.g., 10% of cluster size).
*   **Initial Parameters:** `V = v_reset` (-70mV), `tau ~ N(20ms, 2ms^2)`, `v_th ~ N(-55mV, 2mV^2)`. (`torch.full`, `torch.normal` on `cuda:1`).
*   **Connections:** Form `conn_size=100` connections per new neuron: 50% targeted to low-reward neurons within the cluster (`torch.randint`), 50% random across network (`torch.randint`). Use distance-dependent probability `exp(-d/σ)` for selection (`torch.multinomial`). Check `conn_history` limit (max 3 additions per pair).
*   **Initial Weights:** Uniform `U(0, 0.3)` (`torch.rand * 0.3`) for excitatory outputs, `U(-0.3, 0)` for inhibitory outputs.
*   **Tensor Resizing:** Expand sparse `w` (convert COO, `torch.cat` new indices/values, convert CSR `torch.sparse_csr_tensor`). Rebalance shards using METIS if distributed (executed on CPU).

#### C.3. Pruning Algorithm
*   **Trigger:** Low neuron activity (`rate < 1 Hz` over 10k steps).
*   **Downstream Compensation (Homeostatic Plasticity):** For each downstream neuron `j` losing input from pruned neuron `k`, compute `lost_input[j] = sum(w[k,j])`. Adjust threshold: `v_th[j] -= lost_input[j] * 0.1` (clamped >= -60mV). Executed on 7900 XTX.
*   **Removal Operation:** Delete row/column `k` from sparse `w`: Identify indices `i=k` or `j=k`, remove them, adjust remaining indices (`>k` decremented), remove corresponding values. Rebuild `torch.sparse_csr_tensor`. Update affected shard, broadcast index changes if distributed.

#### C.4. Rewiring Algorithm & Limits
*   **Trigger:** High cluster variance (`std dev > 0.05 Hz` over 1000 steps).
*   **Mechanism:** Active connection formation beyond STDP, based on performance.
*   **Co-Activation Detection:** Compute pairwise co-activation `co_act[i,j] = sum(spike_history[i] * spike_history[j]) / T` for neurons in the unstable cluster (on MI100).
*   **New Connection Formation:** For pairs with `co_act[i,j] > 0.8` and `w[i,j] = 0`, add a new connection `w[i,j] = 0.1` (excitatory) or `-0.1` (inhibitory, maintaining 80:20 ratio), respecting `conn_history` limit (max 3) and overall sparsity limit. Update `w` by appending indices/values (on 7900 XTX).
*   **Limits:**
    *   Max `0.01 * N^2` new connections per rewiring event.
    *   Max 3 additions per pair lifetime (`conn_history`).
    *   Global sparsity target ~95% (prune weakest if exceeded).
    *   Increase inhibitory connections (20 per 100 excitatory) during rewiring.
    *   Reduce `co_act` threshold (e.g., 0.7) if variance remains high.

### D. Adaptive Domain Clustering (Dynamic k and Edge Cases)

#### D.1. Purpose & Mechanism
*   Dynamically identify functional specializations (domains) emerging within the network by grouping neurons with similar activity profiles.
*   Periodically (e.g., every 1000 timesteps), run k-means clustering (`torch.kmeans` on MI100) on neuron firing rates (`rates = torch.sum(spike_history, dim=1) / 50`).

#### D.2. Determining Number of Clusters (k)
*   **Dynamic Selection using Silhouette Score:**
    *   **Method:** Test `k` in range `[k_min, max_k]`.
        *   `k_min = num_domains` (e.g., 8). Ensures minimum granularity reflecting known task domains.
        *   `max_k = min(num_neurons // 50, num_domains * 2)` (e.g., 16 for 1k neurons). Limits complexity.
    *   **Algorithm:** For each `k`, run `torch.kmeans`, compute silhouette score (`(b-a)/max(a,b)`). Choose `k` with highest score (`best_k = argmax(scores)`).
    *   **Adjustment:** Final `k = max(best_k, k_min)`. If silhouette selects `k < k_min`, override with `k_min`.
    *   **Implementation:** Execute on MI100 GPU.

#### D.3. Cluster Assignment & Reward Attribution (Domain Identification)
*   **Assignment:** Assign neurons to clusters based on similarity to centroids (hard assignment `cluster_id[i] = argmax(similarity)`, soft probabilities `probs[i] = softmax(similarity)`).
*   **Reward Attribution:**
    *   Map current input to a cluster based on induced firing pattern: `input_cluster = argmax(sum(probs * rates, dim=0))`.
    *   Attribute global `total_reward` to this cluster: `cluster_rewards[input_cluster] += total_reward`.
    *   Attribute reward to neurons weighted by probability: `neuron_rewards[i] += total_reward * probs[i, input_cluster]`.
*   **Average Reward:** Compute `avg_reward[c] = cluster_rewards[c] / num_inputs[c]` over 1000 steps (handle division by zero, see D.4). Used as growth trigger.
*   **Implementation:** Maintain `cluster_rewards`, `num_inputs`, `neuron_rewards` tensors on MI100.

#### D.4. Edge Case Handling (Small k, Empty Clusters)
*   **Small k:** If dynamic selection yields `k < k_min`, override with `k = k_min` (rerun kmeans if needed). Ensures minimum functional granularity.
*   **Empty Clusters:** If `num_inputs[c] = 0` (no inputs mapped to cluster `c` over 1000 steps), set `avg_reward[c] = 0` (neutral reward) to avoid division by zero. This triggers growth (`avg_reward < 0.5`) for the unused cluster, promoting exploration. Log metrics to SSD.

#### D.5. Adaptation
*   Clusters reflect the current functional organization and guide structural plasticity (growth targets).

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
*   **Consolidating Core Knowledge:**
    *   **Mechanism:** Mark synapses in high-reward, stable pathways (`w > 0.8`, `avg_reward > 0.9` over 10k steps) as "persistent".
    *   **Persistence:** Exempt persistent synapses from decay.
    *   **Implementation:** Use a sparse boolean tensor `persistent` checked during decay (on 7900 XTX).
    *   **Rationale:** Protects essential learned functions while allowing adaptation in non-core pathways, ensuring long-term functional integrity.
*   **Continual Learning vs. Catastrophic Forgetting (Phase 3):**
    *   *Challenge:* Integrating large volumes of novel information without overwriting previously mastered skills.
    *   *Mechanisms:*
        *   **Synaptic Decay:** Slowly weakens unused connections (`w *= 0.99` every 10k steps), making space for new learning while preserving strong pathways.
        *   **STDP/SIE on New Data:** Novelty in SIE (`novelty > 0.5`) can temporarily increase plasticity (`eta *= 1.2`) to facilitate learning new information, while habituation reduces updates for old, mastered information.
        *   **Persistence Tags:** Exempt core, high-reward synapses (`w > 0.8`, `avg_reward > 0.9`) from decay, robustly protecting core competencies.
        *   **Dynamic Balance:** Plasticity (`eta` increase, growth) is balanced against stability mechanisms (`eta` decrease for high variance, inhibition, persistence threshold adjustments) to gracefully integrate new knowledge without catastrophic forgetting. Accuracy on validation sets is monitored to ensure core skills are retained.

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

## 6. Feasibility and Rationale Summary

### A. Why is FUM considered feasible despite its ambitious goals?

FUM's design posits that superintelligence might not require brute-force scaling and massive datasets. It bets on brain-inspired principles:

1.  **Computational Efficiency of SNNs:** Event-driven computation + high sparsity drastically reduces theoretical load. Practical net efficiency shows significant gains (~11x-194x energy savings vs LLM inference), though less than theoretical maximums due to overhead (See Sec 5.E.3).
2.  **Power of Emergence and Self-Organization:** Complex behavior arises from local rules (STDP, intrinsic plasticity) + global feedback (SIE) + inhibition, without explicit design for every capability. Control mechanisms ensure stability (See Sec 2.D, 4, 5.E.4).
3.  **Data Efficiency of Local Learning:** STDP + SIE reinforcement extracts patterns from few examples (80-300 target), leveraging temporal coding and anti-overfitting mechanisms (See Sec 1.A).
4.  **Adaptability through Structural Plasticity:** Autonomous rewiring, growth, pruning enable long-term learning and resource allocation (See Sec 4.C).
5.  **Validation (AMN Predecessor Relevance):** The predecessor AMN model's success up to 10 units (82% accuracy with 3 examples) provides initial validation for the core SNN-STDP-SIE framework.
    *   **Comparability:** AMN shared the core LIF/STDP mechanisms and basic SIE reward, validating the foundational learning approach.
    *   **Differences:** AMN lacked the full SIE complexity (TD, novelty, etc.), advanced structural plasticity (pruning/rewiring), dynamic clustering, and hierarchical temporal encoding present in FUM.
    *   **Predictive Power:** AMN validates the core learning efficiency but doesn't fully predict FUM's emergent capabilities at scale (N=32B+), which rely on the added complexities and scaling strategies. FUM's performance requires phased validation.
*   **Arguments for Outperforming LLMs:** FUM aims to surpass LLMs on specific tasks requiring deep reasoning or temporal understanding through:
    *   *Emergent Graph:* Flexible cross-domain reasoning potentially superior to static attention.
    *   *SNN Temporal Processing:* Natural handling of sequences and multi-step logic.
    *   *SIE Autonomy:* Learning complex tasks from sparse rewards without massive labeled datasets.
    *   *Limitation:* FUM initially lacks the broad, unstructured knowledge of LLMs due to minimal data; it relies on Phase 3 continuous learning to build comparable breadth over time.
6.  **Reliable Emergence of Computational Primitives:** Theoretical backing (STDP for associative learning, RL theory for SIE guidance, graph theory for self-organization) and simulation evidence (AMN at 10 units, FUM at 1k neurons) suggest that fundamental primitives (numerical representation, arithmetic, basic logic) reliably self-assemble using STDP/SIE on minimal data.
    *   *Mechanism:* STDP strengthens correlations (e.g., input "2" with "number" cluster), SIE rewards correct operations (e.g., `r=1` for `A ∧ B = 1`), and inhibitory neurons enable negation.
    *   *Validation:* AMN achieved 82% on quadratic equations, 80% on AND logic. FUM at 1k neurons shows >80% accuracy on basic arithmetic and logic (AND/OR/NOT).
    *   *Risk Mitigation:* If primitives fail to form reliably (e.g., low accuracy on basic logic), mitigations include adjusting the E/I ratio or reinforcing primitives with targeted ground truth feedback during training.

### B. Strategic Foundation: Balancing Initialization and Learning

FUM's design is a strategic combination of neuroscience principles (SNNs, STDP, plasticity, inhibition) and complex systems theory (emergence, self-organization).

*   **Balance:** It balances a minimal seeded structure with knowledge learned purely from minimal data.
    *   **Initialization Contribution (~10-15%):** Provides a scaffold, not significant prior knowledge.
        *   *Distance-Biased Connectivity:* Encourages local clustering (`exp(-d/σ)`, `σ=5`), mimicking biological structure and accelerating initial cluster formation (~20% faster). It's a structural prior, not a knowledge prior (doesn't encode "2+2=4").
        *   *Parameter Distributions:* Heterogeneous LIF parameters (`tau`, `v_th` from `N()`) add variability, enhancing dynamics but not encoding domain knowledge.
        *   *Initial Weights:* Weak and random (`U(0, 0.3)` for E, `U(-0.3, 0)` for I), requiring STDP/SIE to form functional pathways.
    *   **Learning Contribution (~85-90%):** The vast majority of capability (e.g., >85% target accuracy) emerges from STDP/SIE processing the 80-300 training examples, forming strong, functional pathways (`w[i,j] ≈ 0.8`) within the knowledge graph. The minimal data learning claim remains impactful as the initialization primarily accelerates, rather than dictates, learning.
    *   **Sensitivity to Initialization:** Performance shows moderate sensitivity. Changes to distance bias (`σ`) or parameter distributions (`std`) affect clustering speed or dynamics slightly (e.g., ±3-5% accuracy impact), but STDP/SIE learning dominates the final outcome. The chosen scheme optimizes early learning efficiency on constrained hardware.
*   **Core Premise:** The synergistic combination of SNN efficiency, emergent self-organization, data-efficient local learning, and structural adaptability offers a robust and efficient pathway towards advanced AI, contrasting with brute-force scaling. The design's validation lies in demonstrating the coherent emergent intelligence produced during practical implementation.

# References

This list provides potential citations for the concepts, algorithms, and frameworks mentioned in the FUM document. Selections can be integrated into footnotes or a dedicated reference section.

## FUM System Design

* Lietz, J. (2025). *How the Fully Unified Model (FUM) Works*. [Unpublished technical specification / Design document]. (Details the specific FUM architecture and the following potentially novel contributions:
    * **Overall Integrated System:** The synergistic combination of SNNs, the Self-Improvement Engine, detailed structural plasticity, emergent knowledge graph, hybrid computation model, and phased training strategy as a unified system.
    * **Self-Improvement Engine (SIE):** The specific reward formulation `total_reward = TD_error + novelty - habituation + self_benefit`, including the `self_benefit = complexity * impact` calculation, and its direct, quadratically scaled modulation of STDP learning rates via eligibility traces.
    * **Integrated Structural Plasticity System:** The specific set of triggers (low cluster reward, high variance, low neuron activity), detailed algorithms (targeted growth, pruning with homeostatic compensation, co-activation-based rewiring), and defined operational limits (e.g., connection history, sparsity target, E/I balancing during rewiring).
    * **Adaptive Domain Clustering Integration:** Employing K-Means with dynamic `k` selection (via Silhouette Score within defined bounds) to define TD-Learning states and guide reward attribution and structural plasticity within the SNN framework, including specific edge case handling.
    * **Multi-Phase Training Strategy:** The explicit three-phase approach (Seed Sprinkling -> Tandem Complexity Scaling -> Continuous Self-Learning) tailored for minimal data dependency and autonomous operation.
    * **Emergent Knowledge Graph & Routing Mechanism:** Reliance on learned SNN connectivity dynamically shaped by STDP/SIE/plasticity for coordination and information routing, explicitly contrasting with predefined layers or coordinators.
    * **Emergent Energy Landscape Concept:** Proposing network stability as arising naturally from the interplay of local/global rules (measured via variance), rather than an explicitly defined energy function.
    * **Specific Hybrid Computation Model:** The defined heterogeneous GPU workload distribution (LIF kernel vs. PyTorch/SIE/Clustering) and associated data flow/synchronization strategy.
    * **Specific Temporal Encoding Schemes:** The described hierarchical methods for encoding structured data (e.g., code syntax trees, logical propositions) into temporal spike patterns.
    * **Minimal Data Philosophy:** The core design goal targeting high performance from minimal inputs (80-300) based on a defined balance between initialization and learning.)

## SNNs, LIF Model, and Neuron Dynamics

* Burkitt, A. N. (2006). A review of the leaky integrate-and-fire neuron model. *Biological Cybernetics*, *95*(1), 1-19.
* Destexhe, A., & Marder, E. (2004). Plasticity in single neuron and circuit computations. *Nature*, *431*(7010), 789-795.
* Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014). *Neuronal dynamics: From single neurons to networks and models of cognition*. Cambridge University Press.
* Lapicque, L. (1907). Recherches quantitatives sur l'excitation électrique des nerfs traitée comme une polarisation. *Journal de Physiologie et de Pathologie Générale*, *9*, 620-635.
* Maass, W. (1997). Networks of spiking neurons: the third generation of neural network models. *Neural Networks*, *10*(9), 1659-1671.
* Marder, E., & Goaillard, J. M. (2006). Variability, compensation, and homeostasis in neuron and network function. *Nature Reviews Neuroscience*, *7*(7), 563-574.
* Pfeiffer, M., & Pfeil, T. (2018). Deep learning with spiking neurons: opportunities and challenges. *Frontiers in Neuroscience*, *12*, 774. https://doi.org/10.3389/fnins.2018.00774
* Triesch, J. (2007). Synergies between intrinsic and synaptic plasticity mechanisms. *Neural Computation*, *19*(4), 885-909.

## Neural Plasticity & Stability Mechanisms

* Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. *Journal of Neuroscience*, *18*(24), 10464-10472.
* Chklovskii, D. B., Mel, B. W., & Svoboda, K. (2004). Cortical rewiring and information storage. *Nature*, *431*(7010), 782-788.
* Helias, M., Tetzlaff, T., & Diesmann, M. (2014). The correlation structure of local neuronal networks intrinsically results from recurrent connectivity. *PLoS Computational Biology*, *10*(1), e1003458. https://doi.org/10.1371/journal.pcbi.1003458
* Holtmaat, A., & Svoboda, K. (2009). Experience-dependent structural synaptic plasticity in the mammalian brain. *Nature Reviews Neuroscience*, *10*(9), 647-658.
* Markram, H., Lübke, J., Frotscher, M., & Sakmann, B. (1997). Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSPs. *Science*, *275*(5297), 213-215.
* Song, S., Miller, K. D., & Abbott, L. F. (2000). Competitive Hebbian learning through spike-timing-dependent synaptic plasticity. *Nature Neuroscience*, *3*(9), 919-926.
* Turrigiano, G. G., Leslie, K. R., Desai, N. S., Rutherford, L. C., & Nelson, S. B. (1998). Activity-dependent scaling of quantal amplitude in neocortical neurons. *Nature*, *391*(6670), 892-896.
* Vogels, T. P., Sprekeler, H., Zenke, F., Clopath, C., & Gerstner, W. (2011). Inhibitory plasticity balances excitation and inhibition in sensory pathways and memory networks. *Science*, *334*(6062), 1569-1573. https://doi.org/10.1126/science.1211095

## Reinforcement Learning & SIE Components

* Florian, R. V. (2007). Reinforcement learning through modulation of spike-timing-dependent synaptic plasticity. *Neural Computation*, *19*(6), 1468-1502.
* Frémaux, N., & Gerstner, W. (2016). Neuromodulated spike-timing-dependent plasticity, and theory of three-factor learning rules. *Frontiers in Neural Circuits*, *9*, 85. https://doi.org/10.3389/fncir.2015.00085
* Izhikevich, E. M. (2007). Solving the distal reward problem through linkage of STDP and dopamine signaling. *Cerebral Cortex*, *17*(10), 2443-2452. https://doi.org/10.1093/cercor/bhl147
* Marsland, S. (2014). *Machine learning: an algorithmic perspective*. CRC press. (For general concepts potentially underlying novelty/habituation metrics)
* Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. *Machine Learning*, *3*(1), 9-44.
* Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction*. MIT press.

## Emergence, Self-Organization, & Knowledge Graphs

* Bak, P., Tang, C., & Wiesenfeld, K. (1987). Self-organized criticality: An explanation of the 1/f noise. *Physical Review Letters*, *59*(4), 381-384.
* Hogan, A., Blomqvist, E., Cochez, M., d'Amato, C., Melo, G. D., Gutierrez, C., Kirrane, S., Gayo, J. E. L., Navigli, R., Neumaier, S., Ngomo, A. C. N., Polleres, A., Rashid, S. M., Rula, A., Schmelzeisen, L., Sequeda, J., Staab, S., & Zimmermann, A. (2021). Knowledge graphs. *ACM Computing Surveys (CSUR)*, *54*(4), 1-37. https://doi.org/10.1145/3447772
* Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. *Proceedings of the National Academy of Sciences*, *79*(8), 2554-2558.
* Mitchell, M. (2009). *Complexity: A guided tour*. Oxford University Press.

## Clustering & Graph Partitioning

* Karypis, G., & Kumar, V. (1998). A fast and high quality multilevel scheme for partitioning irregular graphs. *SIAM Journal on Scientific Computing*, *20*(1), 359-392.
* MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In *Proceedings of the fifth Berkeley symposium on mathematical statistics and probability* (Vol. 1, pp. 281-297). University of California Press.
* Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, *20*, 53-65.

## I/O Processing

* Brette, R. (2015). Philosophy of the spike: rate-based vs. spike-based theories of computation. *Frontiers in Systems Neuroscience*, *9*, 151. https://doi.org/10.3389/fnsys.2015.00151
* Thorpe, S., Delorme, A., & Van Rullen, R. (2001). Spike-based strategies for rapid processing. *Neural Networks*, *14*(6-7), 715-725.

## Optimization & Frameworks

* AMD. (n.d.). *ROCm Documentation*. Retrieved March 30, 2025, from https://rocm.docs.amd.com/
* Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In *Advances in Neural Information Processing Systems 32* (pp. 8026-8037). Curran Associates, Inc.
* Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization of machine learning algorithms. In *Advances in Neural Information Processing Systems 25* (pp. 2951-2959). Curran Associates, Inc.
