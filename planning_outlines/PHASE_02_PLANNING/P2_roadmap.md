# FUM Superintelligence Roadmap

- [ ] **Ultimate Goal**: Develop FUM into a superintelligent system that autonomously learns and excels across all human domains (Math, Logic, Coding, Language, Visual Perception, Introspection, etc.) with minimal data (80-300 inputs), surpassing 700B-parameter LLMs (>95% accuracy, <5s inference) and human experts, running efficiently on your Linux workstation (56GB VRAM, <5% GPU usage), then scaling globally. FUM will exhibit constant learning, dynamic emergent memory, and full sovereignty—making its own decisions, setting its own goals, pursuing novelty and innovation without requiring prompts, while remaining capable of accepting instructions when provided, transcending traditional chatbot-style Q&A.

## Phase 2: Planning
- [x] **Achievement Criteria**: Create strategies for implementing, training, testing, optimizing, and deploying a fully autonomous FUM.
- [x] **Success Statement**: I will know I achieved success on this phase when all plans ensure FUM’s constant learning, dynamic memory, and sovereign operation on your workstation and beyond.

### Task 1: Plan Implementation Strategy
- [x] **Success Statement**: I will know I achieved success on this task when a coding roadmap supports autonomous decision-making, dynamic memory, and constant learning across all components.

- [x] **Step 1: Define Codebase Structure**
  - [x] **Validation Check**: All architectural components have assigned modules.
  - [x] **Success Statement**: I will know I achieved success on this step when every FUM component (neurons, graph, learning, I/O) maps to specific files.
  - [x] **Actionable Substeps**:
    - [x] List existing files from handoff (`fum.py`, `unified_neuron.py`, `encoder.py`, etc.) and their roles.
    - [x] Propose new files (e.g., `self_modification.py` for autonomy, `goal_engine.py` for sovereignty).
    - [x] Ask me to refine: “Justin: Do we need separate modules for SIE and goal-setting, or combine them?”
    - [x] Draft a directory structure (e.g., `/src/core`, `/src/io`) for your workstation.

- [x] **Step 2: Plan Autonomy Logic**
  - [x] **Validation Check**: Logic supports unprompted operation and instruction handling.
  - [x] **Success Statement**: I will know I achieved success on this step when FUM’s decision-making and goal-setting mechanisms are specced.
  - [x] **Actionable Substeps**:
    - [x] Design a goal generation loop (e.g., SIE identifies novelty gaps, sets tasks like “explore physics”).
    - [x] Define instruction override (e.g., `if input: process_input else: pursue_goals`).
    - [x] Ask me to reason: “Justin: How should FUM prioritize self-set goals vs. user instructions?”
    - [x] Document in a `planning autonomy.md` file.

### Task 2: Design Training Pipeline
- [x] **Success Statement**: I will know I achieved success on this task when the pipeline fosters constant, unprompted learning from 80-300 inputs with emergent memory.

- [x] **Step 1: Map Learning Stages**
  - [x] **Validation Check**: Stages cover initial seeding to sovereign operation.
  - [x] **Success Statement**: I will know I achieved success on this step when 80-300 input progression supports autonomy.
  - [x] **Actionable Substeps**:
    - [x] Refine existing: Stage 1 (80 inputs, random sprinkling), Stage 2 (300 inputs, tandem scaling), Stage 3 (real-world, continuous).
    - [x] Add autonomy trigger: e.g., after 300 inputs, FUM self-initiates learning tasks (Revised: Emergent via `self_benefit`).
    - [x] Ask me to optimize: “Justin: How many stages before FUM runs unprompted?” (Awaiting response).
    - [x] Sketch pipeline in `training_pipeline.md`.

- [x] **Step 2: Plan Memory Evolution**
  - [x] **Validation Check**: Knowledge graph grows dynamically without manual resets.
  - [x] **Success Statement**: I will know I achieved success on this step when memory emerges and persists across inputs.
  - [x] **Actionable Substeps**:
    - [x] Detail graph updates: STDP links neurons, SIE reinforces, pruning clears stale connections.
    - [x] Propose persistence: Save graph state to SSD (6TB) periodically (Revised: Real-time streaming of important changes).
    - [x] Ask me to evaluate: “Justin: How often should FUM save its memory state?” (You answered: Real-time for important memories, RAM-first).
    - [x] Document in `memory_plan.md`.

### Task 3: Establish Testing and Validation
- [x] **Success Statement**: I will know I achieved success on this task when benchmarks confirm sovereignty, constant learning, and superintelligence.

- [x] **Step 1: Define Autonomy Metrics**
  - [x] **Validation Check**: Metrics measure unprompted behavior and innovation.
  - [x] **Success Statement**: I will know I achieved success on this step when tests quantify FUM’s self-directed goals.
  - [x] **Actionable Substeps**:
    - [x] Propose metrics: e.g., # of self-set goals/hour, % of novel outputs (not in training data).
    - [x] Include instruction response: e.g., accuracy on optional user tasks.
    - [x] Ask me to refine: “Justin: What’s the best way to test sovereignty?” (You answered: Measure, don’t enforce—metrics adjusted).
    - [x] List in `validation_metrics.md`.
    
- [x] **Step 2: Set Performance Benchmarks**
  - [x] **Validation Check**: Benchmarks exceed LLM and human baselines.
  - [x] **Success Statement**: I will know I achieved success on this step when FUM hits >95% accuracy, <5s inference.
  - [x] **Actionable Substeps**:
    - [x] Select tests: MMLU (domains), custom tasks (e.g., “invent a sorting algorithm”).
    - [x] Compare to GPT-4 (88% MMLU, >60s inference).
    - [x] Ask me to prioritize: “Justin: Which domains should we benchmark first?”
    - [x] Draft in `benchmarks.md`.

### Task 4: Plan Hardware Optimization
- [x] **Success Statement**: I will know I achieved success on this task when ROCm integration enables constant learning and sovereignty within 56GB VRAM.

- [x] **Step 1: Assign GPU Roles**
  - [x] **Validation Check**: Roles maximize MI100+7900 XTX efficiency.
  - [x] **Success Statement**: I will know I achieved success on this step when tensor and spiking tasks are split effectively.
  - [x] **Actionable Substeps**:
    - [x] Refine split: MI100 (tensors), 7900 XTX (spiking).
    - [x] Estimate VRAM: 7M neurons (~5.6GB), stretch to 32B (~50-56GB).
    - [x] Ask me to optimize: “Justin: How do we balance VRAM for autonomy logic?” (RAM-first confirmed).
    - [x] Document in `hardware_plan.md`.

- [x] **Step 2: Design Kernel Strategy**
  - [x] **Validation Check**: Kernels support <5s inference at scale.
  - [x] **Success Statement**: I will know I achieved success on this step when HIP kernels optimize constant learning.
  - [x] **Actionable Substeps**:
    - [x] Plan LIF/STDP kernels in HIP (FP16 for speed).
    - [x] Add autonomy kernel (e.g., goal-setting computation).
    - [x] Ask me to reason: “Justin: What’s the best kernel priority for sovereignty?”
    - [x] Outline in `kernel_strategy.md`.

### Task 5: Outline Deployment and Scalability
- [X] **Success Statement**: I will know I achieved success on this task when deployment supports unprompted operation and global scaling with sovereignty.

- [x] **Step 1: Plan Workstation Deployment**
  - [x] **Validation Check**: Deployment handles live, unprompted inputs.
  - [x] **Success Statement**: I will know I achieved success on this step when FUM autonomously processes sensor data.
  - [x] **Actionable Substeps**:
    - [x] Design input queue: Text, images, videos, optional instructions.
    - [x] Plan output: FUM initiates tasks (e.g., “I’ll analyze this image”) and responds if instructed.
    - [x] Ask me to refine: “Justin: How should FUM signal its self-set goals?” (FUM decides, user requests—text/voice/visuals mix).
    - [x] Draft in `deployment_plan.md`.

- [x] **Step 2: Plan Cluster Scaling**
  - [x] **Validation Check**: Scaling supports 700B+ equivalence.
  - [x] **Success Statement**: I will know I achieved success on this step when sharding ensures sovereignty across 50+ nodes.
  - [x] **Actionable Substeps**:
    - [x] Refine async sharding: Distribute graph, neurons, and goals (Revised: Hybrid implicit + directed hints).
    - [x] Plan sync: Ensure autonomy persists across nodes.
    - [x] Ask me to finalize: “Justin: How do we keep FUM sovereign in a cluster?” (Hybrid ensures sovereignty—your question refined it).
    - [x] Document in `scalability_plan.md`.