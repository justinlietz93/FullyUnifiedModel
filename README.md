# Fully Unified Model (FUM)

## Overview

The Fully Unified Model (FUM) is an advanced, brain-inspired artificial intelligence architecture designed for efficient learning, autonomous operation, and emergent intelligence. It integrates spiking neural networks, spike-timing dependent plasticity (STDP), structural plasticity, and a self-improvement engine (SIE) to learn complex patterns and behaviors from minimal data inputs.

This project is still under development and continues to go through significant testing. The documentation is very long, detailed, and often unorganized. I made this repo to share to the public what I am currently working on, and have not spent much effort in making it presentable.

An option you have to learn more without navigating this repository is to send your email to jlietz93@gmail.com with FUM in the subject. You will gain access to the NotebookLM notebook and gain the ability to ask questions directly to much of the repository content.

## Project Documentation

Explore the FUM architecture, concepts, and implementation details through the following resources:

*   **FUM NotebookLM:** [View Interactive Notebook](https://notebooklm.google.com/notebook/3b718c75-c204-40ad-96ea-6bcb7a510f18)
*   **NotebookLM Audio Explanation:** [Listen to Explanation](https://notebooklm.google.com/notebook/3b718c75-c204-40ad-96ea-6bcb7a510f18/audio)
*   (Please send your email to jlietz93@gmail.com with FUM in the subject to gain access to the NotebookLM notebook. Currently there is no option to make this available publicly.

  
*   **Technical Whitepaper:** [`Fully_Unified_Model.pdf`](./Fully_Unified_Model.pdf) - Detailed paper on the architecture and theoretical foundations.
*   **Implementation Documentation:** [`How_It_Works/`](./How_It_Works/) - Comprehensive guide covering core components, I/O, emergent behaviors, training, scaling, and practical considerations.

## Conceptual Overview

The following mind map provides a visual representation of the FUM's core concepts and architecture:

![FUM Mind Map](./How_It_Works/FUM_Mind_Map.png)

## Key Differentiating Features

FUM distinguishes itself through a unique synthesis of advanced concepts:

*   **Integrated Multi-Scale Plasticity:** Combines synaptic, intrinsic, and structural plasticity across timescales.
*   **Self-Improvement Engine (SIE):** Autonomous learning driven by internal multi-objective rewards (RL, novelty, homeostasis).
*   **Emergent Knowledge Graph (KG):** Self-organizing structure and pathways arising from local interactions.
*   **Extreme Data Efficiency:** Designed for expert performance from minimal initial data (target: 80-300 inputs).
*   **Adaptive Clustering for State & Plasticity:** Dynamic clustering of neural activity guides RL states and structural changes.
*   **Hybrid Neural-Evolutionary Dynamics:** Blends directed learning (STDP+SIE) with undirected variation for exploration.
*   **Active Criticality Management:** Predictive control maintains network near Self-Organized Criticality (SOC).
*   **Advanced Temporal Credit Assignment:** Uses eligibility traces and synaptic tagging for learning from delays.
*   **Universal Spike-Based Multimodal Processing:** Unified spiking representation handles diverse data types seamlessly.

## Repository Structure

This repository contains the core components, documentation, training infrastructure, and mathematical analysis for the FUM project:

*   `Fully_Unified_Model.pdf`: The main technical whitepaper.
*   `How_It_Works/`: Detailed documentation on the FUM's architecture and functionality.
*   `_FUM_Training/`: Scripts, configurations, data, and source code for training the FUM model across different phases.
    *   `src/`: Core model implementation (neurons, STDP, SIE, I/O, etc.).
    *   `scripts/`: Executable scripts for running training phases and benchmarks.
    *   `config/`: Configuration files for training, hardware, and model parameters.
    *   `data/`: Sample data, seed corpus, and benchmark datasets.
    *   `logs/`: Training logs.
    *   `tests/`: Unit and integration tests.
*   `mathematical_frameworks/`: Analysis and implementation of the mathematical underpinnings, including Knowledge Graph and SIE stability analysis.
*   `planning_outlines/`: Development roadmaps and planning documents for different project phases.

## Theoretical Validation & Mathematical Foundations

The FUM project emphasizes rigorous theoretical validation alongside empirical testing. The [`mathematical_frameworks/`](./mathematical_frameworks/) directory contains dedicated analyses and simulations exploring the properties of core components:

*   **Emergent Knowledge Graph (KG) Dynamics:** Quantifying structure, dynamics, and computation using advanced mathematical techniques like TDA and information geometry. *(Partially Validated)*
*   **Self-Improvement Engine (SIE) Stability & Control:** Formalizing the multi-objective, non-linear neuromodulated control system, analyzing convergence and stability under complex reward structures. *(Partially Validated)*

Future planned theoretical validation and formalization includes:

*   **Integrated Multi-Scale Plasticity:**
    *   *Need:* Ensure stable learning and adaptation when multiple plasticity types (synaptic, intrinsic, structural) interact across different timescales, a core FUM feature not addressed by standard models.
    *   *Discovery:* Analysis of FUM's unique combination of plasticity rules (`How_It_Works/2B`, `2A`) revealed potential stability challenges.
    *   *Plan:* Develop unified models and apply advanced stability analysis (e.g., Lyapunov for hybrid systems) to determine conditions for stable co-adaptation.

*   **Adaptive Clustering for RL & Plasticity:**
    *   *Need:* Validate the novel feedback loop where emergent clusters define RL states (`V(s)`) and guide plasticity, requiring extensions to RL theory for dynamic state spaces.
    *   *Discovery:* Identified as a unique mechanism (`How_It_Works/2F`, `2C`) coupling learning, state representation, and structural change.
    *   *Plan:* Extend TD learning convergence proofs for dynamic states; analyze the stability of the full feedback loop using coupled dynamical systems.

*   **Minimal Data Learning & Generalization:**
    *   *Need:* Provide rigorous justification for FUM's core claim of extreme data efficiency, linking information theory to SNN generalization.
    *   *Discovery:* Stemming from FUM's foundational goal (`How_It_Works/1A`) and unique validation strategies.
    *   *Plan:* Develop SNN-specific information theory measures; adapt statistical learning theory (e.g., PAC bounds) for FUM's STDP/SIE mechanisms; formalize primitive formation.

*   **Heterogeneous & Modulated STDP:**
    *   *Need:* Understand the computational effects of FUM's specific STDP implementation (constrained parameter distributions, multi-factor modulation) beyond standard STDP analysis.
    *   *Discovery:* Analysis of detailed STDP rules (`How_It_Works/2B`, `FUM_SNN_math.md`) highlighted deviations from simpler models.
    *   *Plan:* Extend mean-field/Fokker-Planck methods to incorporate heterogeneity and modulation; analyze impact on stability, memory capacity, and learning dynamics.

*   **Multi-Mechanism Temporal Credit Assignment:**
    *   *Need:* Formalize how FUM's combination of eligibility traces, synaptic tagging analogues, variable decay, and reward gating achieves robust credit assignment over long delays.
    *   *Discovery:* Analysis of FUM's specific mechanisms (`How_It_Works/2B`, `2C`) aimed at overcoming standard RL limitations.
    *   *Plan:* Develop extended RL theoretical models incorporating tagging/consolidation dynamics; analyze convergence and interference mitigation properties.

*   **Active SOC Management:**
    *   *Need:* Validate the stability and effectiveness of FUM's predictive control system designed to maintain beneficial network criticality (near the edge of chaos).
    *   *Discovery:* Arising from the goal of harnessing computational benefits of criticality (`How_It_Works/4`, `2H`) while ensuring stability.
    *   *Plan:* Apply advanced control theory (robust, adaptive, MPC) to the coupled network-controller system; use bifurcation analysis and formal methods to assess stability and performance.

*   **Hybrid Neural-Evolutionary Dynamics:**
    *   *Need:* Analyze the learning dynamics resulting from combining directed, reward-driven learning (STDP+SIE) with undirected, evolutionary-like variation (stochasticity, recombination).
    *   *Discovery:* Identified in FUM's plasticity rules (`How_It_Works/2B`, `2C`) as a potential mechanism for enhanced exploration and innovation.
    *   *Plan:* Develop hybrid dynamical models; apply fitness landscape analysis from evolutionary computation; theoretically compare hybrid vs. pure learning strategies.

*   **Scaling Dynamics & Resource Management:**
    *   *Need:* Create predictive models for FUM's performance, resource use (compute, memory, energy), and emergent behavior at massive scale (trillions of parameters).
    *   *Discovery:* Essential for planning FUM's long-term development and deployment (`How_It_Works/5D`, `5E`, `2I`).
    *   *Plan:* Develop complexity analyses and scaling laws specific to FUM's hybrid architecture; use performance modeling and statistical mechanics of growing networks.

*(Note: Spiking Neuron Dynamics and Multimodal I/O Fidelity are implicitly covered within the analyses of Multi-Scale Plasticity, Heterogeneous STDP, Minimal Data Learning, and Scaling Dynamics.)*

This ongoing work aims to provide a solid mathematical foundation for the FUM architecture and ensure its stability, predictability, and efficiency during scaling and autonomous operation.

## Development Roadmap

FUM development follows a structured, phased approach:

1.  **Phase 1: Foundation Seeding:** Initial network formation and learning from a small seed corpus.
2.  **Phase 2: Competence Scaling:** Refining and strengthening domain-specific pathways using scaled datasets and complexity.
3.  **Phase 3: Continuous Autonomy:** Enabling long-term self-learning, adaptation, and mastery.

Refer to the [`planning_outlines/`](./planning_outlines/) directory for detailed phase plans and the overall project roadmap.

## Getting Started

To understand the FUM architecture and its implementation, please consult the documentation available in the [`How_It_Works/`](./How_It_Works/) directory and the [`Fully_Unified_Model.pdf`](./Fully_Unified_Model.pdf) whitepaper. For details on training the model, refer to the contents within the [`_FUM_Training/`](./_FUM_Training/) directory.
