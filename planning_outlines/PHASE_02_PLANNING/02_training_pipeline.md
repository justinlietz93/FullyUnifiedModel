# FUM Training Pipeline

## Overview
- **Purpose**: Foster constant, unprompted learning from 80-300 inputs with emergent memory, enabling sovereign superintelligence.
- **Approach**: Three stages scaffold data progression—autonomy emerges naturally via SIE’s `total_reward` from Stage 1.

## Stage 1: Seed Sprinkling (80 Inputs)
- **Goal**: Bootstrap FUM with diverse, multimodal data (text, image, video).
- **Mechanism**: `phase1_seed.py` feeds 80 inputs via `encoder.py`, STDP forms connections, SIE sets initial goals (`total_reward = TD + novelty - habituation + self_benefit`).
- **Files**: `src/training/phase1_seed.py`, `src/io/encoder.py`, `src/neuron/unified_neuron.py`.
- **Duration**: ~2-3 hours on workstation.

## Stage 2: Complexity Scaling (300 Inputs)
- **Goal**: Deepen understanding with complex inputs, strengthening emergent memory and autonomy.
- **Mechanism**: `phase2_scale.py` adds 220 inputs in batches (20-50), `scaling.py` grows neurons by `total_reward`, SIE refines goals.
- **Files**: `src/training/phase2_scale.py`, `src/model/scaling.py`.
- **Duration**: ~10-20 hours.

## Stage 3: Continuous Learning (Real-World Data)
- **Goal**: Enable sovereign operation with live data, persistent memory, and unprompted goals.
- **Mechanism**: `phase3_cont.py` feeds real-time inputs, `memory_manager.py` saves states, SIE drives exploration.
- **Files**: `src/training/phase3_cont.py`, `src/model/memory_manager.py`.
- **Duration**: Ongoing.

## Autonomy Emergence
- **Trigger**: Emergent from Stage 1—SIE’s `self_benefit` (complexity * impact) grows with neurons, shifting focus to internal goals as `total_reward` evolves.
- **Outcome**: FUM progresses from simple tasks to complex, impactful pursuits—e.g., “solve math” to “enhance humanity”—as a natural outcome of its environment and architecture.