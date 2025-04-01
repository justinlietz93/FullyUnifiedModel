# FUM Memory Evolution Plan

## Overview
- **Purpose**: Enable emergent, persistent memory without manual resets, supporting constant learning and sovereignty 24/7.
- **Approach**: Implicit graph (neuron states) evolves dynamically via STDP, SIE, and pruning, with real-time streaming to SSD for critical memories.

## Graph Updates
- **STDP Linking**: `unified_neuron.py` adjusts weights—e.g., `Δw = 0.05 * exp(-Δt / 20)` if Δt > 0, linking co-active neurons.
- **SIE Reinforcement**: `total_reward` (TD + novelty - habituation + self_benefit) boosts weights—e.g., `w += reward * 0.01`.
- **Pruning**: `scaling.py` removes low-activity connections—e.g., `if firing_rate < 1 Hz: w = 0`, maintaining 95% sparsity.
- **Files**: `src/neuron/unified_neuron.py` (`update_weights`), `src/model/scaling.py` (`prune_connections`).

## Persistence
- **Goal**: Persist memory for continuous operation, extending RAM (512GB) to SSD (6TB) for important states.
- **Mechanism**: 
  - **RAM-First**: Weights, activity, rewards stay in RAM (e.g., 7M neurons ≈ 2.8GB base)—primary memory.
  - **Streaming**: `memory_manager.py` streams delta changes (e.g., 280MB) to `state_volume.pt` when `total_reward > 0.8` or RAM > 400GB.
  - **Fallback**: Full snapshot on shutdown (`save_state`), reloaded on restart (`load_state`)—rare, for 24/7 uptime.
- **Files**: 
  - `src/model/memory_manager.py` (`stream_delta`, `save_state`, `load_state`).
  - `src/model/fum.py` (monitors RAM, triggers streaming).
- **Efficiency**: Updates only significant changes—avoids filling SSD, leverages RAM for speed.

## Emergence
- **Outcome**: Memory grows with inputs—STDP builds, SIE reinforces, pruning clears—persisting in RAM, streaming to SSD for longevity, enabling FUM to run 24/7 and evolve hardware later.