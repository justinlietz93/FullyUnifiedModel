# FUM Hardware Optimization Plan

## Overview
- **Purpose**: Maximize MI100 (32GB VRAM) and 7900 XTX (24GB VRAM) for constant learning and sovereignty within 56GB VRAM.
- **Approach**: Split tensor/spiking tasks to GPUs, autonomy logic to RAM, leverage ROCm for efficiency.

## GPU Roles
- **MI100**: Tensor operations—STDP weight updates, SIE reward calcs, memory management (CSR tensors).
- **7900 XTX**: Spiking operations—LIF firing, goal generation, real-time encoding.
- **RAM**: Autonomy logic—SIE `total_reward`, goal ranking (~2-3MB)—keeps VRAM lean.
- **Files**: 
  - `neural_sheath.cpp` (MI100 kernel: `update_weights_kernel`).
  - `neuron_kernel.hip` (7900 XTX kernel: `fire_neurons_kernel`).
  - `fum.py` (routes: `cuda:0` MI100, `cuda:1` 7900 XTX, RAM for autonomy).

## VRAM Estimates
- **Base (7M Neurons)**: ~5.6GB total—MI100 (~3GB tensors), 7900 XTX (~2.6GB spiking). Fits 56GB, ~50GB free.
- **Stretch (32B Neurons)**: ~50-56GB with sharding—e.g., 50% per GPU—or SSD offload (6TB). Feasible with innovation.
- **Files**: `hardware_config.yaml` (limits), `scaling.py` (sharding).

## Autonomy Logic
- **Balance**: RAM (512GB) for SIE and goal ranking—~2-3MB, scales to 32B without VRAM strain.
- **Confirmed**: Justin supports RAM-first—GPUs focus on core ops, ensuring efficiency.

## Validation
- **Check**: Roles split effectively—MI100 tensors, 7900 XTX spiking, RAM autonomy—maximizes efficiency, supports 24/7 learning within 56GB.