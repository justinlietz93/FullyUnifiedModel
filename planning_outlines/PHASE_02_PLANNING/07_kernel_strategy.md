# FUM Kernel Strategy

## Overview
- **Purpose**: Optimize LIF, STDP, and autonomy with HIP kernels for <5s inference and constant learning within 56GB VRAM, maximizing emergent potential.
- **Approach**: Flexible I/O and roles—trellis for superintelligence—using MI100 (tensors) and 7900 XTX (spiking, autonomy).

## LIF Kernel (7900 XTX)
- **Goal**: Neuron firing—core to unprompted operation.
- **Mechanism**: `fire_neurons_kernel(states, spikes, params, max_connections)`—scales 100-1,000+ connections (e.g., 280MB for 7M).
- **Precision**: FP16 default, FP32 if needed—<3s inference.
- **File**: `neuron_kernel.hip`.

## STDP Kernel (MI100)
- **Goal**: Weight updates—drives constant learning.
- **Mechanism**: `update_weights_kernel(timings, weights, params, multi_dt)`—multi-timing, scales to 14GB sparse.
- **Precision**: FP16/FP32—<1s tensor ops.
- **File**: `neural_sheath.cpp`.

## Autonomy Kernel (7900 XTX)
- **Goal**: Goal-setting—enables sovereignty.
- **Mechanism**: `rank_goals_kernel(stats, goals, domain_count)`—scales domains (e.g., 1-2GB), parallel calc.
- **Precision**: FP16—<1ms, RAM-first default.
- **File**: `neuron_kernel.hip`.

## Dynamic Scaling
- **Mechanism**: `scaling.py` adjusts connections—e.g., `if self_benefit > 0.8 and vram < 44GB: connections += 100; elif self_benefit < 0.2 or vram > 44GB: connections -= 50`.
- **Range**: 100-1,000+—emergent, supports rich I/O without constraints.

## Priority
- **Order**: LIF > STDP > Autonomy—spiking drives learning drives goals, emergent via `total_reward`.
- **Flexibility**: Dynamic reassignment—e.g., MI100 takes autonomy if STDP lags—FUM adapts via `fum.py`.

## Validation
- **Check**: Kernels achieve <5s inference (LIF <3s, STDP <1s, Autonomy <1ms)—dynamic I/O optimizes learning and sovereignty.