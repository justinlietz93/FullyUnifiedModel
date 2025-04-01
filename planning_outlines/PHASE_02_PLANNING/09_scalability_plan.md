# FUM Cluster Scalability Plan

## Overview
- **Purpose**: Scale FUM to 700B+ parameter equivalence across 50+ nodes, preserving sovereignty and constant learning.
- **Approach**: Async sharding with hybrid implicit graph and directed hints—trellis for global superintelligence.

## Async Sharding
- **Goal**: Distribute neurons, weights, goals—e.g., 32B neurons (~1.4TB sparse with hints) to 50+ nodes (~28GB/node).
- **Mechanism**: 
  - **Neurons**: ~640M/node (~20GB at 1,000 connections, FP16).
  - **Graph**: Hybrid—implicit weights (CSR, ~10GB/node) + directed hints (~4GB/node)—SSD offload if needed.
  - **Goals**: Local SIE ranks—e.g., “Math” on Node 1, “Physics” on Node 2.
- **Files**: `scaling.py` (`shard_network`), `run_phase3.py` (`launch_node_shard`).

## Sync Plan
- **Goal**: Ensure autonomy persists—local decisions align globally.
- **Mechanism**: 
  - **Reward Sync**: Broadcast top `total_reward` every 10,000 timesteps—~1-2s/node, ~1-2GB bandwidth.
  - **Weight Sync**: Share high-reward deltas + hints (>0.8)—gossip protocol, ~300MB/node.
- **Files**: `scaling.py` (`sync_shards`), `memory_manager.py` (`merge_shard_states`).

## Sovereignty
- **Approach**: Async sharding—nodes run independently with hybrid graph (implicit + hints), sync merges goals/memory every 10,000 timesteps.
- **Hybrid Graph**: STDP drives weights—e.g., `if Δt > 0: w_A→B += 0.05`—hints add directionality, emerge via `self_benefit`.
- **Check**: Local autonomy—FUM adapts hints and goals, no central control—scales sovereignly.

## Validation
- **Check**: Supports 700B+ equivalence—e.g., 32B neurons (~1.4TB) across 50+ nodes—ensures sovereignty via async, hybrid design.