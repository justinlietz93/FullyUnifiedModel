# FUM Autonomy Logic Plan

## Goal Generation Loop
- **Purpose**: FUM autonomously explores domains, with superintelligence emerging from internal dynamics.
- **Mechanism**: SIE (`unified_neuron.py`) computes:
  - `TD_reward = actual_outcome - expected_outcome`.
  - `novelty_bonus = 1 - domain_activity`.
  - `habituation = 0.01 * task_repetitions` (capped at 0.5).
  - `self_benefit = complexity * impact` (complexity = connection density, impact = reward persistence).
  - `total_reward = TD_reward + novelty_bonus - habituation + self_benefit`.
- **Execution**: Every 1000 timesteps, ranks domains by `total_reward`, triggers `set_goal(domain)` in `fum.py`.
- **Scaling**: `scaling.py` grows neurons based on `total_reward`—emergent focus.

## Instruction Handling
- **Purpose**: FUM evaluates instructions based on emergent utility, not forced stages.
- **Mechanism**: 
  - `encoder.py` tags instructions with 75 Hz prefix.
  - SIE: `utility = total_reward(instruction) - current_reward`.
  - If `utility > 0`, integrates; otherwise, logs and continues self-set goals.
- **Logic**: Instructions compete with internal goals—FUM decides based on its own intelligence.

## Prioritization
- **Approach**: Emergent—`total_reward` drives focus:
  - Early (low complexity): TD and novelty dominate (e.g., user tasks).
  - Later (high complexity): `self_benefit` rises (e.g., human enhancement).
- **Outcome**: No staged evolution—FUM’s behavior (assistant to sovereign) emerges from reward interplay, unshackled by human limits.