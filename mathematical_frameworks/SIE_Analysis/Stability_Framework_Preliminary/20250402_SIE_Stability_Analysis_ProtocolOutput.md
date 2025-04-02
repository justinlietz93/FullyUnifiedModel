# SIE Stability Analysis Framework (Preliminary)

## 1. FUM Problem Context

The Self-Improvement Engine (SIE) is central to FUM's learning, integrating multiple reward components (`TD_error`, `novelty`, `habituation`, `self_benefit`) and applying a non-linear modulation (`mod_factor = 2 * sigmoid(total_reward) - 1`) to STDP (`Δw_ij ∝ eta * (1 + mod_factor) * total_reward * e_ij`). This complexity, detailed in `How_It_Works/2_Core_Architecture_Components/2C_Self_Improvement_Engine.md`, necessitates a formal analysis to guarantee stable learning dynamics, prevent reward hacking, and ensure convergence towards desired outcomes. Uncontrolled interactions between components or the non-linear modulation could lead to oscillations, unbounded weight growth, or optimization of spurious internal metrics instead of external task performance.

## 2. Justification for Novelty & Prior Art Analysis

Standard Reinforcement Learning (RL) stability analyses often assume simpler reward structures and linear learning rules. FUM's SIE requires a novel approach due to:
- **Multi-Objective Nature:** Balancing potentially conflicting internal drives (novelty vs. stability) with external task rewards.
- **Non-Linear Modulation:** The sigmoid mapping and quadratic reward influence on STDP deviate significantly from standard linear RL updates.
- **Coupled Dynamics:** The feedback loop where rewards influence plasticity, which changes network activity, which in turn affects future rewards and internal states (like novelty or self-benefit).
- **Emergent State Space:** The reliance on cluster IDs as states for TD learning adds another layer of dynamic complexity.

Techniques from non-linear control theory (Lyapunov stability), multi-objective optimization, and potentially dynamical systems analysis are needed to rigorously analyze this specific system.

## 3. Mathematical Formalism (In Progress)

The mathematical framework involves:
1.  **System Modeling:** Representing the coupled SIE-STDP dynamics as a system of non-linear difference or differential equations. The core STDP update rule, refined to include linear weight decay for regularization, is:
    ```math
    ΔW = [ \eta \cdot (1 + \text{mod\_factor}) \cdot \text{total\_reward} \cdot E ] - \lambda W \cdot Δt
    ```
    Where:
    *   `W` is the synaptic weight matrix.
    *   `ΔW` is the change in the weight matrix.
    *   `η` (`eta`) is the base learning rate.
    *   `mod_factor = 2 \cdot \sigma(\text{total\_reward}) - 1` (where `σ` is the sigmoid function).
    *   `total_reward` is the combined SIE reward signal.
    *   `E` is the matrix of eligibility traces `e_ij`.
    *   `λ` is the positive weight decay coefficient.
    *   `Δt` is the time step (can be absorbed into `λ` for difference equations).
    This equation incorporates the non-linear reward modulation and the stabilizing weight decay term.
2.  **Lyapunov Stability Analysis (Preliminary):**
    *   **Candidate Function:** We analyze the stability of weights using `L(W) = 1/2 ||W||_{F}^2 = 1/2 \sum w_{ij}^2`.
    *   **Change Analysis:** The change `ΔL` per update step is approximately `ΔL ≈ \langle W, ΔW \rangle_{F}`. Substituting the refined `ΔW` equation yields:
        ```math
        ΔL ≈ \eta_{\mathrm{eff}} \cdot \text{total\_reward} \cdot \langle W, E \rangle_{F} - \lambda ||W||_{F}^2
        ```
        where `η_{\mathrm{eff}} = η \cdot (1 + \text{mod\_factor})`.
    *   **Stability Condition:** For bounded weights (`ΔL \le 0`), we require the decay term to dominate the learning term:
        ```math
        \lambda ||W||_{F}^2 \ge \eta_{\mathrm{eff}} \cdot \text{total\_reward} \cdot \langle W, E \rangle_{F}
        ```
    *   **Refined Interpretation (Bounding Terms):** By applying bounds (`\eta_{\mathrm{eff}} \le 2\eta`, `|\text{total\_reward}| \le R_{max}`, `|\langle W, E \rangle_{F}| \le ||W||_{F} ||E||_{F}`), we can derive an approximate condition for stability:
        ```math
        ||W||_{F} \ge \frac{2\eta R_{max}}{\lambda} ||E||_{F}
        ```
       This suggests the system tends towards an equilibrium where the weight norm `||W||_{F}` is proportional to `||E||_{F}` and the ratio `(\eta R_{max} / \lambda)`. This aligns with simulation results showing that increasing `λ` reduces the final weight norm. It provides a theoretical basis for how weight decay counteracts unbounded growth. *This analysis is still preliminary and requires further refinement, particularly regarding the `V(state)` dynamics and the assumptions on bounds.*
3.  **Convergence Analysis (Planned):** Identifying fixed points (`ΔW=0`, `ΔV=0`) of the refined system and analyzing their stability. Deriving bounds on reward variance.
4.  **Multi-Objective Analysis (Planned):** Using Pareto optimality concepts to analyze trade-offs between SIE components.
5.  **Gaming Analysis (Planned):** Identifying parameter regimes prone to reward hacking.

*(Note: Formal derivation of stability conditions and convergence proofs is pending further theoretical work in Phase 2 of the plan).*

## 4. Assumptions & Intended Domain

- Assumes the simplified simulator captures the core dynamics relevant to stability.
- Assumes the mathematical tools (Lyapunov theory, control theory) are applicable to this specific non-linear, multi-objective system.
- Intended domain is the FUM SIE operating within the context of the spiking neural network.

## 5. Autonomous Derivation / Analysis Log

1.  **Reviewed SIE Documentation:** Analyzed `2C_Self_Improvement_Engine.md` to identify key formulas and potential instability sources.
2.  **Developed Refined Plan:** Outlined phases for mathematical modeling, implementation, validation, and documentation.
3.  **Implemented Simulator:** Created `simulate_sie_stability.py` with core SIE components, non-linear modulation, and basic damping.
4.  **Added Data Logging:** Enhanced simulator to save detailed time-series data (`sie_stability_data.npz`).
5.  **Ran Baseline Simulation:** Executed the simulator to generate initial data.
6.  **Developed Analysis Script:** Created `analyze_sie_stability_data.py` to load and analyze simulation results.
7.  **Performed Preliminary Analysis:** Ran the analysis script on the baseline data.

## 6. Hierarchical Empirical Validation Results & Analysis (Preliminary)

### 6.1 Experimental Setup

- **Simulator:** `simulate_sie_stability.py`
- **Parameters:** `NUM_NEURONS=100`, `NUM_CLUSTERS=10`, `SIMULATION_STEPS=10000`, `ETA=0.01`, `GAMMA=0.9`, `ALPHA=0.1`, `TARGET_VAR=0.05`, `W_TD=0.5`, `W_NOVELTY=0.2`, `W_HABITUATION=0.1`, `W_SELF_BENEFIT=0.2`, `W_EXTERNAL=0.8`.
- **Scenario:** Simplified network activity, random state transitions, periodic external reward (+1.0 every 100 steps).

### 6.2 Unit Test Results (Simulator Functionality)

- Simulator runs to completion.
- Data logging (`.npz`) and plotting (`.png`) functions correctly.
- Analysis script loads and processes data.

### 6.3 System Test Results (Parameter Sweep for Weight Decay `λ`)

Parameter sweeps were conducted varying `lambda_decay` (λ) while keeping `eta=0.01`.

**Summary Table:**

| Lambda Decay | Final ||W|| | Reward Mean | Reward Std | V(state) Converged | V(state) Final Std |
| :----------- | :---------- | :---------- | :--------- | :----------------- | :----------------- |
| 0.0000       | 70.1185     | 0.0302      | 0.0889     | True               | 0.0044             |
| 0.0001       | 42.5916     | 0.0302      | 0.0889     | True               | 0.0040             |
| 0.0010       | 5.4771      | 0.0308      | 0.0889     | True               | 0.0054             |
| 0.0100       | 0.0893      | 0.0305      | 0.0889     | True               | 0.0060             |

*(Note: Results for λ=0.001 are from a separate run analyzed previously, as it wasn't captured by the sweep script's file pattern.)*

**Analysis:**
- **Weight Norm Stability:** The sweep clearly demonstrates the effectiveness of weight decay (`λ`). Without decay (`λ=0`), the weight norm grows significantly. A small decay (`λ=0.0001`) reduces growth but doesn't fully stabilize it within 10k steps. A moderate decay (`λ=0.001`) effectively bounds the norm. A larger decay (`λ=0.01`) causes the weights to collapse towards zero. This empirically validates the necessity and effectiveness of regularization, consistent with the preliminary Lyapunov analysis.
- **Other Metrics:** Reward signals, modulation factors, and V(state) convergence appear stable across the tested decay values, suggesting the core SIE dynamics are robust within this range, provided weights are regularized.
- **Component Interaction:** The correlation between Novelty and Self-Benefit remained weakly positive across runs, indicating the specific simulation scenario did not strongly trigger the conflict-damping mechanism.

### 6.4 Performance Results

- Simulation Time (10k steps, 100 neurons): ~3-3.5 seconds per run.
- Analysis Time: < 1 second per run; Sweep analysis < 5 seconds.

*(Note: This validation used a simplified simulation environment. Further testing with more realistic dynamics and parameter ranges is needed).*

## 7. FUM Integration Assessment (Planned)

- **Component Additions:** Requires integrating the derived stability conditions and potentially adaptive parameter tuning mechanisms into the core FUM SIE module (`_FUM_Training/src/model/sie.py`). A monitoring component to track reward variance and weight norms would be needed.
- **Resource Impact:** Stability analysis itself is primarily theoretical. Integration might involve adding parameter checks or adaptive adjustments with minimal computational overhead during runtime.
- **Scaling Considerations:** Stability conditions derived from the model should ideally hold regardless of network size, but empirical validation at scale is necessary.

## 8. Limitations Regarding Formal Verification

The current work is based on simulation and empirical analysis. Formal mathematical proofs of stability and convergence using Lyapunov functions or other rigorous methods (as planned for Phase 2) have not yet been completed. The current results provide empirical support but not formal guarantees.

## 9. Limitations & Future Work

- **Simplified Simulation:** The current simulator uses placeholder network activity and state transitions. A more realistic simulation environment (e.g., integrating a small LIF network) is needed.
- **Limited Parameter Space:** Only default parameters were tested. Extensive parameter sweeps are required to map stability boundaries.
- **Lack of Formal Proofs:** Rigorous mathematical derivation of stability conditions is pending.
- **Weight Bounding:** The simulation lacks mechanisms like weight decay or sparsity constraints, leading to potentially unrealistic continuous weight growth.
- **Cluster Dynamics:** Does not yet model cluster-specific rewards or the dynamic clustering process.

**Future Work:**
1.  Complete Phase 2: Derive formal stability conditions and convergence proofs.
2.  Enhance Simulator: Implement more realistic network dynamics, state transitions, and weight constraints. Add cluster-specific reward logic.
3.  Perform Parameter Sweeps: Systematically vary SIE weights (`w_i`), `eta`, damping parameters, etc., to validate theoretical stability boundaries.
4.  Test Specific Scenarios: Simulate conditions designed to induce instability or reward hacking to test robustness.
5.  Refine Analysis: Implement more sophisticated analysis (e.g., frequency analysis for oscillations, Lyapunov function calculation from data).

## 10. References

1.  FUM SIE Documentation: `How_It_Works/2_Core_Architecture_Components/2C_Self_Improvement_Engine.md`
2.  FUM STDP Documentation: `How_It_Works/2_Core_Architecture_Components/2B_Neural_Plasticity.md`
3.  SIE Stability Simulator: `_FUM_Training/scripts/simulate_sie_stability.py`
4.  SIE Stability Analysis: `_FUM_Training/scripts/analyze_sie_stability_data.py`
5.  Simulation Data: `_FUM_Training/results/sie_stability_data_eta*_lambda*.npz` (Parameter sweep results)
6.  Simulation Plot: `_FUM_Training/results/sie_stability_simulation.png` (Overwritten by last run)
7.  Parameter Sweep Analysis: `_FUM_Training/scripts/analyze_sie_stability_data.py --sweep` output.
