### E. Tensor-Based Computation and Hybrid Interface

#### E.1 Hybrid Approach Rationale

##### E.1.i.
*   While SNNs excel at temporal processing, certain operations like analyzing graph properties, calculating complex SIE rewards, managing large state vectors (like eligibility traces `e_ij` or value function `V_states`), or performing clustering are often more efficiently handled using optimized tensor libraries. FUM adopts a hybrid approach, leveraging the strengths of both SNN simulation and tensor computation.

#### E.2 Frameworks & Hardware Roles (Development Context)

##### E.2.i.
*   Utilizes PyTorch for tensor manipulation.

##### E.2.ii.
*   **AMD Radeon 7900 XTX (24GB VRAM):** Primarily runs the custom ROCm HIP kernel (`neuron_kernel.hip`) for high-frequency, parallel LIF updates and spike generation. Also handles the final STDP weight updates (`w += ...`). Stores `V`, `spikes`, `spike_history`, `w`.

##### E.2.iii.
*   **AMD Instinct MI100 (32GB VRAM):** Primarily runs PyTorch tensor operations for tasks like STDP `Δw_ij` calculation, eligibility trace (`e_ij`) updates, SIE component calculations (novelty, habituation, complexity, impact, TD error), value function (`V_states`) updates, and k-means clustering. Stores `e_ij`, `V_states`, `recent_inputs`, `habituation_counter`, etc.

##### E.2.iv.
*   **CPU (AMD Threadripper PRO 5955WX):** Manages overall orchestration, data loading, potentially graph partitioning (METIS), parameter server logic (if scaling beyond node memory), and decoding outputs.

#### E.3 Interface: Data Flow & Synchronization

##### E.3.i.
*   **Frequency:** Interaction occurs primarily after each 50-timestep simulation window. Global operations like clustering or scaling occur less frequently (e.g., every 1000 timesteps).

##### E.3.ii.
*   **Data Flow (SNN -> Tensor):**
    1.  `spike_history` (uint8, ~6KB for 1k neurons) recorded on 7900 XTX by LIF kernel.
    2.  After 50 timesteps, transfer `spike_history` to MI100 (`spike_history_mi100 = spike_history.to('cuda:0')`).
    3.  MI100 computes `Δw_ij`, updates `e_ij`, calculates `rates`, computes SIE components (`novelty`, `habituation`, `complexity`, `impact`, `TD_error`), updates `V_states`.

##### E.3.iii.
*   **Data Flow (Tensor -> SNN):**
    1.  `total_reward` (float16 scalar) calculated on MI100.
    2.  `e_ij` (sparse float16, ~10KB) updated on MI100.
    3.  Transfer `total_reward` and `e_ij` to 7900 XTX (`total_reward.to('cuda:1')`, `e_ij.to('cuda:1')`).
    4.  7900 XTX applies final weight update to `w` using `total_reward` and `e_ij`.

##### E.3.iv.
*   **Synchronization:** Use `torch.cuda.synchronize()` or CUDA events to ensure data transfers are complete before dependent computations begin. Buffering mechanisms (e.g., `rate_buffer` on MI100, appending `rates.to('cuda:0')` every 50 steps) handle aggregation for less frequent operations like k-means (processed when buffer full, e.g., 1000 steps). Timing mismatches are managed by the fixed interaction frequency (every 50 timesteps).
