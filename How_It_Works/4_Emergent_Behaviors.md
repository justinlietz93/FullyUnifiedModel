## 4. Emergent Behaviors and Self-Organization

### A. Emergent Energy Landscape

1.  **Concept & Novelty:**
    *   FUM aims for network stability (analogous to a low-energy state) to **emerge naturally** from the interaction of local learning rules (STDP for excitatory/inhibitory synapses, intrinsic plasticity) and global feedback (SIE), rather than being imposed by a predefined mathematical energy function (like Hopfield networks). **Why is this novel/useful?** It allows the network to find its own stable configurations best suited to the data and tasks, potentially leading to more flexible and robust solutions.
2.  **Mechanism:**
    *   STDP reinforces consistent, reliable pathways. Inhibitory STDP and connections actively balance excitation. Intrinsic plasticity adapts neuron excitability. Synaptic scaling normalizes inputs. SIE feedback guides this process towards rewarded, stable states. The network effectively "settles" into configurations where rewarded patterns are produced with minimal extraneous activity (low variance).
3.  **Stability Metric:**
    *   Firing rate variance (e.g., standard deviation < 0.05 Hz across relevant neuron populations over ~1000 timesteps) is used as a practical, measurable proxy for this emergent stability. High variance indicates inefficient processing and can trigger corrective actions like structural plasticity or SIE penalty (via the 'impact' metric).

### B. Knowledge Graph Evolution (Detailed)

1.  **Process:**
    *   The graph's structure *is* the pattern of learned synaptic weights `w_ij`. It starts sparsely connected with a distance bias (Phase 1). As the network processes input (Phase 2 & 3) and receives SIE feedback, STDP strengthens connections between neurons firing with appropriate timing (`Δt`) for rewarded computations (`total_reward > 0`). Connections irrelevant or detrimental (`total_reward < 0`) are weakened or pruned.
2.  **Outcome:**
    *   Continuous evolution results in a self-organized graph where edge weights implicitly represent learned relationships. Strong paths emerge connecting related concepts (e.g., "calculus" to "algebra") and spanning domains (e.g., visual "square" to mathematical "four sides"). Inhibitory connections shape dynamics and prevent runaway loops.

### C. Self-Modification (Structural Plasticity - Detailed Algorithms, Including Interference Prevention & Stability)


#### D.1. Purpose & Mechanism
*   Dynamically identify functional specializations (domains) emerging within the network by grouping neurons with similar activity profiles.
*   Periodically (e.g., every 1000 timesteps), run k-means clustering (`torch.kmeans` on MI100) on neuron firing rates (`rates = torch.sum(spike_history, dim=1) / 50`).

#### D.2. Determining Number of Clusters (k)
*   **Dynamic Selection using Silhouette Score:**
    *   **Method:** Test `k` in range `[k_min, max_k]`.
        *   `k_min = num_domains` (e.g., 8). Ensures minimum granularity reflecting known task domains.
        *   `max_k = min(num_neurons // 50, num_domains * 2)` (e.g., 16 for 1k neurons). Limits complexity.
    *   **Algorithm:** For each `k`, run `torch.kmeans`, compute silhouette score (`(b-a)/max(a,b)`). Choose `k` with highest score (`best_k = argmax(scores)`).
    *   **Adjustment:** Final `k = max(best_k, k_min)`. If silhouette selects `k < k_min`, override with `k_min`.
    *   **Implementation:** Execute on MI100 GPU.

#### D.3. Cluster Assignment & Reward Attribution (Domain Identification)
*   **Assignment:** Assign neurons to clusters based on similarity to centroids (hard assignment `cluster_id[i] = argmax(similarity)`, soft probabilities `probs[i] = softmax(similarity)`).
*   **Reward Attribution:**
    *   Map current input to a cluster based on induced firing pattern: `input_cluster = argmax(sum(probs * rates, dim=0))`.
    *   Attribute global `total_reward` to this cluster: `cluster_rewards[input_cluster] += total_reward`.
    *   Attribute reward to neurons weighted by probability: `neuron_rewards[i] += total_reward * probs[i, input_cluster]`.
*   **Average Reward:** Compute `avg_reward[c] = cluster_rewards[c] / num_inputs[c]` over 1000 steps (handle division by zero, see D.4). Used as growth trigger.
*   **Implementation:** Maintain `cluster_rewards`, `num_inputs`, `neuron_rewards` tensors on MI100.

#### D.4. Edge Case Handling (Small k, Empty Clusters)
*   **Small k:** If dynamic selection yields `k < k_min`, override with `k = k_min` (rerun kmeans if needed). Ensures minimum functional granularity.
*   **Empty Clusters:** If `num_inputs[c] = 0` (no inputs mapped to cluster `c` over 1000 steps), set `avg_reward[c] = 0` (neutral reward) to avoid division by zero. This triggers growth (`avg_reward < 0.5`) for the unused cluster, promoting exploration. Log metrics to SSD.

#### D.5. Adaptation
*   Clusters reflect the current functional organization and guide structural plasticity (growth targets).
*   **Handling Novel Domains/Inputs:** The clustering mechanism adapts when encountering novel inputs that don't fit well into existing clusters:
    *   **Novelty-Driven Bifurcation:** If an input has high novelty (`novelty > 0.9`) and low similarity to existing cluster centroids (`max_similarity < 0.5`), it can trigger an increase in `k` (`k += 1`) and a re-clustering, potentially forming a new cluster for the emerging domain.
    *   **Temporary Holding Cluster:** Alternatively, novel inputs can be assigned to a temporary "holding" cluster. Rewards associated with these inputs are isolated to this cluster. Once enough novel inputs accumulate (e.g., >10), a new permanent cluster is formed via bifurcation.
    *   **Preventing Misattribution:** If a novel input is initially misclassified into an existing cluster but yields low/negative reward (`total_reward < 0`), mechanisms can trigger reassignment to the holding cluster or prompt bifurcation, preventing the reinforcement of incorrect associations and delaying structural plasticity (growth) for clusters processing potentially misclassified novel inputs.