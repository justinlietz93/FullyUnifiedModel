### F. Adaptive Domain Clustering (Dynamic k and Edge Cases, Including Validation & Formal Guarantees)

#### F.1. Purpose & Mechanism
*   Dynamically identify functional specializations (domains) emerging within the network by grouping neurons with similar activity profiles. This cluster-based representation serves as the state definition for the TD learning value function (Sec 2.C.3).
*   Periodically (e.g., every 1000 timesteps), run k-means clustering (`torch.kmeans` on MI100) on neuron firing rates (`rates = torch.sum(spike_history, dim=1) / 50`).
*   **Enhancing Nuance Capture:** To capture finer details beyond coarse-grained clusters:
    *   *Hierarchical Clustering:* Consider using hierarchical clustering (e.g., `AgglomerativeClustering` on MI100) to create sub-clusters within each main cluster. This allows for a more granular state representation (e.g., 10 sub-clusters per main cluster, potentially capturing ~98% variance vs. ~90% with standard k-means).
    *   *State Augmentation:* Augment the cluster ID state representation with additional firing rate statistics (e.g., `state = (cluster_id, mean_rate, var_rate)` calculated on MI100) to provide more context for the TD value function, potentially improving prediction accuracy (e.g., ~5% improvement expected).

#### F.2. Determining Number of Clusters (k)
*   **Dynamic Selection using Silhouette Score:**
    *   **Method:** Test `k` in range `[k_min, max_k]`.
        *   `k_min = num_domains` (e.g., 8). Ensures minimum granularity reflecting known task domains.
        *   `max_k = min(num_neurons // 50, num_domains * 2)` (e.g., 16 for 1k neurons). Limits complexity.
    *   **Algorithm:** For each `k`, run `torch.kmeans`, compute silhouette score (`(b-a)/max(a,b)`). Choose `k` with highest score (`best_k = argmax(scores)`).
    *   **Adjustment:** Final `k = max(best_k, k_min)`. If silhouette selects `k < k_min`, override with `k_min`.
    *   **Implementation:** Execute on MI100 GPU.

#### F.3. Cluster Assignment & Reward Attribution (Domain Identification)
*   **Assignment:** Assign neurons to clusters based on similarity to centroids (hard assignment `cluster_id[i] = argmax(similarity)`, soft probabilities `probs[i] = softmax(similarity)`).
*   **Reward Attribution:**
    *   Map current input to a cluster based on induced firing pattern: `input_cluster = argmax(sum(probs * rates, dim=0))`.
    *   Attribute global `total_reward` to this cluster: `cluster_rewards[input_cluster] += total_reward`.
    *   Attribute reward to neurons weighted by probability: `neuron_rewards[i] += total_reward * probs[i, input_cluster]`.
*   **Average Reward:** Compute `avg_reward[c] = cluster_rewards[c] / num_inputs[c]` over 1000 steps (handle division by zero, see F.4). Used as growth trigger.
*   **Implementation:** Maintain `cluster_rewards`, `num_inputs`, `neuron_rewards` tensors on MI100.

#### F.4. Edge Case Handling (Small k, Empty Clusters)
*   **Small k:** If dynamic selection yields `k < k_min`, override with `k = k_min` (rerun kmeans if needed). Ensures minimum functional granularity.
*   **Empty Clusters:** If `num_inputs[c] = 0` (no inputs mapped to cluster `c` over 1000 steps), set `avg_reward[c] = 0` (neutral reward) to avoid division by zero. This triggers growth (`avg_reward < 0.5`) for the unused cluster, promoting exploration. Log metrics to SSD.

#### F.5. Adaptation (Including Novel Domains)
*   Clusters reflect the current functional organization and guide structural plasticity (growth targets). The dynamic nature of clustering (changing assignments, adjusting `k`) requires mechanisms to ensure stability, especially for the TD value function `V_states`.
*   **Handling Novel Domains/Inputs:** The clustering mechanism adapts when encountering novel inputs that don't fit well into existing clusters:
    *   **Novelty-Driven Bifurcation:** If an input has high novelty (`novelty > 0.9`) and low similarity to existing cluster centroids (`max_similarity < 0.5`), it can trigger an increase in `k` (`k += 1`) and a re-clustering, potentially forming a new cluster for the emerging domain.
    *   **Temporary Holding Cluster:** Alternatively, novel inputs can be assigned to a temporary "holding" cluster. Rewards associated with these inputs are isolated to this cluster. Once enough novel inputs accumulate (e.g., >10), a new permanent cluster is formed via bifurcation.
    *   **Preventing Misattribution:** If a novel input is initially misclassified into an existing cluster but yields low/negative reward (`total_reward < 0`), mechanisms can trigger reassignment to the holding cluster or prompt bifurcation, preventing the reinforcement of incorrect associations and delaying structural plasticity (growth) for clusters processing potentially misclassified novel inputs.
*   **Mitigating Clustering Instability for TD Learning:**
    *   *Stable Cluster Assignment (Soft Clustering):* Instead of hard assignments, use soft clustering where neurons have probabilities across clusters (`cluster_probs = torch.softmax(-distances, dim=1)` on MI100). Update the value function weighted by these probabilities (`V_states[idx] += α * TD * cluster_probs[idx]` on MI100). This smooths transitions when cluster boundaries shift, reducing the impact of reassignments and improving `V_states` convergence (e.g., 90% stability expected).
    *   *Stable Dynamic k Adjustment:* Adjust `k` incrementally (e.g., `k += 10` if `functional_coherence[c] < 0.8` on MI100) rather than making large jumps. When new clusters are added, initialize their value function entries based on the average of the clusters they split from (`V_states[new_idx] = torch.mean(V_states[old_indices])` on MI100). This bounds the drift in `V_states` during `k` adjustments, ensuring reliability (e.g., `|ΔV_states| < 0.1`, 95% reliability expected).
