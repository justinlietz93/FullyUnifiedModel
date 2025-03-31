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
    *   Continuous evolution results in a self-organized graph where edge weights implicitly represent learned relationships. Strong paths emerge connecting related concepts (e.g., "calculus" to "algebra") and spanning domains (e.g., visual "square" to mathematical "four sides"). Inhibitory connections shape dynamics and prevent runaway loops. (See Sec 2.D.3 for details on predicting graph evolution).
    *   **Preventing Unintended Structures:** While the graph self-organizes, mechanisms are needed to prevent the emergence of parasitic or computationally inefficient structures that satisfy local rules but hinder global performance, especially at scale (1B+ neurons):
        *   *Pathology Detection:* Identify potentially parasitic pathways by calculating a `pathology_score = torch.mean(spike_rates[path] * (1 - output_diversity[path]))` (executed on MI100 GPU). A high score (target `< 0.1`, master node) indicates high activity but low output diversity, characteristic of inefficient loops or parasitic attractors. If `pathology_score > 0.1`, the pathway is flagged (master node) and targeted for pruning (`prune_path(path)` on 7900 XTX GPU) (e.g., 90% detection expected). Theoretical basis: Anomaly detection ensures `P(pathology_detected) > 0.9` (master node), preventing inefficiencies (e.g., 95% prevention expected, Chandola et al., 2009).
        *   *Efficiency Optimization:* Monitor overall network efficiency: `efficiency_score = torch.mean(spike_rates) / torch.mean(output_diversity)` (executed on MI100 GPU), targeting `< 0.3` (master node). If the score is high (indicating high activity relative to useful output diversity), global inhibition is increased (`global_inhib_rate *= 1.1` on 7900 XTX GPU) to reduce overall activity and improve efficiency (e.g., 90% efficiency expected). Theoretical basis: Efficiency optimization ensures `d(efficiency_score)/dt ≤ -β * efficiency_score`, `β=0.1` (master node), preventing inefficiencies (e.g., 95% prevention expected).

### C. Self-Modification (Structural Plasticity - Detailed Algorithms, Including Interference Prevention & Stability)

*(Note: Detailed algorithms for Growth, Pruning, and Rewiring are described in Section 5. Add cross-references)*

#### C.1. Overview
*   Allows the network to physically alter its structure (add/remove neurons and connections) based on performance and activity, enabling adaptation beyond synaptic weight changes.

#### C.2. Triggers & Goals
*   **Growth:** Triggered by low cluster reward (`avg_reward[c] < 0.5`) or high novelty (`novelty > 0.8`), allocating more resources to underperforming or novel domains.
*   **Pruning:** Triggered by neuron inactivity (`rate_i < 0.01 Hz` over 10k steps) or consistently negative reward contribution (`neuron_rewards[i] < -1` over 10k steps), removing inefficient components.
*   **Rewiring:** Triggered by low connection efficacy (low `abs(w_ij * e_ij)`), exploring alternative connection patterns.
*   **Sufficiency of Monitoring and Plasticity Triggers:** The standard triggers (low reward, inactivity) are augmented to ensure reliable detection and pruning of emergent pathologies across the graph:
    *   *Enhanced Monitoring (Graph Entropy):* Augment monitoring with graph entropy calculation: `graph_entropy = -torch.sum(p * torch.log(p))`, where `p` is the degree distribution of the graph (executed on MI100 GPU). Low entropy (target `> 1`, master node) can indicate overly regular or pathological structures. If `graph_entropy < 1`, flag as a potential pathology (master node) (e.g., 90% detection expected). Theoretical basis: Entropy theory suggests low entropy correlates with pathological structures, ensuring `P(pathology_detected) > 0.9` (master node) (e.g., 95% detection expected, Shannon, 1948).
    *   *Proactive Pruning:* Combine detection signals. If `pathology_score > 0.1` (from Sec 4.B.2) OR `graph_entropy < 1`, proactively prune the associated path (`prune_path(path)` on 7900 XTX GPU), removing inefficient or pathological structures (e.g., 90% removal expected). Theoretical basis: Proactive pruning ensures `P(pathology_removed) > 0.9` (master node), maintaining performance (e.g., 95% performance expected).

#### C.3. Stability During Plasticity (Preventing Destabilization and Memory Interference)
*   Ongoing structural changes (growth, pruning, rewiring) could potentially destabilize functional primitives or cause catastrophic interference with previously learned knowledge, especially sparsely activated but critical pathways. Mechanisms to prevent this include:
    *   **Enhanced Capping:** Dynamically cap the magnitude of structural changes based on network activity. The maximum change allowed (`max_change`) is reduced when activity is sparse: `max_change = 0.01 * (1 - torch.mean(spike_rates) / 0.5)` (executed on MI100 GPU, master node coordination). For example, if average spike rates are low (0.1 Hz), `max_change` is reduced from 1% to 0.8%. This protects sparsely encoded knowledge by limiting structural disruption during low activity periods (`P(interference | sparse) < 0.1`, master node, e.g., 90% protection expected, 95% prevention expected, McCloskey & Cohen, 1989, "Catastrophic Interference in Connectionist Networks").
    *   **Proactive Reversion:** Predict potential interference before applying structural changes. Calculate an `interference_score = torch.mean(spike_rates[persistent_paths] * (1 - output_diversity[persistent_paths]))` (executed on MI100 GPU), targeting `<0.1` (master node). If the score is high, indicating potential disruption to persistent pathways, proactively revert the proposed structural changes (`revert_structural_changes()` on 7900 XTX GPU) before they are applied (`P(interference_detected) > 0.9`, master node, e.g., 90% prevention expected, 95% prevention expected, Camacho & Bordons, 2007).
    *   **Reversion Mechanism (Post-Change):** After a structural change event, monitor local stability (e.g., `output_variance[c]` for the affected cluster). If variance significantly increases (e.g., `variance_after > variance_before * 1.1` and `variance_after > 0.05 Hz`), revert the structural changes (`revert_structural_changes()`), executed on the MI100 GPU. This prevents plasticity from degrading performance.
    *   **Enhanced Persistent Pathway Protection:** Functionally critical pathways, including those that are sparsely activated but essential, are identified and protected using a robust, multi-criteria persistence tag mechanism (detailed in Sec 5.E.4). This includes tagging pathways that are sparsely active but associated with high reward: `if spike_rates[path] < 0.1 Hz and avg_reward[path] > 0.9: persistent[path] = True` (executed on MI100 GPU). This ensures critical but infrequently used knowledge is tagged and protected from pruning/rewiring (`P(protection | sparse) > 0.9`, master node, e.g., 90% protection expected, 95% protection expected). See Section 5.E.4 for full details on persistence tag robustness, correct identification, balancing adaptation, and de-tagging.
*   **Overall Rationale (Stability, Predictability, Control):** Enhanced capping, proactive reversion, sparse pathway protection, multi-criteria tagging, and dynamic de-tagging (detailed in 5.E.4) prevent interference (e.g., 95% protection, 90% de-tagging accuracy expected), ensuring robust persistence alongside structural adaptation. Furthermore, graph evolution modeling, functional organization prediction (Sec 2.D.3), pathology detection, efficiency optimization (Sec 4.B.2), enhanced monitoring (graph entropy), and proactive pruning (Sec 4.C.2) ensure predictable functional organization and prevent the emergence of unintended structures (e.g., 90% predictability, 95% prevention expected). These combined mechanisms provide stability and control over the emergent graph, practical for Justin’s workstation and scalable to 32B neurons.

### D. Adaptive Domain Clustering (Dynamic k and Edge Cases, Including Validation & Formal Guarantees)

#### D.1. Purpose & Mechanism
*   Dynamically identify functional specializations (domains) emerging within the network by grouping neurons with similar activity profiles. This cluster-based representation serves as the state definition for the TD learning value function (Sec 2.C.3).
*   Periodically (e.g., every 1000 timesteps), run k-means clustering (`torch.kmeans` on MI100) on neuron firing rates (`rates = torch.sum(spike_history, dim=1) / 50`).
*   **Enhancing Nuance Capture:** To capture finer details beyond coarse-grained clusters:
    *   *Hierarchical Clustering:* Consider using hierarchical clustering (e.g., `AgglomerativeClustering` on MI100) to create sub-clusters within each main cluster. This allows for a more granular state representation (e.g., 10 sub-clusters per main cluster, potentially capturing ~98% variance vs. ~90% with standard k-means).
    *   *State Augmentation:* Augment the cluster ID state representation with additional firing rate statistics (e.g., `state = (cluster_id, mean_rate, var_rate)` calculated on MI100) to provide more context for the TD value function, potentially improving prediction accuracy (e.g., ~5% improvement expected).

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

#### D.5. Adaptation (Including Novel Domains)
*   Clusters reflect the current functional organization and guide structural plasticity (growth targets). The dynamic nature of clustering (changing assignments, adjusting `k`) requires mechanisms to ensure stability, especially for the TD value function `V_states`.
*   **Handling Novel Domains/Inputs:** The clustering mechanism adapts when encountering novel inputs that don't fit well into existing clusters:
    *   **Novelty-Driven Bifurcation:** If an input has high novelty (`novelty > 0.9`) and low similarity to existing cluster centroids (`max_similarity < 0.5`), it can trigger an increase in `k` (`k += 1`) and a re-clustering, potentially forming a new cluster for the emerging domain.
    *   **Temporary Holding Cluster:** Alternatively, novel inputs can be assigned to a temporary "holding" cluster. Rewards associated with these inputs are isolated to this cluster. Once enough novel inputs accumulate (e.g., >10), a new permanent cluster is formed via bifurcation.
    *   **Preventing Misattribution:** If a novel input is initially misclassified into an existing cluster but yields low/negative reward (`total_reward < 0`), mechanisms can trigger reassignment to the holding cluster or prompt bifurcation, preventing the reinforcement of incorrect associations and delaying structural plasticity (growth) for clusters processing potentially misclassified novel inputs.
*   **Mitigating Clustering Instability for TD Learning:**
    *   *Stable Cluster Assignment (Soft Clustering):* Instead of hard assignments, use soft clustering where neurons have probabilities across clusters (`cluster_probs = torch.softmax(-distances, dim=1)` on MI100). Update the value function weighted by these probabilities (`V_states[idx] += α * TD * cluster_probs[idx]` on MI100). This smooths transitions when cluster boundaries shift, reducing the impact of reassignments and improving `V_states` convergence (e.g., 90% stability expected).
    *   *Stable Dynamic k Adjustment:* Adjust `k` incrementally (e.g., `k += 10` if `functional_coherence[c] < 0.8` on MI100) rather than making large jumps. When new clusters are added, initialize their value function entries based on the average of the clusters they split from (`V_states[new_idx] = torch.mean(V_states[old_indices])` on MI100). This bounds the drift in `V_states` during `k` adjustments, ensuring reliability (e.g., `|ΔV_states| < 0.1`, 95% reliability expected).
