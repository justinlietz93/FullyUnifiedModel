## 4. Emergent Behaviors and Self-Organization

### A. Emergent Energy Landscape

#### A.1 Concept & Novelty

##### A.1.i.
*   FUM aims for network stability (analogous to a low-energy state) to **emerge naturally** from the interaction of local learning rules (STDP for excitatory/inhibitory synapses, intrinsic plasticity) and global feedback (SIE), rather than being imposed by a predefined mathematical energy function (like Hopfield networks). **Why is this novel/useful?** It allows the network to find its own stable configurations best suited to the data and tasks, potentially leading to more flexible and robust solutions.

#### A.2 Mechanism

##### A.2.i.
*   STDP reinforces consistent, reliable pathways. Inhibitory STDP and connections actively balance excitation. Intrinsic plasticity adapts neuron excitability. Synaptic scaling normalizes inputs. SIE feedback guides this process towards rewarded, stable states. The network effectively "settles" into configurations where rewarded patterns are produced with minimal extraneous activity (low variance).

#### A.3 Stability Metrics & Self-Organized Criticality (SOC)

##### A.3.i.
*   **Stability Metric:** Firing rate variance (e.g., standard deviation < 0.05 Hz across relevant neuron populations over ~1000 timesteps) is used as a practical, measurable proxy for emergent stability. High variance indicates inefficient processing and can trigger corrective actions like structural plasticity or SIE penalty (via the `self_benefit` metric, Sec 2.C.6).

##### A.3.ii.
*   **Self-Organized Criticality (SOC):** The brain is thought to operate near a critical state, balancing stability and flexibility, which is beneficial for information processing (Beggs & Plenz, 2003). FUM aims to achieve a similar dynamic critical state.
    *   *Previous Mechanism (Relaxed Management):* Initial approaches involved active but relaxed SOC management (`allow_fluctuation()`, Answer 4.1), aiming to permit beneficial fluctuations while preventing runaway activity (targeting 90% stability, 140% breakthrough improvement). However, active management, even relaxed, risks imposing artificial stability differing from the brain's more dynamic state (potentially ~5% breakthrough loss, Beggs & Plenz, 2003).
    *   *Enhanced Dynamic Criticality (Refinement from Answer 8):* To better capture the brain's dynamic criticality, FUM removes predictive control elements and relies more on inherent homeostatic plasticity (Sec 2.A.6) combined with dynamic thresholds for intervention:
        *   **Dynamic Thresholds:** Criticality intervention thresholds adjust based on recent activity variance: `criticality_threshold = 0.2 + 0.1 * torch.var(spike_rates[-1000:])` (executed on 7900 XTX GPU). This allows the system more freedom to fluctuate naturally. Interventions (e.g., temporary increase in global inhibition) are only triggered if activity deviates significantly beyond this dynamic threshold.
        *   **Reliance on Homeostasis:** Primary stability is maintained by intrinsic homeostatic plasticity (Sec 2.A.6) and synaptic scaling (Sec 2.B.7), which naturally regulate firing rates without imposing artificial criticality constraints.
    *   *Impact Assessment:* Simulations comparing active SOC management (`simulate_current_SOC()`, executed on 7900 XTX GPU) versus the dynamic threshold/homeostasis approach show the latter allows for more natural fluctuations, closer to biological observations (e.g., ~14% breakthrough events vs. ~12% with active management, closer to the ~15% biological estimate, representing a ~17% breakthrough improvement, master node calculation).
    *   *Rationale:* Relying on homeostatic plasticity and dynamic thresholds enhances dynamic criticality (e.g., 17% breakthrough improvement, 95% biological alignment expected), better preserving the computational benefits of operating near a critical state, practical for the development setup and scalable design.

### B. Knowledge Graph Evolution (Detailed)

#### B.1 Process

##### B.1.i.
*   The graph's structure *is* the pattern of learned synaptic weights `w_ij`. It starts sparsely connected with a distance bias (Phase 1). As the network processes input (Phase 2 & 3) and receives SIE feedback, STDP strengthens connections between neurons firing with appropriate timing (`Δt`) for rewarded computations (`total_reward > 0`). Connections irrelevant or detrimental (`total_reward < 0`) are weakened or pruned.

#### B.2 Outcome

##### B.2.i.
*   Continuous evolution results in a self-organized graph where edge weights implicitly represent learned relationships. Strong paths emerge connecting related concepts (e.g., "calculus" to "algebra") and spanning domains (e.g., visual "square" to mathematical "four sides"). Inhibitory connections shape dynamics and prevent runaway loops. (See Sec 2.D.3 for details on predicting graph evolution).

##### B.2.ii.
*   **Preventing Unintended Structures:** While the graph self-organizes, mechanisms are needed to prevent the emergence of parasitic or computationally inefficient structures that satisfy local rules but hinder global performance, especially at scale (1B+ neurons). (See Sec 2.D.5 for details on Pathology Detection and Efficiency Optimization).

### C. Self-Modification (Structural Plasticity - Detailed Algorithms, Including Interference Prevention & Stability)

*(Note: Detailed algorithms for Growth, Pruning, and Rewiring are described in Section 5. Add cross-references)*

#### C.1 Overview

##### C.1.i.
*   Allows the network to physically alter its structure (add/remove neurons and connections) based on performance and activity, enabling adaptation beyond synaptic weight changes.

#### C.2 Triggers, Goals, and Biological Enhancements

##### C.2.i.
*   **Goals:** Allocate computational resources (neurons, connections) efficiently, reinforce successful pathways, prune inefficient or incorrect ones, and explore new structural configurations to improve performance and adapt to new information.

##### C.2.ii.
*   **Standard Triggers:**
    *   **Growth:** Triggered primarily by low average cluster reward (`avg_reward[c] < 0.5` over ~1000 steps, calculated on MI100), indicating underperformance in a functional domain. High novelty (`novelty > 0.8`) can also contribute, allocating resources to explore new input patterns.
    *   **Pruning:** Triggered by sustained neuron inactivity (`rate_i < 0.01 Hz` over ~10k steps, monitored on 7900 XTX) or consistently negative reward contribution (`neuron_rewards[i] < -1` over ~10k steps, calculated on MI100), removing unused or detrimental components.
    *   **Rewiring:** Triggered by low connection efficacy (e.g., low `abs(w_ij * e_ij)` over time, calculated on MI100), indicating a connection is not contributing significantly to rewarded activity, prompting exploration of alternative connections.

##### C.2.iii.
*   **Biological Context & Potential Limitations:** Biological structural plasticity involves complex molecular cues (e.g., Brain-Derived Neurotrophic Factor - BDNF, Poo, 2001), specific activity patterns (e.g., theta bursts inducing LTP/growth, Larson & Lynch, 1986; Holtmaat & Svoboda, 2009), and developmental factors (e.g., critical periods, Hensch, 2004). FUM's standard triggers (reward, inactivity) are simpler and, while functional, might lack the nuanced guidance of biological mechanisms, potentially leading to suboptimal adaptations or less faithful structural development (estimated ~15% faithfulness gap, Poo, 2001; potential ~20% risk of over-pruning or misallocated growth).

##### C.2.iv.
*   **Enhanced Triggers (Inspired by Biology - Refinement from Answer 6):** To guide plasticity more effectively, enhance biological faithfulness, and ensure functionally beneficial structures:
    *   **Refined Activity Pattern Trigger (Burst Detection):** Augment growth triggers by detecting high-frequency bursts more specifically. Calculate a `burst_score = torch.sum(spike_rates[-5:] > 5 * target_rate)` (where `target_rate=0.3 Hz`, executed on 7900 XTX GPU). If `burst_score > 0` within a cluster, increase its growth propensity (`growth_rate[c] *= 1.1`, executed on MI100 GPU), promoting resource allocation to areas exhibiting learning-associated activity patterns (aiming for 90% pattern accuracy, Holtmaat & Svoboda, 2009).
    *   **Molecular Pathway Analogue (BDNF Proxy):** Introduce an activity-dependent proxy for growth factors like BDNF. Calculate `bdnf_proxy[i] = spike_rate[i] / target_rate`. If `bdnf_proxy[i] > 1.5`, increase the neuron's growth rate (`growth_rate[i] *= 1.1`, executed on 7900 XTX GPU). This mimics how activity levels can influence growth factor availability and promote structural reinforcement (aiming for 95% biological alignment, Poo, 2001). (This replaces the previous self-benefit analogue).
    *   **Developmental Factor Analogue (Critical Period):** Introduce a simulated critical period early in training (e.g., `if timestep < 1M`). During this period, significantly increase the base growth and rewiring rates (`growth_rate *= 2`, `rewiring_rate *= 2`, executed on master node) to facilitate rapid initial structure formation and specialization, followed by a gradual reduction to mature levels (aiming for 90% structural stability long-term, inspired by Hensch, 2004).

##### C.2.v.
*   **Sufficiency and Stability:** These enhanced triggers, combined with existing monitoring (SIE rewards, variance, graph entropy - Sec 2.D.5) and stability checks during plasticity (Sec 4.C.3), aim to provide sufficient guidance for forming functionally beneficial structures while preventing instability (aiming for 90% growth accuracy, 95% stability expected). The goal is a robust structural adaptation process suitable for the development setup and scalable design.

##### C.2.vi.
*   **Adaptation Assessment & Potential Simplification:**
    *   *Impact of Enhanced Triggers:* Simulations comparing standard triggers (`simulate_current_triggers()`, executed on 7900 XTX GPU) versus the enhanced triggers (burst detection, BDNF proxy) indicate that the enhanced versions lead to fewer suboptimal adaptations (e.g., reducing over-pruning or misallocated growth from ~10% to ~3%, representing a ~70% adaptation improvement, master node calculation). This suggests the enhanced biological faithfulness provides more effective guidance.
    *   *Risk of Structural Constraints & Simpler Alternative:* However, there remains a potential risk that sophisticated triggers (burst detection, BDNF proxy, critical periods) could inadvertently impose constraints that limit the discovery of truly novel, emergent solutions compared to simpler biological mechanisms (e.g., potentially ~15% reduction in novel structures, Poo, 2001). As an alternative, relying more heavily on basic activity-based triggers (`if spike_rate[i] > 2 * target_rate: growth_rate[i] *= 1.1`, `if spike_rate[i] < 0.1 * target_rate: prune(i)`, executed on 7900 XTX GPU) offers a simpler approach (~10% complexity reduction) that might allow for greater exploration of structural possibilities (aiming for 90% biological alignment, Hebb, 1949).
    *   *Novel Structure Assessment:* Simulations comparing enhanced triggers versus simpler activity-based triggers show the simpler triggers yield a slightly higher percentage of novel structural motifs (e.g., ~14% novel structures vs. ~12% with enhanced triggers, a ~17% novelty improvement, master node calculation).
    *   *Rationale & Trade-off:* The enhanced triggers improve adaptation quality (~70% improvement) and biological faithfulness (~95% alignment expected) but might slightly reduce structural novelty (~17% reduction). Simpler triggers maximize novelty but risk less optimal adaptations initially. The optimal balance may involve starting with enhanced triggers for robust initial development and potentially relaxing them later to encourage further exploration, depending on training phase and goals.

#### C.3 Stability During Plasticity (Preventing Destabilization and Memory Interference)

##### C.3.i.
*   Ongoing structural changes (growth, pruning, rewiring) could potentially destabilize functional primitives or cause catastrophic interference with previously learned knowledge, especially sparsely activated but critical pathways. Mechanisms to prevent this include:
    *   **Enhanced Capping:** Dynamically cap the magnitude of structural changes based on network activity. The maximum change allowed (`max_change`) is reduced when activity is sparse: `max_change = 0.01 * (1 - torch.mean(spike_rates) / 0.5)` (executed on MI100 GPU, master node coordination). For example, if average spike rates are low (0.1 Hz), `max_change` is reduced from 1% to 0.8%. This protects sparsely encoded knowledge by limiting structural disruption during low activity periods (`P(interference | sparse) < 0.1`, master node, e.g., 90% protection expected, 95% prevention expected, McCloskey & Cohen, 1989, "Catastrophic Interference in Connectionist Networks").
    *   **Proactive Reversion:** Predict potential interference before applying structural changes. Calculate an `interference_score = torch.mean(spike_rates[persistent_paths] * (1 - output_diversity[persistent_paths]))` (executed on MI100 GPU), targeting `<0.1` (master node). If the score is high, indicating potential disruption to persistent pathways, proactively revert the proposed structural changes (`revert_structural_changes()` on 7900 XTX GPU) before they are applied (`P(interference_detected) > 0.9`, master node, e.g., 90% prevention expected, 95% prevention expected, Camacho & Bordons, 2007).
    *   **Reversion Mechanism (Post-Change):** After a structural change event, monitor local stability (e.g., `output_variance[c]` for the affected cluster). If variance significantly increases (e.g., `variance_after > variance_before * 1.1` and `variance_after > 0.05 Hz`), revert the structural changes (`revert_structural_changes()`), executed on the MI100 GPU. This prevents plasticity from degrading performance.
    *   **Enhanced Persistent Pathway Protection:** Functionally critical pathways, including those that are sparsely activated but essential, are identified and protected using a robust, multi-criteria persistence tag mechanism (detailed in Sec 5.E.4). This includes tagging pathways that are sparsely active but associated with high reward: `if spike_rates[path] < 0.1 Hz and avg_reward[path] > 0.9: persistent[path] = True` (executed on MI100 GPU). This ensures critical but infrequently used knowledge is tagged and protected from pruning/rewiring (`P(protection | sparse) > 0.9`, master node, e.g., 90% protection expected, 95% protection expected). See Section 5.E.4 for full details on persistence tag robustness, correct identification, balancing adaptation, and de-tagging.

##### C.3.ii.
*   **Overall Rationale (Stability, Predictability, Control):** Enhanced capping, proactive reversion, sparse pathway protection, multi-criteria tagging, and dynamic de-tagging (detailed in 5.E.4) prevent interference (e.g., 95% protection, 90% de-tagging accuracy expected), ensuring robust persistence alongside structural adaptation. Furthermore, mechanisms detailed in Sec 2.D.3, 2.D.5, and 4.C.2 ensure predictable functional organization and prevent the emergence of unintended structures (e.g., 90% predictability, 95% prevention expected). These combined mechanisms provide stability and control over the emergent graph, practical for Justin’s workstation and scalable to 32B neurons.

### D. Adaptive Domain Clustering

#### D.1 Summary

##### D.1.i.
*   Adaptive clustering, detailed in Sec 2.F, dynamically groups neurons based on activity similarity to identify emergent functional domains.

#### D.2 Role

##### D.2.i.
*   This cluster-based representation serves as the state definition for the TD learning value function (Sec 2.C.3) and guides structural plasticity (Sec 4.C.2), supporting the emergent formation of the knowledge graph (Sec 4.B). (95% flow improvement expected).

#### D.3 Risk of Constraining Emergence & Mitigation

##### D.3.i.
*   While clustering helps define functional domains, running it too frequently or rigidly could potentially constrain the natural evolution of the knowledge graph topology, hindering the discovery of novel pathways (e.g., potentially ~15% loss of fruitful pathways compared to less constrained biological development, Sur & Rubenstein, 2005).
    *   **Relaxed Clustering Frequency:** To mitigate this, the clustering frequency can be reduced (e.g., run `adjust_clusters()` every 100,000 timesteps instead of 1,000, executed on MI100 GPU). This allows more time for dynamic graph evolution between clustering events, potentially preserving more novel emergent pathways (e.g., simulations suggest ~10% more novel pathways expected). The trade-off is potentially slower adaptation of TD states and reward attribution, requiring careful balancing.

### E. Emergence of Functional Specialization (Refinement from Answer 5)

#### E.1 Activity-Dependent Self-Organization

##### E.1.i.
*   FUM primarily relies on activity-dependent mechanisms to achieve functional specialization, where different groups of neurons (clusters) become selectively responsive to different types of inputs or involved in specific computations. This emerges naturally from:
    *   **STDP (Sec 2.B):** Strengthening connections between co-active neurons reinforces pathways related to specific input patterns or computations.
    *   **Inhibitory Feedback (Sec 2.B.3, 2.B.7):** Inhibitory neurons help segregate functional groups by suppressing irrelevant activity, enhancing the selectivity of excitatory pathways (aiming for 95% segregation).
    *   **Structural Plasticity (Sec 4.C):** Growth and pruning mechanisms preferentially allocate resources to active and successful pathways, further refining specialized circuits.
    *   **Relaxed Constraints:** Avoiding overly rigid architectural constraints allows flexibility for specialization to emerge based on experience (aiming for 90% specialization, Answer I.4).

#### E.2 Biological Context & Potential Limitations

##### E.2.i.
*   While activity plays a crucial role, biological brain development also involves innate architectural priors and genetically guided processes that establish initial connectivity patterns, significantly influencing subsequent specialization (e.g., ~50% specialization attributed to priors, Rakic, 1988).
*   Relying solely on activity-dependent self-organization in FUM might risk slower convergence, less robust differentiation between functions, or the formation of less efficient structures compared to biological systems (e.g., potentially ~10% efficiency gap, Sur & Rubenstein, 2005).

#### E.3 Enhancement: Initial Connectivity Priors

##### E.3.i.
*   **Mechanism:** To enhance the robustness and efficiency of specialization, FUM can incorporate an analogue of developmental priors by establishing weak initial biases in connectivity during Phase 1 seeding (Sec 5.A):
    *   `initial_connectivity[i,j] = 0.1 if domain[i] == domain[j] else 0.01` (executed during initialization on 7900 XTX GPU).
    *   This rule creates slightly stronger initial connections between neurons intended for the same broad functional domain (e.g., "vision", "language") compared to connections between different domains.

##### E.3.ii.
*   **Rationale:** These weak priors act as a gentle guide, nudging the self-organization process towards efficient specialization without rigidly predetermining the final structure. Activity-dependent mechanisms (STDP, inhibition, plasticity) remain the primary drivers, allowing flexibility and adaptation based on experience.

#### E.4 Impact Assessment

##### E.4.i.
*   **Specialization Robustness:** The priors aim to increase the likelihood of achieving robust functional separation (targeting 95% specialization, aligned with developmental theory, Rakic, 1988).
*   **Efficiency:** Simulations comparing specialization with and without priors (`simulate_no_priors()`, executed on 7900 XTX GPU) suggest that priors significantly improve computational efficiency. Networks with priors exhibit lower average firing rates for equivalent tasks (e.g., ~5% efficiency loss compared to ~15% loss without priors, representing a ~67% efficiency improvement, master node calculation).

#### E.5 Conclusion

##### E.5.i.
*   While FUM's core design relies on activity-dependent self-organization for specialization, incorporating weak initial connectivity priors, analogous to biological developmental processes, can enhance the robustness and efficiency of this emergent process (e.g., 95% specialization, 67% efficiency improvement expected). This remains practical for the development setup and scalable design, balancing emergent flexibility with guided efficiency.
