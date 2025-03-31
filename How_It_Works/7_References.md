
# References

This list provides potential citations for the concepts, algorithms, and frameworks mentioned in the FUM document. Selections can be integrated into footnotes or a dedicated reference section.

## FUM System Design

* Lietz, J. (2025). *How the Fully Unified Model (FUM) Works*. [Unpublished technical specification / Design document]. (Details the specific FUM architecture and the following potentially novel contributions:
    * **Overall Integrated System:** The synergistic combination of SNNs, the Self-Improvement Engine, detailed structural plasticity, emergent knowledge graph, hybrid computation model, and phased training strategy as a unified system.
    * **Self-Improvement Engine (SIE):** The specific reward formulation `total_reward = TD_error + novelty - habituation + self_benefit`, including the `self_benefit = complexity * impact` calculation, and its direct, quadratically scaled modulation of STDP learning rates via eligibility traces.
    * **Integrated Structural Plasticity System:** The specific set of triggers (low cluster reward, high variance, low neuron activity), detailed algorithms (targeted growth, pruning with homeostatic compensation, co-activation-based rewiring), and defined operational limits (e.g., connection history, sparsity target, E/I balancing during rewiring).
    * **Adaptive Domain Clustering Integration:** Employing K-Means with dynamic `k` selection (via Silhouette Score within defined bounds) to define TD-Learning states and guide reward attribution and structural plasticity within the SNN framework, including specific edge case handling.
    * **Multi-Phase Training Strategy:** The explicit three-phase approach (Seed Sprinkling -> Tandem Complexity Scaling -> Continuous Self-Learning) tailored for minimal data dependency and autonomous operation.
    * **Emergent Knowledge Graph & Routing Mechanism:** Reliance on learned SNN connectivity dynamically shaped by STDP/SIE/plasticity for coordination and information routing, explicitly contrasting with predefined layers or coordinators.
    * **Emergent Energy Landscape Concept:** Proposing network stability as arising naturally from the interplay of local/global rules (measured via variance), rather than an explicitly defined energy function.
    * **Specific Hybrid Computation Model:** The defined heterogeneous GPU workload distribution (LIF kernel vs. PyTorch/SIE/Clustering) and associated data flow/synchronization strategy.
    * **Specific Temporal Encoding Schemes:** The described hierarchical methods for encoding structured data (e.g., code syntax trees, logical propositions) into temporal spike patterns.
    * **Minimal Data Philosophy:** The core design goal targeting high performance from minimal inputs (80-300) based on a defined balance between initialization and learning.)

## SNNs, LIF Model, and Neuron Dynamics

* Burkitt, A. N. (2006). A review of the leaky integrate-and-fire neuron model. *Biological Cybernetics*, *95*(1), 1-19.
* Destexhe, A., & Marder, E. (2004). Plasticity in single neuron and circuit computations. *Nature*, *431*(7010), 789-795.
* Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014). *Neuronal dynamics: From single neurons to networks and models of cognition*. Cambridge University Press.
* Lapicque, L. (1907). Recherches quantitatives sur l'excitation électrique des nerfs traitée comme une polarisation. *Journal de Physiologie et de Pathologie Générale*, *9*, 620-635.
* Maass, W. (1997). Networks of spiking neurons: the third generation of neural network models. *Neural Networks*, *10*(9), 1659-1671.
* Marder, E., & Goaillard, J. M. (2006). Variability, compensation, and homeostasis in neuron and network function. *Nature Reviews Neuroscience*, *7*(7), 563-574.
* Pfeiffer, M., & Pfeil, T. (2018). Deep learning with spiking neurons: opportunities and challenges. *Frontiers in Neuroscience*, *12*, 774. https://doi.org/10.3389/fnins.2018.00774
* Triesch, J. (2007). Synergies between intrinsic and synaptic plasticity mechanisms. *Neural Computation*, *19*(4), 885-909.

## Neural Plasticity & Stability Mechanisms

* Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. *Journal of Neuroscience*, *18*(24), 10464-10472.
* Chklovskii, D. B., Mel, B. W., & Svoboda, K. (2004). Cortical rewiring and information storage. *Nature*, *431*(7010), 782-788.
* Helias, M., Tetzlaff, T., & Diesmann, M. (2014). The correlation structure of local neuronal networks intrinsically results from recurrent connectivity. *PLoS Computational Biology*, *10*(1), e1003458. https://doi.org/10.1371/journal.pcbi.1003458
* Holtmaat, A., & Svoboda, K. (2009). Experience-dependent structural synaptic plasticity in the mammalian brain. *Nature Reviews Neuroscience*, *10*(9), 647-658.
* Markram, H., Lübke, J., Frotscher, M., & Sakmann, B. (1997). Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSPs. *Science*, *275*(5297), 213-215.
* Song, S., Miller, K. D., & Abbott, L. F. (2000). Competitive Hebbian learning through spike-timing-dependent synaptic plasticity. *Nature Neuroscience*, *3*(9), 919-926.
* Turrigiano, G. G., Leslie, K. R., Desai, N. S., Rutherford, L. C., & Nelson, S. B. (1998). Activity-dependent scaling of quantal amplitude in neocortical neurons. *Nature*, *391*(6670), 892-896.
* Vogels, T. P., Sprekeler, H., Zenke, F., Clopath, C., & Gerstner, W. (2011). Inhibitory plasticity balances excitation and inhibition in sensory pathways and memory networks. *Science*, *334*(6062), 1569-1573. https://doi.org/10.1126/science.1211095

## Reinforcement Learning & SIE Components

* Florian, R. V. (2007). Reinforcement learning through modulation of spike-timing-dependent synaptic plasticity. *Neural Computation*, *19*(6), 1468-1502.
* Frémaux, N., & Gerstner, W. (2016). Neuromodulated spike-timing-dependent plasticity, and theory of three-factor learning rules. *Frontiers in Neural Circuits*, *9*, 85. https://doi.org/10.3389/fncir.2015.00085
* Izhikevich, E. M. (2007). Solving the distal reward problem through linkage of STDP and dopamine signaling. *Cerebral Cortex*, *17*(10), 2443-2452. https://doi.org/10.1093/cercor/bhl147
* Marsland, S. (2014). *Machine learning: an algorithmic perspective*. CRC press. (For general concepts potentially underlying novelty/habituation metrics)
* Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. *Machine Learning*, *3*(1), 9-44.
* Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction*. MIT press.

## Emergence, Self-Organization, & Knowledge Graphs

* Bak, P., Tang, C., & Wiesenfeld, K. (1987). Self-organized criticality: An explanation of the 1/f noise. *Physical Review Letters*, *59*(4), 381-384.
* Hogan, A., Blomqvist, E., Cochez, M., d'Amato, C., Melo, G. D., Gutierrez, C., Kirrane, S., Gayo, J. E. L., Navigli, R., Neumaier, S., Ngomo, A. C. N., Polleres, A., Rashid, S. M., Rula, A., Schmelzeisen, L., Sequeda, J., Staab, S., & Zimmermann, A. (2021). Knowledge graphs. *ACM Computing Surveys (CSUR)*, *54*(4), 1-37. https://doi.org/10.1145/3447772
* Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. *Proceedings of the National Academy of Sciences*, *79*(8), 2554-2558.
* Mitchell, M. (2009). *Complexity: A guided tour*. Oxford University Press.

## Clustering & Graph Partitioning

* Karypis, G., & Kumar, V. (1998). A fast and high quality multilevel scheme for partitioning irregular graphs. *SIAM Journal on Scientific Computing*, *20*(1), 359-392.
* MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In *Proceedings of the fifth Berkeley symposium on mathematical statistics and probability* (Vol. 1, pp. 281-297). University of California Press.
* Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, *20*, 53-65.

## I/O Processing

* Brette, R. (2015). Philosophy of the spike: rate-based vs. spike-based theories of computation. *Frontiers in Systems Neuroscience*, *9*, 151. https://doi.org/10.3389/fnsys.2015.00151
* Thorpe, S., Delorme, A., & Van Rullen, R. (2001). Spike-based strategies for rapid processing. *Neural Networks*, *14*(6-7), 715-725.

## Optimization & Frameworks

* AMD. (n.d.). *ROCm Documentation*. Retrieved March 30, 2025, from https://rocm.docs.amd.com/
* Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In *Advances in Neural Information Processing Systems 32* (pp. 8026-8037). Curran Associates, Inc.
* Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization of machine learning algorithms. In *Advances in Neural Information Processing Systems 25* (pp. 2951-2959). Curran Associates, Inc.
