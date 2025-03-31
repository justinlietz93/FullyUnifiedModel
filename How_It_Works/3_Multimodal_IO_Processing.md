## 3. Multimodal Input/Output Processing

### A. Encoder Mechanism: From Raw Data to Spike Trains

#### A.1 Purpose & Contrast with LLM Input

##### A.1.i.
*   To act as the sensory interface, translating diverse raw input data from various modalities (text, images, video, potentially audio, touch, etc.) into a **universal spike-based format** that the SNN core can process uniformly. **Why?** This allows the core network to be modality-agnostic, simplifying the architecture and enabling seamless integration of new sensor types.

##### A.1.ii.
*   This differs markedly from LLMs which typically use tokenization (breaking text into sub-words) followed by embedding layers to convert input into dense vectors. FUM uses temporal spike patterns.

#### A.2 Enhanced Encoding Methods (Hierarchical & Spike Pattern)

##### A.2.i.
*   **Addressing Potential Bottleneck:** Simple rate encoding may not capture sufficient complexity from the minimal input set (80-300 examples) to achieve expert-level mastery (>85% on benchmarks). To resolve this potential information bottleneck, FUM employs enhanced, brain-inspired encoding strategies:

##### A.2.ii.
*   **Hierarchical Encoding:** Emulates the brain's hierarchical processing (e.g., V1-V4 in visual cortex, Felleman & Van Essen, 1991).
    *   *Mechanism:* `hierarchical_encoding = encode_hierarchy(input, layers=3)`, executed on the 7900 XTX GPU.
    *   *Text Example:* Layer 1 encodes characters (e.g., 1 Hz/char), Layer 2 encodes words (e.g., 2 Hz/word), Layer 3 encodes sentences (e.g., 5 Hz/sentence) (master node logic).
    *   *Image Example:* Layer 1 encodes pixels (e.g., 0-10 Hz intensity), Layer 2 encodes edges/textures (e.g., 5 Hz/feature), Layer 3 encodes objects/regions (e.g., 10 Hz/object) (master node logic).

##### A.2.iii.
*   **Spike Pattern Encoding:** Uses temporal patterns within the encoding window to increase information capacity, inspired by temporal coding theories (Buzsáki, 2010).
    *   *Mechanism:* `spike_pattern = encode_pattern(input, max_rate=10 Hz, duration=50ms)`, executed on the 7900 XTX GPU. Encodes features not just by rate but by the precise timing of spikes within the 50ms window (master node logic). For example, 'A' might be encoded as [1Hz @ 0-10ms, 2Hz @ 10-20ms, ...].

##### A.2.iv.
*   **Increased Information Capture:** These enhanced methods significantly increase the information captured per input compared to simple rate encoding.
    *   *Estimate:* Yields approximately **~2255-8460 bits per input** (assuming 50 timesteps, max 10Hz rate, pattern encoding providing ~5x more info than rate encoding).
    *   *Sufficiency for Mastery:* For 300 inputs, this provides ~2.54M bits total, sufficient to constrain the ~12.8T synapses at the 32B neuron scale (Answer 4), supporting the target of >85% accuracy on complex benchmarks like MATH/GPQA subsets (Answer 3.2) and aligning with the minimal data mastery goal (Sec 1.A) (95% goal alignment expected).

#### A.3 Poisson Spike Generation Details

##### A.3.i.
*   **Formula:** Spikes are generated using a Poisson process based on target frequency `f` and timestep `dt=1ms`.
    *   Probability of spike per timestep: `p = f * dt`. (e.g., `f=50Hz` -> `p=0.05`).
    *   Algorithm: For each timestep `t`, if `torch.rand(1) < p`, emit spike `spike[t] = 1`.

##### A.3.ii.
*   **Refractory Period:** A 5ms refractory period (5 timesteps) is imposed on input neurons after spiking.
    *   **Implementation:** Maintain `refractory[i]` counter. If `spike[t]=1`, set `refractory[i]=5`. Only generate spike if `refractory[i]==0`. Decrement counter each step.
    *   **Rationale:** Prevents unrealistically high firing rates (caps at 200 Hz), aligning with biological limits.

#### A.4 Output & Extensibility

##### A.4.i.
*   Output is a tensor `I_encoded` (shape: `[num_input_neurons, T_total]`) containing spike trains (0s and 1s) fed into the SNN core.

##### A.4.ii.
*   Adding a new sensor only requires designing a new encoder module mapping its data to spike trains.

### B. Decoder Mechanism: From Spike Trains to Structured Output

#### B.1 Purpose

##### B.1.i.
*   To translate the internal spiking activity patterns of designated output neurons back into a human-understandable format (e.g., text, classification label, numerical value, code, logical steps), relevant to the task performed.

#### B.2 Decoding Methods (Rate & Temporal)

##### B.2.i.
*   **Rate Decoding (Simple Outputs):** Average firing rates of output neurons over a window `T` (e.g., 50 timesteps) are mapped to symbols.
    *   *Classification:* Highest firing rate indicates the class.
    *   *Numerical:* `symbol = int(rate * 2)` (e.g., `rate = torch.sum(spike_history[output_neuron]) / 50`, so 2 Hz -> '4').

##### B.2.ii.
*   **Temporal Decoding (Structured Outputs):** Generate sequences by interpreting firing rates of output neurons over sequential time windows.
    *   *Code Generation:* `print(2+2)` -> Window 1: Neuron 'print' fires at 10Hz; Window 2: Neuron '(' fires at 11Hz; Window 3: Neuron '2' fires at 12Hz, etc. Map rates (`rate = torch.sum(...) / 50`) to tokens using a lookup table (`token = lookup[rate]`).
    *   *Logical Deduction:* Output steps ("Given A=1", "A ∧ B = 1", "C = 1") sequentially, mapping tokens to firing rates in successive windows.

#### B.3 Emergent Formation

##### B.3.i.
*   STDP and SIE reinforce connections from internal processing clusters to the appropriate output neurons, ensuring they fire at the correct rates/times to produce the desired output, guided by rewards (`r=1`) for successful task completion.

#### B.4 Implementation

##### B.4.i.
*   Decoding typically occurs on the CPU after retrieving spike history or firing rates from the GPU, logging outputs to SSD (`torch.save(outputs, 'outputs.pt')`).
