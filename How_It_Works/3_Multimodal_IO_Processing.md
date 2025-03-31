## 3. Multimodal Input/Output Processing

### A. Encoder Mechanism: From Raw Data to Spike Trains

#### A.1. Purpose & Contrast with LLM Input
*   To act as the sensory interface, translating diverse raw input data from various modalities (text, images, video, potentially audio, touch, etc.) into a **universal spike-based format** that the SNN core can process uniformly. **Why?** This allows the core network to be modality-agnostic, simplifying the architecture and enabling seamless integration of new sensor types.
*   This differs markedly from LLMs which typically use tokenization (breaking text into sub-words) followed by embedding layers to convert input into dense vectors. FUM uses temporal spike patterns.

#### A.2. Encoding Methods (Rate & Temporal)
*   **Rate Encoding (Primary):** Maps features to firing frequencies `f` over a window `T` (e.g., 50 timesteps).
    *   *Text:* ASCII value `c` -> `f = (ord(c) % 50) Hz`.
    *   *Images:* Pixel intensity `p` (0-255) -> `f = (p / 2.55) Hz`.
*   **Temporal Encoding (Structured Inputs):** For complex inputs like code syntax trees or logical propositions, use hierarchical temporal encoding:
    *   *Code Syntax Trees:* Parse tree (e.g., using `ast`). Encode node type/value using frequency bands (e.g., Call: 10-20Hz) modulated by value (e.g., `print`: 15Hz). Encode hierarchy over sequential time windows (e.g., 50 timesteps per level: Root -> Children -> Grandchildren).
    *   *Logical Propositions:* Encode variables (A: 10Hz) and operators (∧: 30Hz, →: 35Hz) as frequencies in a temporal sequence (e.g., [A, ∧, B, →, C] over 250 timesteps).

#### A.3. Poisson Spike Generation Details
*   **Formula:** Spikes are generated using a Poisson process based on target frequency `f` and timestep `dt=1ms`.
    *   Probability of spike per timestep: `p = f * dt`. (e.g., `f=50Hz` -> `p=0.05`).
    *   Algorithm: For each timestep `t`, if `torch.rand(1) < p`, emit spike `spike[t] = 1`.
*   **Refractory Period:** A 5ms refractory period (5 timesteps) is imposed on input neurons after spiking.
    *   **Implementation:** Maintain `refractory[i]` counter. If `spike[t]=1`, set `refractory[i]=5`. Only generate spike if `refractory[i]==0`. Decrement counter each step.
    *   **Rationale:** Prevents unrealistically high firing rates (caps at 200 Hz), aligning with biological limits.

#### A.4. Output & Extensibility
*   Output is a tensor `I_encoded` (shape: `[num_input_neurons, T_total]`) containing spike trains (0s and 1s) fed into the SNN core.
*   Adding a new sensor only requires designing a new encoder module mapping its data to spike trains.

### B. Decoder Mechanism: From Spike Trains to Structured Output

#### B.1. Purpose
*   To translate the internal spiking activity patterns of designated output neurons back into a human-understandable format (e.g., text, classification label, numerical value, code, logical steps), relevant to the task performed.

#### B.2. Decoding Methods (Rate & Temporal)
*   **Rate Decoding (Simple Outputs):** Average firing rates of output neurons over a window `T` (e.g., 50 timesteps) are mapped to symbols.
    *   *Classification:* Highest firing rate indicates the class.
    *   *Numerical:* `symbol = int(rate * 2)` (e.g., `rate = torch.sum(spike_history[output_neuron]) / 50`, so 2 Hz -> '4').
*   **Temporal Decoding (Structured Outputs):** Generate sequences by interpreting firing rates of output neurons over sequential time windows.
    *   *Code Generation:* `print(2+2)` -> Window 1: Neuron 'print' fires at 10Hz; Window 2: Neuron '(' fires at 11Hz; Window 3: Neuron '2' fires at 12Hz, etc. Map rates (`rate = torch.sum(...) / 50`) to tokens using a lookup table (`token = lookup[rate]`).
    *   *Logical Deduction:* Output steps ("Given A=1", "A ∧ B = 1", "C = 1") sequentially, mapping tokens to firing rates in successive windows.

#### B.3. Emergent Formation
*   STDP and SIE reinforce connections from internal processing clusters to the appropriate output neurons, ensuring they fire at the correct rates/times to produce the desired output, guided by rewards (`r=1`) for successful task completion.

#### B.4. Implementation
*   Decoding typically occurs on the CPU after retrieving spike history or firing rates from the GPU, logging outputs to SSD (`torch.save(outputs, 'outputs.pt')`).

