
### Task 2: Integrate Multimodal Inputs
- [ ] **Success Statement**: I will know I achieved success on this task when enhanced encoders/decoders support autonomous interpretation and generation across modalities, validated through testing.
- **Current Status**: *Partial*—Basic encoders/decoder implemented and tested. Advanced features, integration, and streaming pending.

- [ ] **Step 1: Implement Enhanced Multimodal Encoders**
  - [ ] **Validation Check**: Encoders process text, image, video, audio using hierarchical and temporal schemes effectively (Ref: 3A.2).
  - [ ] **Success Statement**: I will know I achieved success on this step when all modalities stream information-rich spikes continuously.
  - [ ] **Actionable Substeps**:
    - [x] Code basic `encoder.py` sub-encoders (Ref: 3A).
    - [ ] Implement Poisson spike generation with 5ms refractory period for input neurons (Ref: 3A.3.ii).
    - [ ] Implement hierarchical encoding enhancements (e.g., text: char/word/sentence layers; image: pixel/edge/object layers) (Ref: 3A.2.ii).
    - [ ] Implement spike pattern encoding enhancements (encode features via precise timing within 50ms window) (Ref: 3A.2.iii).
    - [ ] Implement encoding robustness: Apply low-pass filter to input frequencies (Ref: 5E.5.ii).
    - [x] Test: Unit test basic encoders (`test_io.py`).
    - [ ] Test: Unit test input neuron refractory period implementation.
    - [ ] Test: Unit test enhanced encoding schemes (hierarchical layers, pattern generation).
    - [ ] Test: Unit test encoding robustness filter application.
    - [ ] Integrate encoders with `run_phase3.py` for continuous/streaming operation.
    - [ ] Test: Validate handling of multi-domain inputs (temporal separation, sequential cluster activation, inhibitory suppression) (Ref: 2D.4.iv).
  - [ ] **Test: Unit test Step 1 functionalities.**

- [ ] **Step 2: Implement Flexible Decoder**
  - [ ] **Validation Check**: Decoder outputs varied formats (text, structured, silent) based on context/SIE state accurately (Ref: 3B).
  - [ ] **Success Statement**: I will know I achieved success on this step when FUM signals goals flexibly, appropriately, and accurately.
  - [ ] **Actionable Substeps**:
    - [x] Code basic rate decoder (`decode_text_rate`) in `decoder.py` (Ref: 3B.2).
    - [ ] Implement temporal decoding for structured output (map rates in sequential windows to tokens via lookup table) (Ref: 3B.2.ii).
    - [ ] Implement mode selection logic (choose text, voice, visual, silent based on SIE state / context) (Ref: Roadmap/3B).
    - [ ] Implement decoder execution on CPU, logging outputs to SSD (Ref: 3B.4.i).
    - [x] Test: Unit test basic text decoding (`test_io.py`).
    - [ ] Test: Unit test temporal decoding logic (windowing, mapping).
    - [ ] Test: Unit test mode selection logic.
    - [ ] Test: Unit test voice/visual output generation (placeholder/basic implementation).
    - [ ] Test: Unit test CPU execution and logging mechanism.
  - [ ] **Test: Unit test Step 2 functionalities.**

- [ ] **Test: Integration test Task 2 components (Encoders, Decoders).**
- [ ] **Test: Unit tests for Task 2.**