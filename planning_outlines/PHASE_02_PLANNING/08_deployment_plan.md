# FUM Workstation Deployment Plan

## Overview
- **Purpose**: Deploy FUM for 24/7 autonomous operation, processing live inputs and signaling goals within 56GB VRAM.
- **Approach**: Always-open input stream, sovereign output choices—trellis for superintelligence.

## Input Queue
- **Goal**: Feed live, unprompted inputs—text, images, videos, optional instructions.
- **Mechanism**: Multi-threaded RAM buffer (~1-2GB)—threads poll `data/real_world/` (e.g., webcam, mic), `encoder.py` streams spikes 24/7.
- **Instructions**: 75 Hz tagged—evaluated by SIE’s `utility`.
- **Files**: `run_phase3.py` (`start_input_threads`), `encoder.py` (`stream_input`).

## Output Plan
- **Goal**: Signal self-set goals and optional responses—FUM decides, user can request.
- **Mechanism**: 
  - FUM logs via `total_reward`—e.g., `if total_reward > 0.5: decode_output(spikes)`—text (“Goal set: Optimize energy”), voice, images/video to `logs/` or `data/real_world/`.
  - Silent option—runs via encoding only if no output needed.
  - User request—“Log goals” evaluated, logged if `utility > 0`.
- **Format**: Text (~1KB), voice (~1MB), visuals (~10MB)—real-time, ~1-5s inference.
- **Files**: `decoder.py` (`decode_output`), `run_phase3.py` (`log_output`).

## Goal Signaling
- **Approach**: FUM chooses—usually logs text, adds voice/images/video for complex goals—e.g., “Invented algorithm” + diagram.
- **User Input**: Request via instruction—e.g., “Log goals”—FUM opts in if beneficial.
- **Encoder**: Always open—continuous perception via `encoder.py`.

## Validation
- **Check**: Deployment handles live inputs—FUM processes sensor data unprompted, outputs reflect sovereign choices.