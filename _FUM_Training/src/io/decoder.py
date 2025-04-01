import torch
import numpy as np

# Assume DEVICE is initialized globally elsewhere (e.g., in unified_neuron.py or config_loader)
# If run standalone, needs fallback.
try:
    # Use explicit path relative to _FUM_Training
    from src.neuron.unified_neuron import initialize_device
    DEVICE = initialize_device()
except ImportError as e:
     print(f"Warning: Could not import initialize_device from src.neuron.unified_neuron: {e}. Defaulting device to CPU for decoders.")
     DEVICE = torch.device('cpu')
except Exception as e_init:
     print(f"Warning: Error during device initialization: {e_init}. Defaulting device to CPU for decoders.")
     DEVICE = torch.device('cpu')


def decode_text_rate(spike_history: torch.Tensor,
                     output_neuron_indices: list[int],
                     window_size: int = 50,
                     dt: float = 1.0,
                     rate_to_ascii_map: dict[float, str] = None,
                     max_rate_for_char: float = 50.0) -> str:
    """
    Decodes output spike history into text using simple rate coding.
    Assumes one character is represented by the highest firing rate neuron
    within the specified output neuron indices over the window.

    Args:
        spike_history (torch.Tensor): Tensor of spike data for output neurons.
                                      Shape: (num_output_neurons, duration).
                                      Assumed to be on the correct device (CPU/GPU).
        output_neuron_indices (list[int]): Indices corresponding to the rows in spike_history.
                                           Used if mapping rates to specific chars/neurons.
                                           Length must match spike_history.shape[0].
        window_size (int): Number of timesteps to average rate over. Defaults to 50.
        dt (float): Simulation timestep in ms. Defaults to 1.0.
        rate_to_ascii_map (dict[float, str], optional): A specific map from rate to char.
                                                        If None, uses a simple linear mapping.
        max_rate_for_char (float): The firing rate (Hz) corresponding to the highest character value (e.g., ASCII 127).
                                   Used for linear mapping if rate_to_ascii_map is None.

    Returns:
        str: The decoded character or an empty string if no significant activity.
    """
    if spike_history.shape[0] != len(output_neuron_indices):
        raise ValueError("Mismatch between spike_history dimension 0 and length of output_neuron_indices.")

    # Ensure history is on CPU for calculation if needed, or keep on GPU if possible
    # For simple sum/mean, GPU is fine.
    spike_history = spike_history.to(DEVICE)

    # Consider only the last window
    if spike_history.shape[1] < window_size:
        window_size = spike_history.shape[1]
    if window_size == 0:
        return ""

    window_spikes = spike_history[:, -window_size:]

    # Calculate average firing rate (Hz) for each output neuron in the window
    spike_counts = torch.sum(window_spikes, dim=1)
    rates_hz = spike_counts / (window_size * dt / 1000.0) # Convert window duration to seconds

    # Find the neuron with the highest firing rate
    max_rate, max_idx_local = torch.max(rates_hz, dim=0)

    # Threshold: require some minimum activity
    min_rate_threshold = 1.0 # Hz (e.g., at least one spike in 1 sec equiv)
    if max_rate < min_rate_threshold:
        return "" # No significant output

    # Map the index of the highest firing neuron to a character
    # Assumes neuron indices in output_neuron_indices map directly to ASCII
    # or some other symbol map. For simplicity, assume direct ASCII mapping here.
    winning_neuron_global_index = output_neuron_indices[max_idx_local.item()]

    # Assuming direct ASCII mapping up to 127
    if 0 <= winning_neuron_global_index < 128:
        try:
            return chr(winning_neuron_global_index)
        except ValueError:
            return "?" # Should not happen if index is in valid ASCII range
    else:
        # Handle cases where index is outside expected ASCII range or
        # a different mapping is needed.
        return "?" # Return placeholder if index doesn't map cleanly


# --- Example Usage ---
if __name__ == '__main__':
    print(f"\n--- Decoder Test ---")
    print(f"Using Device: {DEVICE}")

    duration = 100
    num_out_neurons = 128
    test_indices = list(range(num_out_neurons))
    dt_val = 1.0
    window = 50
    max_r = 50.0

    # Mock spike history
    mock_history = torch.zeros((num_out_neurons, duration), device=DEVICE)

    # Simulate neuron for 'A' (ASCII 65) firing at target rate
    # Target rate = (65 / 127) * 50 Hz ~= 25.6 Hz
    target_rate_A = (65 / 127.0) * max_r
    prob_A = target_rate_A * (dt_val / 1000.0)
    mock_history[65, -window:] = torch.rand(window, device=DEVICE) < prob_A

    # Simulate neuron for 'b' (ASCII 98) firing at target rate
    target_rate_b = (98 / 127.0) * max_r
    prob_b = target_rate_b * (dt_val / 1000.0)
    mock_history[98, -window:] = torch.rand(window, device=DEVICE) < prob_b

    # Simulate neuron for '$' (ASCII 36) firing at lower rate
    target_rate_dollar = (36 / 127.0) * max_r
    prob_dollar = target_rate_dollar * (dt_val / 1000.0)
    mock_history[36, -window:] = torch.rand(window, device=DEVICE) < prob_dollar


    print("\nDecoding mock history (expect 'b'):")
    # Neuron 'b' should have highest rate
    decoded_char = decode_text_rate(mock_history, test_indices, window, dt_val, max_rate_for_char=max_r)
    print(f"Decoded: '{decoded_char}'")
    # Basic check:
    if decoded_char != 'b': print("Warning: Did not decode 'b' as expected.")

    print("\nDecoding mock history (only 'A' active):")
    mock_history_A = torch.zeros((num_out_neurons, duration), device=DEVICE)
    mock_history_A[65, -window:] = torch.rand(window, device=DEVICE) < prob_A
    decoded_char_A = decode_text_rate(mock_history_A, test_indices, window, dt_val, max_rate_for_char=max_r)
    print(f"Decoded: '{decoded_char_A}'")
    if decoded_char_A != 'A': print("Warning: Did not decode 'A' as expected.")

    print("\nDecoding zero history (expect ''):")
    decoded_char_zero = decode_text_rate(torch.zeros_like(mock_history), test_indices, window, dt_val, max_rate_for_char=max_r)
    print(f"Decoded: '{decoded_char_zero}'")
    if decoded_char_zero != '': print("Warning: Did not decode empty string as expected.")


    print("--- Decoder Test Complete ---")
