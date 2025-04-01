# File: resonance_enhanced_stdp.py (REVISED for Tensor Operations and Trace-Based STDP)
# Purpose: Implement Resonance-Enhanced STDP using PyTorch tensors,
#          modulating eligibility trace decay (gamma) based on PLV.
#          Uses pre/post synaptic traces for STDP calculation.

import torch
from typing import Optional

# Assume DEVICE is initialized globally elsewhere (e.g., in unified_neuron.py)
# If run standalone, needs fallback.
try:
    from neuron.unified_neuron import DEVICE
except ImportError:
    print("Warning: Could not import global DEVICE. Defaulting to CPU for STDP.")
    DEVICE = torch.device('cpu')

class ResonanceEnhancedSTDP_TraceModulation_Tensor:
    """
    Implements Resonance-Enhanced STDP rule using PyTorch Tensors.
    - Modulates eligibility trace decay (gamma) based on phase synchronization (PLV).
    - Uses pre/post synaptic traces for STDP calculation.
    - Operates on boolean spike tensors per timestep.
    """
    def __init__(self, num_pre: int, num_post: int,
                 eta: float = 0.01, a_plus: float = 0.1, a_minus: float = 0.05,
                 tau_plus: float = 20.0, tau_minus: float = 20.0,
                 tau_trace: float = 25.0, # Time constant for pre/post synaptic traces (ms)
                 gamma_min: float = 0.50, gamma_max: float = 0.95, # Eligibility trace decay range
                 plv_k: float = 10.0, plv_mid: float = 0.5,       # Gamma sigmoid params
                 w_min: float = -1.0, w_max: float = 1.0,
                 device: torch.device = DEVICE):
        """
        Initialize Resonance-Enhanced STDP (Tensor Version) parameters.

        Args:
            num_pre (int): Number of pre-synaptic neurons.
            num_post (int): Number of post-synaptic neurons.
            eta (float): Base learning rate for weight updates from eligibility traces.
            a_plus (float): Potentiation amplitude scaling factor.
            a_minus (float): Depression amplitude scaling factor.
            tau_plus (float): Time constant for potentiation effect on eligibility trace (ms).
            tau_minus (float): Time constant for depression effect on eligibility trace (ms).
            tau_trace (float): Time constant for pre/post synaptic traces (ms).
            gamma_min (float): Minimum eligibility trace decay factor (at low PLV).
            gamma_max (float): Maximum eligibility trace decay factor (at high PLV).
            plv_k (float): Steepness factor for the gamma(PLV) sigmoid function.
            plv_mid (float): Midpoint (PLV value) for the gamma(PLV) sigmoid transition.
            w_min (float): Minimum synaptic weight.
            w_max (float): Maximum synaptic weight.
            device (torch.device): The compute device to use for tensors.
        """
        self.num_pre = num_pre
        self.num_post = num_post
        self.eta = eta
        self.a_plus = a_plus
        self.a_minus = a_minus
        # Precompute decay factors from time constants (assuming dt=1ms)
        self.decay_plus = torch.exp(torch.tensor(-1.0 / tau_plus, device=device))
        self.decay_minus = torch.exp(torch.tensor(-1.0 / tau_minus, device=device))
        self.decay_trace = torch.exp(torch.tensor(-1.0 / tau_trace, device=device))
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.plv_k = plv_k
        self.plv_mid = plv_mid
        self.w_min = w_min
        self.w_max = w_max
        self.device = device

        # --- Internal State Tensors ---
        # Eligibility traces (one per synapse)
        self.eligibility_traces = torch.zeros((num_pre, num_post), device=self.device, dtype=torch.float32)
        # Pre-synaptic traces (one per pre-neuron)
        self.pre_trace = torch.zeros(num_pre, device=self.device, dtype=torch.float32)
        # Post-synaptic traces (one per post-neuron)
        self.post_trace = torch.zeros(num_post, device=self.device, dtype=torch.float32)

        print(f"Initialized ResonanceEnhancedSTDP_TraceModulation_Tensor on device: {self.device}")

    def _calculate_gamma(self, plv: float) -> float:
        """Calculates PLV-dependent eligibility trace decay factor using a sigmoid function."""
        # Using torch equivalents
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) / (1 + torch.exp(-self.plv_k * (torch.tensor(plv, device=self.device) - self.plv_mid)))
        return torch.clip(gamma, self.gamma_min, self.gamma_max).item() # Return scalar

    @torch.no_grad() # Disable gradient tracking for performance if not training with autograd
    def update(self, pre_spikes_t: torch.Tensor, post_spikes_t: torch.Tensor,
               weights: torch.Tensor, plv: float, reward_signal: float = 1.0,
               hints: Optional[torch.Tensor] = None, hint_factor: float = 0.1) -> torch.Tensor:
        """
        Updates synaptic traces, eligibility traces, and weights for a single timestep.
        Incorporates optional hints to bias learning.

        Args:
            pre_spikes_t (torch.Tensor): Boolean tensor indicating pre-synaptic spikes at current time t (shape: [num_pre]).
            post_spikes_t (torch.Tensor): Boolean tensor indicating post-synaptic spikes at current time t (shape: [num_post]).
            weights (torch.Tensor): Current synaptic weights tensor (shape: [num_pre, num_post]).
            plv (float): Phase-locking value (PLV) for this update context.
            reward_signal (float): Pre-modulated reward signal (incorporating eta and SIE modulation).
            hints (Optional[torch.Tensor]): Optional tensor of hints biasing weight changes (shape: [num_pre, num_post]).
                                            Positive values encourage potentiation, negative encourage depression.
            hint_factor (float): Scaling factor for the influence of hints.

        Returns:
            torch.Tensor: Updated synaptic weights tensor.
        """
        if pre_spikes_t.device != self.device or post_spikes_t.device != self.device or weights.device != self.device:
             raise ValueError("Input tensors must be on the same device as the STDP handler.")
        if pre_spikes_t.shape[0] != self.num_pre or post_spikes_t.shape[0] != self.num_post or weights.shape != (self.num_pre, self.num_post):
             raise ValueError("Input tensor shapes do not match handler configuration.")

        # --- 1. Update Pre- and Post-Synaptic Traces ---
        # Decay existing traces
        self.pre_trace *= self.decay_trace
        self.post_trace *= self.decay_trace
        # Increase traces for neurons that spiked (add 1.0)
        self.pre_trace[pre_spikes_t] = 1.0 # Reset trace to max on spike
        self.post_trace[post_spikes_t] = 1.0 # Reset trace to max on spike

        # --- 2. Calculate STDP-based Eligibility Trace Updates ---
        # Potentiation: Occurs when post-synaptic neuron fires. Depends on pre-synaptic trace.
        # Depression: Occurs when pre-synaptic neuron fires. Depends on post-synaptic trace.

        # Expand traces for broadcasting: pre [pre, 1], post [1, post]
        pre_trace_expanded = self.pre_trace.unsqueeze(1)
        post_trace_expanded = self.post_trace.unsqueeze(0)

        # Calculate potential change contributions at synapses
        # Potentiation contribution where post spiked: a_plus * pre_trace
        delta_eligibility_pot = self.a_plus * pre_trace_expanded * post_spikes_t.unsqueeze(0).float()
        # Depression contribution where pre spiked: -a_minus * post_trace
        delta_eligibility_dep = -self.a_minus * post_trace_expanded * pre_spikes_t.unsqueeze(1).float()

        # Combine contributions (potentiation where post fired, depression where pre fired)
        delta_eligibility = delta_eligibility_pot + delta_eligibility_dep

        # --- 3. Update Eligibility Traces ---
        # Decay existing eligibility traces based on PLV
        current_gamma = self._calculate_gamma(plv)
        self.eligibility_traces *= current_gamma
        # Add the new contributions from this timestep
        self.eligibility_traces += delta_eligibility

        # --- 4. Update Weights ---
        # Apply weight updates using eligibility traces and the pre-modulated reward signal.
        # The reward_signal already incorporates eta and the SIE modulation factor.
        base_delta_weights = reward_signal * self.eligibility_traces

        # Apply hint bias if hints are provided
        if hints is not None:
            if hints.shape != weights.shape:
                 raise ValueError(f"Hints shape {hints.shape} must match weights shape {weights.shape}")
            # Simple additive bias based on hint value and factor
            # Note: Ensure hints are on the correct device
            hint_bias = hint_factor * hints.to(self.device) * torch.abs(reward_signal) # Scale hint by reward magnitude? Or fixed factor? Let's use fixed factor for now.
            # hint_bias = hint_factor * hints.to(self.device) # Simpler fixed bias
            # Let's try scaling by reward signal magnitude to make hints stronger when reward is significant
            hint_bias = hint_factor * hints.to(self.device) * torch.abs(reward_signal)

            delta_weights = base_delta_weights + hint_bias
            # print(f"Hint bias applied: mean={hint_bias.mean().item():.4f}") # Debug
        else:
            delta_weights = base_delta_weights

        weights += delta_weights

        # Apply weight clipping
        weights.clip_(min=self.w_min, max=self.w_max)

        return weights

# --- Remove old simulation runner code ---
# The simulation logic will now reside elsewhere (e.g., within the main training loop
# that uses the UnifiedNeuronModel).
