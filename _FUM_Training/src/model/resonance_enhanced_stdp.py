# File: resonance_enhanced_stdp.py (REVISED for Tensor Operations and Trace-Based STDP)
# Purpose: Implement Resonance-Enhanced STDP using PyTorch tensors,
#          modulating eligibility trace decay (gamma) based on PLV.
#          Uses pre/post synaptic traces for STDP calculation.
#          Includes constrained biological diversity and jitter mitigation.

import torch
from typing import Optional, Dict, Any, List
import math # For sign

# Assume DEVICE is initialized globally elsewhere (e.g., in unified_neuron.py)
# If run standalone, needs fallback.
try:
    # Assuming unified_neuron might be refactored or DEVICE passed differently
    # For now, try importing, but handle failure gracefully.
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
    - Incorporates constrained biological diversity for potentiation parameters.
    - Includes jitter mitigation techniques.
    Ref: How_It_Works/2_Core_Architecture_Components/2B_Neural_Plasticity.md (esp. B.4.iv, B.2.iii, B.2.vi, B.7.iv)
    Ref: How_It_Works/5_Training_and_Scaling/5D_Scaling_Strategy.md (esp. D.2.vii)
    """
    def __init__(self, num_pre: int, num_post: int,
                 eta: float = 0.01, a_plus: float = 0.1, a_minus: float = 0.05, # Note: a_plus is now base mean
                 tau_plus: float = 20.0, tau_minus: float = 20.0, # Note: tau_plus is now base mean
                 tau_trace: float = 25.0, # Time constant for pre/post synaptic traces (ms)
                 gamma_min: float = 0.50, gamma_max: float = 0.95, # Eligibility trace decay range
                 plv_k: float = 10.0, plv_mid: float = 0.5,       # Gamma sigmoid params
                 w_min: float = -1.0, w_max: float = 1.0,
                 target_rate: float = 0.3, # Hz, for rate dependency modulation (Ref: 2B.4.iv)
                 temporal_filter_width: int = 4, # Width for moving average filter (Ref: 2B.2.vi)
                 reward_sigmoid_scale: float = 1.0, # Scale factor for sigmoid input (from SIE config)
                 device: torch.device = DEVICE):
        """
        Initialize Resonance-Enhanced STDP (Tensor Version) parameters. Incorporates constrained biological diversity.

        Args:
            num_pre (int): Number of pre-synaptic neurons.
            num_post (int): Number of post-synaptic neurons.
            eta (float): Base learning rate for weight updates from eligibility traces.
            a_plus (float): *Mean* potentiation amplitude scaling factor for base variability.
            a_minus (float): Depression amplitude scaling factor (kept fixed for now).
            tau_plus (float): *Mean* time constant for potentiation effect (ms) for base variability.
            tau_minus (float): Time constant for depression effect on eligibility trace (ms).
            tau_trace (float): Time constant for pre/post synaptic traces (ms).
            gamma_min (float): Minimum eligibility trace decay factor (at low PLV).
            gamma_max (float): Maximum eligibility trace decay factor (at high PLV).
            plv_k (float): Steepness factor for the gamma(PLV) sigmoid function.
            plv_mid (float): Midpoint (PLV value) for the gamma(PLV) sigmoid transition.
            w_min (float): Minimum synaptic weight.
            w_max (float): Maximum synaptic weight.
            target_rate (float): Target firing rate in Hz for rate dependency modulation.
            temporal_filter_width (int): Number of past timesteps (plus current) for temporal noise filter.
            reward_sigmoid_scale (float): Scale factor applied to total_reward before sigmoid in mod_factor calc.
            device (torch.device): The compute device to use for tensors.
        """
        self.num_pre = num_pre
        self.num_post = num_post
        self.eta = eta
        self.a_minus = a_minus
        self.target_rate = target_rate
        self.temporal_filter_width = max(1, temporal_filter_width)
        self.reward_sigmoid_scale = reward_sigmoid_scale

        # --- Initialize Base Parameters with Constrained Variability (Ref: 2B.4.iv) ---
        a_plus_std_dev = 0.05
        a_plus_base_unclamped = torch.normal(mean=a_plus, std=a_plus_std_dev, size=(num_pre, num_post), device=device, dtype=torch.float32)
        self.a_plus_base = torch.clamp(a_plus_base_unclamped, min=0.05, max=0.15)

        tau_plus_std_dev = 5.0
        tau_plus_base_unclamped = torch.normal(mean=tau_plus, std=tau_plus_std_dev, size=(num_pre, num_post), device=device, dtype=torch.float32)
        self.tau_plus_base = torch.clamp(tau_plus_base_unclamped, min=15.0, max=25.0)
        # TODO: If tau_plus becomes heterogeneous, decay_plus needs dynamic calculation in update()

        # Precompute fixed decay factors
        self.tau_minus = tau_minus
        self.decay_minus = torch.exp(torch.tensor(-1.0 / self.tau_minus, device=device))
        self.tau_trace = tau_trace
        self.decay_trace = torch.exp(torch.tensor(-1.0 / self.tau_trace, device=device))

        # Other parameters
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.plv_k = plv_k
        self.plv_mid = plv_mid
        self.w_min = w_min
        self.w_max = w_max
        self.device = device

        # --- Internal State Tensors ---
        self.eligibility_traces = torch.zeros((num_pre, num_post), device=self.device, dtype=torch.float32)
        self.pre_trace = torch.zeros(num_pre, device=self.device, dtype=torch.float32)
        self.post_trace = torch.zeros(num_post, device=self.device, dtype=torch.float32)

        # --- STC Analogue State (Ref: 2B.5.ix) ---
        self.stc_tag_threshold = 0.05
        self.stc_consolidation_duration = 100000
        # TODO: Optimize tag history storage
        self.tag_history = torch.zeros((num_pre, num_post, self.stc_consolidation_duration), dtype=torch.bool, device=self.device)
        self.tag_history_ptr = 0

        print(f"Initialized ResonanceEnhancedSTDP_TraceModulation_Tensor on device: {self.device}")
        print(f"  A+ Base Mean: {self.a_plus_base.mean():.4f}, Std: {self.a_plus_base.std():.4f}")
        print(f"  Tau+ Base Mean: {self.tau_plus_base.mean():.4f}, Std: {self.tau_plus_base.std():.4f}")


    def _calculate_gamma(self, plv: float) -> float:
        """Calculates PLV-dependent eligibility trace decay factor using a sigmoid function."""
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) / (1 + torch.exp(-self.plv_k * (torch.tensor(plv, device=self.device) - self.plv_mid)))
        return torch.clip(gamma, self.gamma_min, self.gamma_max).item()

    @torch.no_grad()
    def update(self, pre_spikes_t: torch.Tensor, post_spikes_t: torch.Tensor,
               weights: torch.Tensor, plv: float,
               total_reward: float = 0.0, # Use raw total_reward now
               # Inputs for biological diversity modulation (Ref: 2B.4.iv)
               pre_spike_rates: Optional[torch.Tensor] = None,
               cluster_assignments: Optional[torch.Tensor] = None,
               cluster_rewards: Optional[torch.Tensor] = None,
               # Inputs for Jitter Mitigation (Ref: 5D.2.vii)
               max_latency: float = 0.0,
               latency_error: float = 0.0,
               spike_history_buffer: Optional[List[Dict[str, torch.Tensor]]] = None,
               # Optional hints
               hints: Optional[torch.Tensor] = None, hint_factor: float = 0.1) -> torch.Tensor:
        """
        Updates synaptic traces, eligibility traces, and weights for a single timestep.
        Incorporates constrained biological diversity, jitter mitigation, STC, and optional hints.
        Applies reward signal correctly to modulate learning direction and magnitude.

        Args:
            pre_spikes_t (torch.Tensor): Boolean tensor indicating pre-synaptic spikes [num_pre].
            post_spikes_t (torch.Tensor): Boolean tensor indicating post-synaptic spikes [num_post].
            weights (torch.Tensor): Current synaptic weights tensor [num_pre, num_post].
            plv (float): Phase-locking value (PLV) for this update context.
            total_reward (float): Raw reward signal from SIE (or external). Sign determines direction.
            pre_spike_rates (Optional[torch.Tensor]): Firing rates of pre-synaptic neurons (Hz) [num_pre].
            cluster_assignments (Optional[torch.Tensor]): Cluster ID for each post-synaptic neuron [num_post].
            cluster_rewards (Optional[torch.Tensor]): Average reward for each cluster [num_clusters].
            max_latency (float): Estimated max latency in ms.
            latency_error (float): Estimated latency error std dev in ms.
            spike_history_buffer (Optional[List[Dict[str, torch.Tensor]]]): Buffer of recent spike tensors.
            hints (Optional[torch.Tensor]): Optional tensor of hints biasing weight changes [num_pre, num_post].
            hint_factor (float): Scaling factor for the influence of hints.

        Returns:
            torch.Tensor: Updated synaptic weights tensor.
        """
        original_weights = weights.clone()
        try:
            # --- Input Validation ---
            if pre_spikes_t.device != self.device or post_spikes_t.device != self.device or weights.device != self.device:
                raise ValueError("Input tensors must be on the same device as the STDP handler")
            if pre_spikes_t.shape[0] != self.num_pre or post_spikes_t.shape[0] != self.num_post or weights.shape != (self.num_pre, self.num_post):
                raise ValueError("Input tensor shapes do not match handler configuration")

            # --- Jitter Mitigation: Temporal Noise Filtering ---
            effective_pre_spikes = pre_spikes_t.float()
            effective_post_spikes = post_spikes_t.float()
            if spike_history_buffer and len(spike_history_buffer) >= self.temporal_filter_width - 1:
                relevant_history = spike_history_buffer[-(self.temporal_filter_width - 1):]
                pre_spike_stack = torch.stack([hist['pre'].float() for hist in relevant_history] + [pre_spikes_t.float()])
                post_spike_stack = torch.stack([hist['post'].float() for hist in relevant_history] + [post_spikes_t.float()])
                effective_pre_spikes = (torch.mean(pre_spike_stack, dim=0) > 0.5).float()
                effective_post_spikes = (torch.mean(post_spike_stack, dim=0) > 0.5).float()

            # --- Calculate Effective A+ based on Modulations ---
            a_plus_effective = self.a_plus_base.clone()
            if cluster_assignments is not None and cluster_rewards is not None:
                # Apply SIE modulation (using cluster rewards)
                if cluster_assignments.shape[0] != self.num_post: raise ValueError("cluster_assignments shape must match num_post")
                if cluster_rewards.ndim != 1: raise ValueError("cluster_rewards must be a 1D tensor")
                try:
                    post_cluster_rewards = cluster_rewards.to(self.device)[cluster_assignments]
                    reward_modulation = torch.clamp(post_cluster_rewards.unsqueeze(0), min=0.0, max=1.0)
                    a_plus_effective *= reward_modulation
                except Exception as e: print(f"Warning: Error during SIE modulation: {e}")
            if pre_spike_rates is not None:
                # Apply rate dependency modulation
                if pre_spike_rates.shape[0] != self.num_pre: raise ValueError("pre_spike_rates shape must match num_pre")
                safe_target_rate = self.target_rate if self.target_rate > 1e-6 else 1e-6
                rate_factor = torch.clamp(pre_spike_rates.unsqueeze(1).to(self.device) / safe_target_rate, min=0.1, max=2.0)
                a_plus_effective *= rate_factor

            # --- Jitter Mitigation: Adaptive STDP Window (Scale A+) ---
            adaptive_window_scale = 1.0 + 0.1 * torch.clamp(torch.tensor(max_latency / 20.0, device=self.device), max=1.0)
            a_plus_effective *= adaptive_window_scale

            # --- 1. Update Pre- and Post-Synaptic Traces ---
            self.pre_trace = self.pre_trace * self.decay_trace + effective_pre_spikes
            self.post_trace = self.post_trace * self.decay_trace + effective_post_spikes
            self.pre_trace.clamp_(max=1.0)
            self.post_trace.clamp_(max=1.0)

            # --- 2. Calculate STDP-based Eligibility Trace Updates ---
            pre_trace_expanded = self.pre_trace.unsqueeze(1)
            post_trace_expanded = self.post_trace.unsqueeze(0)
            # Use a_plus_effective for potentiation
            delta_eligibility_pot = a_plus_effective * pre_trace_expanded * effective_post_spikes.unsqueeze(0)
            # Use fixed a_minus for depression, triggered by *effective* pre-spike, depends on post-trace
            delta_eligibility_dep = -self.a_minus * post_trace_expanded * effective_pre_spikes.unsqueeze(1)
            delta_eligibility = delta_eligibility_pot + delta_eligibility_dep

            # --- Jitter Mitigation: Latency-Aware Scaling ---
            if max_latency > 0 and latency_error < max_latency:
                latency_scale_factor = torch.clamp(torch.tensor(1.0 - (latency_error / max_latency), device=self.device), min=0.5, max=1.0)
                delta_eligibility *= latency_scale_factor

            # --- STC Analogue: Tagging ---
            # Tag based on the potentiation component *before* latency scaling
            current_tags = (delta_eligibility_pot > self.stc_tag_threshold).bool()
            self.tag_history[:, :, self.tag_history_ptr] = current_tags
            self.tag_history_ptr = (self.tag_history_ptr + 1) % self.stc_consolidation_duration

            # --- 3. Update Eligibility Traces ---
            current_gamma = self._calculate_gamma(plv)
            self.eligibility_traces *= current_gamma
            self.eligibility_traces += delta_eligibility # Add potentially scaled delta

            # --- STC Analogue: Consolidation Check ---
            # TODO: Optimize this check.
            persistent_tags_mask = torch.sum(self.tag_history, dim=2) == self.stc_consolidation_duration
            consolidation_bonus = 0.1

            # --- 4. Calculate Final Weight Change ---
            # Calculate SIE modulation factor (Ref: 2C.3.ii)
            mod_factor = 2.0 * torch.sigmoid(torch.tensor(self.reward_sigmoid_scale * total_reward, device=self.device)) - 1.0
            # Calculate effective learning rate magnitude (Ref: 2C.3.iii)
            eta_magnitude = self.eta * (1.0 + mod_factor)

            # Core logic: Apply reward-modulated learning rate to eligibility trace
            # Direction depends on sign of reward, magnitude depends on eta_magnitude
            # Use torch.sign for robustness with zero reward
            reward_sign = torch.sign(torch.tensor(total_reward, device=self.device))
            # If reward is zero, delta should be zero (before bonus/noise)
            base_delta_weights = torch.where(reward_sign != 0,
                                             eta_magnitude * reward_sign * self.eligibility_traces,
                                             torch.zeros_like(self.eligibility_traces))

            # Add consolidation bonus (applied after reward modulation)
            base_delta_weights[persistent_tags_mask] += consolidation_bonus

            # Apply hint bias
            if hints is not None:
                if hints.shape != weights.shape: raise ValueError(f"Hints shape {hints.shape} must match weights shape {weights.shape}")
                # Scale hint by absolute magnitude of effective learning rate?
                hint_bias = hint_factor * hints.to(self.device) * torch.abs(eta_magnitude)
                delta_weights = base_delta_weights + hint_bias
            else:
                delta_weights = base_delta_weights

            # Add Exploration Noise
            stochastic_noise = 0.01 * torch.randn_like(weights)
            delta_weights += stochastic_noise
            # TODO: Implement Neutral Drift logic

            # --- 5. Apply Updates and Check Validity ---
            # Check for NaNs/Infs in the calculated delta *before* applying
            if torch.isnan(delta_weights).any() or torch.isinf(delta_weights).any():
                 print("ERROR: NaN or Inf detected in calculated delta_weights. Reverting step.")
                 return original_weights

            updated_weights = weights + delta_weights
            updated_weights.clamp_(min=self.w_min, max=self.w_max) # Apply clipping

            # Final check for NaNs/Infs *after* applying update and clipping
            if torch.isnan(updated_weights).any() or torch.isinf(updated_weights).any():
                print("ERROR: NaN or Inf detected in weights after update/clipping. Reverting step.")
                return original_weights

            return updated_weights # Return the updated weights

        # Correctly placed except block for the entire update logic
        except Exception as e:
            print(f"ERROR during STDP update: {e}. Skipping weight update for this step.")
            # TODO: Implement more robust logging (log_stdp_error)
            return original_weights # Return original weights on error


    def apply_synaptic_scaling(self, weights: torch.Tensor, is_excitatory_mask: torch.Tensor) -> torch.Tensor:
        """
        Applies synaptic scaling to normalize total excitatory input.
        Should be called periodically (e.g., every 1000 steps) AFTER STDP/SIE updates.
        Includes placeholder logic for gating/protection.
        Ref: 2B.7.ii

        Args:
            weights (torch.Tensor): Current weight matrix [num_pre, num_post].
            is_excitatory_mask (torch.Tensor): Boolean mask indicating which pre-synaptic neurons are excitatory [num_pre].

        Returns:
            torch.Tensor: Updated weights after scaling.
        """
        print("Applying Synaptic Scaling (Placeholder Implementation)...")
        is_excitatory_mask = is_excitatory_mask.to(self.device).bool()
        target_total_exc = 1.0

        # Vectorized approach
        excitatory_weights = weights[is_excitatory_mask, :] # Shape [num_exc, num_post]
        positive_excitatory_weights = torch.clamp(excitatory_weights, min=0) # Consider only positive weights for sum
        total_exc_per_post = torch.sum(positive_excitatory_weights, dim=0) # Shape [num_post]

        needs_scaling_mask = total_exc_per_post > target_total_exc
        if torch.any(needs_scaling_mask):
            # Avoid division by zero if total_exc is zero (shouldn't happen if needs_scaling_mask is true)
            safe_total_exc = torch.where(needs_scaling_mask, total_exc_per_post, torch.tensor(1.0, device=self.device))
            scale_factors = torch.ones_like(total_exc_per_post)
            scale_factors[needs_scaling_mask] = target_total_exc / safe_total_exc[needs_scaling_mask]

            # Expand scale_factors to match weights shape for broadcasting
            # Only apply to excitatory inputs to the specific post-synaptic neurons needing scaling
            # TODO: Add gating/protection logic here before applying scaling
            apply_scaling_mask_full = torch.zeros_like(weights, dtype=torch.bool)
            # Ensure is_excitatory_mask aligns with the first dimension of weights
            if is_excitatory_mask.sum() > 0: # Avoid error if no excitatory neurons
                 apply_scaling_mask_full[is_excitatory_mask, :] = needs_scaling_mask.unsqueeze(0).expand(is_excitatory_mask.sum(), self.num_post)


            # Apply scaling only where needed and only to positive weights
            positive_weights_mask = weights > 0
            final_scaling_mask = apply_scaling_mask_full & positive_weights_mask

            # Apply scaling using the expanded scale_factors
            weights[final_scaling_mask] *= scale_factors.unsqueeze(0).expand_as(weights)[final_scaling_mask]


            print(f"Synaptic Scaling Applied to {needs_scaling_mask.sum()} neurons.")

        weights.clamp_(min=self.w_min, max=self.w_max) # Ensure clipping after scaling
        return weights


# --- Remove old simulation runner code ---
# The simulation logic will now reside elsewhere (e.g., within the main training loop
# that uses the UnifiedNeuronModel).
