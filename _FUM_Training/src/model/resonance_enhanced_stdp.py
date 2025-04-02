# File: resonance_enhanced_stdp.py (REVISED for Tensor Operations and Trace-Based STDP)
# Purpose: Implement Resonance-Enhanced STDP using PyTorch tensors,
#          modulating eligibility trace decay (gamma) based on PLV.
#          Uses pre/post synaptic traces for STDP calculation.
#          Includes constrained biological diversity.

import torch
from typing import Optional, Dict, Any

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
    Ref: How_It_Works/2_Core_Architecture_Components/2B_Neural_Plasticity.md (esp. B.4.iv)
    """
    def __init__(self, num_pre: int, num_post: int,
                 eta: float = 0.01, a_plus: float = 0.1, a_minus: float = 0.05, # Note: a_plus is now base mean
                 tau_plus: float = 20.0, tau_minus: float = 20.0, # Note: tau_plus is now base mean
                 tau_trace: float = 25.0, # Time constant for pre/post synaptic traces (ms)
                 gamma_min: float = 0.50, gamma_max: float = 0.95, # Eligibility trace decay range
                 plv_k: float = 10.0, plv_mid: float = 0.5,       # Gamma sigmoid params
                 w_min: float = -1.0, w_max: float = 1.0,
                 target_rate: float = 0.3, # Hz, for rate dependency modulation (Ref: 2B.4.iv)
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
            device (torch.device): The compute device to use for tensors.
        """
        self.num_pre = num_pre
        self.num_post = num_post
        self.eta = eta
        # self.a_plus = a_plus # Replaced by dynamic calculation
        self.a_minus = a_minus # Keep a_minus fixed for now
        self.target_rate = target_rate # Store target rate for modulation

        # --- Initialize Base Parameters with Constrained Variability (Ref: 2B.4.iv) ---
        # A_plus_base: Mean a_plus, Std 0.05*a_plus (relative std dev), Clamped [0.05, 0.15] (absolute bounds)
        # Using absolute std dev for simplicity now, as in docs example
        a_plus_std_dev = 0.05 # As per docs example
        a_plus_base_unclamped = torch.normal(mean=a_plus, std=a_plus_std_dev, size=(num_pre, num_post), device=device, dtype=torch.float32)
        self.a_plus_base = torch.clamp(a_plus_base_unclamped, min=0.05, max=0.15)

        # Tau_plus_base: Mean tau_plus, Std 5ms, Clamped [15ms, 25ms]
        tau_plus_std_dev = 5.0 # ms, as per docs example
        tau_plus_base_unclamped = torch.normal(mean=tau_plus, std=tau_plus_std_dev, size=(num_pre, num_post), device=device, dtype=torch.float32)
        self.tau_plus_base = torch.clamp(tau_plus_base_unclamped, min=15.0, max=25.0)
        # TODO: If tau_plus becomes heterogeneous, decay_plus needs dynamic calculation in update()

        # Precompute fixed decay factors
        self.tau_minus = tau_minus # Keep tau_minus fixed for now
        self.decay_minus = torch.exp(torch.tensor(-1.0 / self.tau_minus, device=device))
        self.tau_trace = tau_trace # Keep tau_trace fixed for now
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
        # Eligibility traces (one per synapse)
        self.eligibility_traces = torch.zeros((num_pre, num_post), device=self.device, dtype=torch.float32)
        # Pre-synaptic traces (one per pre-neuron)
        self.pre_trace = torch.zeros(num_pre, device=self.device, dtype=torch.float32)
        # Post-synaptic traces (one per post-neuron)
        self.post_trace = torch.zeros(num_post, device=self.device, dtype=torch.float32)

        # --- STC Analogue State (Ref: 2B.5.ix) ---
        self.stc_tag_threshold = 0.05 # Δw threshold for tagging
        self.stc_consolidation_duration = 100000 # Timesteps tag must persist
        # Store recent tags efficiently (e.g., using a buffer or packed bits if memory is tight)
        # For simplicity now, store full history (memory intensive!)
        # TODO: Optimize tag history storage
        self.tag_history = torch.zeros((num_pre, num_post, self.stc_consolidation_duration), dtype=torch.bool, device=self.device)
        self.tag_history_ptr = 0 # Pointer for circular buffer logic

        print(f"Initialized ResonanceEnhancedSTDP_TraceModulation_Tensor on device: {self.device}")
        print(f"  A+ Base Mean: {self.a_plus_base.mean():.4f}, Std: {self.a_plus_base.std():.4f}")
        print(f"  Tau+ Base Mean: {self.tau_plus_base.mean():.4f}, Std: {self.tau_plus_base.std():.4f}")


    def _calculate_gamma(self, plv: float) -> float:
        """Calculates PLV-dependent eligibility trace decay factor using a sigmoid function."""
        # Using torch equivalents
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) / (1 + torch.exp(-self.plv_k * (torch.tensor(plv, device=self.device) - self.plv_mid)))
        return torch.clip(gamma, self.gamma_min, self.gamma_max).item() # Return scalar

    @torch.no_grad() # Disable gradient tracking for performance if not training with autograd
    def update(self, pre_spikes_t: torch.Tensor, post_spikes_t: torch.Tensor,
               weights: torch.Tensor, plv: float, reward_signal: float = 1.0,
               # Inputs for biological diversity modulation (Ref: 2B.4.iv)
               pre_spike_rates: Optional[torch.Tensor] = None, # Shape [num_pre]
               cluster_assignments: Optional[torch.Tensor] = None, # Shape [num_post] - Post-synaptic neuron's cluster ID
               cluster_rewards: Optional[torch.Tensor] = None, # Shape [num_clusters] - Avg reward per cluster
               # Optional hints
               hints: Optional[torch.Tensor] = None, hint_factor: float = 0.1) -> torch.Tensor:
        """
        Updates synaptic traces, eligibility traces, and weights for a single timestep.
        Incorporates constrained biological diversity and optional hints.

        Args:
            pre_spikes_t (torch.Tensor): Boolean tensor indicating pre-synaptic spikes at current time t (shape: [num_pre]).
            post_spikes_t (torch.Tensor): Boolean tensor indicating post-synaptic spikes at current time t (shape: [num_post]).
            weights (torch.Tensor): Current synaptic weights tensor (shape: [num_pre, num_post]).
            plv (float): Phase-locking value (PLV) for this update context.
            reward_signal (float): Pre-modulated reward signal (incorporating eta and SIE modulation).
            pre_spike_rates (Optional[torch.Tensor]): Firing rates of pre-synaptic neurons (Hz). Shape [num_pre].
            cluster_assignments (Optional[torch.Tensor]): Cluster ID for each post-synaptic neuron. Shape [num_post].
            cluster_rewards (Optional[torch.Tensor]): Average reward for each cluster. Shape [num_clusters].
            hints (Optional[torch.Tensor]): Optional tensor of hints biasing weight changes (shape: [num_pre, num_post]).
            hint_factor (float): Scaling factor for the influence of hints.

        Returns:
            torch.Tensor: Updated synaptic weights tensor.
        """
        if pre_spikes_t.device != self.device or post_spikes_t.device != self.device or weights.device != self.device:
             raise ValueError("Input tensors must be on the same device as the STDP handler.")
        if pre_spikes_t.shape[0] != self.num_pre or post_spikes_t.shape[0] != self.num_post or weights.shape != (self.num_pre, self.num_post):
             raise ValueError("Input tensor shapes do not match handler configuration.")

        # --- Calculate Effective A+ based on Modulations (Ref: 2B.4.iv) ---
        a_plus_effective = self.a_plus_base.clone() # Start with base heterogeneous values

        # Apply SIE modulation (neuromodulatory constraint)
        # A_+ = A_+_base * (cluster_reward[c] / max_reward)
        if cluster_assignments is not None and cluster_rewards is not None:
            if cluster_assignments.shape[0] != self.num_post:
                 raise ValueError("cluster_assignments shape must match num_post.")
            if cluster_rewards.ndim != 1:
                 raise ValueError("cluster_rewards must be a 1D tensor.")
            try:
                # Get reward for the cluster of each post-synaptic neuron
                # Ensure cluster_assignments are valid indices for cluster_rewards
                post_cluster_rewards = cluster_rewards.to(self.device)[cluster_assignments] # Shape [num_post]
                # Expand for broadcasting: [1, num_post]
                post_cluster_rewards_expanded = post_cluster_rewards.unsqueeze(0)
                # Assuming max_reward = 1.0, clamp reward to avoid negative A+ or excessive values
                reward_modulation = torch.clamp(post_cluster_rewards_expanded, min=0.0, max=1.0) # Ensure modulation factor is reasonable
                a_plus_effective *= reward_modulation
                # print(f"DEBUG: Reward Mod mean: {reward_modulation.mean():.4f}") # Debug
            except IndexError as e:
                 print(f"Warning: Cluster assignment index out of bounds: {e}. Skipping SIE modulation.")
            except Exception as e:
                 print(f"Warning: Error during SIE modulation: {e}. Skipping SIE modulation.")


        # Apply rate dependency modulation
        # A_+ *= spike_rate[i] / target_rate
        if pre_spike_rates is not None:
            if pre_spike_rates.shape[0] != self.num_pre:
                 raise ValueError("pre_spike_rates shape must match num_pre.")
            # Expand for broadcasting: [num_pre, 1]
            pre_rates_expanded = pre_spike_rates.unsqueeze(1).to(self.device)
            # Clamp rate factor to avoid extreme values (e.g., division by zero or huge rates)
            # Ensure target_rate is not zero
            safe_target_rate = self.target_rate if self.target_rate != 0 else 1e-6
            rate_factor = torch.clamp(pre_rates_expanded / safe_target_rate, min=0.1, max=2.0) # Example clamping
            a_plus_effective *= rate_factor
            # print(f"DEBUG: Rate Factor mean: {rate_factor.mean():.4f}") # Debug

        # TODO: Optional: Apply synapse-specific variability based on location
        # TODO: Optional: Apply neuron-type dependency for tau_plus (would require dynamic decay_plus)

        # --- 1. Update Pre- and Post-Synaptic Traces ---
        # Decay existing traces
        self.pre_trace *= self.decay_trace
        self.post_trace *= self.decay_trace
        # Increase traces for neurons that spiked (add 1.0)
        # Use float() to ensure compatibility if spikes are bool
        self.pre_trace += pre_spikes_t.float() # Add 1 if spiked
        self.post_trace += post_spikes_t.float() # Add 1 if spiked
        # Clamp traces to avoid unbounded growth (optional, but good practice)
        self.pre_trace.clamp_(max=1.0)
        self.post_trace.clamp_(max=1.0)


        # --- 2. Calculate STDP-based Eligibility Trace Updates ---
        # Potentiation: Occurs when post-synaptic neuron fires. Depends on pre-synaptic trace.
        # Depression: Occurs when pre-synaptic neuron fires. Depends on post-synaptic trace.

        # Expand traces for broadcasting: pre [pre, 1], post [1, post]
        pre_trace_expanded = self.pre_trace.unsqueeze(1)
        post_trace_expanded = self.post_trace.unsqueeze(0)

        # Calculate potential change contributions at synapses
        # Potentiation contribution where post spiked: a_plus_effective * pre_trace
        # Using the dynamically calculated a_plus_effective
        delta_eligibility_pot = a_plus_effective * pre_trace_expanded * post_spikes_t.unsqueeze(0).float()
        # Depression contribution where pre spiked: -a_minus * post_trace (a_minus is fixed for now)
        delta_eligibility_dep = -self.a_minus * post_trace_expanded * pre_spikes_t.unsqueeze(1).float()

        # Combine contributions (potentiation where post fired, depression where pre fired)
        delta_eligibility = delta_eligibility_pot + delta_eligibility_dep

        # --- STC Analogue: Tagging (Ref: 2B.5.ix) ---
        # Tag synapses where potentiation contribution exceeds threshold
        current_tags = (delta_eligibility_pot > self.stc_tag_threshold).bool()

        # Update tag history (using circular buffer logic)
        self.tag_history[:, :, self.tag_history_ptr] = current_tags
        self.tag_history_ptr = (self.tag_history_ptr + 1) % self.stc_consolidation_duration

        # --- 3. Update Eligibility Traces (Incorporate STC Tag) ---
        # Decay existing eligibility traces based on PLV
        current_gamma = self._calculate_gamma(plv)
        self.eligibility_traces *= current_gamma
        # Add the new contributions from this timestep, modulated by tag
        # Only tagged potentiation contributes fully? Or scale based on tag?
        # Docs: e_ij(t) = ... + Δw_ij(t) * tag_ij(t) -> This implies scaling the *entire* delta_eligibility by the tag?
        # Let's reinterpret: Add delta_eligibility normally, but tag influences consolidation/final update.
        # Original plan: Modify trace update. Let's stick to that for now as per docs interpretation.
        # This means only add delta_eligibility if tag is true? Seems too harsh.
        # Alternative: Add delta_eligibility normally, tag is used for consolidation bonus later.
        # Let's implement the trace update modification as per B.5.ix first:
        # self.eligibility_traces += delta_eligibility * current_tags.float() # Scale contribution by tag
        # Re-evaluating B.5.ix: "Modify trace update ... to incorporate the tag, focusing reinforcement on tagged synapses"
        # This is ambiguous. Let's revert to the simpler interpretation: update traces normally, use tags for consolidation bonus.
        self.eligibility_traces += delta_eligibility # Add full delta

        # --- STC Analogue: Consolidation Check (Ref: 2B.5.ix) ---
        # TODO: This check is very expensive (summing over history). Optimize or run less frequently.
        # Check if tags have been active for the required duration
        # Sum boolean history along the time dimension
        persistent_tags_mask = torch.sum(self.tag_history, dim=2) == self.stc_consolidation_duration
        consolidation_bonus = 0.1 # As per docs example

        # --- 4. Update Weights ---
        # Apply weight updates using eligibility traces and the pre-modulated reward signal.
        # The reward_signal already incorporates eta and the SIE modulation factor.
        base_delta_weights = reward_signal * self.eligibility_traces

        # Add consolidation bonus where tags were persistent
        # Ensure bonus is only added once after consolidation? Need state tracking?
        # For now, add bonus if mask is true.
        # TODO: Refine consolidation logic (e.g., add only once, reset history after consolidation?)
        base_delta_weights[persistent_tags_mask] += consolidation_bonus

        # Apply hint bias if hints are provided
        if hints is not None:
            if hints.shape != weights.shape:
                 raise ValueError(f"Hints shape {hints.shape} must match weights shape {weights.shape}")
            # Simple additive bias based on hint value and factor
            # Note: Ensure hints are on the correct device
            # Let's try scaling by reward signal magnitude to make hints stronger when reward is significant
            hint_bias = hint_factor * hints.to(self.device) * torch.abs(torch.tensor(reward_signal, device=self.device)) # Ensure reward_signal is tensor for abs

            delta_weights = base_delta_weights + hint_bias
            # print(f"Hint bias applied: mean={hint_bias.mean().item():.4f}") # Debug
        else:
            delta_weights = base_delta_weights

        # --- Add Exploration Noise (Ref: 2B.8.iii) ---
        # Stochastic STDP (Random Mutation Analogue) - Add small Gaussian noise
        stochastic_noise = 0.01 * torch.randn_like(weights)
        delta_weights += stochastic_noise

        # Neutral Drift (Add smaller noise if reward is high and stable - requires reward/stability info)
        # TODO: Implement Neutral Drift logic - requires reward/stability context passed to update()
        # Example placeholder:
        # if reward_signal > 0.9 * self.eta: # Check if effective reward is high
        #     neutral_noise = 0.005 * torch.randn_like(weights)
        #     delta_weights += neutral_noise

        weights += delta_weights

        # Apply weight clipping
        weights.clip_(min=self.w_min, max=self.w_max)

        return weights

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
        # Ensure mask is boolean and on the correct device
        is_excitatory_mask = is_excitatory_mask.to(self.device).bool()

        # Iterate over post-synaptic neurons (can be vectorized)
        for j in range(self.num_post):
            # Select incoming excitatory weights for neuron j
            excitatory_weights_j = weights[is_excitatory_mask, j]
            positive_excitatory_weights_j = excitatory_weights_j[excitatory_weights_j > 0]

            total_exc = torch.sum(positive_excitatory_weights_j)
            target_total_exc = 1.0 # Target sum for excitatory inputs

            if total_exc > target_total_exc:
                # --- Gating/Protection Logic (Placeholders) ---
                # TODO: Implement actual gating based on reward stability, update recency etc.
                apply_scaling = True # Assume scaling should be applied for now

                if apply_scaling:
                    scale_factor = target_total_exc / total_exc
                    # Apply scaling only to positive excitatory weights
                    # TODO: Potentially protect strong weights (w > 0.8)
                    weights[is_excitatory_mask, j][excitatory_weights_j > 0] *= scale_factor

        # Ensure weights remain clipped after scaling
        weights.clip_(min=self.w_min, max=self.w_max)
        print("Synaptic Scaling Applied.")
        return weights


# --- Remove old simulation runner code ---
# The simulation logic will now reside elsewhere (e.g., within the main training loop
# that uses the UnifiedNeuronModel).
