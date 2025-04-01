import torch
import torch.nn.functional as F

class SelfImprovementEngine:
    """
    Encapsulates the logic for the Self-Improvement Engine (SIE), calculating
    a reward signal based on Temporal Difference error, novelty, habituation,
    and self-benefit (homeostasis).
    """
    def __init__(self, config, device, input_dim, num_clusters):
        """
        Initializes the SelfImprovementEngine.

        Args:
            config (dict): Configuration dictionary containing SIE parameters.
                           Expected keys: 'gamma', 'td_alpha', 'novelty_history_size',
                                          'habituation_decay', 'target_variance',
                                          'reward_sigmoid_scale'.
            device (torch.device): The compute device (e.g., 'cuda', 'cpu').
            input_dim (int): The dimensionality of the input encodings.
            num_clusters (int): The number of clusters representing network states.
        """
        self.device = device
        self.gamma = config.get('gamma', 0.9)
        self.td_alpha = config.get('td_alpha', 0.1)
        self.novelty_history_size = config.get('novelty_history_size', 100)
        self.habituation_decay = config.get('habituation_decay', 0.95)
        self.target_var = config.get('target_variance', 0.05)
        self.reward_sigmoid_scale = config.get('reward_sigmoid_scale', 1.0) # Scale factor for sigmoid input

        # --- State Tensors ---
        # Value function per cluster state
        self.V_states = torch.zeros(num_clusters, device=self.device)

        # History for novelty calculation
        self.recent_inputs = torch.zeros((self.novelty_history_size, input_dim), device=self.device)
        self.novelty_history_idx = 0

        # Counters for habituation calculation (one per history slot)
        self.habituation_counters = torch.zeros(self.novelty_history_size, device=self.device)

        # History for self-benefit (homeostasis) calculation
        self.spike_rate_history_size = 1000 # As per documentation
        self.recent_spike_rates = torch.zeros(self.spike_rate_history_size, device=self.device) # Assuming scalar average rate for now
        self.spike_rate_history_idx = 0

        print(f"SIE Initialized on device: {self.device}")
        print(f"SIE Config: gamma={self.gamma}, td_alpha={self.td_alpha}, novelty_hist={self.novelty_history_size}, "
              f"habit_decay={self.habituation_decay}, target_var={self.target_var}, sigmoid_scale={self.reward_sigmoid_scale}")


    def _calculate_novelty(self, current_input_encoding):
        """Calculates novelty based on cosine similarity to recent inputs."""
        if self.recent_inputs.shape[1] != current_input_encoding.shape[0]:
             raise ValueError(f"Input encoding dimension mismatch in SIE Novelty. Expected {self.recent_inputs.shape[1]}, got {current_input_encoding.shape[0]}")

        # Ensure input is on the correct device and flattened
        current_input_encoding = current_input_encoding.to(self.device).view(1, -1)

        # Handle initial state where history is empty
        if self.novelty_history_idx < self.novelty_history_size:
             # Fill history first before calculating similarity meaningfully
             self.recent_inputs[self.novelty_history_idx] = current_input_encoding.squeeze()
             self.novelty_history_idx += 1
             # Return max novelty until history is somewhat populated
             return torch.tensor(1.0, device=self.device), torch.tensor(-1, device=self.device) # Max novelty, no match index

        # Calculate cosine similarity with history
        similarities = F.cosine_similarity(current_input_encoding, self.recent_inputs, dim=1)
        max_similarity, matched_idx = torch.max(similarities, dim=0)

        novelty = 1.0 - max_similarity

        # Update history (circular buffer)
        current_idx = self.novelty_history_idx % self.novelty_history_size
        self.recent_inputs[current_idx] = current_input_encoding.squeeze()
        # # Reset habituation counter for the replaced entry - REMOVED: This was incorrect logic.
        # self.habituation_counters[current_idx] = 0.0
        self.novelty_history_idx += 1


        return novelty, matched_idx

    def _calculate_habituation(self, matched_idx):
        """Calculates habituation based on repeated matches."""
        # Apply decay to all counters first
        self.habituation_counters *= self.habituation_decay

        habituation = torch.tensor(0.0, device=self.device)
        if matched_idx >= 0: # If a match was found in novelty calculation
            # Increment counter for the matched pattern, capped at 1
            self.habituation_counters[matched_idx] = torch.min(
                self.habituation_counters[matched_idx] + 0.1,
                torch.tensor(1.0, device=self.device)
            )
            habituation = self.habituation_counters[matched_idx]

        return habituation

    def _calculate_self_benefit(self, current_avg_spike_rate):
        """Calculates self-benefit based on homeostasis (variance of recent spike rates)."""
        # Ensure rate is a tensor on the correct device
        current_avg_spike_rate = torch.tensor(current_avg_spike_rate, device=self.device)

        # Update history (circular buffer)
        current_idx = self.spike_rate_history_idx % self.spike_rate_history_size
        self.recent_spike_rates[current_idx] = current_avg_spike_rate
        self.spike_rate_history_idx += 1

        # Calculate variance only if buffer is sufficiently full
        if self.spike_rate_history_idx < self.spike_rate_history_size:
            return torch.tensor(0.5, device=self.device) # Default value until history is full

        variance = torch.var(self.recent_spike_rates)
        # Avoid division by zero if target_var is 0 or variance is exactly target_var
        if self.target_var <= 1e-6:
             benefit = 1.0 if torch.abs(variance) < 1e-6 else 0.0
        else:
             benefit = 1.0 - torch.abs(variance - self.target_var) / self.target_var

        # Clamp to [0, 1]
        self_benefit = torch.clamp(benefit, 0.0, 1.0)

        return self_benefit

    def _calculate_td_error(self, current_cluster_id, next_cluster_id, external_reward):
        """Calculates the Temporal Difference error and updates the value function."""
        if current_cluster_id is None or next_cluster_id is None:
             # Cannot calculate TD error without valid states
             return torch.tensor(0.0, device=self.device)

        # Ensure IDs are valid indices
        if not (0 <= current_cluster_id < len(self.V_states) and 0 <= next_cluster_id < len(self.V_states)):
             print(f"Warning: Invalid cluster ID received in SIE TD calculation. Current: {current_cluster_id}, Next: {next_cluster_id}. Max index: {len(self.V_states)-1}")
             # Option 1: Return 0 error
             # return torch.tensor(0.0, device=self.device)
             # Option 2: Clamp IDs (might hide issues)
             current_cluster_id = max(0, min(current_cluster_id, len(self.V_states)-1))
             next_cluster_id = max(0, min(next_cluster_id, len(self.V_states)-1))


        V_current = self.V_states[current_cluster_id]
        V_next = self.V_states[next_cluster_id] # V(s')

        # Ensure external_reward is a tensor
        external_reward = torch.tensor(external_reward, device=self.device)

        td_error = external_reward + self.gamma * V_next - V_current

        # Update value function V(s)
        self.V_states[current_cluster_id] += self.td_alpha * td_error

        return td_error

    def calculate_total_reward(self, current_input_encoding, current_avg_spike_rate, current_cluster_id, next_cluster_id, external_reward=0):
        """
        Calculates the total SIE reward by combining its components.

        Args:
            current_input_encoding (torch.Tensor): Encoded input for the current step.
            current_avg_spike_rate (float): Average spike rate of the network.
            current_cluster_id (int): Index of the current network state cluster.
            next_cluster_id (int): Index of the next network state cluster.
            external_reward (float, optional): External reward signal, if available. Defaults to 0.

        Returns:
            torch.Tensor: The calculated total reward.
        """
        novelty, matched_idx = self._calculate_novelty(current_input_encoding)
        habituation = self._calculate_habituation(matched_idx)
        self_benefit = self._calculate_self_benefit(current_avg_spike_rate)
        td_error = self._calculate_td_error(current_cluster_id, next_cluster_id, external_reward)

        # Combine components (as per documentation formula)
        total_reward = td_error + novelty - habituation + self_benefit

        # Optional: Add logging for component values
        # print(f"SIE Components: TD={td_error.item():.4f}, Nov={novelty.item():.4f}, Hab={habituation.item():.4f}, SB={self_benefit.item():.4f} -> Total={total_reward.item():.4f}")

        return total_reward

    def get_modulation_factor(self, total_reward):
        """
        Maps the total reward to a modulation factor in [-1, 1] using a scaled sigmoid.
        """
        # Scale the input to the sigmoid
        scaled_reward = total_reward * self.reward_sigmoid_scale
        mod_factor = 2 * torch.sigmoid(scaled_reward) - 1
        return mod_factor

# Example Usage (for basic verification)
if __name__ == '__main__':
    print("Running basic SIE verification...")
    mock_config = {
        'gamma': 0.9,
        'td_alpha': 0.1,
        'novelty_history_size': 5,
        'habituation_decay': 0.9,
        'target_variance': 0.05,
        'reward_sigmoid_scale': 1.0
    }
    mock_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mock_input_dim = 10
    mock_num_clusters = 3

    sie = SelfImprovementEngine(mock_config, mock_device, mock_input_dim, mock_num_clusters)

    # Simulate some steps
    inputs = [torch.randn(mock_input_dim) for _ in range(10)]
    rates = [0.1, 0.15, 0.2, 0.18, 0.22, 0.1, 0.05, 0.1, 0.12, 0.15]
    clusters = [(0, 1), (1, 1), (1, 2), (2, 0), (0, 1), (1, 0), (0, 0), (0, 1), (1, 2), (2, 2)]
    rewards = [0, 0, 1, 0, 0, -1, 0, 0, 0, 1] # External rewards

    print("\nSimulating steps:")
    for i in range(10):
        print(f"\n--- Step {i+1} ---")
        print(f"Input: {i+1}, Rate: {rates[i]}, Clusters: {clusters[i]}, Ext Reward: {rewards[i]}")
        total_reward = sie.calculate_total_reward(
            inputs[i],
            rates[i],
            clusters[i][0], # current_cluster_id
            clusters[i][1], # next_cluster_id
            rewards[i]
        )
        mod_factor = sie.get_modulation_factor(total_reward)
        print(f"Total Reward: {total_reward.item():.4f}, Mod Factor: {mod_factor.item():.4f}")
        print(f"Value States: {sie.V_states.cpu().numpy()}")
        print(f"Habituation Counters: {sie.habituation_counters.cpu().numpy()}")

    print("\nBasic SIE verification complete.")
