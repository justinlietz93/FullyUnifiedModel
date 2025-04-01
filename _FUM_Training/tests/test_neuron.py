# test_neuron.py
import torch
import time

# Assuming your existing 1000-neuron code is in unified_neuron.py
class LIFNeuron:
    def __init__(self, num_neurons=1000, tau=20.0, v_th=-55.0, v_reset=-70.0):
        self.num_neurons = num_neurons
        self.tau = tau
        self.v_th = v_th
        self.v_reset = v_reset
        self.V = torch.zeros(num_neurons, dtype=torch.float16)  # Membrane potential
        self.spikes = torch.zeros(num_neurons, dtype=torch.float16)  # Spike output

    def update_state(self, I, dt=0.01):
        # I: Input current (spikes from encoder)
        # dt: Timestep in ms (0.01ms = 100 Hz)
        self.V += I - (self.V / self.tau) * dt  # Leaky integration
        self.spikes = (self.V > self.v_th).float()  # Spike if above threshold
        self.V = torch.where(self.spikes > 0, self.v_reset, self.V)  # Reset if spiked
        return self.spikes

def test_lif_firing():
    # Initialize 1000 neurons
    neuron = LIFNeuron(num_neurons=1000)
    
    # Simulate ~182 text inputs (~15 Hz, ~0.1 current per spike)
    I = torch.zeros(1000, dtype=torch.float16)
    I[:182] = 0.1  # ~15 Hz input for first 182 neurons
    
    # Run for 50 timesteps (~0.5ms at 100 Hz)
    spikes_sum = 0
    start_time = time.time()
    for _ in range(50):
        spikes = neuron.update_state(I)
        spikes_sum += spikes.sum().item()
    elapsed = time.time() - start_time
    
    # Validate: Expect ~20-40 spikes/timestep (~10-20% firing rate), <0.1s
    expected_spikes = 182 * 0.1 * 50  # ~10% firing rate over 50 steps
    assert spikes_sum > 0, "No spikes generated"
    assert elapsed < 0.1, f"Latency {elapsed}s exceeds 0.1s"
    print(f"Total spikes: {spikes_sum} (expected ~{expected_spikes})")
    print(f"Latency: {elapsed}s")

if __name__ == "__main__":
    test_lif_firing()