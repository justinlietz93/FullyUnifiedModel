import unittest
import torch
import numpy as np
import sys
import os

# Add src directory to path to allow importing UnifiedNeuronModel
# Assumes this test script is in _FUM_Training/tests/
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import the class to be tested
# Note: This might fail if dependencies like config_loader aren't set up correctly
# or if the environment doesn't match where UnifiedNeuronModel expects to run.
try:
    # Import necessary components including the global variable
    from neuron.unified_neuron import UnifiedNeuronModel, initialize_device, COMPUTE_BACKEND
    # Initialize device once for all tests in this module
    TEST_DEVICE = initialize_device()
    print(f"Test device initialized to: {TEST_DEVICE}")
    _model_import_success = True
except ImportError as e:
    print(f"ERROR: Failed to import UnifiedNeuronModel: {e}")
    print("Ensure you are running tests from the project root or have configured PYTHONPATH.")
    _model_import_success = False
    # Define a dummy class if import fails, so tests can be skipped
    class UnifiedNeuronModel:
        def __init__(self, *args, **kwargs): pass
        def update_state(self, *args, **kwargs): return torch.tensor([])
        def apply_intrinsic_plasticity(self, *args, **kwargs): pass
    TEST_DEVICE = torch.device('cpu') # Fallback device

# --- Mock Configuration for Tests ---
MOCK_CONFIG = {
    'stdp_params': {}, # STDP not tested here
    'sie_params': {}   # SIE not tested here
}
MOCK_INPUT_DIM = 10 # Dummy value
MOCK_NUM_CLUSTERS = 5 # Dummy value

@unittest.skipIf(not _model_import_success, "UnifiedNeuronModel could not be imported.")
class TestUnifiedNeuronModel(unittest.TestCase):

    def setUp(self):
        """Set up common resources for tests."""
        self.num_neurons = 1000
        self.dt = 1.0 # ms
        # Ensure model uses the globally initialized device
        self.model = UnifiedNeuronModel(
            num_neurons=self.num_neurons,
            config=MOCK_CONFIG,
            input_dim=MOCK_INPUT_DIM,
            num_clusters=MOCK_NUM_CLUSTERS,
            dt=self.dt
        )
        # Ensure model attributes are on the correct device
        self.model.tau = self.model.tau.to(TEST_DEVICE)
        self.model.v_th = self.model.v_th.to(TEST_DEVICE)
        self.model.voltage = self.model.voltage.to(TEST_DEVICE)
        self.model.spike_counts = self.model.spike_counts.to(TEST_DEVICE)


    def test_initialization(self):
        """Test basic model initialization."""
        self.assertEqual(self.model.num_neurons, self.num_neurons)
        self.assertEqual(self.model.dt, self.dt)
        self.assertEqual(self.model.voltage.shape[0], self.num_neurons)
        self.assertEqual(self.model.tau.shape[0], self.num_neurons)
        self.assertEqual(self.model.v_th.shape[0], self.num_neurons)
        self.assertEqual(self.model.voltage.device, TEST_DEVICE)
        self.assertEqual(self.model.tau.device, TEST_DEVICE)
        self.assertEqual(self.model.v_th.device, TEST_DEVICE)

    def test_heterogeneity_initialization(self):
        """Test initialization of heterogeneous parameters tau and v_th."""
        # Check shapes
        self.assertEqual(self.model.tau.shape, (self.num_neurons,))
        self.assertEqual(self.model.v_th.shape, (self.num_neurons,))

        # Check means (should be close to the specified means)
        # Use float32 for calculations, convert model params if needed
        tau_mean = self.model.tau.float().mean().item()
        v_th_mean = self.model.v_th.float().mean().item()
        self.assertAlmostEqual(tau_mean, 20.0, delta=1.0) # Mean tau around 20ms
        self.assertAlmostEqual(v_th_mean, -55.0, delta=1.0) # Mean v_th around -55mV

        # Check standard deviations (should be non-zero and roughly match)
        tau_std = self.model.tau.float().std().item()
        v_th_std = self.model.v_th.float().std().item()
        self.assertGreater(tau_std, 0.5) # Check std dev is reasonable
        self.assertLess(tau_std, 4.0)
        self.assertGreater(v_th_std, 0.5)
        self.assertLess(v_th_std, 4.0)

        # Check that not all values are identical (confirming heterogeneity)
        self.assertGreater(len(torch.unique(self.model.tau)), 1)
        self.assertGreater(len(torch.unique(self.model.v_th)), 1)

    def test_lif_update_and_firing(self):
        """Test the LIF update logic and spike generation."""
        # Constant supra-threshold current
        # If V_th=-55, V_rest=-65, R=10, tau=20, steady V = -65 + 10*I
        # To reach -55, need 10*I = 10, so I=1.0 nA
        input_current = torch.full((self.num_neurons,), 1.1, device=TEST_DEVICE) # Slightly above threshold

        # Initial state
        self.model.voltage[:] = self.model.v_reset # Start all at reset potential

        # Simulate one step
        spikes = self.model.update_state(input_current)

        # Check voltage increase (should increase towards steady state > v_th)
        # dV = (-(V_reset - V_rest) + R*I) / tau * dt
        # dV = (-(-70 - (-65)) + 10*1.1) / 20 * 1.0 = (5 + 11) / 20 = 16/20 = 0.8 mV
        expected_v = self.model.v_reset + 0.8
        # Need to handle potential device mismatch if comparing directly
        self.assertTrue(torch.all(self.model.voltage.cpu() > self.model.v_reset)) # Voltage should increase

        # Simulate enough steps to guarantee firing for most neurons
        fired_mask = torch.zeros_like(spikes)
        for _ in range(int(self.model.tau.max() * 2 / self.dt)): # Simulate ~2*tau max
            spikes = self.model.update_state(input_current)
            fired_mask = fired_mask | spikes
            if fired_mask.all(): # Stop if all neurons fired at least once
                break

        self.assertGreater(fired_mask.sum().item(), 0, "No neurons fired with supra-threshold current.")
        # Check reset potential
        self.assertTrue(torch.all(self.model.voltage[spikes].cpu() == self.model.v_reset),
                        "Voltage of spiking neurons not reset correctly.")

    def test_intrinsic_plasticity_adjustments(self):
        """Test the intrinsic plasticity adjustment logic."""
        initial_v_th_mean = self.model.v_th.mean().item()
        initial_tau_mean = self.model.tau.mean().item()

        # --- Test Hyperactive Case ---
        # Force high firing rate by setting voltage above threshold
        self.model.voltage[:] = self.model.v_th.max() + 1.0 # Ensure all fire
        self.model.spike_counts.zero_() # Reset counts
        self.model.steps_since_ip_update = self.model.ip_update_interval -1 # Trigger IP on next step

        # Simulate one step to trigger IP
        _ = self.model.update_state(torch.zeros_like(self.model.voltage)) # Zero current

        # Check adjustments for hyperactive neurons (all should be hyperactive)
        # v_th should increase, tau should decrease
        self.assertGreater(self.model.v_th.mean().item(), initial_v_th_mean)
        self.assertLess(self.model.tau.mean().item(), initial_tau_mean)
        # Check clamping
        self.assertTrue(torch.all(self.model.v_th >= self.model.ip_v_th_bounds[0]))
        self.assertTrue(torch.all(self.model.v_th <= self.model.ip_v_th_bounds[1]))
        self.assertTrue(torch.all(self.model.tau >= self.model.ip_tau_bounds[0]))
        self.assertTrue(torch.all(self.model.tau <= self.model.ip_tau_bounds[1]))

        # --- Test Hypoactive Case ---
        # Reset params and force low firing rate (no spikes)
        self.setUp() # Re-initialize model to reset params
        initial_v_th_mean = self.model.v_th.mean().item()
        initial_tau_mean = self.model.tau.mean().item()
        self.model.spike_counts.zero_()
        self.model.steps_since_ip_update = self.model.ip_update_interval -1

        # Simulate one step with zero current (no spikes)
        _ = self.model.update_state(torch.zeros_like(self.model.voltage))

        # Check adjustments for hypoactive neurons (all should be hypoactive)
        # v_th should decrease, tau should increase
        self.assertLess(self.model.v_th.mean().item(), initial_v_th_mean)
        self.assertGreater(self.model.tau.mean().item(), initial_tau_mean)
        # Check clamping
        self.assertTrue(torch.all(self.model.v_th >= self.model.ip_v_th_bounds[0]))
        self.assertTrue(torch.all(self.model.v_th <= self.model.ip_v_th_bounds[1]))
        self.assertTrue(torch.all(self.model.tau >= self.model.ip_tau_bounds[0]))
        self.assertTrue(torch.all(self.model.tau <= self.model.ip_tau_bounds[1]))

        # Check counter reset
        self.assertEqual(self.model.steps_since_ip_update, 0)

    @unittest.skipUnless(COMPUTE_BACKEND == 'amd_hip', "Skipping HIP kernel test (backend is not amd_hip)")
    def test_hip_kernel_equivalence(self):
        """Test if HIP kernel output matches PyTorch implementation for a few steps."""
        print("\nTesting HIP kernel equivalence...")
        # Use a smaller number of neurons for faster test
        num_test_neurons_hip = 128
        dt_hip = 1.0
        model_hip = UnifiedNeuronModel(
            num_neurons=num_test_neurons_hip, config=MOCK_CONFIG,
            input_dim=MOCK_INPUT_DIM, num_clusters=MOCK_NUM_CLUSTERS, dt=dt_hip
        )
        # Ensure tensors are on the correct device (should be HIP device)
        model_hip.voltage = model_hip.voltage.to(TEST_DEVICE)
        model_hip.tau = model_hip.tau.to(TEST_DEVICE)
        model_hip.v_th = model_hip.v_th.to(TEST_DEVICE)

        # Consistent input current
        input_current_hip = torch.rand(num_test_neurons_hip, device=TEST_DEVICE) * 2.0 # Random currents

        # --- Run HIP Path ---
        # Reset state
        model_hip.voltage[:] = model_hip.v_reset
        hip_voltages = []
        hip_spikes_list = []
        try:
            # Force backend check within update_state to use HIP path
            # This relies on the global COMPUTE_BACKEND being 'amd_hip'
            for _ in range(5): # Simulate a few steps
                spikes = model_hip.update_state(input_current_hip)
                hip_voltages.append(model_hip.voltage.clone().cpu())
                hip_spikes_list.append(spikes.clone().cpu())
            hip_final_voltage = model_hip.voltage.clone().cpu()
            print("HIP path executed.")
        except Exception as e:
            self.fail(f"HIP kernel execution failed: {e}")


        # --- Run PyTorch Path (Simulate CPU/CUDA fallback for comparison) ---
        # Create a new model instance or reset state carefully
        # Easiest is often a new instance, forcing CPU/CUDA path if possible
        # Forcing requires modifying the global or passing backend override (not implemented)
        # Alternative: Manually implement the PyTorch logic here for comparison
        print("Simulating PyTorch path for comparison...")
        voltage_py = torch.full((num_test_neurons_hip,), model_hip.v_reset, device='cpu', dtype=torch.float32)
        # Use the same initial heterogeneous params from model_hip, moved to CPU
        tau_py = model_hip.tau.clone().cpu()
        v_th_py = model_hip.v_th.clone().cpu()
        v_rest_py = model_hip.v_rest
        r_mem_py = model_hip.r_mem
        v_reset_py = model_hip.v_reset
        input_current_py = input_current_hip.clone().cpu()

        py_voltages = []
        py_spikes_list = []
        for _ in range(5):
            dV = (-(voltage_py - v_rest_py) + r_mem_py * input_current_py) / tau_py * dt_hip
            voltage_py += dV
            spikes_py = voltage_py >= v_th_py
            voltage_py[spikes_py] = v_reset_py
            py_voltages.append(voltage_py.clone())
            py_spikes_list.append(spikes_py.clone())
        py_final_voltage = voltage_py.clone()
        print("PyTorch path simulated.")

        # --- Compare Results ---
        # Compare final voltage state (allow small tolerance for float precision)
        self.assertTrue(torch.allclose(hip_final_voltage, py_final_voltage, atol=1e-2), # Increased tolerance for float16 vs float32?
                        f"Final voltages differ significantly. HIP mean: {hip_final_voltage.mean()}, PyTorch mean: {py_final_voltage.mean()}")

        # Compare spike outputs at each step
        for i in range(5):
            self.assertTrue(torch.equal(hip_spikes_list[i], py_spikes_list[i]),
                            f"Spike outputs differ at step {i+1}")

        print("HIP kernel equivalence test passed.")


if __name__ == '__main__':
    # Ensure device is initialized before running tests
    if TEST_DEVICE is None:
         print("ERROR: Test device could not be initialized.")
    elif not _model_import_success:
         print("Skipping tests as UnifiedNeuronModel could not be imported.")
    else:
         print(f"Running tests on device: {TEST_DEVICE}")
         unittest.main()
