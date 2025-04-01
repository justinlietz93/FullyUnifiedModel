# fum.py (Linux workstation)
import torch
import ctypes
import time

# Load HIP kernel
lib = ctypes.CDLL('./lif_kernel.so')
lib.launch_lif_kernel.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
    ctypes.c_float, ctypes.c_float, ctypes.c_float, 
    ctypes.c_int, ctypes.c_float, ctypes.c_int
]

class FUM:
    def __init__(self, num_neurons=1000):  # Start with 1000
        self.num_neurons = num_neurons
        self.V_d = torch.zeros(num_neurons, dtype=torch.float16, device='cuda:1')  # 7900 XTX
        self.spikes_d = torch.zeros(num_neurons, dtype=torch.float16, device='cuda:1')
        self.I_d = torch.zeros(num_neurons, dtype=torch.float16, device='cuda:1')
        self.tau = 20.0
        self.v_th = -55.0
        self.v_reset = -70.0

    def update_lif(self, I):
        self.I_d.copy_(I)
        lib.launch_lif_kernel(
            self.V_d.data_ptr(), self.spikes_d.data_ptr(), self.I_d.data_ptr(),
            self.tau, self.v_th, self.v_reset, self.num_neurons, 0.01, 1000
        )
        return self.spikes_d

# Test 1000 neurons on GPU
if __name__ == "__main__":
    fum = FUM(num_neurons=1000)
    I = torch.ones(1000, dtype=torch.float16, device='cuda:1') * 0.1  # ~15 Hz
    start = time.time()
    for _ in range(50):  # 50 timesteps
        spikes = fum.update_lif(I)
    elapsed = time.time() - start
    print(f"1000 neurons, 50 timesteps: {elapsed}s")
    print(f"Total spikes: {spikes.sum().item()}")
    assert elapsed < 0.1, f"Latency {elapsed}s exceeds 0.1s"