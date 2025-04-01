// neuron_kernel.hip
#include <hip/hip_runtime.h>

__global__ void fire_neurons_kernel(
    half* V, half* spikes, half* I, half tau, half v_th, half v_reset, 
    int num_neurons, float dt, int max_connections) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_neurons) {
        half v = V[idx] + I[idx] - (V[idx] / tau) * dt;  // Leaky integration
        half spike = (v > v_th) ? 1.0 : 0.0;  // Spike if above threshold
        V[idx] = spike > 0 ? v_reset : v;  // Reset if spiked
        spikes[idx] = spike;
    }
}

// Host code (to be called from fum.py)
extern "C" void launch_lif_kernel(
    half* V, half* spikes, half* I, half tau, half v_th, half v_reset, 
    int num_neurons, float dt, int max_connections) {
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);
    hipLaunchKernelGGL(fire_neurons_kernel, grid, block, 0, 0,
                       V, spikes, I, tau, v_th, v_reset, num_neurons, dt, max_connections);
}