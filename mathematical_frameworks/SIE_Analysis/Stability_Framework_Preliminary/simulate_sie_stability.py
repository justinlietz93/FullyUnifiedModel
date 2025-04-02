import numpy as np
import matplotlib.pyplot as plt
import time
import os # Import os module
import argparse # Import argparse for command-line arguments

# --- Simulation Parameters ---
NUM_NEURONS = 100 # Simplified network size for simulation
NUM_CLUSTERS = 10 # Number of functional clusters
SIMULATION_STEPS = 10000
ETA = 0.01 # Base STDP learning rate
GAMMA = 0.9 # TD discount factor
ALPHA = 0.1 # TD learning rate
TARGET_VAR = 0.05 # Target variance for self_benefit calculation
LAMBDA = 0.001 # Weight decay coefficient (NEW)

# SIE Component Weights (Example - will be varied in tests)
W_TD = 0.5
W_NOVELTY = 0.2
W_HABITUATION = 0.1
W_SELF_BENEFIT = 0.2
W_EXTERNAL = 0.8 # Weight for external reward when available

# --- Helper Functions ---

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_sie_components(current_state, recent_inputs, habituation_counters, spike_rates_history, V_states, external_reward=None):
    """Calculates the components of the SIE reward signal."""
    # --- TD Error ---
    # Simplified: Assume we know current/next state cluster IDs and external reward
    # In a real system, this depends on clustering and environment interaction
    current_state_idx = current_state['cluster_id']
    next_state_idx = np.random.randint(NUM_CLUSTERS) # Placeholder for next state
    r = external_reward if external_reward is not None else 0 # Use external reward if provided
    td_error = r + GAMMA * V_states[next_state_idx] - V_states[current_state_idx]

    # --- Novelty ---
    # Simplified: Compare current input pattern to history
    current_input = current_state['input_pattern']
    if len(recent_inputs) > 0:
        similarities = [np.dot(current_input, inp) / (np.linalg.norm(current_input) * np.linalg.norm(inp)) for inp in recent_inputs]
        max_similarity = np.max(similarities) if similarities else 0
        novelty = 1.0 - max_similarity
        matched_idx = np.argmax(similarities) if similarities else -1
    else:
        novelty = 1.0
        matched_idx = -1
    
    # --- Habituation ---
    habituation = 0.0
    if matched_idx != -1 and max_similarity > 0.9:
        habituation_counters[matched_idx] = min(1.0, habituation_counters[matched_idx] + 0.1)
        habituation = habituation_counters[matched_idx]
    # Decay counters
    habituation_counters *= 0.995 

    # --- Self-Benefit (Homeostasis-Based) ---
    if len(spike_rates_history) > 100: # Need enough history
        current_variance = np.var(spike_rates_history[-100:])
        self_benefit = 1.0 - min(1.0, np.abs(current_variance - TARGET_VAR) / TARGET_VAR) # Clamped deviation
    else:
        self_benefit = 0.5 # Default value until enough history

    # --- Normalization & Weighting ---
    # --- Normalization & Damping ---
    # Normalize components to be roughly comparable (e.g., clip TD error)
    td_norm = np.clip(td_error, -1, 1) 
    novelty_norm = novelty 
    habituation_norm = habituation
    self_benefit_norm = self_benefit 
    
    # Calculate damping factor alpha based on novelty vs. self_benefit (impact proxy)
    # alpha balances exploration (novelty) vs stability (self_benefit)
    alpha_damping = 1.0 - np.tanh(np.abs(novelty_norm - self_benefit_norm)) # High diff -> low alpha -> less exploration/stability drive
    beta_damping = 1.0 - alpha_damping # High diff -> high beta -> more TD drive? Revisit logic. Let's damp novelty/sb directly.
    
    # Apply damping: Reduce influence of novelty/habituation and self_benefit when they conflict
    damped_novelty_term = alpha_damping * (W_NOVELTY * novelty_norm - W_HABITUATION * habituation_norm)
    damped_self_benefit_term = alpha_damping * (W_SELF_BENEFIT * self_benefit_norm)

    # Apply external reward weight
    w_r = W_EXTERNAL if external_reward is not None else (1 - W_EXTERNAL)
    w_internal = 1 - w_r
    
    # Combine components with damping and weighting
    # Prioritize TD error, apply damped internal drives
    internal_reward = (W_TD * td_norm + 
                       damped_novelty_term + 
                       damped_self_benefit_term)
                       
    total_reward = w_r * r + w_internal * internal_reward
    # Clip total_reward to prevent extreme values before sigmoid? Optional.
    # total_reward = np.clip(total_reward, -5, 5) 

    # --- Update History ---
    if len(recent_inputs) >= 50: # Keep limited history
        recent_inputs.pop(0)
        # Note: Need to handle habituation counter indices correctly when popping
    recent_inputs.append(current_input)

    return total_reward, td_error, novelty, habituation, self_benefit, next_state_idx

def update_weights(W, eligibility_trace, total_reward, mod_factor, eta, decay_lambda):
    """Applies the modulated STDP update with weight decay."""
    eta_effective = eta * (1 + mod_factor)
    hebbian_update = eta_effective * total_reward * eligibility_trace
    decay_term = decay_lambda * W
    
    # Apply weight changes (Hebbian + Decay)
    W += hebbian_update - decay_term
    
    # Add weight clipping as a simple constraint
    W = np.clip(W, -1.0, 1.0) # Example bounds, can be adjusted
    
    # Add noise/drift if modeling Phase 2 equations require it
    # W += 0.001 * np.random.randn(*W.shape) 
    return W

# --- Simulation Loop ---

def run_simulation(params):
    """Runs the SIE stability simulation."""
    print(f"Running simulation with params: {params}")
    
    # Initialize state
    W = np.random.rand(NUM_NEURONS, NUM_NEURONS) * 0.1 # Synaptic weights
    V_states = np.zeros(NUM_CLUSTERS) # TD Value estimates
    eligibility_trace = np.zeros((NUM_NEURONS, NUM_NEURONS)) # STDP eligibility trace
    recent_inputs = []
    habituation_counters = np.zeros(50) # Size matches recent_inputs history limit
    spike_rates_history = []

    # History tracking for analysis
    reward_history = []
    mod_factor_history = []
    weight_norm_history = []
    v_state_history = []
    component_history = {'td': [], 'nov': [], 'hab': [], 'sb': []}

    start_time = time.time()
    for step in range(SIMULATION_STEPS):
        # --- 1. Simulate Network Activity (Slightly More Realistic) ---
        # Generate mock input pattern
        current_input_pattern = np.random.rand(NUM_NEURONS) 
        
        # Calculate simple neuron activation based on input and weights
        # activation = W.T @ current_input_pattern # Input drives neurons via incoming weights
        # Use mean activation to avoid excessive computation in this simple sim
        mean_activation = np.mean(W) * np.mean(current_input_pattern) * NUM_NEURONS 
        # Simulate firing rates based on activation (e.g., sigmoid) + noise
        base_rate = sigmoid(mean_activation - 0.5) # Shifted sigmoid for baseline rate
        current_spike_rates = np.clip(base_rate + np.random.randn(NUM_NEURONS) * 0.05, 0, 1) # Add noise, clip rate [0,1] Hz
        
        spike_rates_history.append(np.mean(current_spike_rates))
        current_cluster_id = np.random.randint(NUM_CLUSTERS) # Mock cluster ID
        current_state = {'input_pattern': current_input_pattern, 'cluster_id': current_cluster_id}
        
        # Update eligibility trace (simplified decay)
        eligibility_trace *= 0.95
        # Add contribution from current spikes (placeholder)
        eligibility_trace += np.outer(current_spike_rates, current_spike_rates) * 0.01 
        eligibility_trace = np.clip(eligibility_trace, 0, 1)

        # --- 2. Calculate SIE Reward ---
        # Simulate occasional external reward signal
        external_reward = 1.0 if step % 100 == 0 else None # Example: reward every 100 steps
        
        total_reward, td, nov, hab, sb, next_state_idx = calculate_sie_components(
            current_state, recent_inputs, habituation_counters, spike_rates_history, V_states, external_reward
        )
        
        # --- 3. Update TD Value Function ---
        V_states[current_cluster_id] += ALPHA * td

        # --- 4. Calculate Modulation Factor ---
        mod_factor = 2 * sigmoid(total_reward) - 1

        # --- 5. Update Weights ---
        W = update_weights(W, eligibility_trace, total_reward, mod_factor, ETA, params['lambda_decay']) # Pass lambda

        # --- 6. Track History ---
        reward_history.append(total_reward)
        mod_factor_history.append(mod_factor)
        weight_norm_history.append(np.linalg.norm(W))
        v_state_history.append(np.mean(V_states))
        component_history['td'].append(td)
        component_history['nov'].append(nov)
        component_history['hab'].append(hab)
        component_history['sb'].append(sb)

        if step % 1000 == 0:
            print(f"Step {step}/{SIMULATION_STEPS} - Reward: {total_reward:.3f}, ModFactor: {mod_factor:.3f}, ||W||: {weight_norm_history[-1]:.2f}")

    end_time = time.time()
    print(f"Simulation finished in {end_time - start_time:.2f} seconds.")

    # --- 7. Plot Results ---
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 2, 1)
    plt.plot(reward_history)
    plt.title('Total Reward')
    plt.xlabel('Step')
    plt.ylabel('Reward')

    plt.subplot(3, 2, 2)
    plt.plot(mod_factor_history)
    plt.title('Modulation Factor')
    plt.xlabel('Step')
    plt.ylabel('Factor')

    plt.subplot(3, 2, 3)
    plt.plot(weight_norm_history)
    plt.title('Weight Matrix Norm ||W||')
    plt.xlabel('Step')
    plt.ylabel('Norm')

    plt.subplot(3, 2, 4)
    plt.plot(v_state_history)
    plt.title('Average V(state)')
    plt.xlabel('Step')
    plt.ylabel('Avg Value')

    plt.subplot(3, 2, 5)
    plt.plot(component_history['nov'], label='Novelty', alpha=0.7)
    plt.plot(component_history['hab'], label='Habituation', alpha=0.7)
    plt.plot(component_history['sb'], label='Self-Benefit', alpha=0.7)
    plt.title('SIE Components (Internal)')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.legend()
    
    plt.subplot(3, 2, 6)
    plt.plot(component_history['td'], label='TD Error', alpha=0.7)
    plt.title('SIE Component (TD Error)')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    # Construct the correct path relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '..', 'results') # Go up one level to _FUM_Training, then into results
    os.makedirs(results_dir, exist_ok=True) # Ensure the directory exists
    save_path = os.path.join(results_dir, 'sie_stability_simulation.png')
    
    plt.savefig(save_path)
    print(f"Saved simulation plot to {save_path}")
    # plt.show() # Uncomment to display plot interactively

    # --- 8. Save Detailed History Data ---
    # Create a filename based on parameters
    param_str = f"eta{params['eta']}_lambda{params['lambda_decay']}"
    data_filename = f'sie_stability_data_{param_str}.npz'
    data_save_path = os.path.join(results_dir, data_filename)
    
    history_data = {
        'reward': np.array(reward_history),
        'mod_factor': np.array(mod_factor_history),
        'weight_norm': np.array(weight_norm_history),
        'v_state_avg': np.array(v_state_history),
        'td_error': np.array(component_history['td']),
        'novelty': np.array(component_history['nov']),
        'habituation': np.array(component_history['hab']),
        'self_benefit': np.array(component_history['sb']),
        'params': params # Include simulation parameters
    }
    # data_save_path = os.path.join(results_dir, 'sie_stability_data.npz') # Old path
    np.savez(data_save_path, **history_data)
    print(f"Saved detailed simulation data to {data_save_path}")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SIE Stability Simulation.')
    parser.add_argument('--eta', type=float, default=ETA, help=f'Base STDP learning rate (default: {ETA})')
    parser.add_argument('--lambda_decay', type=float, default=LAMBDA, help=f'Weight decay coefficient (default: {LAMBDA})')
    # Add arguments for other parameters like W_TD, W_NOVELTY etc. if needed for sweeps
    
    args = parser.parse_args()

    # Define parameter set using command-line arguments or defaults
    simulation_params = {
        'eta': args.eta,
        'w_td': W_TD, # Keep other weights fixed for now, focus on eta/lambda
        'w_novelty': W_NOVELTY,
        'w_habituation': W_HABITUATION,
        'w_self_benefit': W_SELF_BENEFIT,
        'lambda_decay': args.lambda_decay 
        # Add other parameters to test (e.g., damping factors)
    }
    run_simulation(simulation_params)
