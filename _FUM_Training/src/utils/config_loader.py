import yaml
import os
from typing import Dict, Any

# Define the expected path to the config file relative to this script's directory structure
# Assumes this file is in src/utils and config is in config/
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'hardware_config.yaml')

_cached_config: Dict[str, Any] | None = None

def get_hardware_config() -> Dict[str, Any]:
    """
    Loads the hardware configuration from hardware_config.yaml.

    Caches the configuration after the first load.

    Returns:
        A dictionary containing the hardware configuration.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    # Recalculate path relative to the *actual* location of this file at runtime
    # This makes it more robust if the script is called from different working directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', '..', 'config', 'hardware_config.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Hardware configuration file not found at: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if config is None: # Handle empty file case
                config = {}
        _cached_config = config
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing hardware configuration file: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading hardware config: {e}")
        raise

def get_compute_backend() -> str:
    """
    Retrieves the specified compute backend from the hardware configuration.

    Defaults to 'cpu' if not specified or if the config file is empty/invalid.

    Returns:
        The compute backend string ('nvidia_cuda', 'amd_hip', 'cpu').
    """
    try:
        config = get_hardware_config()
        backend = config.get('compute_backend', 'cpu') # Default to 'cpu'
        if backend not in ['nvidia_cuda', 'amd_hip', 'cpu']:
            print(f"Warning: Invalid compute_backend '{backend}' in config. Defaulting to 'cpu'.")
            return 'cpu'
        return backend
    except (FileNotFoundError, yaml.YAMLError):
        print("Warning: Hardware config not found or invalid. Defaulting compute backend to 'cpu'.")
        return 'cpu'
    except Exception as e:
        # Catch other potential errors during loading
        print(f"Warning: Error loading hardware config ({e}). Defaulting compute backend to 'cpu'.")
        return 'cpu'


if __name__ == '__main__':
    # Example usage/test when running this file directly
    try:
        # Recalculate path for direct execution test
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path_test = os.path.join(current_dir, '..', '..', 'config', 'hardware_config.yaml')
        print(f"Loading hardware config from: {config_path_test}")

        hw_config = get_hardware_config()
        print("Hardware Config:", hw_config)
        backend = get_compute_backend()
        print("Selected Compute Backend:", backend)
    except Exception as e:
        print(f"Error during config loading test: {e}")
