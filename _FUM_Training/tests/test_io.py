import unittest
import torch
import numpy as np
import cv2
import librosa
import soundfile as sf # Need this for creating dummy audio
import sys
import os
import time

# --- Adjust sys.path to find project modules ---
# Add the '_FUM_Training' directory to sys.path to allow imports like 'from src.io...'
fum_training_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if fum_training_dir not in sys.path:
    sys.path.insert(0, fum_training_dir)

# --- Attempt Imports ---
try:
    # Use explicit path relative to _FUM_Training to avoid conflict with built-in 'io'
    from src.io.encoder import BaseEncoder, TextEncoder, ImageEncoder, VideoEncoder, AudioEncoder
    from src.io.decoder import decode_text_rate # Added decoder import
    from src.neuron.unified_neuron import initialize_device # Use this to get consistent device
    _imports_ok = True
except ImportError as e:
    print(f"Error importing necessary modules for test_io: {e}")
    _imports_ok = False

# Determine device
try:
    DEVICE = initialize_device()
except NameError:
     DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     print(f"Warning: Could not use initialize_device from unified_neuron. Defaulting test device to {DEVICE}")

# --- Test Data Paths ---
TEST_DIR = os.path.dirname(__file__)
DUMMY_IMG_PATH = os.path.join(TEST_DIR, "dummy_test_image.png")
DUMMY_VID_PATH = os.path.join(TEST_DIR, "dummy_test_video.avi")
DUMMY_AUD_PATH = os.path.join(TEST_DIR, "dummy_test_audio.wav")

@unittest.skipIf(not _imports_ok, "Skipping IO tests due to import errors.")
class TestEncoders(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up common parameters and create dummy files."""
        cls.device = DEVICE
        cls.duration = 100 # ms (timesteps if dt=1.0)
        cls.dt = 1.0
        cls.max_rate = 100.0 # Hz
        cls.test_dir = TEST_DIR # Store test dir path

        # Create dummy image
        img_arr = np.zeros((30, 30), dtype=np.uint8)
        img_arr[10:20, 10:20] = 255 # White square in the middle
        cv2.imwrite(DUMMY_IMG_PATH, img_arr)
        print(f"Created dummy image: {DUMMY_IMG_PATH}")

        # Create dummy video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_size = (32, 32)
        out = cv2.VideoWriter(DUMMY_VID_PATH, fourcc, 10.0, frame_size) # 10 fps
        if out.isOpened():
            for i in range(30): # 3 seconds
                frame = np.random.randint(0, 50, (*frame_size, 3), dtype=np.uint8) # Low intensity background
                if 10 <= i < 20: # Moving square
                    x = 5 + i
                    y = 10
                    cv2.rectangle(frame, (x, y), (x+5, y+5), (0, 255, 0), -1) # Green square
                out.write(frame)
            out.release()
            print(f"Created dummy video: {DUMMY_VID_PATH}")
        else:
            print(f"Error: Could not open VideoWriter for {DUMMY_VID_PATH}")
            # Consider raising an error or skipping video tests

        # Create dummy audio
        try:
            sr_aud = 22050
            duration_aud = 1 # second
            frequency = 440
            t_aud = np.linspace(0., duration_aud, int(sr_aud * duration_aud), endpoint=False)
            amplitude = np.iinfo(np.int16).max * 0.5
            data_aud = (amplitude * np.sin(2. * np.pi * frequency * t_aud)).astype(np.int16)
            sf.write(DUMMY_AUD_PATH, data_aud, sr_aud)
            print(f"Created dummy audio: {DUMMY_AUD_PATH}")
            cls.audio_created = True
        except ImportError:
            print("Warning: soundfile not installed. Cannot create dummy audio file. Skipping audio tests.")
            cls.audio_created = False
        except Exception as e:
            print(f"Error creating dummy audio file: {e}")
            cls.audio_created = False


    @classmethod
    def tearDownClass(cls):
        """Remove dummy files after tests."""
        if os.path.exists(DUMMY_IMG_PATH): os.remove(DUMMY_IMG_PATH)
        if os.path.exists(DUMMY_VID_PATH): os.remove(DUMMY_VID_PATH)
        if os.path.exists(DUMMY_AUD_PATH): os.remove(DUMMY_AUD_PATH)
        print("Cleaned up dummy files.")

    def test_01_text_encoder(self):
        """Test TextEncoder output shape and basic rate encoding."""
        num_neurons = 128
        encoder = TextEncoder(num_neurons=num_neurons, duration=self.duration, dt=self.dt, max_rate=self.max_rate)
        text = "Hello"
        spikes = encoder.encode(text)

        self.assertEqual(spikes.shape, (num_neurons, self.duration))
        self.assertEqual(spikes.device.type, self.device.type)
        self.assertTrue(torch.all((spikes == 0) | (spikes == 1))) # Check binary

        # Check if the correct neuron spiked (ASCII 'H' = 72)
        neuron_idx = ord('H')
        self.assertLess(neuron_idx, num_neurons)
        # Check if *some* spikes occurred for the target neuron (probabilistic)
        self.assertGreater(spikes[neuron_idx, :].sum().item(), 0, "Neuron for 'H' did not spike.")
        # Check if other neurons remained mostly silent
        other_neurons_mask = torch.ones(num_neurons, dtype=torch.bool)
        other_neurons_mask[neuron_idx] = False
        self.assertLess(spikes[other_neurons_mask, :].sum().item(), self.duration * 0.1, "Too many spikes in non-target neurons.") # Allow some noise

        # Test empty string
        spikes_empty = encoder.encode("")
        self.assertEqual(spikes_empty.sum().item(), 0)

    def test_02_image_encoder(self):
        """Test ImageEncoder output shape and basic intensity mapping."""
        target_size = (8, 8)
        num_neurons = target_size[0] * target_size[1]
        encoder = ImageEncoder(num_neurons=num_neurons, duration=self.duration, dt=self.dt, target_size=target_size, max_rate=self.max_rate)

        spikes = encoder.encode(DUMMY_IMG_PATH)

        self.assertEqual(spikes.shape, (num_neurons, self.duration))
        self.assertEqual(spikes.device.type, self.device.type)
        self.assertTrue(torch.all((spikes == 0) | (spikes == 1)))

        # Check if neurons corresponding to the white square spiked more (probabilistic)
        # Need to map the white square in original image (10:20, 10:20) to the resized (8,8)
        # Approx center pixels in 8x8: rows 3-4, cols 3-4
        center_indices = []
        for r in range(3, 5):
            for c in range(3, 5):
                center_indices.append(r * target_size[1] + c)

        center_mask = torch.zeros(num_neurons, dtype=torch.bool)
        center_mask[center_indices] = True
        corner_mask = ~center_mask # Simple approximation

        avg_center_spikes = spikes[center_mask, :].sum().item() / len(center_indices)
        avg_corner_spikes = spikes[corner_mask, :].sum().item() / (num_neurons - len(center_indices))

        self.assertGreater(avg_center_spikes, avg_corner_spikes + 1, # Expect significantly more spikes in center
                           f"Center pixels (avg spikes={avg_center_spikes:.2f}) did not spike significantly more than corners (avg spikes={avg_corner_spikes:.2f}).")

    def test_03_video_encoder(self):
        """Test VideoEncoder output shape and responsiveness to change."""
        target_size = (10, 10)
        num_neurons = target_size[0] * target_size[1]
        encoder = VideoEncoder(num_neurons=num_neurons, duration=self.duration, dt=self.dt, target_size=target_size, max_rate=self.max_rate, frame_skip=1)

        spikes = encoder.encode(DUMMY_VID_PATH)

        self.assertEqual(spikes.shape, (num_neurons, self.duration))
        self.assertEqual(spikes.device.type, self.device.type)
        self.assertTrue(torch.all((spikes == 0) | (spikes == 1)))

        # Check if total spikes are non-zero (indicating frame diffs were processed)
        # Note: This test is basic, doesn't verify specific motion detection.
        self.assertGreater(spikes.sum().item(), 0, "Video encoding produced zero spikes, check frame diff logic.")

    @unittest.skipIf(not os.path.exists(DUMMY_AUD_PATH), "Skipping audio test because dummy audio file doesn't exist.")
    def test_04_audio_encoder(self):
        """Test AudioEncoder output shape and basic processing."""
        n_mfcc = 13
        num_neurons = n_mfcc
        encoder = AudioEncoder(num_neurons=num_neurons, duration=self.duration, dt=self.dt, n_mfcc=n_mfcc, max_rate=self.max_rate)

        spikes = encoder.encode(DUMMY_AUD_PATH)

        self.assertEqual(spikes.shape, (num_neurons, self.duration))
        self.assertEqual(spikes.device.type, self.device.type)
        self.assertTrue(torch.all((spikes == 0) | (spikes == 1)))

        # Check if *some* spikes were generated (basic check)
        self.assertGreater(spikes.sum().item(), 0, "Audio encoding produced zero spikes.")

    def test_05_decoder_text_rate(self):
        """Test the decode_text_rate function."""
        num_neurons = 128
        duration = 100
        window = 50
        dt = 1.0
        max_r = 50.0
        test_indices = list(range(num_neurons))

        # Test case 1: Neuron for 'C' (ASCII 67) has highest rate (deterministic spikes)
        history1 = torch.zeros((num_neurons, duration), device=self.device)
        rate_C = (67 / 127.0) * max_r # Target rate Hz
        # Calculate expected spikes in window deterministically
        expected_spikes_C = int(round(rate_C * (window * dt / 1000.0)))
        # Distribute spikes evenly in the window for neuron 67
        if expected_spikes_C > 0:
            spike_indices_C = torch.linspace(0, window - 1, expected_spikes_C, device=self.device).long()
            history1[67, -window + spike_indices_C] = 1.0

        # Add lower noise rate to another neuron (deterministic)
        rate_X = (88 / 127.0) * max_r * 0.5 # Lower rate for 'X'
        expected_spikes_X = int(round(rate_X * (window * dt / 1000.0)))
        if expected_spikes_X > 0:
             # Ensure indices don't overlap with C's if possible (simple offset)
             spike_indices_X = torch.linspace(1, window - 2, expected_spikes_X, device=self.device).long()
             history1[88, -window + spike_indices_X] = 1.0

        decoded1 = decode_text_rate(history1, test_indices, window, dt, max_rate_for_char=max_r)
        # Use assertAlmostEqual for rate calculation or check decoded char directly
        self.assertEqual(decoded1, 'C', f"Expected 'C', got '{decoded1}'")

        # Test case 2: No significant activity
        history2 = torch.zeros((num_neurons, duration), device=self.device)
        decoded2 = decode_text_rate(history2, test_indices, window, dt, max_rate_for_char=max_r)
        self.assertEqual(decoded2, '', f"Expected empty string, got '{decoded2}'")

        # Test case 3: Activity below threshold (deterministic)
        history3 = torch.zeros((num_neurons, duration), device=self.device)
        # Calculate spikes needed for just below threshold rate (e.g., 0.9 Hz)
        min_rate_threshold = 1.0 # From decoder function
        spikes_below_thresh = int(np.floor((min_rate_threshold * 0.9) * (window * dt / 1000.0)))
        if spikes_below_thresh > 0:
             spike_indices_low = torch.linspace(0, window - 1, spikes_below_thresh, device=self.device).long()
             history3[70, -window + spike_indices_low] = 1.0 # Neuron 70 fires just below threshold

        decoded3 = decode_text_rate(history3, test_indices, window, dt, max_rate_for_char=max_r)
        self.assertEqual(decoded3, '', f"Expected empty string for low activity (spikes={spikes_below_thresh}), got '{decoded3}'")


if __name__ == '__main__':
    if not _imports_ok:
        print("Cannot run tests due to import errors.")
    else:
        print(f"Running IO tests on device: {DEVICE}")
        # Need to ensure dummy files are created before running tests
        # setUpClass handles this, but good practice to check
        if not os.path.exists(DUMMY_IMG_PATH):
             print("Error: Dummy image not found for tests.")
        elif not os.path.exists(DUMMY_VID_PATH):
             print("Error: Dummy video not found for tests.")
        # Audio creation handled in setUpClass with skip logic

        unittest.main()
