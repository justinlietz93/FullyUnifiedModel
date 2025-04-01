import torch
import numpy as np
import cv2 # For image/video
import librosa # For audio
from abc import ABC, abstractmethod
import time
import sys # Added sys import
import os # Added os import

# --- Adjust sys.path to find project modules ---
# Add the '_FUM_Training' directory to sys.path to allow imports like 'from src.neuron...'
fum_training_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if fum_training_dir not in sys.path:
    sys.path.insert(0, fum_training_dir)

# --- Attempt Imports and Determine Device ---
try:
    # Use explicit path relative to _FUM_Training
    from src.neuron.unified_neuron import initialize_device
    DEVICE = initialize_device()
except ImportError as e:
     print(f"Warning: Could not import initialize_device from src.neuron.unified_neuron: {e}. Defaulting device to CPU for encoders.")
     DEVICE = torch.device('cpu')
except Exception as e_init:
     print(f"Warning: Error during device initialization: {e_init}. Defaulting device to CPU for encoders.")
     DEVICE = torch.device('cpu')


class BaseEncoder(ABC):
    """Abstract base class for all modality encoders."""

    def __init__(self, num_neurons: int, duration: int, dt: float = 1.0, refractory_period_ms: int = 5):
        """
        Initialize the base encoder.

        Args:
            num_neurons (int): Number of neurons assigned to this encoder.
            duration (int): Duration of the encoded output spike train in timesteps (ms if dt=1.0).
            dt (float): Simulation timestep in ms. Defaults to 1.0.
            refractory_period_ms (int): Refractory period in ms. Defaults to 5.
        """
        self.num_neurons = num_neurons
        self.duration = duration
        self.dt = dt
        self.refractory_steps = int(refractory_period_ms / dt)
        self.device = DEVICE
        print(f"Initialized {self.__class__.__name__} for {num_neurons} neurons, duration {duration} steps, on {self.device}")

    @abstractmethod
    def encode(self, data) -> torch.Tensor:
        """
        Encodes the input data into a spike train tensor.

        Args:
            data: The raw input data for the specific modality.

        Returns:
            torch.Tensor: A tensor of shape (num_neurons, duration) containing binary spikes (0 or 1).
        """
        pass

    def _generate_poisson_spikes(self, rates: torch.Tensor) -> torch.Tensor:
        """
        Generates Poisson spike trains based on input rates with refractory period.

        Args:
            rates (torch.Tensor): Tensor of firing rates (in Hz) for each neuron.
                                  Shape: (num_neurons,).

        Returns:
            torch.Tensor: Spike train tensor of shape (num_neurons, duration).
        """
        if rates.shape[0] != self.num_neurons:
            raise ValueError(f"Rates tensor shape ({rates.shape}) does not match num_neurons ({self.num_neurons})")
        if rates.max() > (1000.0 / self.dt / (1 + self.refractory_steps)):
             print(f"Warning: Max requested rate ({rates.max():.2f} Hz) exceeds effective limit due to refractory period. Clipping rates.")
             rates = torch.clamp(rates, max=(1000.0 / self.dt / (1 + self.refractory_steps)))

        rates = rates.to(self.device)
        # Probability of spiking in one timestep (dt is in ms)
        prob = rates * (self.dt / 1000.0)
        prob = prob.unsqueeze(1).expand(-1, self.duration) # Shape: (num_neurons, duration)

        # Generate spikes based on probability
        spikes = torch.rand_like(prob) < prob

        # Apply refractory period
        refractory_counters = torch.zeros(self.num_neurons, device=self.device, dtype=torch.int32)
        for t in range(self.duration):
            # Neurons in refractory period cannot spike
            spikes[:, t] = spikes[:, t] & (refractory_counters == 0)
            # Reset counter for neurons that spiked
            refractory_counters[spikes[:, t]] = self.refractory_steps + 1 # +1 because we decrement immediately
            # Decrement counters (minimum 0)
            refractory_counters = torch.clamp(refractory_counters - 1, min=0)

        return spikes.float() # Return as float tensor (0.0 or 1.0)


class TextEncoder(BaseEncoder):
    """Encodes text into spike trains based on ASCII values."""

    def __init__(self, num_neurons: int, duration: int, dt: float = 1.0, max_rate: float = 50.0):
        """
        Args:
            max_rate (float): Maximum firing rate (Hz) corresponding to highest ASCII value.
        """
        if num_neurons < 128:
            print(f"Warning: TextEncoder ideally needs at least 128 neurons for full ASCII range. Got {num_neurons}.")
            # Could implement mapping or truncation if needed
        super().__init__(num_neurons, duration, dt)
        self.max_rate = max_rate
        # Assign first 128 neurons to ASCII characters if possible
        self.neuron_map_size = min(num_neurons, 128)

    def encode(self, text: str) -> torch.Tensor:
        """
        Encodes text using rate coding based on ASCII values.
        Each character activates one neuron for the full duration.

        Args:
            text (str): The input text string.

        Returns:
            torch.Tensor: Spike train tensor.
        """
        print(f"Encoding text: '{text[:50]}...'")
        rates = torch.zeros(self.num_neurons, device=self.device)
        if not text:
            return self._generate_poisson_spikes(rates)

        # Simple approach: Use the first character's ASCII value
        # More complex: Average ASCII, sequential encoding, etc.
        # Roadmap example: "ASCII % 50 Hz" - let's interpret as rate = (ASCII / 127) * max_rate
        # Assigning one neuron per ASCII value up to neuron_map_size
        char_ord = ord(text[0])
        if 0 <= char_ord < self.neuron_map_size:
            rate = (char_ord / 127.0) * self.max_rate
            rates[char_ord] = rate
        elif char_ord >= self.neuron_map_size: # Map higher ASCII to last available neuron
             rate = (char_ord / 127.0) * self.max_rate # Keep rate calculation consistent
             rates[self.neuron_map_size - 1] = rate


        # Alternative: Sequential encoding (each char gets a time slice) - More complex
        # time_per_char = self.duration // len(text) if len(text) > 0 else self.duration
        # all_spikes = torch.zeros((self.num_neurons, self.duration), device=self.device)
        # for i, char in enumerate(text):
        #     char_ord = ord(char)
        #     if 0 <= char_ord < self.neuron_map_size:
        #         rate = torch.zeros(self.num_neurons, device=self.device)
        #         rate[char_ord] = (char_ord / 127.0) * self.max_rate
        #         # Generate spikes for this char's time slice
        #         # Need a modified _generate_poisson_spikes for slices... complex.
        # Stick to simpler rate encoding for now.

        return self._generate_poisson_spikes(rates)


class ImageEncoder(BaseEncoder):
    """Encodes images into spike trains based on pixel intensity."""

    def __init__(self, num_neurons: int, duration: int, dt: float = 1.0,
                 target_size: tuple[int, int] = (10, 10), max_rate: float = 100.0):
        """
        Args:
            target_size (tuple[int, int]): Resize input image to this (width, height).
                                           Number of neurons must be >= width * height.
            max_rate (float): Max firing rate (Hz) for max pixel intensity (255).
        """
        self.target_width, self.target_height = target_size
        required_neurons = self.target_width * self.target_height
        if num_neurons < required_neurons:
            raise ValueError(f"ImageEncoder needs at least {required_neurons} neurons for target size {target_size}. Got {num_neurons}.")
        super().__init__(num_neurons, duration, dt)
        self.max_rate = max_rate

    def encode(self, image_path: str) -> torch.Tensor:
        """
        Encodes an image file using rate coding based on pixel intensity.
        Each pixel maps to one neuron.

        Args:
            image_path (str): Path to the image file.

        Returns:
            torch.Tensor: Spike train tensor.
        """
        print(f"Encoding image: {image_path}")
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Could not read image file: {image_path}")

            img_resized = cv2.resize(img, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
            # Flatten the image and normalize pixel values (0-1)
            img_flat = img_resized.flatten().astype(np.float32) / 255.0

            rates = torch.zeros(self.num_neurons, device=self.device)
            # Map pixel intensities to rates for the assigned neurons
            rates[:len(img_flat)] = torch.tensor(img_flat, device=self.device) * self.max_rate

            return self._generate_poisson_spikes(rates)

        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            # Return zero spikes on error
            return torch.zeros((self.num_neurons, self.duration), device=self.device)


class VideoEncoder(BaseEncoder):
    """Encodes video into spike trains based on frame differences."""

    def __init__(self, num_neurons: int, duration: int, dt: float = 1.0,
                 target_size: tuple[int, int] = (10, 10), max_rate: float = 100.0,
                 frame_skip: int = 1):
        """
        Args:
            target_size (tuple[int, int]): Resize video frames to this (width, height).
            max_rate (float): Max firing rate (Hz) for max pixel difference.
            frame_skip (int): Process every Nth frame.
        """
        self.target_width, self.target_height = target_size
        required_neurons = self.target_width * self.target_height
        if num_neurons < required_neurons:
            raise ValueError(f"VideoEncoder needs at least {required_neurons} neurons for target size {target_size}. Got {num_neurons}.")
        super().__init__(num_neurons, duration, dt)
        self.max_rate = max_rate
        self.frame_skip = max(1, frame_skip)

    def encode(self, video_path: str) -> torch.Tensor:
        """
        Encodes a video file using rate coding based on absolute frame differences.
        Each pixel difference maps to one neuron. Spikes are generated over time.

        Args:
            video_path (str): Path to the video file.

        Returns:
            torch.Tensor: Spike train tensor.
        """
        print(f"Encoding video: {video_path}")
        all_spikes = torch.zeros((self.num_neurons, self.duration), device=self.device)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return all_spikes

        prev_frame_gray = None
        frame_count = 0
        timestep = 0
        required_neurons = self.target_width * self.target_height

        while timestep < self.duration:
            ret, frame = cap.read()
            if not ret:
                break # End of video

            frame_count += 1
            if frame_count % self.frame_skip != 0:
                continue

            try:
                frame_resized = cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
                frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

                if prev_frame_gray is not None:
                    # Calculate absolute difference
                    frame_diff = cv2.absdiff(frame_gray, prev_frame_gray)
                    # Flatten and normalize (0-1)
                    diff_flat = frame_diff.flatten().astype(np.float32) / 255.0

                    rates = torch.zeros(self.num_neurons, device=self.device)
                    rates[:required_neurons] = torch.tensor(diff_flat, device=self.device) * self.max_rate

                    # Generate spikes for one timestep
                    # This is inefficient - generating full duration each frame diff.
                    # Better: Generate spikes only for the current timestep 't'
                    # Modify _generate_poisson_spikes or implement inline?

                    # --- Inline Poisson generation for one step ---
                    if rates.shape[0] != self.num_neurons: continue # Safety check
                    rates = torch.clamp(rates, max=(1000.0 / self.dt / (1 + self.refractory_steps)))
                    prob = rates * (self.dt / 1000.0)
                    frame_spikes = torch.rand_like(rates) < prob
                    # Apply refractory (assuming counters are maintained outside, or reset each frame?)
                    # For simplicity here, let's ignore refractory across frames for now.
                    all_spikes[:required_neurons, timestep] = frame_spikes[:required_neurons].float()
                    # --- End Inline ---

                    timestep += 1
                    if timestep >= self.duration: break

                prev_frame_gray = frame_gray

            except Exception as e:
                print(f"Error processing video frame {frame_count}: {e}")
                # Continue to next frame if possible

        cap.release()
        if timestep == 0:
             print("Warning: No frames processed or video too short for duration.")
        elif timestep < self.duration:
             print(f"Warning: Video ended before filling duration. Encoded {timestep} steps.")

        return all_spikes


class AudioEncoder(BaseEncoder):
    """Encodes audio into spike trains based on MFCC features."""

    def __init__(self, num_neurons: int, duration: int, dt: float = 1.0,
                 n_mfcc: int = 13, hop_length: int = 512, max_rate: float = 100.0):
        """
        Args:
            n_mfcc (int): Number of Mel-frequency cepstral coefficients to extract.
                          num_neurons must be >= n_mfcc.
            hop_length (int): Number of audio samples between successive MFCC columns.
            max_rate (float): Max firing rate (Hz) corresponding to max MFCC value.
        """
        if num_neurons < n_mfcc:
            raise ValueError(f"AudioEncoder needs at least {n_mfcc} neurons for {n_mfcc} MFCCs. Got {num_neurons}.")
        super().__init__(num_neurons, duration, dt)
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.max_rate = max_rate
        self._sr = None # Store sample rate

    def encode(self, audio_path: str) -> torch.Tensor:
        """
        Encodes an audio file using rate coding based on MFCC features over time.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            torch.Tensor: Spike train tensor.
        """
        print(f"Encoding audio: {audio_path}")
        all_spikes = torch.zeros((self.num_neurons, self.duration), device=self.device)

        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None) # Load with native sample rate
            self._sr = sr
            if sr is None: raise ValueError("Could not determine sample rate.")

            # Extract MFCCs
            # duration_sec = len(y) / sr
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length)

            # Normalize MFCCs (simple min-max scaling per coefficient)
            min_vals = np.min(mfccs, axis=1, keepdims=True)
            max_vals = np.max(mfccs, axis=1, keepdims=True)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1 # Avoid division by zero
            mfccs_normalized = (mfccs - min_vals) / range_vals # Shape: (n_mfcc, num_frames)

            num_audio_frames = mfccs_normalized.shape[1]
            # Map audio frames to simulation timesteps
            # Each audio frame corresponds to hop_length / sr seconds
            time_per_audio_frame_ms = (self.hop_length / sr) * 1000.0
            steps_per_audio_frame = max(1, int(time_per_audio_frame_ms / self.dt))

            print(f"  Audio frames: {num_audio_frames}, Steps/frame: {steps_per_audio_frame}")

            timestep = 0
            for frame_idx in range(num_audio_frames):
                if timestep >= self.duration: break

                # Get normalized MFCC values for this frame
                frame_mfccs = mfccs_normalized[:, frame_idx] # Shape: (n_mfcc,)

                rates = torch.zeros(self.num_neurons, device=self.device)
                rates[:self.n_mfcc] = torch.tensor(frame_mfccs, device=self.device) * self.max_rate

                # Generate spikes for the duration this audio frame represents
                end_step = min(timestep + steps_per_audio_frame, self.duration)
                num_steps_this_frame = end_step - timestep

                if num_steps_this_frame > 0:
                    # --- Inline Poisson generation for slice ---
                    if rates.shape[0] != self.num_neurons: continue
                    rates_clamped = torch.clamp(rates, max=(1000.0 / self.dt / (1 + self.refractory_steps)))
                    prob = rates_clamped * (self.dt / 1000.0)
                    prob = prob.unsqueeze(1).expand(-1, num_steps_this_frame)
                    frame_spikes = torch.rand_like(prob) < prob
                    # Apply refractory (simplified: reset each frame slice)
                    ref_counters = torch.zeros(self.num_neurons, device=self.device, dtype=torch.int32)
                    for t_slice in range(num_steps_this_frame):
                         frame_spikes[:, t_slice] = frame_spikes[:, t_slice] & (ref_counters == 0)
                         ref_counters[frame_spikes[:, t_slice]] = self.refractory_steps + 1
                         ref_counters = torch.clamp(ref_counters - 1, min=0)
                    # --- End Inline ---

                    all_spikes[:, timestep:end_step] = frame_spikes.float()

                timestep = end_step

            if timestep == 0:
                 print("Warning: No audio frames processed or audio too short.")
            elif timestep < self.duration:
                 print(f"Warning: Audio ended before filling duration. Encoded {timestep} steps.")

            return all_spikes

        except Exception as e:
            print(f"Error encoding audio {audio_path}: {e}")
            # Return zero spikes on error
            return torch.zeros((self.num_neurons, self.duration), device=self.device)


# --- Example Usage ---
if __name__ == '__main__':
    print(f"\n--- Encoder Test ---")
    print(f"Using Device: {DEVICE}")

    # Config (assuming 100 neurons total, 100ms duration)
    total_neurons = 128+100+100+13 # text+img+vid+aud
    duration_ms = 200
    dt_val = 1.0

    # --- Text ---
    try:
        text_encoder = TextEncoder(num_neurons=128, duration=duration_ms, dt=dt_val, max_rate=50)
        text_spikes = text_encoder.encode("Test!")
        print(f"Text spikes shape: {text_spikes.shape}, Sum: {text_spikes.sum()}")
        # Verify non-zero spikes if text is valid
        if text_spikes.sum() == 0: print("Warning: Text encoding produced zero spikes.")
    except Exception as e: print(f"TextEncoder Error: {e}")

    # --- Image ---
    # Create a dummy image file for testing
    dummy_img_path = "_FUM_Training/tests/dummy_image.png"
    if not os.path.exists(os.path.dirname(dummy_img_path)): os.makedirs(os.path.dirname(dummy_img_path))
    cv2.imwrite(dummy_img_path, np.random.randint(0, 256, (50, 50), dtype=np.uint8))
    try:
        img_encoder = ImageEncoder(num_neurons=100, duration=duration_ms, dt=dt_val, target_size=(10, 10), max_rate=100)
        img_spikes = img_encoder.encode(dummy_img_path)
        print(f"Image spikes shape: {img_spikes.shape}, Sum: {img_spikes.sum()}")
        if img_spikes.sum() == 0: print("Warning: Image encoding produced zero spikes.")
    except Exception as e: print(f"ImageEncoder Error: {e}")

    # --- Video ---
    # Create a dummy video file for testing (requires cv2 writer)
    dummy_vid_path = "_FUM_Training/tests/dummy_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(dummy_vid_path, fourcc, 20.0, (64, 64))
    for _ in range(60): # 3 seconds of video
        frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    try:
        vid_encoder = VideoEncoder(num_neurons=100, duration=duration_ms, dt=dt_val, target_size=(10, 10), max_rate=100, frame_skip=2)
        vid_spikes = vid_encoder.encode(dummy_vid_path)
        print(f"Video spikes shape: {vid_spikes.shape}, Sum: {vid_spikes.sum()}")
        # Video diff might produce zero spikes if frames are identical
    except Exception as e: print(f"VideoEncoder Error: {e}")

    # --- Audio ---
    # Create a dummy audio file for testing (requires soundfile or similar)
    # Using librosa's example tone for simplicity if soundfile not installed
    dummy_aud_path = "_FUM_Training/tests/dummy_audio.wav"
    try:
        import soundfile as sf
        sr_aud = 22050
        duration_aud = 2 # seconds
        frequency = 440
        t_aud = np.linspace(0., duration_aud, int(sr_aud * duration_aud))
        amplitude = np.iinfo(np.int16).max * 0.5
        data_aud = (amplitude * np.sin(2. * np.pi * frequency * t_aud)).astype(np.int16)
        sf.write(dummy_aud_path, data_aud, sr_aud)
    except ImportError:
        print("Warning: soundfile not installed. Cannot create dummy audio file.")
        dummy_aud_path = librosa.ex('trumpet') # Use librosa example if available

    try:
        aud_encoder = AudioEncoder(num_neurons=13, duration=duration_ms, dt=dt_val, n_mfcc=13, max_rate=100)
        if os.path.exists(dummy_aud_path):
             aud_spikes = aud_encoder.encode(dummy_aud_path)
             print(f"Audio spikes shape: {aud_spikes.shape}, Sum: {aud_spikes.sum()}")
             if aud_spikes.sum() == 0: print("Warning: Audio encoding produced zero spikes.")
        else:
             print("Skipping audio test as dummy file could not be created.")
    except Exception as e: print(f"AudioEncoder Error: {e}")


    print("--- Encoder Test Complete ---")
