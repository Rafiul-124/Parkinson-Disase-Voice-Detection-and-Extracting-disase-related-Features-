import parselmouth
from parselmouth.praat import call
import librosa
import numpy as np
import os

# **********************************************
# **** Input WAV Path (No conversion needed) ****
# **********************************************
AUDIO_WAV_PATH = r"C:\Users\USER\Desktop\Perkinson\sample_voice.wav"


def extract_voice_features(audio_file_path):
    """Extract Jitter, Shimmer, F0, HNR + MFCC features."""

    # Load WAV with Parselmouth
    try:
        sound = parselmouth.Sound(audio_file_path)
    except Exception as e:
        print(f"ERROR: Parselmouth ফাইল লোড করতে ব্যর্থ: {e}")
        return None

    # Pitch extraction
    f0_min, f0_max = 75, 600
    pitch = call(sound, "To Pitch", 0.0, f0_min, f0_max)

    # Praat voice measurements
    jitter_local = call(pitch, "Get jitter (local)", 0, 0.02, 1.3)
    shimmer_local = call([sound, pitch], "Get shimmer (local)", 0, 0.02, 1.3, 1.6)

    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0_min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")

    # MFCC using Librosa
    y, sr = librosa.load(audio_file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # Convert MFCC to dict
    mfccs_features = {f"MFCC_{i+1}": float(v) for i, v in enumerate(mfccs_mean)}

    # Combine all results
    results = {
        "Jitter (Local)": jitter_local,
        "Shimmer (Local)": shimmer_local,
        "HNR (dB)": hnr,
        "F0 Mean (Hz)": f0_mean,
    }
    results.update(mfccs_features)

    return results


# **********************************************
# **** MAIN EXECUTION ****
# **********************************************
if __name__ == "__main__":
    if not os.path.exists(AUDIO_WAV_PATH):
        print(f"ERROR: File not found: {AUDIO_WAV_PATH}")
    else:
        features = extract_voice_features(AUDIO_WAV_PATH)

        if features:
            print("\n--- Extracted Voice Features ---")
            for key, value in features.items():
                print(f"{key:<20}: {value:.4f}")
            print("--------------------------------\n")
