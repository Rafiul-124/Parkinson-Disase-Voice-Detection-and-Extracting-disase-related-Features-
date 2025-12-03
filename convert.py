import os
import math
import numpy as np
import parselmouth
from parselmouth.praat import call, PraatError
import librosa
from scipy.stats import entropy

# Path to your WAV file
WAV_PATH = r"C:\Users\USER\Desktop\Perkinson\sample_voice.wav"


def safe_call(object_or_list, command, *args):
    """
    Call Praat function and return np.nan on failure instead of raising.
    object_or_list: either a parselmouth object or a list of objects (for two-input functions)
    command: string name of Praat command
    args: additional arguments for the command
    """
    try:
        return call(object_or_list, command, *args)
    except PraatError as e:
        print(f"Warning: PraatError for '{command}': {e}")
        return float("nan")
    except Exception as e:
        print(f"Warning: Exception for '{command}': {e}")
        return float("nan")


def RPDE_feature(x, bins=50):
    # Relative permutation entropy proxy using histogram entropy
    hist, _ = np.histogram(x, bins=bins, density=True)
    return entropy(hist + 1e-12)


def DFA_feature(x):
    # Detrended fluctuation analysis (simple implementation)
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 50:
        return float("nan")
    y = np.cumsum(x - np.mean(x))
    scales = np.floor(np.logspace(1, math.log10(n/4), num=20)).astype(int)
    fluctuations = []
    for s in np.unique(scales):
        if s < 4:
            continue
        segments = n // s
        if segments < 2:
            continue
        segs = y[:segments * s].reshape((segments, s))
        local_fluct = []
        t = np.arange(s)
        A = np.vstack([t, np.ones(s)]).T
        for seg in segs:
            # linear fit
            coeffs, *_ = np.linalg.lstsq(A, seg, rcond=None)
            trend = A @ coeffs
            local_fluct.append(np.sqrt(np.mean((seg - trend) ** 2)))
        fluctuations.append(np.mean(local_fluct))
    fluctuations = np.array(fluctuations)
    valid = fluctuations > 0
    if valid.sum() < 2:
        return float("nan")
    log_s = np.log(np.unique(scales)[valid])
    log_f = np.log(fluctuations[valid])
    alpha = np.polyfit(log_s, log_f, 1)[0]
    return alpha


def extract_parkinson_features(wav_path):
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"File not found: {wav_path}")

    # Load with parselmouth (Praat)
    sound = parselmouth.Sound(wav_path)

    # Create Pitch object
    pitch = safe_call(sound, "To Pitch", 0.0, 75, 600)  # time step 0.0 (auto), f0min 75, f0max 600

    # Fundamental frequency parameters (note: new Praat needs interpolation argument)
    Fo = safe_call(pitch, "Get mean", 0, 0, "Hertz", "Parabolic")
    Fhi = safe_call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    Flo = safe_call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")

    # Jitter measures
    # These argument lists are chosen to be robust; if your Praat complains you will get nan
    jitter_local = safe_call(pitch, "Get jitter (local)", 0, 0.02, 1.3)
    jitter_abs = safe_call(pitch, "Get jitter (local, absolute)", 0, 0.02, 1.3)
    rap = safe_call(pitch, "Get jitter (rap)", 0, 0.02, 1.3)
    ppq = safe_call(pitch, "Get jitter (ppq5)", 0, 0.02, 1.3)
    ddp = float(rap) * 3 if not math.isnan(rap) else float("nan")  # DDP often defined as 3*RAP

    # Shimmer measures (these need both sound and pitch objects)
    shimmer_local = safe_call([sound, pitch], "Get shimmer (local)", 0, 0.02, 1.3, 1.6)
    shimmer_db = safe_call([sound, pitch], "Get shimmer (local_dB)", 0, 0.02, 1.3, 1.6)
    apq3 = safe_call([sound, pitch], "Get shimmer (apq3)", 0, 0.02, 1.3, 1.6)
    apq5 = safe_call([sound, pitch], "Get shimmer (apq5)", 0, 0.02, 1.3, 1.6)
    apq11 = safe_call([sound, pitch], "Get shimmer (apq11)", 0, 0.02, 1.3, 1.6)
    dda = float(apq3) * 3 if not math.isnan(apq3) else float("nan")

    # Harmonicity / HNR and NHR
    harmonicity = safe_call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    HNR = safe_call(harmonicity, "Get mean", 0, 0) if not math.isnan(harmonicity) else float("nan")
    # Praat has different harmonicity measures — use the cc-method for mean HNR above.
    # For NHR we try a Praat call; if it errors, fallback to nan
    NHR = safe_call(sound, "To Harmonicity (noise-to-harmonics)", 0.01, 75, 0.1, 1.0)

    # Load waveform with librosa for time-series features
    y, sr = librosa.load(wav_path, sr=22050)
    if y.size == 0:
        raise ValueError("Loaded audio is empty")

    # RPDE (approx via histogram entropy)
    RPDE = RPDE_feature(y)

    # DFA
    DFA_val = DFA_feature(y)

    # Spread1 & Spread2 (use mel-spectrogram statistics)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S + 1e-12)
    # spread1: mean of spectral slope (approx using frame-to-frame differences)
    spread1 = float(np.mean(np.diff(S_db, axis=1)))
    spread2 = float(np.std(S_db))

    # D2 (approximate correlation dimension proxy)
    denom = np.mean(np.abs(np.diff(y)))
    if denom == 0 or np.isnan(denom):
        D2 = float("nan")
    else:
        D2 = float(np.log(np.std(y) / denom + 1e-12))

    # PPE (Pitch Period Entropy) — compute from Praat pitch values
    try:
        pitch_array = pitch.selected_array['frequency']
        pitch_vals = pitch_array[pitch_array > 0]
        if pitch_vals.size == 0:
            PPE = float("nan")
        else:
            hist, _ = np.histogram(pitch_vals, bins=30, density=True)
            PPE = float(entropy(hist + 1e-12))
    except Exception as e:
        print(f"Warning: Could not compute PPE: {e}")
        PPE = float("nan")

    # Build result dict in the exact column order you showed earlier
    features = {
        "MDVP:Fo(Hz)": float(Fo) if not math.isnan(Fo) else float("nan"),
        "MDVP:Fhi(Hz)": float(Fhi) if not math.isnan(Fhi) else float("nan"),
        "MDVP:Flo(Hz)": float(Flo) if not math.isnan(Flo) else float("nan"),
        "MDVP:Jitter(%)": float(jitter_local) if not math.isnan(jitter_local) else float("nan"),
        "MDVP:Jitter(Abs)": float(jitter_abs) if not math.isnan(jitter_abs) else float("nan"),
        "MDVP:RAP": float(rap) if not math.isnan(rap) else float("nan"),
        "MDVP:PPQ": float(ppq) if not math.isnan(ppq) else float("nan"),
        "Jitter:DDP": float(ddp) if not math.isnan(ddp) else float("nan"),
        "MDVP:Shimmer": float(shimmer_local) if not math.isnan(shimmer_local) else float("nan"),
        "MDVP:Shimmer(dB)": float(shimmer_db) if not math.isnan(shimmer_db) else float("nan"),
        "Shimmer:APQ3": float(apq3) if not math.isnan(apq3) else float("nan"),
        "Shimmer:APQ5": float(apq5) if not math.isnan(apq5) else float("nan"),
        "MDVP:APQ": float(apq11) if not math.isnan(apq11) else float("nan"),
        "Shimmer:DDA": float(dda) if not math.isnan(dda) else float("nan"),
        "NHR": float(NHR) if not math.isnan(NHR) else float("nan"),
        "HNR": float(HNR) if not math.isnan(HNR) else float("nan"),
        "RPDE": float(RPDE),
        "DFA": float(DFA_val) if not math.isnan(DFA_val) else float("nan"),
        "spread1": float(spread1),
        "spread2": float(spread2),
        "D2": float(D2),
        "PPE": float(PPE),
    }

    return features


if __name__ == "__main__":
    try:
        feats = extract_parkinson_features(WAV_PATH)
        print("\n--- Extracted Parkinson-like Features ---")
        for k, v in feats.items():
            # Print numeric nicely
            try:
                print(f"{k:<22}: {v:.6f}")
            except Exception:
                print(f"{k:<22}: {v}")
    except Exception as e:
        print(f"\nFatal error: {e}")
        raise
