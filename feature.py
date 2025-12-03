import parselmouth
from parselmouth.praat import call
import numpy as np
import librosa
from scipy.signal import hilbert
from python_speech_features import mfcc
import warnings
warnings.filterwarnings("ignore")

file_path = r"C:\Users\USER\Desktop\Perkinson\my_recording.wav"
def extract_voice_features(file_path):

    # ============================
    # Load audio
    # ============================
    snd = parselmouth.Sound(file_path)
    y, sr = librosa.load(file_path, sr=None)

    # =============================================================
    # I. JITTER (Using Praat)
    # =============================================================
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    pointProcess = call(snd, "To PointProcess (periodic, cc)", 75, 600)

    local_jitter = call(pointProcess, "Get jitter (local)", 0, 0, 75, 600, 1.3)
    local_abs_jitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 75, 600, 1.3)
    rap = call(pointProcess, "Get jitter (rap)", 0, 0, 75, 600, 1.3)
    ppq5 = call(pointProcess, "Get jitter (ppq5)", 0, 0, 75, 600, 1.3)
    ddp = call(pointProcess, "Get jitter (ddp)", 0, 0, 75, 600, 1.3)

    # =============================================================
    # II. SHIMMER (Using Praat)
    # =============================================================
    local_shimmer = call([snd, pointProcess], "Get shimmer (local)", 0, 0, 75, 600, 1.3, 1.6)
    apq3 = call([snd, pointProcess], "Get shimmer (apq3)", 0, 0, 75, 600, 1.3, 1.6)
    apq5 = call([snd, pointProcess], "Get shimmer (apq5)", 0, 0, 75, 600, 1.3, 1.6)
    dda = call([snd, pointProcess], "Get shimmer (dda)", 0, 0, 75, 600, 1.3, 1.6)

    # =============================================================
    # III. FUNDAMENTAL FREQUENCY
    # =============================================================
    f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
    f0_min = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    f0_max = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    f0_sd  = call(pitch, "Get standard deviation", 0, 0, "Hertz")

    # =============================================================
    # IV. HNR / NHR
    # =============================================================
    hnr = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr_val = call(hnr, "Get mean", 0, 0)
    nhr_val = 1 / hnr_val if hnr_val != 0 else 0

    # =============================================================
    # V. NONLINEAR DYNAMICS (approximate)
    # =============================================================
    # RPDE & DFA from librosa
    rpde = librosa.feature.spectral_flatness(y=y)[0].mean()
    ppe  = np.std(librosa.yin(y, fmin=75, fmax=600, sr=sr))
    # DFA via detrended fluctuation
    def dfa(data):
        n = len(data)
        X = np.cumsum(data - np.mean(data))
        L = np.floor(np.logspace(1, np.log10(n/4), 20)).astype(int)
        F = []
        for l in L:
            segments = n // l
            RMS = []
            for i in range(segments):
                idx = slice(i*l, (i+1)*l)
                t = np.arange(l)
                p = np.polyfit(t, X[idx], 1)
                RMS.append(np.sqrt(np.mean((X[idx] - np.polyval(p, t))**2)))
            F.append(np.mean(RMS))
        return np.polyfit(np.log(L), np.log(F), 1)[0]

    dfa_val = dfa(y)

    # =============================================================
    # VI. MFCC + FORMANTS
    # =============================================================
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    formants = extract_formants(snd)

    # =============================================================
    # Done
    # =============================================================
    return {
        "Jitter_Local": local_jitter,
        "Jitter_Abs": local_abs_jitter,
        "RAP": rap,
        "PPQ5": ppq5,
        "DDP": ddp,

        "Shimmer_Local": local_shimmer,
        "APQ3": apq3,
        "APQ5": apq5,
        "DDA": dda,

        "F0_Mean": f0_mean,
        "F0_Min": f0_min,
        "F0_Max": f0_max,
        "F0_SD": f0_sd,

        "HNR": hnr_val,
        "NHR": nhr_val,

        "RPDE": rpde,
        "PPE": ppe,
        "DFA": dfa_val,

        "MFCCs": mfccs,
        "Formants": formants,
    }

# =============================
# Helper: Extract Formants
# =============================
def extract_formants(sound):
    formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.02, 50)
    f1 = call(formant, "Get mean", 1, 0, 0, "Hertz")
    f2 = call(formant, "Get mean", 2, 0, 0, "Hertz")
    f3 = call(formant, "Get mean", 3, 0, 0, "Hertz")
    return (f1, f2, f3)

# =============================
# Run example
# =============================
if __name__ == "__main__":
    result = extract_voice_features(file_path)
    for k, v in result.items():
        print(k, ":", v)
