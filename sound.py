import sounddevice as sd
from scipy.io.wavfile import write

# === Settings ===
SAMPLE_RATE = 44100      # CD quality
DURATION = 5             # seconds to record
OUTPUT_FILE = "recording.wav"

print("Recording...")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
sd.wait()  # Wait until recording is finished
write(OUTPUT_FILE, SAMPLE_RATE, audio)
print(f"Saved to {OUTPUT_FILE}")

