import sounddevice as sd
import soundfile as sf
import numpy as np

FILENAME = "my_recording.wav"
SAMPLE_RATE = 44100

recording = []
is_recording = False

def audio_callback(indata, frames, time, status):
    if is_recording:
        recording.append(indata.copy())

print("Press ENTER to start recording...")
input()

is_recording = True
print("Recording... Press ENTER to stop.")

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
    input()  # wait for second ENTER
    is_recording = False

if len(recording) == 0:
    print("No audio captured — check microphone permissions or device selection.")
else:
    audio_np = np.concatenate(recording, axis=0)
    sf.write(FILENAME, audio_np, SAMPLE_RATE)
    print(f"Saved to {FILENAME}")
