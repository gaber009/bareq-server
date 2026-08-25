import wave, io
import numpy as np
from faster_whisper import WhisperModel

wav_path = r"c:\Users\ahjbs\OneDrive\Desktop\Bareq_Server\debug_last_utterance.wav"
with wave.open(wav_path, 'rb') as wf:
    raw_bytes = wf.readframes(wf.getnframes())

audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

print("Testing WhisperModel on debug_last_utterance.wav with various parameters...")

model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=6)

print("\n1. Standard transcribe (language='ar'):")
segs, _ = model.transcribe(audio_np, language="ar")
text1 = " ".join([s.text.strip() for s in segs])
print(f"   Result: '{text1}'")

print("\n2. Transcribe without initial_prompt:")
segs, _ = model.transcribe(audio_np, language="ar", condition_on_previous_text=False, no_speech_threshold=0.9)
text2 = " ".join([s.text.strip() for s in segs])
print(f"   Result: '{text2}'")

print("\n3. Transcribe auto language detection:")
segs, info = model.transcribe(audio_np, condition_on_previous_text=False)
text3 = " ".join([s.text.strip() for s in segs])
print(f"   Detected Lang: {info.language} (prob={info.language_probability:.2f}) | Result: '{text3}'")
