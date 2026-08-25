import os, sys, wave, io
import numpy as np

wav_path = r"c:\Users\ahjbs\OneDrive\Desktop\Bareq_Server\debug_last_utterance.wav"

if not os.path.exists(wav_path):
    print("No debug WAV file found yet.")
    sys.exit(0)

with wave.open(wav_path, 'rb') as wf:
    n_channels = wf.getnchannels()
    sampwidth = wf.getsampwidth()
    framerate = wf.getframerate()
    n_frames = wf.getnframes()
    raw_bytes = wf.readframes(n_frames)

audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
duration = len(audio_np) / framerate
rms = float(np.sqrt(np.mean(audio_np**2)))
peak = float(np.max(np.abs(audio_np)))

print(f"DEBUG WAV DIAGNOSTICS:")
print(f"  Channels:   {n_channels}")
print(f"  Sample Rate:{framerate} Hz")
print(f"  Duration:   {duration:.2f} seconds")
print(f"  Raw RMS:    {rms:.6f}")
print(f"  Raw Peak:   {peak:.6f}")

# Test ASR on RAW audio vs NORMALIZED (gain boosted) audio
from asr_engine import get_model_manager, transcribe_audio
from plate_decoder import get_decoder

print("\nRunning ASR on raw audio...")
res_raw = transcribe_audio(audio_np, is_partial=False)
print(f"  RAW ASR: '{res_raw.get('text')}'")

# Apply Peak Gain Normalization (Boost volume if quiet)
if peak > 0.001:
    audio_boosted = audio_np / peak * 0.85
    print("\nRunning ASR on VOLUME-NORMALIZED audio (Peak normalized to 0.85)...")
    res_boosted = transcribe_audio(audio_boosted, is_partial=False)
    print(f"  BOOSTED ASR: '{res_boosted.get('text')}'")
    
    decoder = get_decoder()
    dec = decoder.decode_final(res_boosted.get('text', ''))
    print(f"  DECODED: '{dec}'")
