import time, os, psutil, numpy as np
from asr_engine import get_model_manager, transcribe_audio, detect_hardware

print("1. Detecting hardware...")
hw = detect_hardware()
print(f"Device selected: {hw['device']} | compute_type: {hw['compute_type']}")
for n in hw.get('notes', []):
    print(f"  Note: {n}")

print("\n2. Loading model...")
t0 = time.time()
mgr = get_model_manager()
model = mgr.load_model()
load_time = time.time() - t0
ram_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
print(f"Model loaded in {load_time:.2f}s | RAM: {ram_mb:.0f} MB")

# 3. Benchmark Partial Latency (0.8s audio)
sample_rate = 16000
duration = 1.5
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
audio = 0.2 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.sin(2 * np.pi * 600 * t)

print("\n3. Testing Partial Latency (0.8s audio)...")
t0 = time.time()
res_p = transcribe_audio(audio[:12800], is_partial=True)
lat_p = (time.time() - t0) * 1000
print(f"Partial Latency: {lat_p:.1f}ms | error: {res_p.get('error')}")

print("\n4. Testing Final Latency (1.5s audio)...")
t0 = time.time()
res_f = transcribe_audio(audio, is_partial=False)
lat_f = (time.time() - t0) * 1000
print(f"Final Latency: {lat_f:.1f}ms | error: {res_f.get('error')}")
