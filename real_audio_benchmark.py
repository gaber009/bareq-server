"""
real_audio_benchmark.py — Real Audio End-to-End Benchmark for License Plate Recognition
========================================================================================
Performs REAL AUDIO testing across the complete pipeline:
  Audio (WAV) -> PCM16 Array -> Whisper ASR (Raw Text) -> PlateDecoder -> Final Plate

Measures:
  1. RAW Whisper ASR output for each audio utterance
  2. DECODED Plate & Validity
  3. Composite Confidence Score
  4. End-to-End Latency per audio utterance
  5. Real Audio Exact Plate Match Accuracy
"""

import time
import wave
import io
import os
import sys
import numpy as np

# Ensure server directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from asr_engine import get_model_manager, transcribe_audio, pcm16_to_wav
from plate_decoder import get_decoder

def create_real_arabic_speech_wav(text: str) -> bytes:
    """
    Synthesizes real Arabic speech audio WAV (16kHz PCM16 Mono).
    """
    from gtts import gTTS
    tts = gTTS(text=text, lang='ar', slow=False)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    
    # Convert MP3 to WAV using PyAV
    import av
    container = av.open(mp3_fp)
    resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
    pcm_chunks = []
    for frame in container.decode(audio=0):
        resampled_frames = resampler.resample(frame)
        for rf in resampled_frames:
            pcm_chunks.append(rf.to_ndarray().tobytes())
    all_pcm = b''.join(pcm_chunks)
    return pcm16_to_wav(all_pcm, 16000)


def run_real_audio_benchmark():
    print("=" * 70)
    print("      REAL AUDIO END-TO-END BENCHMARK (Whisper ASR + PlateDecoder)")
    print("=" * 70)

    print("\n1. Initializing ASR Engine & Model...")
    t0 = time.time()
    mgr = get_model_manager()
    model = mgr.load_model()
    print(f"   Model ready in {time.time() - t0:.2f}s")

    decoder = get_decoder()

    # ── Real Spoken Arabic Plate Utterances ──
    audio_test_utterances = [
        ("ألف باء جيم واحد اتنين تلاتة أربعة", "أ ب ج 1234"),
        ("ألف باء دال خمسة ستة سبعة ثمانية", "أ ب د 5678"),
        ("طاء دال لام أربعة سبعة ثمانية اثنين", "ط د ل 4782"),
        ("راء كاف عين سبعة خمسة واحد واحد", "ر ك ع 7511"),
        ("دال باء صاد واحد اتنين تلاتة أربعة", "د ب ص 1234"),
        ("حاء باء سين تسعة خمسة صفر صفر", "ح ب س 9500"),
        ("عين قاف كاف واحد صفر اتنين تلاتة", "ع ق ك 1023"),
        ("سين شين صاد أربعة خمسة ستة سبعة", "س ش ص 4567"),
    ]

    print(f"\n2. Synthesizing and testing {len(audio_test_utterances)} REAL ARABIC AUDIO samples...")

    correct_matches = 0
    total_samples = len(audio_test_utterances)

    for idx, (target_speech, expected_plate) in enumerate(audio_test_utterances, 1):
        print("\n" + "-" * 50)
        print(f"Sample #{idx}: Spoken Input = \"{target_speech}\"")
        
        # 1. Generate Real Arabic WAV Audio
        t_gen = time.time()
        wav_bytes = create_real_arabic_speech_wav(target_speech)
        gen_ms = (time.time() - t_gen) * 1000
        
        # Parse WAV to Float32 numpy array
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, 'rb') as wf:
            raw_pcm = wf.readframes(wf.getnframes())
            audio_np = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0

        audio_duration_s = len(raw_pcm) / (16000 * 2)

        # 2. Run RAW ASR (Whisper)
        t_asr = time.time()
        asr_res = transcribe_audio(audio_np, is_partial=False)
        asr_latency_ms = (time.time() - t_asr) * 1000
        raw_text = asr_res.get("text", "").strip()
        segments = asr_res.get("segments", [])
        logprobs = [s.get("avg_logprob", 0.0) for s in segments if "avg_logprob" in s]

        print(f"Audio Duration:   {audio_duration_s:.2f}s")
        print(f"RAW ASR Output:   \"{raw_text}\" (took {asr_latency_ms:.0f}ms)")

        # 3. Run Plate Decoder
        t_dec = time.time()
        dec = decoder.decode_final(raw_text, asr_segment_confidences=logprobs)
        dec_latency_ms = (time.time() - t_dec) * 1000
        
        decoded_plate = dec["plate"]
        is_valid = dec["valid"]
        confidence = dec["confidence"]

        print(f"NORMALIZED Text:  \"{dec['normalized']}\"")
        print(f"DECODED Plate:    \"{decoded_plate}\"")
        print(f"Expected Plate:   \"{expected_plate}\"")
        print(f"Validity & Conf:  valid={is_valid} | confidence={confidence*100:.1f}%")

        if is_valid and decoded_plate == expected_plate:
            correct_matches += 1
            print("RESULT:           [OK] MATCH PERFECT")
        else:
            print("RESULT:           [WARN] MISMATCH OR INCOMPLETE")

    accuracy = (correct_matches / total_samples) * 100

    print("\n" + "=" * 70)
    print("               REAL AUDIO BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total Real Audio Samples:    {total_samples}")
    print(f"Exact Plate Match Accuracy:  {accuracy:.2f}% ({correct_matches}/{total_samples})")
    print("=" * 70)

if __name__ == "__main__":
    run_real_audio_benchmark()
