"""Cut listening samples (kept + rejected-by-reason) from a processed shard for human review.
Writes /workspace/review/samples.json with base64 MP3 audio + metadata."""
import base64, json, os, subprocess, sys, tarfile, random
import numpy as np, pandas as pd, soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_lib import decode_chunks, SR_NATIVE

OUT_DIR = os.environ.get("OUT_DIR", "/workspace/test_out")
RAW = "/workspace/yodas3_raw/data/ar/audio/0000.tar"
N_KEEP, N_PER_REASON, MAX_SEC = 20, 2, 12.0
REASONS = ["multi_speaker", "caption_speaker_turn", "overlap", "music", "music_fr", "singing", "noisy",
           "noisy_um", "low_quality", "clipping", "cut_edge", "synthetic", "word_mismatch",
           "bandwidth", "crowd_babble", "multi_speaker_dz"]
random.seed(0)

man = pd.read_parquet(f"{OUT_DIR}/manifest/0000.parquet")
man["fails"] = man.fail.fillna("").str.split(",")
pick = []
kept = man[man.keep]
pick += [("kept", r) for _, r in kept.sample(min(N_KEEP, len(kept)), random_state=0).iterrows()]
for reason in REASONS:
    sub = man[man.fails.map(lambda f: f == [reason])]  # rejected for this reason only
    if len(sub) == 0:
        sub = man[man.fails.map(lambda f: reason in f)]
    pick += [(reason, r) for _, r in sub.sample(min(N_PER_REASON, len(sub)), random_state=0).iterrows()]

need = {}
for cat, r in pick:
    need.setdefault(r.vid, []).append((cat, r))
samples = []
with tarfile.open(RAW) as tf:
    for m in tf:
        key, ext = m.name.rsplit(".", 1)
        if ext != "webm" or key not in need:
            continue
        b = tf.extractfile(m).read()
        items = need[key]
        for off, x in decode_chunks(b):
            blk_end = off + x.shape[1] / SR_NATIVE
            for cat, r in items:
                if off <= r.start and r.end <= blk_end:
                    a = int((r.start - off) * SR_NATIVE)
                    b = int((min(r.end, r.start + MAX_SEC) - off) * SR_NATIVE)
                    seg = x[:, a:b]
                    mono = seg.mean(0)
                    wav = f"/tmp/_rv.wav"
                    sf.write(wav, mono, SR_NATIVE)
                    mp3 = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", wav, "-ar", "44100", "-ac", "1",
                                          "-b:a", "160k", "-f", "mp3", "-"], capture_output=True).stdout
                    g = lambda k: (None if k not in r or pd.isna(r[k]) else (round(float(r[k]), 3) if isinstance(r[k], (float, np.floating)) else r[k]))
                    samples.append({
                        "cat": cat, "id": r.id, "salvaged": isinstance(r.get("salvaged_from"), str), "dur": round(float(r.dur), 2), "fail": r.fail,
                        "caption": r.text, "tashkeel": g("text_tashkeel"),
                        "scores": {k: g(k) for k in ["dnsmos_ovrl", "dnsmos_bak", "um_mos_overall", "aes_pc",
                                                     "tag_music", "fr_music_p95", "seg_n_spk", "dz_2nd_spk_sec",
                                                     "spk_min_sim", "tag_synth", "bw_hz", "clip_frac", "asr_cer"]},
                        "mp3": base64.b64encode(mp3).decode()})
os.makedirs("/workspace/review", exist_ok=True)
json.dump(samples, open("/workspace/review/samples.json", "w"), ensure_ascii=False)
print(len(samples), "samples", sum(len(s["mp3"]) for s in samples) / 1e6, "MB b64")
