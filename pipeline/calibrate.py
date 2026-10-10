import json, sys, tarfile, time, os
import numpy as np, pandas as pd, torch, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_lib import decode_chunks, make_chunks, signal_metrics, to_rates, attach_text, SR_NATIVE
from silero_vad import load_silero_vad, get_speech_timestamps

N_VID = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUT = "/workspace/calib"; os.makedirs(f"{OUT}/wav", exist_ok=True)
torch.set_num_threads(8)
vad = load_silero_vad(onnx=True)
tf = tarfile.open("/workspace/yodas3_raw/data/ar/audio/0000.tar")
rows, w16s = [], []
t0 = time.time(); audio_s = 0; nv = 0; pend = {}
for m in tf:
    key, ext = m.name.rsplit(".", 1)
    pend.setdefault(key, {})[ext] = tf.extractfile(m).read()
    if not {"json", "webm"} <= pend[key].keys():
        continue
    d = pend.pop(key); meta = json.loads(d["json"]); nv += 1
    for off, x in decode_chunks(d["webm"]):
        mono = x.mean(0); audio_s += x.shape[1] / SR_NATIVE
        w16full, _ = to_rates(mono)
        ts = get_speech_timestamps(torch.from_numpy(w16full), vad, sampling_rate=16000, threshold=0.5,
                                   min_silence_duration_ms=100, speech_pad_ms=30, return_seconds=True)
        sp = [(t["start"], t["end"]) for t in ts]
        for (s, e, ss, se, inner) in make_chunks(sp, x.shape[1] / SR_NATIVE):
            seg = x[:, int(s * SR_NATIVE):int(e * SR_NATIVE)]
            mt, mono_seg = signal_metrics(seg, ss - s, se - s)
            w16, w24 = to_rates(mono_seg)
            text, nw = attach_text(meta.get("transcript"), off + s, off + e)
            cid = f"{key}_{int((off+s)*1000):08d}"
            rows.append(dict(id=cid, vid=key, start=off + s, end=off + e, dur=e - s, speech_ratio=inner / (e - s),
                             est_bw_meta=meta.get("estimated_bandwidth_hz"), n_distinct=meta.get("n_distinct_channels"),
                             text=text, n_words=nw, **mt))
            w16s.append(w16)
            sf.write(f"{OUT}/wav/{cid}.flac", w24, 24000)
    if nv >= N_VID:
        break
t_cpu = time.time() - t0
print(f"CPU: {nv} videos {audio_s/3600:.2f} h -> {len(rows)} chunks ({sum(r['dur'] for r in rows)/3600:.2f} h) in {t_cpu:.0f}s single-proc", flush=True)

from gpu_models import Judges
J = Judges()
seg_h = sum(len(w) for w in w16s) / 16000 / 3600
times = {}
res = [dict() for _ in rows]
for name in ["seg", "spk", "ced", "dnsmos", "aes", "utmos"]:
    fn = getattr(J, f"run_{name}")
    fn(w16s[:8]); torch.cuda.synchronize()  # warmup
    t = time.time(); out = fn(w16s); torch.cuda.synchronize(); dt = time.time() - t
    times[name] = dt
    print(f"{name:7s} {dt:7.1f}s  -> {seg_h*3600/dt:7.0f}x realtime", flush=True)
    for r, o in zip(res, out):
        r.update(o)
for r, o in zip(rows, res):
    o = dict(o); emb = o.pop("spk_emb"); r.update(o)
df = pd.DataFrame(rows)
df.to_parquet(f"{OUT}/calib.parquet")
print(df.describe().T.to_string())
