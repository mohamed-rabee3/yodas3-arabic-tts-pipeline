"""Stream YODAS v3 Arabic shards -> strict audio-quality audit -> clean 24 kHz FLAC chunks.

Per shard: download raw tar -> CPU workers (decode, VAD, pause-aware chunking, signal checks)
-> GPU judge cascade -> keep only chunks passing every check -> WebDataset tar + full manifest
-> delete raw tar. Resumable: finished shards have done/<shard>.done.

usage: python run_shards.py [first_shard] [last_shard]
"""
import io, json, os, re, sys, tarfile, threading, time, queue, traceback, math
import multiprocessing as mp
from collections import deque

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REPO = "espnet/yodas3"
LANG = "ar"
RAW = "/workspace/yodas3_raw"
OUT = os.environ.get("OUT_DIR", "/workspace/yodas3_ar_clean")
N_WORKERS = 12
PREFETCH = 2          # raw shards on disk ahead of the one being processed
GPU_BATCH_SEC = 1800  # seconds of candidate audio per GPU cascade batch

# ------------------------------------------------------------------ thresholds (single strict tier)
# A chunk is kept only if it passes EVERY rule. Rule names are recorded as reject reasons.
SIGNAL_RULES = {
    "clipping":      lambda m: m["clip_frac"] < 1e-5 and m["plateau_runs"] <= 1 and m["peak"] <= 1.0,
    "level":         lambda m: -35 <= m["speech_dbfs"] <= -6,
    "bandwidth":     lambda m: m["bw_hz"] >= 10000,
    "stereo_phase":  lambda m: m["lr_corr"] >= 0.6,
    "dc_offset":     lambda m: abs(m["dc"]) < 0.01,
    "low_speech":    lambda m: m["speech_ratio"] >= 0.6,
    "caption_speaker_turn": lambda m: not (m["text"] and ">>" in m["text"]),
    "cut_edge":      lambda m: all(not (m[f"edge_{e}_excess_db"] > 12 and m[f"edge_{e}_db"] > -30)
                                   for e in ("head", "tail")),
}
# Thresholds for the neural judges. Values marked (AITW) were chosen on the AITW human-labelled
# YODAS/Emilia set (see aitw_eval.py) to catch >= 95% of bad clips per category.
THRESH = dict(
    seg_overlap_max=0.01,
    music_ced_max=0.10, singing_ced_max=0.05, crowd_max=0.05, laughter_max=0.05,
    noise_tag_max=0.20, effects_max=0.20, speech_min=0.60, synth_max=0.30,
    fr_music_p95_max=0.50, fr_singing_p95_max=0.50,
    dnsmos_bak_min=3.8, dnsmos_ovrl_min=2.8, dnsmos_sig_min=3.3,      # (AITW)
    um_overall_min=3.3, um_scoreq_min=3.3,                           # (AITW)
    dz_2nd_spk_max=0.2, dz_overlap_max=0.01,                         # (AITW)
    spk_min_sim=0.5,
    aes_pc_max=2.6,
)
T = THRESH
GPU_STAGES = [  # (model, rules) in increasing cost order; later models only see survivors
    ("seg", {   # pyannote seg-3.0: cheap pre-filter
        "multi_speaker": lambda m: m["seg_n_spk"] == 1,
        "overlap":       lambda m: m["seg_overlap_frac"] <= T["seg_overlap_max"],
        "low_speech":    lambda m: m["seg_speech_frac"] >= 0.6,
    }),
    ("ced", {
        "music":         lambda m: m["tag_music"] <= T["music_ced_max"],
        "singing":       lambda m: m["tag_singing"] <= T["singing_ced_max"],
        "crowd_babble":  lambda m: m["tag_crowd"] <= T["crowd_max"],
        "laughter":      lambda m: m["tag_laughter"] <= T["laughter_max"],
        "noise_tag":     lambda m: m["tag_noise"] <= T["noise_tag_max"],
        "sound_effects": lambda m: m["tag_effects"] <= T["effects_max"],
        "not_speech":    lambda m: m["tag_speech"] >= T["speech_min"],
        "synthetic":     lambda m: m["tag_synth"] <= T["synth_max"],
    }),
    ("fraed", {  # FireRedVAD AED: second, independent music/singing detector
        "music_fr":      lambda m: m["fr_music_p95"] <= T["fr_music_p95_max"],
        "singing_fr":    lambda m: m["fr_singing_p95"] <= T["fr_singing_p95_max"],
    }),
    ("dnsmos", {
        "noisy":         lambda m: m["dnsmos_bak"] >= T["dnsmos_bak_min"],
        "low_quality":   lambda m: m["dnsmos_ovrl"] >= T["dnsmos_ovrl_min"] and m["dnsmos_sig"] >= T["dnsmos_sig_min"],
    }),
    ("um", {     # UrgentMOS (WavLM, multi-metric): best noise detector on AITW
        "noisy_um":      lambda m: m["um_mos_overall"] >= T["um_overall_min"] and m["um_scoreq"] >= T["um_scoreq_min"],
    }),
    ("dz", {     # DiariZen WavLM-Conformer EEND: best multi-speaker / overlap detector on AITW
        "multi_speaker_dz": lambda m: m["dz_2nd_spk_sec"] <= T["dz_2nd_spk_max"],
        "overlap_dz":       lambda m: m["dz_overlap_frac"] <= T["dz_overlap_max"],
    }),
    ("spk", {
        "speaker_shift": lambda m: math.isnan(m["spk_min_sim"]) or m["spk_min_sim"] >= T["spk_min_sim"],
    }),
    ("aes", {
        "busy_mix":      lambda m: m["aes_pc"] <= T["aes_pc_max"],
    }),
]
ASR_RULES = {
    # tashkeel ASR (diacritics stripped) must agree with the independent YouTube caption:
    # catches non-Arabic speech, mumbling, hallucination, and caption/audio misalignment
    "no_caption_reference": lambda m: m["n_words"] >= 3,
    # every word must agree with the YouTube caption (tashkeel, punctuation and spelling variants ignored)
    "word_mismatch":        lambda m: m["words_match"],
    "asr_rate":             lambda m: 5.0 <= m["asr_cps"] <= 22.0,
    "number_conflict":      lambda m: m["number_conflicts"] == 0,
}
VIDEO_SYNTH_MEDIAN = 0.20   # whole video dropped if its median synth-speech prob exceeds this
MIN_VIDEO_BW_META = 22050   # skip videos whose metadata says effective SR < 22.05 kHz (upsampled narrowband)
BLOCK_GUARD = 0.5           # s; drop chunks whose speech is this close to a decode-block / file edge


MIN_FREE_GB_START = 40.0   # wait for this much free host memory before loading models / starting a shard
MIN_FREE_GB_RUN = 6.0      # pause feeding new videos below this


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    return 0.0


def wait_for_memory(min_gb, what):
    t0, last = time.time(), 0.0
    while mem_available_gb() < min_gb:
        if time.time() - last > 600:
            print(f"[mem] waiting for {min_gb:.0f} GB free host memory before {what} "
                  f"(now {mem_available_gb():.1f} GB, waited {(time.time() - t0) / 60:.0f} min)", flush=True)
            last = time.time()
        time.sleep(30)


def failed(rules, m):
    return [n for n, f in rules.items() if not f(m)]


# ------------------------------------------------------------------ CPU worker
_vad = None
_fvad = None


def _init_worker():
    global _vad, _fvad
    import torch
    torch.set_num_threads(1)
    from silero_vad import load_silero_vad
    _vad = load_silero_vad(onnx=True)
    from fireredvad.vad import FireRedVad, FireRedVadConfig
    cfg = FireRedVadConfig()
    cfg.use_gpu = False
    cfg.min_speech_frame = 5        # keep short blips: they must block cut points
    cfg.min_silence_frame = 10      # 100 ms pauses are resolvable
    cfg.max_speech_frame = 10 ** 7  # never force-split long speech (would create fake pauses)
    cfg.chunk_max_frame = 10 ** 7
    _fvad = FireRedVad.from_pretrained("/workspace/models/FireRedVAD/VAD", cfg)


def process_video(args):
    key, meta_b, webm = args
    _t0 = time.time()
    import torch, soundfile as sf
    from silero_vad import get_speech_timestamps
    from audit_lib import decode_chunks, make_chunks, refine_bounds, signal_metrics, to_rates, caption_window, union_intervals, SR_NATIVE
    meta = json.loads(meta_b) if meta_b else {}
    vinfo = {"vid": key, "length": meta.get("length"), "est_bw_meta": meta.get("estimated_bandwidth_hz"),
             "n_distinct": meta.get("n_distinct_channels"), "locale_lang": meta.get("locale_lang"),
             "transcript_lang": meta.get("transcript_lang")}
    bwm = meta.get("estimated_bandwidth_hz")
    if bwm is not None and not (isinstance(bwm, float) and math.isnan(bwm)) and bwm < MIN_VIDEO_BW_META:
        return vinfo, [], "video_bandwidth", 0.0, 0.0
    tr = meta.get("transcript")
    if isinstance(tr, str):
        tr = json.loads(tr)
    rows, speech_s, audio_s = [], 0.0, 0.0
    try:
        for off, x in decode_chunks(webm, chunk_sec=300.0):
            dur = x.shape[1] / SR_NATIVE
            audio_s += dur
            w16full = __import__('soxr').resample(x.mean(0), SR_NATIVE, 16000).astype(np.float32)
            ts = get_speech_timestamps(torch.from_numpy(w16full), _vad, sampling_rate=16000, threshold=0.5,
                                       min_silence_duration_ms=100, speech_pad_ms=30, return_seconds=True)
            sp_silero = [(t["start"], t["end"]) for t in ts]
            pcm = np.clip(w16full * 32768, -32768, 32767).astype(np.int16)
            fr = _fvad.detect((pcm, 16000))
            fr = fr[0] if isinstance(fr, tuple) else fr
            sp_fire = [(float(a), float(b)) for a, b in fr["timestamps"]]
            # a pause is a valid cut point only if BOTH VADs call it silence
            sp = union_intervals(sp_silero, sp_fire)
            speech_s += sum(e - s for s, e in sp)
            mono48 = x.mean(0)
            for (s, e, ss, se, inner) in refine_bounds(mono48, make_chunks(sp, dur)):
                # speech touching a decode-block edge (or the file start/end) may be a truncated word
                if ss < BLOCK_GUARD or se > dur - BLOCK_GUARD:
                    continue
                seg = x[:, int(s * SR_NATIVE):int(e * SR_NATIVE)]
                mt, mono = signal_metrics(seg, ss - s, se - s)
                cw = caption_window(tr, off + s, off + e)
                text, nw = (" ".join(cw["core"]).strip() or None), len(cw["core"])
                r = dict(id=f"{key}_{int(round((off + s) * 1000)):08d}", vid=key,
                         start=round(off + s, 3), end=round(off + e, 3), dur=round(e - s, 3),
                         speech_ratio=inner / (e - s), text=text, n_words=nw, **mt)
                r["_capwin"] = cw
                r["fail"] = failed(SIGNAL_RULES, r)
                if not r["fail"]:
                    w16, w24 = to_rates(mono)
                    pk = float(np.abs(w24).max())
                    if pk > 0.999:  # resampler overshoot guard (never clip on int16 write)
                        w24 *= 0.999 / pk
                    r["_w16"] = w16
                    r["_w24"] = w24
                    # speech regions inside the chunk (relative s) -> internal pauses for re-split salvage
                    r["_regions"] = [(max(a, s) - s, min(b, e) - s) for (a, b) in sp if b > s and a < e]
                rows.append(r)
    except Exception as ex:  # corrupt media -> drop video, keep going
        return vinfo, [], f"decode_error:{type(ex).__name__}", speech_s, audio_s
    vinfo["cpu_s"] = time.time() - _t0
    return vinfo, rows, None, speech_s, audio_s


# ------------------------------------------------------------------ GPU cascade
def asr_stage(asr, rows):
    from arabic_text import strip_tashkeel, norm_text, cer, caption_guided_digits, exact_word_match
    alive = [r for r in rows if not r["fail"]]
    if not alive:
        return
    txt = asr.run([r["_w16"] for r in alive])
    for r, t in zip(alive, txt):
        dig, n_conv, n_conf = caption_guided_digits(t, r["text"])
        hyp = norm_text(dig)          # keeps Arabic + Latin letters + digits (mixed ar/en is fine)
        ref = norm_text(r["text"])
        sp = max(r["dur"] * r["speech_ratio"], 1e-3)
        r["text_tashkeel_spelled"] = t          # verbatim ASR, numbers spelled out
        r["text_tashkeel"] = dig                # numbers as Western digits
        r["text_plain"] = " ".join(strip_tashkeel(dig).split())
        r["has_latin"] = bool(re.search(r"[A-Za-z]", dig))
        r["n_numbers"] = n_conv
        r["number_conflicts"] = n_conf
        r["asr_cer"] = cer(hyp, ref) if ref else float("nan")
        r["words_match"] = exact_word_match(dig, r["_capwin"])
        r["asr_cps"] = len(hyp.replace(" ", "")) / sp
        r["fail"] = failed(ASR_RULES, r)


SALVAGE_FAILS = {"word_mismatch", "number_conflict", "asr_rate"}  # audio was clean, only the words disagreed
SALVAGE_MIN_GAP = 0.3    # same pause rule as the main chunker: never cut inside a word
SALVAGE_MIN_PIECE = 2.0
SALVAGE_MAX_JOIN = 20.0


def _quiet_cut(w24, a, b, sr=24000, frame=0.01):
    """Quietest 10 ms frame inside the pause [a, b] (seconds, chunk-relative)."""
    hop = int(frame * sr)
    i0, i1 = int(a / frame), int(b / frame)
    if i1 <= i0:
        return (a + b) / 2
    e = [float((w24[k * hop:(k + 1) * hop] ** 2).mean()) for k in range(i0, i1)]
    return (i0 + int(np.argmin(e))) * frame + frame / 2


def _pieces(r):
    """Split a chunk at its internal pauses into >= SALVAGE_MIN_PIECE pieces. Returns [(a, b)] chunk-relative."""
    reg = r["_regions"]
    dur = r["dur"]
    cuts = [0.0]
    for (s0, e0), (s1, e1) in zip(reg, reg[1:]):
        if s1 - e0 >= SALVAGE_MIN_GAP:
            cuts.append(_quiet_cut(r["_w24"], e0 + 0.08, s1 - 0.08))
    cuts.append(dur)
    out, start = [], cuts[0]
    for c in cuts[1:]:
        if c - start >= SALVAGE_MIN_PIECE:
            out.append((start, c))
            start = c
    if out and dur - out[-1][1] > 1e-6:          # short tail -> extend the last piece
        out[-1] = (out[-1][0], dur)
    return out if len(out) >= 2 else []


def salvage_stage(asr, videos):
    """Recover exact-match sub-chunks from chunks rejected only for word disagreement."""
    from arabic_text import strip_tashkeel, caption_guided_digits, exact_word_match
    from audit_lib import caption_window_timed
    jobs = []  # (video_rows, parent, a, b)
    for _, rows in videos:
        for r in rows:
            if r["fail"] and set(r["fail"]) <= SALVAGE_FAILS and "_w24" in r:
                for a, b in _pieces(r):
                    jobs.append((rows, r, a, b))
    if not jobs:
        return 0
    w16 = [j[1]["_w16"][int(j[2] * 16000):int(j[3] * 16000)] for j in jobs]
    txt = asr.run(w16)
    matched = []
    for (rows, r, a, b), t in zip(jobs, txt):
        cw = caption_window_timed(r["_capwin"]["timed"], r["start"] + a, r["start"] + b)
        dig, _, conf = caption_guided_digits(t, " ".join(cw["core"]))
        ok = bool(cw["core"]) and conf == 0 and exact_word_match(dig, cw)
        matched.append((rows, r, a, b, t, dig, ok))
    # join consecutive matching pieces of the same parent back into longer chunks (<= 20 s)
    n_new, i = 0, 0
    while i < len(matched):
        rows, r, a, b, t, dig, ok = matched[i]
        if not ok:
            i += 1
            continue
        grp = [matched[i]]
        j = i + 1
        while (j < len(matched) and matched[j][1] is r and matched[j][6] and matched[j][2] == grp[-1][3]
               and matched[j][3] - grp[0][2] <= SALVAGE_MAX_JOIN):
            grp.append(matched[j])
            j += 1
        a0, b1 = grp[0][2], grp[-1][3]
        spelled = " ".join(g[4] for g in grp)
        digits = " ".join(g[5] for g in grp)
        cw = caption_window_timed(r["_capwin"]["timed"], r["start"] + a0, r["start"] + b1)
        if len(grp) > 1 and not exact_word_match(digits, cw):   # joined text must still match exactly
            grp, b1 = grp[:1], grp[0][3]
            spelled, digits = grp[0][4], grp[0][5]
            cw = caption_window_timed(r["_capwin"]["timed"], r["start"] + a0, r["start"] + b1)
        nr = {k: v for k, v in r.items() if k not in ("_regions", "_capwin")}
        nr.update(id=f"{r['vid']}_{int(round((r['start'] + a0) * 1000)):08d}s", start=round(r["start"] + a0, 3),
                  end=round(r["start"] + b1, 3), dur=round(b1 - a0, 3), salvaged_from=r["id"],
                  text=" ".join(cw["core"]), n_words=len(cw["core"]),
                  text_tashkeel_spelled=spelled, text_tashkeel=digits,
                  text_plain=" ".join(strip_tashkeel(digits).split()), words_match=True, fail=[],
                  asr_cer=0.0, number_conflicts=0)
        nr["_w16"] = r["_w16"][int(a0 * 16000):int(b1 * 16000)]
        nr["_w24"] = r["_w24"][int(a0 * 24000):int(b1 * 24000)]
        rows.append(nr)
        n_new += 1
        i += len(grp)
    return n_new


def gpu_cascade(J, videos, asr=None):
    """videos: list of (vinfo, rows). Mutates rows: adds scores, 'fail' list."""
    cand = [r for _, rows in videos for r in rows if not r["fail"]]
    for stage, rules in GPU_STAGES:
        alive = [r for r in cand if not r["fail"]]
        if not alive:
            break
        t0 = time.time()
        out = getattr(J, f"run_{stage}")([r["_w16"] for r in alive])
        TIMING[stage] = TIMING.get(stage, 0.0) + time.time() - t0
        TIMING[stage + "_audio_s"] = TIMING.get(stage + "_audio_s", 0.0) + sum(r["dur"] for r in alive)
        for r, o in zip(alive, out):
            r.update(o)
            r["fail"] = failed(rules, r)
        if stage == "ced":  # video-level synthetic-voice prior (computed below)
            by_vid = {}
            for r in alive:
                by_vid.setdefault(r["vid"], []).append(r["tag_synth"])
            for r in alive:
                if np.median(by_vid[r["vid"]]) > VIDEO_SYNTH_MEDIAN and "synthetic" not in r["fail"]:
                    r["fail"].append("synthetic_video")
    if asr is not None:
        t0 = time.time()
        asr_stage(asr, cand)
        TIMING["asr"] = TIMING.get("asr", 0.0) + time.time() - t0
        t0 = time.time()
        TIMING["salvaged_n"] = TIMING.get("salvaged_n", 0) + salvage_stage(asr, videos)
        TIMING["salvage"] = TIMING.get("salvage", 0.0) + time.time() - t0


# ------------------------------------------------------------------ shard driver
def download(shard):
    from huggingface_hub import hf_hub_download
    for attempt in range(20):
        try:
            return hf_hub_download(REPO, f"data/{LANG}/audio/{shard}.tar", repo_type="dataset", local_dir=RAW)
        except Exception as ex:
            print(f"[dl] {shard} attempt {attempt} failed: {ex}", flush=True)
            time.sleep(min(300, 15 * (attempt + 1)))
    raise RuntimeError(f"download failed {shard}")


def iter_tar(path):
    """Yield (key, json_bytes, webm_bytes) pairing members by key."""
    pend = {}
    with tarfile.open(path) as tf:
        for m in tf:
            if not m.isfile():
                continue
            key, ext = m.name.rsplit("/", 1)[-1].rsplit(".", 1)
            pend.setdefault(key, {})[ext] = tf.extractfile(m).read()
            if {"json", "webm"} <= pend[key].keys():
                d = pend.pop(key)
                yield key, d["json"], d["webm"]
    for key, d in pend.items():
        if "webm" in d:
            yield key, d.get("json"), d["webm"]


SCORE_COLS_DROP = {"_w16", "_w24", "_flac", "spk_emb", "_capwin", "_regions"}


def finish_videos(videos, wtar, manifest, embs, stats):
    for vinfo, rows in videos:
        for r in rows:
            keep = not r["fail"]
            rec = {k: v for k, v in r.items() if k not in SCORE_COLS_DROP}
            rec["fail"] = ",".join(r["fail"])
            rec["keep"] = keep
            rec.update({k: vinfo[k] for k in ("locale_lang", "transcript_lang", "est_bw_meta", "n_distinct")})
            manifest.append(rec)
            for f in r["fail"]:
                stats["reject_" + f] = stats.get("reject_" + f, 0.0) + r["dur"]
            if keep:
                js = {k: v for k, v in rec.items() if k not in ("fail", "keep")}
                js = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in js.items()}
                buf = io.BytesIO()
                sf.write(buf, r["_w24"], 24000, format="FLAC", subtype="PCM_16")
                for name, data in ((f"{r['id']}.flac", buf.getvalue()),
                                   (f"{r['id']}.json", json.dumps(js, ensure_ascii=False).encode())):
                    ti = tarfile.TarInfo(name)
                    ti.size = len(data)
                    ti.mtime = int(time.time())
                    wtar.addfile(ti, io.BytesIO(data))
                embs.append((r["id"], r["spk_emb"]))
                stats["kept_s"] = stats.get("kept_s", 0.0) + r["dur"]
                stats["kept_n"] = stats.get("kept_n", 0) + 1
            if not r.get("salvaged_from"):
                stats["chunk_s"] = stats.get("chunk_s", 0.0) + r["dur"]
            elif keep:
                stats["salvaged_s"] = stats.get("salvaged_s", 0.0) + r["dur"]


ASR = None
TIMING = {}


def process_shard(shard, raw_path, pool, J):
    import pandas as pd
    t0 = time.time()
    for d in ("audio", "manifest", "spk_emb", "done", "stats"):
        os.makedirs(f"{OUT}/{d}", exist_ok=True)
    tmp_tar = f"{OUT}/audio/{shard}.tar.partial"
    wtar = tarfile.open(tmp_tar, "w")
    manifest, embs, stats, vid_rows = [], [], {"videos": 0}, []
    inflight, pending, pend_sec = deque(), [], 0.0
    src = iter_tar(raw_path)
    exhausted = False
    max_videos = int(os.environ.get("MAX_VIDEOS", "0"))
    gq, gerr = queue.Queue(maxsize=2), []

    def gpu_worker():  # GPU cascade + writing overlap with CPU decoding
        while True:
            batch = gq.get()
            if batch is None:
                return
            try:
                gpu_cascade(J, batch, ASR)
                finish_videos(batch, wtar, manifest, embs, stats)
            except Exception:
                gerr.append(traceback.format_exc())

    gt = threading.Thread(target=gpu_worker, daemon=True)
    gt.start()
    submitted = 0
    while True:
        while mem_available_gb() < MIN_FREE_GB_RUN and inflight:
            time.sleep(2)   # host memory is tight: drain in-flight work before feeding more
        while not exhausted and len(inflight) < N_WORKERS * 4:
            if max_videos and submitted >= max_videos:
                exhausted = True
                break
            try:
                inflight.append(pool.apply_async(process_video, (next(src),)))
                submitted += 1
            except StopIteration:
                exhausted = True
        if not inflight:
            break
        vinfo, rows, vfail, sp_s, au_s = inflight.popleft().get()
        stats["videos"] += 1
        stats["audio_s"] = stats.get("audio_s", 0.0) + au_s
        stats["speech_s"] = stats.get("speech_s", 0.0) + sp_s
        vid_rows.append({**vinfo, "video_fail": vfail, "speech_s": sp_s, "audio_s": au_s, "n_chunks": len(rows)})
        if vfail:
            stats["video_" + vfail.split(":")[0]] = stats.get("video_" + vfail.split(":")[0], 0) + 1
        pending.append((vinfo, rows))
        pend_sec += sum(r["dur"] for r in rows if not r["fail"])
        if pend_sec >= GPU_BATCH_SEC:
            gq.put(pending)
            pending, pend_sec = [], 0.0
    if pending:
        gq.put(pending)
    gq.put(None)
    gt.join()
    if gerr:
        raise RuntimeError("GPU stage failed:\n" + gerr[0])
    wtar.close()
    os.replace(tmp_tar, f"{OUT}/audio/{shard}.tar")
    pd.DataFrame(manifest).to_parquet(f"{OUT}/manifest/{shard}.parquet")
    pd.DataFrame(vid_rows).to_parquet(f"{OUT}/manifest/{shard}.videos.parquet")
    if embs:
        np.savez(f"{OUT}/spk_emb/{shard}.npz", ids=np.array([e[0] for e in embs]),
                 emb=np.stack([e[1] for e in embs]).astype(np.float16))
    stats["wall_s"] = time.time() - t0
    stats["gpu_timing"] = dict(TIMING)
    TIMING.clear()
    json.dump(stats, open(f"{OUT}/stats/{shard}.json", "w"), indent=1)
    open(f"{OUT}/done/{shard}.done", "w").write("ok")
    a = stats.get("audio_s", 0) / 3600
    k = stats.get("kept_s", 0) / 3600
    print(f"[shard {shard}] {stats['videos']} videos {a:.1f} h -> kept {stats.get('kept_n',0)} chunks {k:.2f} h "
          f"({100*k/max(a,1e-9):.1f}% of audio) in {stats['wall_s']/60:.1f} min", flush=True)


REMOTE = os.environ.get("REMOTE", "hetzner:yodas3-arabic-tts-clean")


def upload_shard(shard):
    """Copy one finished shard's outputs to the storage box, verify sizes, mark uploaded."""
    import subprocess
    files = [f"audio/{shard}.tar", f"manifest/{shard}.parquet", f"manifest/{shard}.videos.parquet",
             f"spk_emb/{shard}.npz", f"stats/{shard}.json"]
    files = [f for f in files if os.path.exists(f"{OUT}/{f}")]
    lst = f"{OUT}/done/{shard}.files"
    open(lst, "w").write("\n".join(files) + "\n")
    for attempt in range(10):
        r = subprocess.run(["rclone", "copy", OUT, REMOTE, "--files-from", lst, "--transfers", "4",
                            "--retries", "3", "--low-level-retries", "20"], capture_output=True, text=True)
        if r.returncode == 0:
            chk = subprocess.run(["rclone", "check", OUT, REMOTE, "--files-from", lst, "--size-only", "--one-way"],
                                 capture_output=True, text=True)
            if chk.returncode == 0:
                open(f"{OUT}/done/{shard}.uploaded", "w").write("ok")
                print(f"[upload] {shard} -> {REMOTE} ok", flush=True)
                return True
            print(f"[upload] {shard} verify failed: {chk.stderr[-500:]}", flush=True)
        else:
            print(f"[upload] {shard} attempt {attempt} failed: {r.stderr[-500:]}", flush=True)
        time.sleep(30 * (attempt + 1))
    print(f"[upload] {shard} GAVE UP (will retry on next start)", flush=True)
    return False


def main():
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    last = int(sys.argv[2]) if len(sys.argv) > 2 else 207
    shards = [f"{i:04d}" for i in range(first, last + 1)
              if not os.path.exists(f"{OUT}/done/{i:04d}.done")]
    print(f"{len(shards)} shards to process", flush=True)
    q = queue.Queue(maxsize=PREFETCH)

    def downloader():
        for s in shards:
            p = f"{RAW}/data/{LANG}/audio/{s}.tar"
            if not os.path.exists(p):
                p = download(s)
            q.put((s, p))
        q.put(None)

    threading.Thread(target=downloader, daemon=True).start()
    uq = queue.Queue()
    upload_on = not os.environ.get("NO_UPLOAD")

    def uploader():
        while True:
            s = uq.get()
            if s is None:
                return
            upload_shard(s)

    ut = threading.Thread(target=uploader, daemon=True)
    ut.start()
    if upload_on and os.path.isdir(f"{OUT}/done"):  # finished earlier but not uploaded yet
        for f in sorted(os.listdir(f"{OUT}/done")):
            if f.endswith(".done") and not os.path.exists(f"{OUT}/done/{f[:-5]}.uploaded"):
                uq.put(f[:-5])
    wait_for_memory(MIN_FREE_GB_START, "loading models")
    import torch
    torch.set_grad_enabled(False)
    from gpu_models import Judges
    J = Judges()
    global ASR
    from gpu_models import TashkeelASR
    ASR = TashkeelASR()
    ctx = mp.get_context("spawn")
    with ctx.Pool(N_WORKERS, initializer=_init_worker, maxtasksperchild=200) as pool:
        while True:
            item = q.get()
            if item is None:
                break
            s, p = item
            wait_for_memory(MIN_FREE_GB_START, f"shard {s}")
            try:
                process_shard(s, p, pool, J)
                if upload_on:
                    uq.put(s)
                if not os.environ.get("KEEP_RAW"):
                    os.remove(p)
            except Exception:
                traceback.print_exc()
                print(f"[shard {s}] FAILED, raw tar kept for retry", flush=True)
    uq.put(None)
    ut.join()
    print("all done", flush=True)


if __name__ == "__main__":
    main()
