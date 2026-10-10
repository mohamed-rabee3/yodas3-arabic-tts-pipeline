"""Audio-quality audit primitives for YODAS v3 Arabic -> TTS-grade chunks.

CPU side : decode (Opus/WebM -> 48 kHz stereo), Silero VAD, pause-aware chunking,
           signal metrics (clipping, level, bandwidth, edge silence, L/R correlation).
GPU side : DNSMOS P.835, UTMOS22-strong, Audiobox-Aesthetics, CED-base AudioSet tags,
           pyannote segmentation-3.0 (overlap / local speaker count),
           WeSpeaker ResNet34 (intra-chunk speaker consistency + chunk embedding).
"""
import io
import math

import av
import numpy as np
import soxr
from scipy.signal import welch

SR_NATIVE = 48000
SR_OUT = 24000
SR_M = 16000

# ---------------------------------------------------------------- decoding

def decode_chunks(webm_bytes, chunk_sec=600.0):
    """Yield (offset_sec, float32 [2, N] @48k) blocks of ~chunk_sec, streaming (bounded memory)."""
    with av.open(io.BytesIO(webm_bytes)) as c:
        st = c.streams.audio[0]
        rs = av.AudioResampler(format="fltp", layout="stereo", rate=SR_NATIVE)
        buf, n, off = [], 0, 0
        lim = int(chunk_sec * SR_NATIVE)

        def frames():
            for fr in c.decode(st):
                yield from rs.resample(fr)
            yield from rs.resample(None)

        for f in frames():
            a = f.to_ndarray()
            buf.append(a)
            n += a.shape[1]
            if n >= lim:
                x = np.concatenate(buf, 1)
                yield off / SR_NATIVE, x
                off += x.shape[1]
                buf, n = [], 0
        if buf:
            yield off / SR_NATIVE, np.concatenate(buf, 1)


# ---------------------------------------------------------------- chunking

def union_intervals(*lists):
    """Union of several [(s,e)] interval lists (speech by ANY VAD counts as speech)."""
    iv = sorted(x for l in lists for x in l)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def split_regions(regions, min_dur, max_dur, min_split_gap):
    """Recursively split a list of speech regions [(s,e),...] at the largest pause
    until every piece is <= max_dur. Only pauses >= min_split_gap are valid cut points."""
    s, e = regions[0][0], regions[-1][1]
    if e - s <= max_dur:
        return [regions]
    gaps = [(regions[i + 1][0] - regions[i][1], i) for i in range(len(regions) - 1)]
    gaps = [g for g in gaps if g[0] >= min_split_gap]
    if not gaps:
        return []  # one long breathless stretch: no safe cut point -> drop
    gaps.sort(reverse=True)
    # prefer the largest pause that leaves both sides >= min_dur
    pick = None
    for g, i in gaps:
        if regions[i][1] - s >= min_dur and e - regions[i + 1][0] >= min_dur:
            pick = i
            break
    if pick is None:
        pick = gaps[0][1]
    return (split_regions(regions[: pick + 1], min_dur, max_dur, min_split_gap)
            + split_regions(regions[pick + 1:], min_dur, max_dur, min_split_gap))


def make_chunks(speech, total, min_dur=2.0, max_dur=20.0, phrase_gap=1.0,
                min_split_gap=0.3, pad=0.25):
    """speech: sorted [(s,e)] seconds. Returns [(start, end, speech_start, speech_end, inner)]
    where start/end include up to `pad` s of real silence (never more than half the neighbouring pause)."""
    if not speech:
        return []
    phrases, cur = [], [speech[0]]
    for r in speech[1:]:
        if r[0] - cur[-1][1] >= phrase_gap:
            phrases.append(cur)
            cur = [r]
        else:
            cur.append(r)
    phrases.append(cur)

    pieces = []
    for ph in phrases:
        pieces += split_regions(ph, min_dur, max_dur, min_split_gap)
    out = []
    flat_bounds = [(p[0][0], p[-1][1]) for p in pieces]
    for k, p in enumerate(pieces):
        ss, se = p[0][0], p[-1][1]
        if se - ss < min_dur:
            continue
        # find neighbouring speech (anything, not just kept pieces) to bound padding
        prev_end = max([e for (s, e) in speech if e <= ss + 1e-6] or [0.0]) if k == 0 else flat_bounds[k - 1][1]
        next_start = flat_bounds[k + 1][0] if k + 1 < len(flat_bounds) else total
        lp = min(pad, max(0.0, (ss - prev_end) / 2))
        rp = min(pad, max(0.0, (next_start - se) / 2))
        inner = sum(r[1] - r[0] for r in p)
        # silence available for boundary search: up to the middle of each neighbouring pause
        lo = max(ss - 0.6, (prev_end + ss) / 2)
        hi = min(se + 0.6, (se + next_start) / 2)
        out.append((ss - lp, se + rp, ss, se, inner, lo, hi))
    return out


def refine_bounds(mono48, chunks, target_pad=0.2, min_pad=0.08, frame=0.01, tol_db=3.0):
    """Move each chunk boundary to the quietest 10 ms frame inside the pause next to it
    (among frames within tol_db of the minimum, the one closest to target_pad from speech)."""
    hop = int(frame * 48000)
    nfr = len(mono48) // hop
    if nfr == 0:
        return chunks
    e = db((mono48[: nfr * hop].reshape(nfr, hop).astype(np.float64) ** 2).mean(1))

    def pick(a, b, target):
        ia, ib = max(int(a / frame), 0), min(int(b / frame), nfr - 1)
        if ib <= ia:
            return None
        seg = e[ia:ib + 1]
        cand = np.where(seg <= seg.min() + tol_db)[0] + ia
        t = cand[np.argmin(np.abs(cand * frame - target))]
        return t * frame

    out = []
    for (s, en, ss, se, inner, lo, hi) in chunks:
        ns = pick(lo, ss - min_pad, ss - target_pad)
        ne = pick(se + min_pad, hi, se + target_pad)
        out.append((ns if ns is not None else s, (ne + frame) if ne is not None else en, ss, se, inner))
    return out


# ---------------------------------------------------------------- signal metrics

def db(x, eps=1e-12):
    return 10.0 * np.log10(np.maximum(x, eps))


def bandwidth_hz(mono48, thr_db=50.0):
    """Highest frequency whose long-term PSD is within thr_db of the PSD peak
    (same idea as YODAS' urgent2025 estimate, but per chunk)."""
    f, p = welch(mono48, fs=SR_NATIVE, nperseg=2048, noverlap=1024)
    pd = db(p)
    above = np.where(pd >= pd.max() - thr_db)[0]
    return float(f[above[-1]]) if len(above) else 0.0


def signal_metrics(x48, ss_rel, se_rel):
    """x48: [2, N] float32 for the chunk (incl. padding). ss_rel/se_rel: speech span inside chunk (s)."""
    L, R = x48[0].astype(np.float64), x48[1].astype(np.float64)
    mono = 0.5 * (L + R)
    n = mono.shape[0]
    m = {}
    absx = np.abs(x48)
    peak = float(absx.max())
    m["peak"] = peak
    # hard clipping: samples at/over full scale on any channel
    m["clip_frac"] = float((absx >= 0.99).any(0).mean())
    # flat-top / limiter: mass of samples crowding the chunk's own peak
    m["near_peak_frac"] = float((np.abs(mono) >= 0.97 * np.abs(mono).max()).mean()) if peak > 0 else 0.0
    # consecutive plateau runs at the peak on either channel (classic clipping signature)
    runs = 0
    for ch in (x48[0], x48[1]):
        a = np.abs(ch)
        hi = a >= 0.985 * max(a.max(), 1e-9)
        if hi.any():
            d = np.diff(np.concatenate([[0], hi.astype(np.int8), [0]]))
            st, en = np.where(d == 1)[0], np.where(d == -1)[0]
            runs = max(runs, int(((en - st) >= 4).sum()))
    m["plateau_runs"] = runs
    m["dc"] = float(mono.mean())
    rms = math.sqrt(float((mono ** 2).mean()) + 1e-20)
    m["rms_dbfs"] = 20 * math.log10(rms + 1e-12)
    # speech-span level and edge silence level (truncated words show up as loud edges)
    a, b = int(ss_rel * SR_NATIVE), int(se_rel * SR_NATIVE)
    sp = mono[a:b] if b > a else mono
    sp_rms = math.sqrt(float((sp ** 2).mean()) + 1e-20)
    m["speech_dbfs"] = 20 * math.log10(sp_rms + 1e-12)
    w = int(0.04 * SR_NATIVE)
    head = mono[: max(w, 1)]
    tail = mono[-max(w, 1):]
    m["edge_head_db"] = 20 * math.log10(math.sqrt(float((head ** 2).mean()) + 1e-20) / sp_rms)
    m["edge_tail_db"] = 20 * math.log10(math.sqrt(float((tail ** 2).mean()) + 1e-20) / sp_rms)
    # noise floor estimate: 10th percentile of 20ms frame energies vs 90th
    fl = int(0.02 * SR_NATIVE)
    nf = n // fl
    if nf >= 10:
        fe = (mono[: nf * fl].reshape(nf, fl) ** 2).mean(1)
        floor = float(db(np.percentile(fe, 10)))
        m["dyn_range_db"] = float(db(np.percentile(fe, 95))) - floor
    else:
        floor = 20 * math.log10(sp_rms + 1e-12)
        m["dyn_range_db"] = float("nan")
    # truncation check: an edge that is well above the chunk's own noise floor *and* close to
    # speech level means the cut landed on a word (or on a non-silent event)
    m["floor_dbfs"] = floor
    m["edge_head_excess_db"] = 20 * math.log10(math.sqrt(float((head ** 2).mean()) + 1e-20)) - floor
    m["edge_tail_excess_db"] = 20 * math.log10(math.sqrt(float((tail ** 2).mean()) + 1e-20)) - floor
    m["bw_hz"] = bandwidth_hz(mono)
    ll, rr = float((L ** 2).sum()), float((R ** 2).sum())
    m["lr_corr"] = float((L * R).sum() / math.sqrt(ll * rr + 1e-20))
    m["lr_bal_db"] = float(db(ll + 1e-12) - db(rr + 1e-12))
    return m, mono.astype(np.float32)


def to_rates(mono48):
    w16 = soxr.resample(mono48, SR_NATIVE, SR_M).astype(np.float32)
    w24 = soxr.resample(mono48, SR_NATIVE, SR_OUT).astype(np.float32)
    return w16, w24


def attach_text(transcript, start, end):
    """Caption words whose start time falls inside [start, end). transcript = parsed JSON list or None."""
    if not transcript:
        return None, 0
    s_ms, e_ms = start * 1000, end * 1000
    words = []
    for seg in transcript:
        st = seg.get("start", 0)
        if "words" in seg and seg["words"]:
            if st > e_ms or st + seg.get("duration", 0) + 15000 < s_ms:
                continue
            for w in seg["words"]:
                t = st + w["t"]
                if s_ms <= t < e_ms:
                    words.append((t, w["w"]))
        else:
            mid = st + seg.get("duration", 0) / 2
            if s_ms <= mid < e_ms:
                words.append((st, seg.get("text", "")))
    words.sort()
    # rolling auto-captions repeat words across segments -> dedupe identical (t, w)
    dedup, seen = [], set()
    for t, w in words:
        if (t, w) in seen:
            continue
        seen.add((t, w))
        dedup.append(w)
    return " ".join(dedup).strip() or None, len(dedup)


def caption_window(transcript, start, end, margin=0.35):
    """Caption words for [start, end) plus the boundary-jitter candidates needed for exact matching:
    prev = words starting in [start-margin, start), next = words in [end, end+margin),
    head_n / tail_n = how many core words start within `margin` of the chunk edges."""
    if not transcript:
        return dict(timed=[], core=[], prev=[], next=[], head_n=0, tail_n=0)
    s_ms, e_ms, m_ms = start * 1000, end * 1000, margin * 1000
    words, seen = [], set()
    for seg in transcript:
        st = seg.get("start", 0)
        if "words" in seg and seg["words"]:
            if st > e_ms + m_ms or st + seg.get("duration", 0) + 15000 < s_ms - m_ms:
                continue
            items = [(st + w["t"], w["w"]) for w in seg["words"]]
        else:  # uploader subtitles: segment-level only
            items = [(st + seg.get("duration", 0) / 2, seg.get("text", ""))]
        for t, w in items:
            if s_ms - m_ms <= t < e_ms + m_ms and (t, w) not in seen:
                seen.add((t, w))
                words.append((t, w))
    words.sort()
    return _window(words, s_ms, e_ms, m_ms)


def caption_window_timed(timed, start, end, margin=0.35):
    """Same as caption_window() but from an already-collected [(t_ms, word)] list."""
    return _window(sorted(timed), start * 1000, end * 1000, margin * 1000)


def _window(words, s_ms, e_ms, m_ms):
    words = [(t, w) for t, w in words if s_ms - m_ms <= t < e_ms + m_ms]
    core = [(t, w) for t, w in words if s_ms <= t < e_ms]
    return dict(timed=words, core=[w for _, w in core],
                prev=[w for t, w in words if t < s_ms],
                next=[w for t, w in words if t >= e_ms],
                head_n=sum(1 for t, _ in core if t < s_ms + m_ms),
                tail_n=sum(1 for t, _ in core if t >= e_ms - m_ms))
