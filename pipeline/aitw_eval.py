"""Score every AITW clip with current + challenger judges, then report ROC-AUC per human label.

usage: python aitw_eval.py score   # compute features -> /workspace/aitw/features.parquet
       python aitw_eval.py report  # AUC table
"""
import os, sys, time, glob
import numpy as np, pandas as pd

A = "/workspace/aitw"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# feature -> direction (+1: higher means MORE likely bad, -1: higher means cleaner)
FEATS = {
    # multi-speaker / overlap
    "seg_n_spk": 1, "seg_overlap_frac": 1, "dz_n_spk": 1, "dz_overlap_frac": 1, "dz_2nd_spk_sec": 1,
    "spk_min_sim": -1, "spk293_min_sim": -1, "spk_min_pair": -1, "spk293_min_pair": -1,
    # music / singing
    "tag_music": 1, "tag_singing": 1, "fr_music_ratio": 1, "fr_music_p95": 1, "fr_singing_p95": 1, "aes_pc": 1,
    # noise / quality
    "dnsmos_bak": -1, "dnsmos_ovrl": -1, "dnsmos_sig": -1, "utmos": -1, "utmosv2": -1, "aes_pq": -1, "aes_ce": -1,
    "um_mos_overall": -1, "um_nisqa_mos": -1, "um_distill_mos": -1, "um_scoreq": -1, "um_utmos": -1,
    "tag_noise": 1, "tag_crowd": 1,
    # synthetic
    "tag_synth": 1,
}
LABELS = {"multi_speaker": lambda d: d.num_speakers >= 2, "background_music": lambda d: d.background_music == 1,
          "noise": lambda d: d.noise == 1, "synthetic": lambda d: d.synthetic == 1,
          "any_bad": lambda d: (d.num_speakers >= 2) | (d.background_music == 1) | (d.noise == 1) | (d.synthetic == 1)}


def score():
    import soundfile as sf, soxr, torch
    torch.set_grad_enabled(False)
    lab = pd.read_csv(f"{A}/eval_labels.csv")
    lab["path"] = A + "/clips/" + lab.filename.str.split("/").str[-1]
    lab = lab[lab.path.map(os.path.exists)].drop_duplicates("filename").reset_index(drop=True)
    lab = lab[(lab.duration >= 1.0)].reset_index(drop=True)
    wavs = []
    for p in lab.path:
        w, sr = sf.read(p, dtype="float32")
        w = w if w.ndim == 1 else w.mean(1)
        wavs.append(soxr.resample(w, sr, 16000).astype(np.float32))
    print(len(wavs), "clips", sum(len(w) for w in wavs) / 16000 / 3600, "h", flush=True)
    feats = pd.DataFrame(index=lab.index)
    from gpu_models import Judges
    J = Judges(load_utmos=True)
    import candidates as C
    judges = [("seg", J.run_seg), ("spk", J.run_spk), ("ced", J.run_ced), ("dnsmos", J.run_dnsmos),
              ("utmos", J.run_utmos), ("aes", J.run_aes)]
    judges += [("dz", J.run_dz), ("fraed", J.run_fraed), ("um", J.run_um)]
    for cls in ["WeSpeaker293", "UTMOSv2"]:
        judges.append((cls, getattr(C, cls)().run))
    for name, fn in judges:
        t = time.time()
        out = fn(wavs)
        for k in out[0]:
            if k in ("spk_emb", "tag_top"):
                continue
            feats[k] = [o.get(k, np.nan) for o in out]
        print(f"{name} done {time.time()-t:.0f}s", flush=True)
        pd.concat([lab, feats], axis=1).to_parquet(f"{A}/features.parquet")


def report():
    from sklearn.metrics import roc_auc_score
    d = pd.read_parquet(f"{A}/features.parquet")
    print(f"{len(d)} clips | positives:", {k: int(f(d).sum()) for k, f in LABELS.items()})
    rows = []
    for feat, sgn in FEATS.items():
        if feat not in d:
            continue
        x = d[feat].astype(float)
        r = {"feature": feat}
        for lname, f in LABELS.items():
            y = f(d).values
            ok = ~np.isnan(x.values)
            r[lname] = roc_auc_score(y[ok], sgn * x.values[ok]) if y[ok].any() and (~y[ok]).any() else np.nan
        rows.append(r)
    t = pd.DataFrame(rows).set_index("feature").round(3)
    pd.set_option("display.width", 200)
    print(t.to_string())
    t.to_csv(f"{A}/auc_report.csv")




# rule -> (feature, direction, label it targets); direction +1: reject if feature > thr
RULES = {
    "dz_2nd_spk_sec": (1, "multi_speaker"), "dz_overlap_frac": (1, "multi_speaker"),
    "seg_overlap_frac": (1, "multi_speaker"), "spk_min_sim": (-1, "multi_speaker"),
    "tag_music": (1, "background_music"), "fr_music_p95": (1, "background_music"),
    "fr_singing_p95": (1, "background_music"), "aes_pc": (1, "background_music"),
    "dnsmos_bak": (-1, "noise"), "dnsmos_ovrl": (-1, "noise"),
    "um_mos_overall": (-1, "noise"), "um_scoreq": (-1, "noise"),
    "tag_synth": (1, "synthetic"),
}


def calib(recall=0.95):
    d = pd.read_parquet(f"{A}/features.parquet")
    clean = ~LABELS["any_bad"](d)
    print(f"{len(d)} clips, clean {clean.sum()}")
    out = {}
    for feat, (sgn, lab) in RULES.items():
        y = LABELS[lab](d)
        x = d[feat].astype(float)
        pos = x[y & x.notna()]
        if len(pos) < 5:
            continue
        # threshold catching `recall` of positives
        thr = np.nanpercentile(pos, 100 * (1 - recall)) if sgn > 0 else np.nanpercentile(pos, 100 * recall)
        rej = (x > thr) if sgn > 0 else (x < thr)
        out[feat] = thr
        print(f"{feat:18s} [{lab:16s}] thr {thr:8.3f}  recall {rej[y].mean():.2f}  "
              f"clean-rejected {rej[clean].mean():.2f}  (n_pos {y.sum()})")
    return out


def simulate(th):
    """Recall per label and clean retention for a full rule set {feat: (op, thr)}."""
    d = pd.read_parquet(f"{A}/features.parquet")
    rej = pd.Series(False, index=d.index)
    for feat, (op, thr) in th.items():
        x = d[feat].astype(float)
        rej |= (x > thr) if op == ">" else (x < thr)
    clean = ~LABELS["any_bad"](d)
    for lname, f in LABELS.items():
        y = f(d)
        print(f"  {lname:16s} caught {rej[y].mean():.3f}  (n={y.sum()})")
    print(f"  clean kept {1 - rej[clean].mean():.3f}  (n={clean.sum()})")
    return rej


if __name__ == "__main__":
    {"score": score, "report": report, "calib": calib}[sys.argv[1]]()
