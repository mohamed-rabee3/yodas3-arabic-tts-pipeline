"""Fetch AITW-labelled YODAS clips: download the densest yodas2 tars, cut the labelled spans, delete tars."""
import os, sys, tarfile, io, concurrent.futures as cf
import numpy as np, pandas as pd, soundfile as sf
from huggingface_hub import hf_hub_download

A = "/workspace/aitw"
N_TARS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
os.makedirs(f"{A}/clips", exist_ok=True)
os.makedirs(f"{A}/tmp", exist_ok=True)
m = pd.read_csv(f"{A}/yodas_file_map.csv")
sets = pd.concat([pd.read_csv(f"{A}/{f}").assign(split=f.split("_")[0])
                  for f in ["test_set.csv", "validation_set.csv", "finetuning_set.csv"]])
y = sets[sets.filename.str.startswith("yodas")].merge(m, left_on="filename", right_on="AITW File Path")
y["repo_path"] = y["Original File Path"].str.extract(r"resolve/main/(.+)$")[0]
tars = y.groupby("repo_path").size().sort_values(ascending=False).head(N_TARS).index.tolist()
y[y.repo_path.isin(tars)].to_csv(f"{A}/eval_labels.csv", index=False)


def job(rp):
    rows = y[y.repo_path == rp]
    if all(os.path.exists(f"{A}/clips/{os.path.basename(f)}") for f in rows.filename):
        return rp, len(rows)
    p = hf_hub_download("espnet/yodas2", rp, repo_type="dataset", local_dir=f"{A}/tmp/{abs(hash(rp))}")
    want = {}
    for _, r in rows.iterrows():
        want.setdefault(r["YODAS File Name"], []).append(r)
    n = 0
    with tarfile.open(p, "r:gz") as tf:
        for mem in tf:
            base = os.path.basename(mem.name)
            if base in want:
                w, sr = sf.read(io.BytesIO(tf.extractfile(mem).read()), dtype="float32")
                for r in want[base]:
                    seg = w[int(r["Timestamp Start"] * sr):int(r["Timestamp End"] * sr)]
                    sf.write(f"{A}/clips/{os.path.basename(r['filename'])}", seg, sr)
                    n += 1
    os.remove(p)
    return rp, n


with cf.ThreadPoolExecutor(6) as ex:
    done = 0
    for rp, n in ex.map(job, tars):
        done += 1
        print(f"{done}/{len(tars)} {rp} {n} clips", flush=True)
print("FETCH DONE", flush=True)
