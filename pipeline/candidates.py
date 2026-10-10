"""Challenger judges evaluated against the current ones on AITW human labels."""
import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda"
SR = 16000


def _windows(x, win):
    n = len(x)
    if n <= win:
        return [np.pad(x, (0, win - n))], [n]
    starts = list(range(0, n - win + 1, win))
    if starts[-1] != n - win:
        starts.append(n - win)
    return [x[s:s + win] for s in starts], [win] * len(starts)


class DiariZenJudge:
    """Overlap fraction + local speaker count from DiariZen's WavLM-Conformer EEND (16 s windows)."""

    def __init__(self):
        from diarizen_seg import load_diarizen
        self.model, self.ps, self.chunk = load_diarizen(DEV)

    @torch.inference_mode()
    def run(self, wavs, bs=32, min_spk_sec=0.4):
        win = self.chunk * SR
        W, N, owner = [], [], []
        for i, w in enumerate(wavs):
            ws, ns = _windows(w, win)
            W += ws
            N += ns
            owner += [i] * len(ws)
        outs = []
        for i in range(0, len(W), bs):
            x = torch.from_numpy(np.stack(W[i:i + bs])).to(DEV).unsqueeze(1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp = self.model(x)
            outs.append(self.ps.to_multilabel(lp.float()).cpu())
        ML = torch.cat(outs).numpy()
        fr = ML.shape[1]
        fps = fr / self.chunk
        owner = np.array(owner)
        res = []
        for i in range(len(wavs)):
            ov = sp = tot = 0
            nspk, spk_sec = 0, []
            for k in np.where(owner == i)[0]:
                v = int(fr * N[k] / win)
                m = ML[k, :v]
                act = m.sum(1)
                sp += int((act >= 1).sum())
                ov += int((act >= 2).sum())
                tot += v
                per = m.sum(0) / fps
                nspk = max(nspk, int((per >= min_spk_sec).sum()))
                spk_sec.append(np.sort(per)[::-1])
            second = max((s[1] for s in spk_sec if len(s) > 1), default=0.0)
            res.append({"dz_overlap_frac": ov / max(sp, 1), "dz_speech_frac": sp / max(tot, 1),
                        "dz_n_spk": nspk, "dz_2nd_spk_sec": float(second)})
        return res


class FireRedAED:
    """FireRedVAD audio-event model: frame-level speech / singing / music probabilities."""

    def __init__(self, model_dir="/workspace/models/FireRedVAD/AED"):
        from fireredvad.aed import FireRedAed, FireRedAedConfig
        cfg = FireRedAedConfig()
        cfg.use_gpu = True
        self.aed = FireRedAed.from_pretrained(model_dir, cfg)

    @torch.inference_mode()
    def run(self, wavs):
        res = []
        for w in wavs:
            pcm = np.clip(w * 32768, -32768, 32767).astype(np.int16)
            feat, _ = self.aed.audio_feat.extract((pcm, SR))
            probs, _ = self.aed.model.forward(feat.cuda().unsqueeze(0))
            p = probs[0].float().cpu().numpy()  # (T, 3) speech, singing, music
            res.append({"fr_speech_ratio": float((p[:, 0] >= 0.4).mean()),
                        "fr_singing_ratio": float((p[:, 1] >= 0.5).mean()),
                        "fr_music_ratio": float((p[:, 2] >= 0.5).mean()),
                        "fr_music_p95": float(np.percentile(p[:, 2], 95)),
                        "fr_singing_p95": float(np.percentile(p[:, 1], 95))})
        return res


class UrgentMOS:
    def __init__(self, ckpt="/workspace/models/urgent-mos/model.pt"):
        from urgent_mos.utils import load_model_from_checkpoint
        self.model = load_model_from_checkpoint(ckpt, DEV).eval()

    @torch.inference_mode()
    def run(self, wavs):
        from urgent_mos.api.infer import infer
        out = infer(self.model, [torch.from_numpy(w) for w in wavs], sample_rate=SR, num_workers=0)
        return [{f"um_{k}": float(v) for k, v in o.items() if isinstance(v, (int, float))} for o in out]


class UTMOSv2:
    def __init__(self):
        import utmosv2
        self.model = utmosv2.create_model(pretrained=True, device=DEV)

    def run(self, wavs):
        res = []
        for w in wavs:
            s = self.model.predict(data=w, sr=SR, device=DEV, verbose=False)
            res.append({"utmosv2": float(np.asarray(s).reshape(-1)[0])})
        return res


class WeSpeaker293:
    """WeSpeaker ResNet293-LM (VoxCeleb) — stronger speaker embeddings for intra-chunk consistency."""

    def __init__(self, repo="Wespeaker/wespeaker-voxceleb-resnet293-LM"):
        from huggingface_hub import hf_hub_download
        from pyannote.audio.models.embedding.wespeaker import WeSpeakerResNet293
        p = hf_hub_download(repo, "avg_model.pt")
        sd = torch.load(p, map_location="cpu", weights_only=False)
        m = WeSpeakerResNet293(sample_rate=SR)
        tgt = m.state_dict()
        mapped = {}
        for k, v in sd.items():
            for cand in (f"resnet.{k}", k):
                if cand in tgt and tgt[cand].shape == v.shape:
                    mapped[cand] = v
                    break
        missing = [k for k in tgt if k not in mapped and "num_batches" not in k]
        if missing:
            raise RuntimeError(f"ResNet293 weights did not map: {missing[:5]} (+{len(missing)})")
        m.load_state_dict(mapped, strict=False)
        self.m = m.to(DEV).eval()

    @torch.inference_mode()
    def run(self, wavs, piece=3 * SR):
        res = []
        for w in wavs:
            full = F.normalize(self.m(torch.from_numpy(w)[None, None].to(DEV)).float(), dim=-1)[0]
            n = len(w) // piece
            if n >= 2:
                P = torch.from_numpy(np.stack([w[k * piece:(k + 1) * piece] for k in range(n)])).to(DEV).unsqueeze(1)
                pe = F.normalize(self.m(P).float(), dim=-1)
                res.append({"spk293_min_sim": float((pe @ full).min()), "spk293_min_pair": float((pe @ pe.T).min())})
            else:
                res.append({"spk293_min_sim": float("nan"), "spk293_min_pair": float("nan")})
        return res
