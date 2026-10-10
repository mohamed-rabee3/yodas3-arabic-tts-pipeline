"""Batched GPU wrappers for the neural quality judges. All take lists of 16 kHz float32 numpy chunks."""
import os

import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda"
SR = 16000

# AudioSet labels we care about (CED uses the standard 527-class ontology names)
BAD_TAGS = {
    "music": ["Music", "Musical instrument", "Background music", "Theme music", "Soundtrack music",
              "Pop music", "Electronic music", "Drum", "Guitar", "Piano", "Synthesizer", "Percussion"],
    "singing": ["Singing", "Choir", "Chant", "Rapping", "Humming", "Song", "Vocal music"],
    "crowd": ["Crowd", "Hubbub, speech noise, speech babble", "Chatter", "Cheering", "Applause",
              "Children shouting", "Clapping"],
    "laughter": ["Laughter", "Giggle", "Chuckle, chortle", "Belly laugh", "Baby laughter", "Snicker"],
    "noise": ["Noise", "Static", "Hum", "Mains hum", "White noise", "Pink noise", "Environmental noise",
              "Wind", "Wind noise (microphone)", "Rain", "Rumble", "Vehicle", "Traffic noise, roadway noise",
              "Engine", "Air conditioning", "Mechanical fan", "Distortion", "Echo", "Reverberation",
              "Inside, large room or hall", "Outside, urban or manmade", "Television", "Radio"],
    "effects": ["Sound effect", "Beep, bleep", "Whoosh, swoosh, swish", "Explosion", "Gunshot, gunfire",
                "Telephone", "Ringtone", "Alarm", "Bell", "Dog", "Animal", "Bird", "Cough", "Sneeze",
                "Crying, sobbing", "Baby cry, infant cry", "Shout", "Screaming"],
    "synth": ["Speech synthesizer"],
}


def windows(x, win, hop=None, tile_short=False):
    """Split 1-D array into fixed windows; last window aligned to the end. Returns (list, weights)."""
    n = len(x)
    if n <= win:
        if tile_short:
            reps = int(np.ceil(win / max(n, 1)))
            return [np.tile(x, reps)[:win]], [1.0]
        return [np.pad(x, (0, win - n))], [n / win]
    hop = hop or win
    starts = list(range(0, n - win + 1, hop))
    if starts[-1] != n - win:
        starts.append(n - win)
    return [x[s:s + win] for s in starts], [1.0] * len(starts)


class Judges:
    def __init__(self, models_dir="/workspace/models", hf_token=None, load_utmos=False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        hf_token = hf_token or os.environ.get("HF_TOKEN")
        # DNSMOS P.835 (ONNX -> torch)
        from onnx2torch import convert
        self.dnsmos = convert(f"{models_dir}/sig_bak_ovr.onnx").to(DEV).eval()
        # UTMOS22 strong (superseded by UrgentMOS in the pipeline; kept for evaluation)
        if load_utmos:
            self.utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV).eval()
        # challengers that won on AITW: DiariZen (multi-speaker), FireRedVAD-AED (music), UrgentMOS (noise)
        import candidates as C
        self._dz = C.DiariZenJudge()
        self._fraed = C.FireRedAED()
        self._um = C.UrgentMOS()
        # Audiobox aesthetics
        from audiobox_aesthetics.infer import initialize_predictor
        self.aes = initialize_predictor()
        self.aes.model.to(DEV).eval()
        # CED-base AudioSet tagger
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        self.ced_fe = AutoFeatureExtractor.from_pretrained("mispeech/ced-base", trust_remote_code=True)
        self.ced = AutoModelForAudioClassification.from_pretrained("mispeech/ced-base", trust_remote_code=True).to(DEV).eval()
        id2l = self.ced.config.id2label
        l2i = {v: int(k) for k, v in id2l.items()}
        self.tag_idx = {g: [l2i[n] for n in names if n in l2i] for g, names in BAD_TAGS.items()}
        self.speech_idx = [l2i[n] for n in ["Speech", "Male speech, man speaking", "Female speech, woman speaking",
                                            "Narration, monologue", "Conversation"] if n in l2i]
        self.id2label = id2l
        # pyannote segmentation-3.0 + WeSpeaker
        from pyannote.audio import Model
        from pyannote.audio.utils.powerset import Powerset
        self.seg = Model.from_pretrained("pyannote/segmentation-3.0", token=hf_token).to(DEV).eval()
        spec = self.seg.specifications
        self.powerset = Powerset(len(spec.classes), spec.powerset_max_classes).to(DEV)
        self.emb = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", token=hf_token).to(DEV).eval()

    def run_dz(self, wavs):
        return self._dz.run(wavs)

    def run_fraed(self, wavs):
        return self._fraed.run(wavs)

    def run_um(self, wavs):
        return self._um.run(wavs)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _stack(ws):
        return torch.from_numpy(np.stack(ws)).to(DEV)

    def _batched(self, items, bs, fn):
        out = []
        for i in range(0, len(items), bs):
            out.append(fn(items[i:i + bs]))
        return torch.cat(out) if out and isinstance(out[0], torch.Tensor) else sum(out, [])

    # ------------------------------------------------------------ DNSMOS P.835
    @torch.inference_mode()
    def run_dnsmos(self, wavs, bs=256):
        win = int(9.01 * SR)
        W, owner = [], []
        for i, w in enumerate(wavs):
            ws, _ = windows(w, win, tile_short=True)  # non-overlapping 9 s windows, last aligned to end
            W += ws
            owner += [i] * len(ws)
        raw = self._batched(W, bs, lambda b: self.dnsmos(self._stack(b)).float())  # [N,3] sig,bak,ovr
        raw = raw.cpu().numpy()
        p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        owner = np.array(owner)
        res = []
        for i in range(len(wavs)):
            r = raw[owner == i].mean(0)
            res.append({"dnsmos_sig": float(p_sig(r[0])), "dnsmos_bak": float(p_bak(r[1])),
                        "dnsmos_ovrl": float(p_ovr(r[2]))})
        return res

    # ------------------------------------------------------------ UTMOS22 (length-bucketed)
    @torch.inference_mode()
    def run_utmos(self, wavs, max_samples=16000 * 160):
        order = np.argsort([len(w) for w in wavs])
        res = [None] * len(wavs)
        i = 0
        while i < len(order):
            L = len(wavs[order[i]])
            j = i
            # bucket: lengths within 10% and total padded samples bounded
            while j < len(order) and len(wavs[order[j]]) <= L * 1.1 + 1 and (j - i + 1) * len(wavs[order[j]]) <= max_samples:
                j += 1
            j = max(j, i + 1)
            idx = order[i:j]
            T = max(len(wavs[k]) for k in idx)
            b = torch.zeros(len(idx), T, device=DEV)
            for r, k in enumerate(idx):
                b[r, : len(wavs[k])] = torch.from_numpy(wavs[k]).to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = self.utmos(b, SR).float().cpu().numpy()
            for r, k in enumerate(idx):
                res[k] = {"utmos": float(s[r])}
            i = j
        return res

    # ------------------------------------------------------------ Audiobox aesthetics (10 s windows + masks)
    @torch.inference_mode()
    def run_aes(self, wavs, bs=64):
        win = 10 * SR
        W, M, Wt, owner = [], [], [], []
        for i, w in enumerate(wavs):
            for s in range(0, len(w), win):
                seg = w[s:s + win]
                n = len(seg)
                if n < SR and s > 0:  # skip <1 s tail
                    continue
                W.append(np.pad(seg, (0, win - n)))
                m = np.zeros(win, dtype=bool)
                m[:n] = True
                M.append(m)
                Wt.append(n / win)
                owner.append(i)
        axes = ["CE", "CU", "PC", "PQ"]
        preds = {a: [] for a in axes}
        for i in range(0, len(W), bs):
            wav = self._stack(W[i:i + bs]).unsqueeze(1)
            mask = torch.from_numpy(np.stack(M[i:i + bs])).to(DEV).unsqueeze(1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = self.aes.model({"wav": wav, "mask": mask})
            for a in axes:
                preds[a].append(self.aes.target_transform[a].inverse(p[a].float()).cpu())
        owner, Wt = np.array(owner), np.array(Wt)
        res = []
        P = {a: torch.cat(preds[a]).numpy().reshape(-1) for a in axes}
        for i in range(len(wavs)):
            sel = owner == i
            res.append({f"aes_{a.lower()}": float((P[a][sel] * Wt[sel]).sum() / Wt[sel].sum()) for a in axes})
        return res

    # ------------------------------------------------------------ CED AudioSet tags (max over 10 s windows)
    @torch.inference_mode()
    def run_ced(self, wavs, bs=64):
        win = 10 * SR
        W, owner = [], []
        for i, w in enumerate(wavs):
            ws, _ = windows(w, win)
            W += ws
            owner += [i] * len(ws)
        probs = []
        for i in range(0, len(W), bs):
            feats = self.ced_fe(list(W[i:i + bs]), sampling_rate=SR, return_tensors="pt")["input_values"].to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = self.ced(feats).logits.float()
            probs.append(lg.cpu())  # CED head already outputs sigmoid probabilities
        P = torch.cat(probs).numpy()
        owner = np.array(owner)
        res = []
        for i in range(len(wavs)):
            p = P[owner == i]
            pm = p.max(0)
            r = {f"tag_{g}": float(pm[ix].max()) if ix else 0.0 for g, ix in self.tag_idx.items()}
            r["tag_speech"] = float(p[:, self.speech_idx].max(1).min())  # worst window's speech prob
            top = np.argsort(-pm)[:3]
            r["tag_top"] = ";".join(f"{self.id2label[int(k)]}:{pm[k]:.2f}" for k in top)
            res.append(r)
        return res

    # ------------------------------------------------------------ pyannote seg-3.0: overlap / speaker count
    @torch.inference_mode()
    def run_seg(self, wavs, bs=128):
        win = 10 * SR
        W, Wn, owner = [], [], []
        for i, w in enumerate(wavs):
            ws, _ = windows(w, win)  # 1 window if <=10 s, else [0,10] + [..end] (overlapping for <=20 s)
            W += ws
            Wn += [min(len(w), win)] * len(ws)
            owner += [i] * len(ws)
        outs = []
        for i in range(0, len(W), bs):
            x = self._stack(W[i:i + bs]).unsqueeze(1)
            lp = self.seg(x)  # [B, F, 7] log-probs (powerset)
            ml = self.powerset.to_multilabel(lp)  # hard [B, F, 3]
            outs.append(ml.cpu())
        ML = torch.cat(outs).numpy()  # [N, F, 3]
        Fr = ML.shape[1]
        owner = np.array(owner)
        res = []
        for i in range(len(wavs)):
            sel = np.where(owner == i)[0]
            ov, sp, nspk, tot = 0, 0, 0, 0
            for k in sel:
                valid = int(Fr * Wn[k] / win)
                m = ML[k, :valid]
                act = m.sum(1)
                sp += int((act >= 1).sum())
                ov += int((act >= 2).sum())
                tot += valid
                # a speaker "counts" if active >= 0.4 s (~24 frames of 16.9 ms) in this window
                nspk = max(nspk, int((m.sum(0) >= 24).sum()))
            res.append({"seg_overlap_frac": ov / max(sp, 1), "seg_speech_frac": sp / max(tot, 1),
                        "seg_n_spk": nspk})
        return res

    # ------------------------------------------------------------ WeSpeaker: chunk embedding + consistency
    @torch.inference_mode()
    def run_spk(self, wavs, piece=3 * SR, bs=256):
        W, owner, kind = [], [], []
        for i, w in enumerate(wavs):
            W.append(w)
            owner.append(i)
            kind.append(0)  # full chunk
            n = len(w) // piece
            if n >= 2:
                for k in range(n):
                    W.append(w[k * piece:(k + 1) * piece])
                    owner.append(i)
                    kind.append(1)
        # embed full chunks one length-bucket at a time, pieces in big batches
        E = [None] * len(W)
        full = [k for k in range(len(W)) if kind[k] == 0]
        parts = [k for k in range(len(W)) if kind[k] == 1]
        full.sort(key=lambda k: len(W[k]))
        i = 0
        while i < len(full):
            L = len(W[full[i]])
            j = i
            while j < len(full) and len(W[full[j]]) <= L * 1.1 + 1 and j - i < 64:
                j += 1
            idx = full[i:j]
            T = max(len(W[k]) for k in idx)
            b = torch.zeros(len(idx), 1, T, device=DEV)
            for r, k in enumerate(idx):
                # repeat-pad instead of zero-pad so statistics pooling is not biased
                w = W[k]
                b[r, 0] = torch.from_numpy(np.resize(w, T)).to(DEV)
            e = F.normalize(self.emb(b).float(), dim=-1).cpu().numpy()
            for r, k in enumerate(idx):
                E[k] = e[r]
            i = j
        for i in range(0, len(parts), bs):
            idx = parts[i:i + bs]
            b = self._stack([W[k] for k in idx]).unsqueeze(1)
            e = F.normalize(self.emb(b).float(), dim=-1).cpu().numpy()
            for r, k in enumerate(idx):
                E[k] = e[r]
        owner, kind = np.array(owner), np.array(kind)
        res = []
        for i in range(len(wavs)):
            sel = np.where(owner == i)[0]
            fe = E[sel[kind[sel] == 0][0]]
            pe = [E[k] for k in sel[kind[sel] == 1]]
            if len(pe) >= 2:
                pe = np.stack(pe)
                sims = pe @ fe
                pair = pe @ pe.T
                r = {"spk_min_sim": float(sims.min()), "spk_min_pair": float(pair.min())}
            else:
                r = {"spk_min_sim": float("nan"), "spk_min_pair": float("nan")}
            r["spk_emb"] = fe.astype(np.float16)
            res.append(r)
        return res


# ---------------------------------------------------------------- Arabic text normalisation / CER
import re as _re

_DIAC = _re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_NON_AR = _re.compile(r"[^ء-غف-ي ]+")
_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه", "ى": "ي", "ؤ": "و", "ئ": "ي"})


def strip_tashkeel(t):
    return _DIAC.sub("", t or "")


def norm_ar(t):
    t = strip_tashkeel(t).translate(_MAP)
    t = _NON_AR.sub(" ", t)
    return " ".join(t.split())


def cer(hyp, ref):
    from rapidfuzz.distance import Levenshtein
    if not ref:
        return float("nan")
    return Levenshtein.distance(hyp, ref) / len(ref)


class TashkeelASR:
    """NAMAA-Space/Cohere-Speech-Tashkeel-2B: Arabic speech -> fully diacritised text."""

    def __init__(self, repo="NAMAA-Space/Cohere-Speech-Tashkeel-2B"):
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration
        self.proc = AutoProcessor.from_pretrained(repo)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(repo, dtype=torch.bfloat16).to(DEV).eval()
        self.prompt = self.proc.get_decoder_prompt_ids(language="ar", punctuation=True)

    @torch.inference_mode()
    def run(self, wavs, bs=64):
        order = np.argsort([len(w) for w in wavs])
        res = [None] * len(wavs)
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            feats = self.proc.feature_extractor([wavs[k] for k in idx], sampling_rate=SR, return_tensors="pt",
                                                padding="longest", return_attention_mask=True)
            aci = feats.pop("audio_chunk_index")
            dec = torch.tensor([self.prompt] * len(idx), dtype=torch.long, device=DEV)
            out = self.model.generate(input_features=feats["input_features"].to(DEV, torch.bfloat16),
                                      attention_mask=feats["attention_mask"].to(DEV),
                                      decoder_input_ids=dec, max_new_tokens=448)
            txt = self.proc.decode(out, skip_special_tokens=True, audio_chunk_index=aci, language="ar")
            for k, t in zip(idx, txt):
                res[k] = t.strip()
        return res
