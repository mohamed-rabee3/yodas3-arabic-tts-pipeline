# YODAS v3 Arabic: clean TTS data pipeline

This pipeline turns the Arabic part of YODAS v3 (about 33,000 hours of YouTube audio) into a strictly clean TTS training set. The output is 24 kHz mono FLAC chunks of 2 to 20 seconds. Each chunk has one speaker and no music, background noise, overlapping speech, clipping or cut words. Each also has a fully diacritized (tashkeel) transcript whose words match the YouTube caption exactly.

Results are uploaded shard by shard to the Hetzner storage box, folder `yodas3-arabic-tts-clean/`.

## Status (2026-10-10)

| | |
|---|---|
| Shards done and uploaded | 1 of 208 (`0000`) |
| Shard 0000 | 674 videos, 155.0 h in, 773 chunks / 1.32 h kept (0.85 %), of which 0.62 h came from re-split salvage |
| Projected total | about 275 h of clean audio from all 208 shards |
| Speed on GB10 | 19.7 min per shard (GPU-bound), so about 68 h for the remaining 207 shards |
| Blocker | On Vast instance 55188341, about 112 GB of the GB10's 121 GB shared memory is held outside the container (not by this pipeline). The service waits for 40 GB free before it starts. Move to a machine that meets the requirements below. |

## Data source

- Dataset: [espnet/yodas3](https://huggingface.co/datasets/espnet/yodas3) (license CC-BY-3.0; please cite the paper on the card if you release derived data)
- Arabic files: [data/ar](https://huggingface.co/datasets/espnet/yodas3/tree/main/data/ar), 1.77 TB in total
  - audio: `data/ar/audio/0000.tar` … `0207.tar` (WebDataset, about 8.6 GB each, one `<id>.webm` Opus file per video plus `<id>.json`)
  - metadata: `data/ar/metadata/0000.parquet` … `0207.parquet` (captions with word timestamps, bandwidth, channels)
- Direct URL pattern: `https://huggingface.co/datasets/espnet/yodas3/resolve/main/data/ar/audio/NNNN.tar`
- Paper: [YODAS v3, Interspeech 2026](https://www.isca-archive.org/interspeech_2026/chen26d_interspeech.html)

The pipeline downloads each tar itself, keeps at most 2 tars ahead on disk, and deletes each raw tar once its shard is uploaded.

## Credentials

The real values live in `secrets.env` in the repo root. Git ignores that file, so it never goes to GitHub. A copy is kept on the original machine (`/workspace/yodas3-arabic-tts-pipeline/secrets.env`). Create it on a new machine like this:

```bash
cat > secrets.env <<'X'
HF_TOKEN=hf_...                         # Hugging Face token, account Rabe3
HETZNER_HOST=u685260.your-storagebox.de
HETZNER_USER=u685260
HETZNER_PASS='...'                      # storage box password
X
chmod 600 secrets.env
```

| What | Value |
|---|---|
| Hugging Face account | `Rabe3`. It must have accepted the gated [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) (done) |
| Hetzner storage box | `u685260.your-storagebox.de`, user `u685260`, SFTP on port 23, rclone remote name `hetzner` |
| Output folder on the box | `yodas3-arabic-tts-clean/` |

`setup/install.sh` loads `secrets.env` automatically.

## What the machine must have

| Resource | Minimum | Why |
|---|---|---|
| GPU | NVIDIA, 24 GB VRAM or more, bf16 support (Ampere or newer) | 9 neural models are loaded at once (weights about 7 GB); the Cohere ASR alone peaked at 6.8 GB at batch 64 |
| NVIDIA driver | Supports CUDA 13.0 (driver ≥ 580) for the default `cu130` wheels; otherwise set `TORCH_INDEX=https://download.pytorch.org/whl/cu128` (CUDA ≥ 12.8) | Blackwell GPUs (GB10, RTX 50xx, B200) need CUDA ≥ 12.8 builds |
| System RAM | 64 GB free on a discrete-GPU machine. On unified-memory machines (GB10 / DGX Spark, GH200) at least 60 GB must be free, because GPU memory comes out of the same pool | Measured 37 GB process memory with 16 CPU workers; the default is now 12 |
| CPU | 16 cores or more | 12 decode/VAD workers at about 90× real time each, plus the GPU feeder |
| Disk | 150 GB free | 3 raw shards (26 GB) + venv (7 GB) + models (about 6 GB) + outputs (about 120 MB per shard) |
| Network | 50 MB/s or more from huggingface.co; outbound TCP 23 to the storage box | one 8.6 GB shard every 20 min |
| OS | Linux, x86_64 or aarch64, Python 3.12, ffmpeg, rclone, git | |
| Container | Unprivileged Docker is fine; supervisor is optional | |

Tested on Vast instance 55188341: NVIDIA GB10 (DGX Spark, 128 GB unified memory, aarch64, 20 cores), Ubuntu 24.04, driver 610.57, Python 3.12.14, torch 2.14.1+cu130, torchaudio 2.11.0+cu130, transformers 5.19.0, pyannote.audio 4.0.7. A faster datacenter GPU (H100, RTX Pro 6000) should beat the 20 min/shard measured on GB10, but that hasn't been measured.

## Setup on a new machine

```bash
git clone git@github.com:<you>/yodas3-arabic-tts-pipeline.git /workspace/yodas3-arabic-tts-pipeline
cd /workspace/yodas3-arabic-tts-pipeline
# put secrets.env in the repo root first (see Credentials)
bash setup/install.sh
```

`setup/install.sh` does the following:
- creates `/venv/main`
- installs torch and the pinned packages in `setup/requirements.txt`
- clones DiariZen, FireRedVAD, URGENT-MOS and UTMOSv2 at the exact commits used
- downloads every model and test-loads all of them
- configures the `hetzner` rclone remote
- links `pipeline/` to `/workspace/pipeline`
- installs the supervisor service

All paths are fixed under `/workspace`.

### Resume from where the run stopped

Shard 0000 is already done. Mark it done so the new machine skips it, and leave `.uploaded` in place so it is not uploaded again:

```bash
mkdir -p /workspace/yodas3_ar_clean/done
touch /workspace/yodas3_ar_clean/done/0000.done /workspace/yodas3_ar_clean/done/0000.uploaded
```

## Run, monitor, stop

```bash
supervisorctl reread && supervisorctl update        # starts service yodas_ar_audit (shards 0000-0207)
supervisorctl status yodas_ar_audit
tail -f /workspace/yodas3_ar_clean/run.log          # one line per shard, upload and memory wait
ls /workspace/yodas3_ar_clean/done/                 # NNNN.done = processed, NNNN.uploaded = on Hetzner
supervisorctl stop yodas_ar_audit                   # safe: the current shard restarts from scratch next time
```

Without supervisor: `cd /workspace/pipeline && nohup python run_shards.py 0 207 >> /workspace/yodas3_ar_clean/run.log 2>&1 &`

Safety behaviour:
- **Resumable.** Finished shards have `done/NNNN.done` and are skipped. A failed shard keeps its raw tar and is retried on the next start.
- **Verified uploads.** Each finished shard is copied with `rclone copy`, then checked with `rclone check --size-only`, then marked `.uploaded`. Shards finished but not yet uploaded are uploaded on the next start.
- **Memory guard.** Before loading models and before each shard, the run waits for 40 GB of free host memory. It pauses feeding videos below 6 GB.
- **Crash restart.** If the process is killed (for example when the host runs out of memory), the service exits non-zero and supervisor restarts it.

## What the pipeline does

Each video goes through these steps in order. A chunk is kept only if it passes every rule. Later models only see what earlier ones kept, which keeps the run fast.

1. **Decode** the Opus/WebM to 48 kHz stereo in 5-minute blocks. Skip videos whose metadata bandwidth is under 22.05 kHz.
2. **Find speech** with Silero VAD v6 ∪ FireRedVAD: speech by either counts as speech, so a cut is only made where both hear silence.
3. **Chunk** into 2–20 s pieces, cut only in pauses of at least 0.3 s, splitting at the largest pause first. Each boundary moves to the quietest 10 ms of its pause. Chunks within 0.5 s of a decode-block or file edge are dropped, since the video may start or end mid-word.
4. **Signal checks** (CPU), measured on the 48 kHz source:

   | Rule | Keep if |
   |---|---|
   | clipping | fraction of samples ≥ 0.99 full scale < 1e-5, at most 1 flat-top plateau, peak ≤ 1.0 |
   | level | speech level between −35 and −6 dBFS |
   | bandwidth | real content up to at least 10 kHz (−50 dB rule) |
   | stereo phase | L/R correlation ≥ 0.6 |
   | DC offset | \|mean\| < 0.01 |
   | speech ratio | VAD speech ≥ 60 % of the chunk |
   | caption speaker turn | the YouTube caption has no `>>` speaker-change marker inside the chunk |
   | cut edge | neither edge is both > 12 dB above the chunk's noise floor and within 30 dB of speech level |

5. **Neural judges** (GPU), cheapest first:

   | Model | Checks | Keep if |
   |---|---|---|
   | [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) | speakers, overlap | exactly 1 speaker, overlap ≤ 1 %, speech ≥ 60 % |
   | [mispeech/ced-base](https://huggingface.co/mispeech/ced-base) (AudioSet tagger) | music, singing, crowd/babble, laughter, noise, effects, synthetic voice | music ≤ 0.10, singing ≤ 0.05, crowd ≤ 0.05, laughter ≤ 0.05, noise ≤ 0.20, effects ≤ 0.20, speech ≥ 0.60, synthetic ≤ 0.30, and the video's median synthetic score ≤ 0.20 |
   | [FireRedTeam/FireRedVAD](https://huggingface.co/FireRedTeam/FireRedVAD) AED | music, singing (second opinion) | 95th-percentile music ≤ 0.5, singing ≤ 0.5 |
   | [DNSMOS P.835](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS) | noise, quality | background ≥ 3.8, overall ≥ 2.8, signal ≥ 3.3 |
   | [UrgentMOS](https://huggingface.co/urgent-challenge/urgent-mos-f1c1m5dcorpus) | noise, quality | overall MOS ≥ 3.3, ScoreQ ≥ 3.3 |
   | [DiariZen WavLM-Large-s80-v2](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2) | second speaker, overlap | second speaker active ≤ 0.2 s, overlap ≤ 1 % |
   | [WeSpeaker ResNet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) | speaker change inside a chunk | every 3 s piece has cosine ≥ 0.5 to the whole chunk |
   | [Meta Audiobox Aesthetics](https://huggingface.co/facebook/audiobox-aesthetics) | busy mix (music beds) | production complexity ≤ 2.6 |

6. **Transcript** with [NAMAA-Space/Cohere-Speech-Tashkeel-2B](https://huggingface.co/NAMAA-Space/Cohere-Speech-Tashkeel-2B), which outputs fully diacritized Arabic.
   - Numbers become Western digits (1, 2, 3) when the spoken number matches a digit in the caption, or when the number is unambiguous (two or more words, or 11 and up). Words like `واحد` ("someone") stay as words.
   - **Word rule:** the chunk is kept only if every word matches the YouTube caption once tashkeel, punctuation and spelling variants are ignored. Ignored variants: alef/hamza forms, ة/ه, ى/ي, Eastern digits, word spacing, `ما حدش`=`محدش`, `يقول لك`=`يقولك`. Caption words starting within 0.35 s of an edge may count on either side of it.
   - Chunks are also dropped when the caption has fewer than 3 words, the speech rate is outside 5–22 characters/s, or the caption and ASR give different numbers.
   - Mixed Arabic + English is allowed.
7. **Re-split salvage.** Some chunks pass every audio check but fail only on words. These are split at internal pauses of at least 0.3 s into pieces of 2 s or more, and each piece is re-transcribed. Pieces whose words match exactly are kept. Neighbouring matching pieces are re-joined (up to 20 s) and checked again. Their ids end in `s`.

All thresholds are in `pipeline/run_shards.py` (`SIGNAL_RULES`, `THRESH`, `ASR_RULES`).

## Output on Hetzner (`yodas3-arabic-tts-clean/`)

| Path | Content |
|---|---|
| `audio/NNNN.tar` | WebDataset with `<id>.flac` (24 kHz, mono, 16-bit, original level) and `<id>.json` for every kept chunk |
| `manifest/NNNN.parquet` | every chunk, kept and rejected, with all scores, `keep` and `fail` (comma-separated reasons) |
| `manifest/NNNN.videos.parquet` | per video: length, speech found, chunks, skip reason, source metadata |
| `spk_emb/NNNN.npz` | `ids` + `emb` (float16, 256-d WeSpeaker embedding) per kept chunk, for speaker clustering or balancing |
| `stats/NNNN.json` | hours in and kept, seconds rejected per rule, per-model GPU time |

Chunk id: `<video id>_<start ms, 8 digits>`, with an `s` suffix for re-split pieces. Main JSON fields:
- `text_tashkeel`: final diacritized transcript with digits
- `text_tashkeel_spelled`: the same with numbers spelled out
- `text_plain`: no tashkeel
- `text`: the YouTube caption words
- `start` / `end`: seconds in the source video
- every judge score

Rejected chunks do not keep audio. Loosening a threshold later means re-processing those shards.

## How the models were chosen

Every judge was scored against human labels: 655 clips from [AITW](https://zenodo.org/records/16937418), hand-labelled YODAS clips (CC BY 4.0). ROC-AUC: 1.0 separates perfectly, 0.5 is chance. Full table: `results/aitw_auc_report.csv`.

| Check | Best | Others tested |
|---|---|---|
| Two or more speakers | DiariZen 0.951 | WeSpeaker-34 0.906, WeSpeaker-293 0.893, pyannote 3.0 0.844 |
| Background music | FireRedVAD 0.975, CED 0.970 | Audiobox PC 0.947, DNSMOS BAK 0.823 |
| Background noise | UrgentMOS ScoreQ 0.785 | Audiobox PC 0.772, DNSMOS BAK 0.743, UTMOS22 0.735, UTMOSv2 0.619 |
| Synthetic voice | CED 0.855 | none usable |

With the final thresholds, the rule set catches 100 % of multi-speaker, music and synthetic clips and 96 % of noisy clips, and keeps 24.7 % of the clips humans called clean. VAD choice comes from the [FireRedVAD paper](https://arxiv.org/pdf/2603.10420): FireRedVAD has a lower false-alarm rate than Silero, and the pipeline uses both.

A human listening check covered 53 clips from the shard 0000 test, using the listening page at https://claude.ai/artifact/CS1cbstmkD4nVrf2sDq3fa. All kept clips with complete audio were judged clean. 6 of 32 rejected clips were judged clean: the noise and second-speaker rules sit close to the limit, and one video was flagged as synthetic.

## Pipeline files

| File | Role |
|---|---|
| `pipeline/run_shards.py` | driver: download, CPU workers, GPU cascade, salvage, writing, upload, memory guard |
| `pipeline/audit_lib.py` | decoding, VAD chunking, boundary refinement, signal metrics, caption windows |
| `pipeline/gpu_models.py` | batched wrappers for DNSMOS, CED, Audiobox, pyannote, WeSpeaker, Cohere tashkeel ASR |
| `pipeline/candidates.py` | DiariZen, FireRedVAD AED, UrgentMOS, UTMOSv2, WeSpeaker-293 wrappers |
| `pipeline/diarizen_seg.py` | loads DiariZen's model without its pinned pyannote-3.1 fork |
| `pipeline/arabic_text.py` | Arabic normalisation, exact word matching, caption-guided number→digit conversion |
| `pipeline/aitw_fetch.py`, `aitw_eval.py` | builds the AITW benchmark and scores and calibrates all judges |
| `pipeline/review_samples.py`, `build_review.py` | builds the listening-review page |
| `pipeline/calibrate.py` | first calibration on 40 videos (historical) |
| `deploy/supervisor/` | service wrapper and config |
| `setup/` | `install.sh`, `requirements.txt` |
| `results/` | AITW AUC table, shard 0000 stats |

## Open questions and limits

- **Synthetic voice.** All 243 chunks of video `132709369b7c8f613c2ffafd196f98ea` score as synthetic (median 0.69). If that is a real person, consider deciding the synthetic check per video only, with a higher threshold.
- **Reverb.** There is no dedicated reverb model. [pyannote/brouhaha](https://huggingface.co/pyannote/brouhaha) (SNR + C50) is gated; accept its terms to add it.
- **DiariZen license.** Its weights are CC BY-NC 4.0. Check this is acceptable for a commercial TTS model.
- **Yield.** About 0.85 % of the raw audio is kept. The biggest losses are word mismatches, bandwidth, multiple speakers, noise and clipping (see `results/shard_0000_stats.json`).
