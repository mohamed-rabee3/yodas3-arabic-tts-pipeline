#!/usr/bin/env bash
# Rebuild the complete environment for the YODAS v3 Arabic TTS audit on a fresh Linux + NVIDIA machine.
# Paths are fixed to /workspace (the code expects them). Run as root (or with sudo) from anywhere:
#   bash setup/install.sh        (reads secrets.env in the repo root; see README.md, Credentials)
set -euo pipefail
_REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "$_REPO/secrets.env" ]; then set -a; . "$_REPO/secrets.env"; set +a; fi
: "${HF_TOKEN:?export HF_TOKEN first (see README.md, Credentials)}"
: "${HETZNER_HOST:?export HETZNER_HOST}" "${HETZNER_USER:?export HETZNER_USER}" "${HETZNER_PASS:?export HETZNER_PASS}"

W=/workspace
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV=${VENV:-/venv/main}
# CUDA 13.0 wheels (needed for Blackwell, e.g. GB10 / RTX 50xx / B200). cu128 also works on any driver >= 12.8.
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
export HF_HOME=$W/.hf_home

echo "== 1/8 system packages"
if command -v apt-get >/dev/null; then
  apt-get update -qq && apt-get install -y -qq ffmpeg rclone git curl python3.12 python3.12-venv >/dev/null
fi

echo "== 2/8 python venv ($VENV)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -x "$VENV/bin/python" ] || uv venv -p 3.12 "$VENV"
source "$VENV/bin/activate"

echo "== 3/8 torch"
uv pip install torch torchaudio torchvision --index-url "$TORCH_INDEX"
uv pip install -r "$REPO/setup/requirements.txt"

echo "== 4/8 third-party code at the exact commits used"
clone() {  # name url commit
  [ -d "$W/$1" ] || git clone -q --recursive "$2" "$W/$1"
  git -C "$W/$1" fetch -q --all && git -C "$W/$1" checkout -q "$3"
}
clone DiariZen   https://github.com/BUTSpeechFIT/DiariZen.git   844f5555b0a98acd0931511fc641a8c5b8ba92c7
clone FireRedVAD https://github.com/FireRedTeam/FireRedVAD.git  c30ec49e8cc69642b0ee65362eba11b9d11c6e54
clone URGENT-MOS https://github.com/vvwangvv/URGENT-MOS.git     79fb39277ba8989d1dea28bf109a2d1631530fbe
clone UTMOSv2    https://github.com/sarulab-speech/UTMOSv2.git  cc2700db57bb83ee13dc31ebe1b868c254e15d09  # evaluation only
# --no-deps: their pins (torch 2.1, numpy 1.26, pyannote 3.1 fork) would break the stack above
uv pip install --no-deps -e "$W/FireRedVAD" -e "$W/URGENT-MOS" -e "$W/UTMOSv2"
# DiariZen is NOT pip-installed; pipeline/diarizen_seg.py imports its model code from $W/DiariZen.

echo "== 5/8 Hugging Face login + model weights"
hf auth login --token "$HF_TOKEN"
mkdir -p "$W/models"
for f in sig_bak_ovr.onnx model_v8.onnx; do
  [ -f "$W/models/$f" ] || curl -sL -o "$W/models/$f" "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/$f"
done
hf download BUT-FIT/diarizen-wavlm-large-s80-md-v2      --local-dir "$W/models/diarizen/large-s80-v2"
hf download FireRedTeam/FireRedVAD                      --local-dir "$W/models/FireRedVAD"
hf download urgent-challenge/urgent-mos-f1c1m5dcorpus   --local-dir "$W/models/urgent-mos"

echo "== 6/8 link pipeline, warm every model cache (downloads CED, Audiobox, pyannote, WeSpeaker, Cohere ASR)"
ln -sfn "$REPO/pipeline" "$W/pipeline"
mkdir -p "$W/yodas3_raw" "$W/yodas3_ar_clean"
(cd "$W/pipeline" && python -c "
import torch; torch.set_grad_enabled(False)
from gpu_models import Judges, TashkeelASR
Judges(); TashkeelASR(); print('all judges load OK')")

echo "== 7/8 Hetzner storage box remote 'hetzner' (SFTP, port 23)"
rclone config create hetzner sftp host "$HETZNER_HOST" user "$HETZNER_USER" port 23 \
  pass "$(rclone obscure "$HETZNER_PASS")" shell_type unix md5sum_command none sha1sum_command none >/dev/null
rclone lsd hetzner: >/dev/null && echo "hetzner remote OK"

echo "== 8/8 supervisor service (optional; skipped if supervisor is absent)"
if [ -d /etc/supervisor/conf.d ]; then
  cp "$REPO/deploy/supervisor/yodas_ar_audit.sh" /opt/supervisor-scripts/ 2>/dev/null || {
    mkdir -p /opt/supervisor-scripts && cp "$REPO/deploy/supervisor/yodas_ar_audit.sh" /opt/supervisor-scripts/; }
  chmod +x /opt/supervisor-scripts/yodas_ar_audit.sh
  cp "$REPO/deploy/supervisor/yodas_ar_audit.conf" /etc/supervisor/conf.d/
  echo "service installed; start with: supervisorctl reread && supervisorctl update"
else
  echo "no supervisor: run manually with  cd $W/pipeline && nohup python run_shards.py 0 207 >> $W/yodas3_ar_clean/run.log 2>&1 &"
fi
echo "done"
