#!/bin/bash
# YODAS v3 Arabic -> strict audio audit + tashkeel ASR + exact caption agreement -> Hetzner
utils=/opt/supervisor-scripts/utils
# Vast base-image helpers (log forwarding, /workspace/.env); optional elsewhere
[ -f "${utils}/logging.sh" ] && . "${utils}/logging.sh"
[ -f "${utils}/environment.sh" ] && . "${utils}/environment.sh"
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export PYTHONUNBUFFERED=1
cd /workspace/pipeline
python run_shards.py 0 207 2>&1 | tee -a /workspace/yodas3_ar_clean/run.log
rc=${PIPESTATUS[0]}
# a silent kill (e.g. host out of memory) must not look like success
if [ "$rc" -eq 0 ] && ! tail -n 3 /workspace/yodas3_ar_clean/run.log | grep -q "all done"; then rc=1; fi
exit $rc
