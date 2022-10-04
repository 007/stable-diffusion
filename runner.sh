#!/bin/bash
set -euo pipefail


GEN_GRID=false
GEN_GRID=true

PROMPT="${1:-a farm landscape featuring a brown cow looking into the distance, oil painting in the style of vermeer}"

SEED="${2:-314159265}"
SCALE=${3:-$((RANDOM % 25 + 1))}

HASHDIR="outputs/$(md5sum <<< "${PROMPT} / ${SEED}" | awk '{print $1}')"
echo "Creating output dir $HASHDIR"
mkdir -p "${HASHDIR}"

echo "Prompt: \"${PROMPT}\", Scale: ${SCALE}, Seed: ${SEED}"

jq -c . > "${HASHDIR}/metadata.json" <<< '{"prompt":'$(jq -R 'gsub("\\s+$";"")' <<< "$PROMPT")',"seed":'${SEED}',"scale":'${SCALE}'}'

if [ "$GEN_GRID" == "true" ] ; then

  RESULT_IMG="grid-0001.png"
  python3 scripts/txt2img.py \
  --skip_watermark --skip_safety_check \
  --n_iter 3 --n_samples 4 \
  --W 384 --H 384 \
  --ddim_steps 50 \
  --scale ${SCALE} \
  --seed "${SEED}" \
  --outdir "${HASHDIR}/" \
  --prompt "${PROMPT}"

#   --n_iter 2 --n_samples 3 \
#   --W 512 --H 512 \
#  --W 640 --H 576 \

else
  RESULT_IMG="samples/00000.png"
  python3 scripts/txt2img.py \
  --skip_safety_check \
  --skip_watermark \
  --skip_grid \
  --n_iter 1 --n_samples 1 \
  --H 512 --W 512 \
  --scale ${3:-10} \
  --seed "${SEED}" \
  --outdir "${HASHDIR}/" \
  --prompt "${PROMPT}"
#   --save_steps \
#   --ddim_steps 100 \
#  --H 1024 --W 576 \
#  --W 1600 --H 896 \
#  --ultra_low_vram \
#  --W 1024 --H 1024 \
#  --save_steps \
#  --W 512 --H 512 \
#  --ddim_steps 120 \
#  --scale 5 \

fi

echo "Results stored in $HASHDIR"
echo "$PROMPT"
display $HASHDIR/${RESULT_IMG} &
echo ""
echo ""
