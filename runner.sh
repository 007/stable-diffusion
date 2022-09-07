#!/bin/bash
set -euo pipefail


GEN_GRID=false
#GEN_GRID=true

PROMPT="${1:-a farm landscape featuring a brown cow looking into the distance, oil painting in the style of vermeer}"

SEED="${2:-314159265}"

HASHDIR="outputs/$(md5sum <<< "${PROMPT} / ${SEED}" | awk '{print $1}')"
echo "Creating output dir $HASHDIR"
mkdir -p "${HASHDIR}"

jq -c . > "${HASHDIR}/metadata.json" <<< '{"prompt":'$(jq -R 'gsub("\\s+$";"")' <<< "$PROMPT")',"seed":'${SEED}'}'

if [ "$GEN_GRID" == "true" ] ; then

  RESULT_IMG="grid-0001.png"
  python3 scripts/txt2img.py \
  --skip_watermark --skip_safety_check \
  --n_iter 3 --n_samples 6 \
  --W 640 --H 576 \
  --ddim_steps 80 \
  --seed "${SEED}" \
  --outdir "${HASHDIR}/" \
  --prompt "${PROMPT}"

else
  RESULT_IMG="samples/00000.png"
  python3 scripts/txt2img.py \
  --skip_safety_check \
  --skip_watermark \
  --save_steps \
  --skip_grid \
  --n_iter 1 --n_samples 1 \
  --ddim_steps 80 \
  --W 1024 --H 576 \
  --seed "${SEED}" \
  --outdir "${HASHDIR}/" \
  --prompt "${PROMPT}"
#  --W 1024 --H 1024 \
#  --W 512 --H 512 \
#  --ddim_steps 120 \
#  --scale 5 \

fi

echo "Results stored in $HASHDIR"
echo "$PROMPT"
display $HASHDIR/${RESULT_IMG} &
echo ""
echo ""
