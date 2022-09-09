#!/bin/bash

PATTERN="${1}"

PATTERN_MATCH_ALL="${PATTERN}/samples/*-*.png"
PATTERN_MATCH_FEW="${PATTERN}/samples/00000-0?9.png"
echo $PATTERN_MATCH

ffmpeg -r 3 -pattern_type glob -i "${PATTERN_MATCH_ALL}" -f matroska - | mplayer -vo x11 -

for x in ${PATTERN_MATCH_FEW}; do display "$x";done
