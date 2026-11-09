#!/usr/bin/env bash
# Downloads the ONNX models and Mini LibriSpeech test data.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models data/raw
R=https://github.com/k2-fsa/sherpa-onnx/releases/download
(cd models
 curl -sSL $R/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 | tar xj
 curl -sSLO $R/speaker-recongition-models/nemo_en_titanet_small.onnx
 curl -sSL $R/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2 | tar xj
 rm -f sherpa-onnx-whisper-tiny.en/tiny.en-encoder.onnx sherpa-onnx-whisper-tiny.en/tiny.en-decoder.onnx)
(cd data/raw
 for s in dev-clean-2 train-clean-5; do
   curl -sSL https://openslr.elda.org/resources/31/$s.tar.gz | tar xz
 done)
