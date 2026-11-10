#!/usr/bin/env bash
# Downloads the ONNX models and the AMI meetings used for evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models data/ami/audio data/ami/rttm
R=https://github.com/k2-fsa/sherpa-onnx/releases/download
(cd models
 curl -sSL $R/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 | tar xj
 for m in nemo_en_titanet_small wespeaker_en_voxceleb_resnet34_LM 3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k; do
   curl -sSLO $R/speaker-recongition-models/$m.onnx
 done
 curl -sSL $R/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2 | tar xj
 rm -f sherpa-onnx-whisper-tiny.en/tiny.en-encoder.onnx sherpa-onnx-whisper-tiny.en/tiny.en-decoder.onnx)

# AMI Meeting Corpus (CC BY 4.0): headset mix audio + reference speaker turns.
# The Edinburgh server sometimes answers 403 or drops the connection, so
# every file is retried and resumed until its size matches.
AMI=https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus
RTTM=https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/only_words/rttms/test
for m in ES2004a ES2004b IS1009a IS1009b IS1009c TS3003a TS3003b TS3003c EN2002a EN2002b EN2002c; do
  curl -sSL -o data/ami/rttm/$m.rttm $RTTM/$m.rttm
  f=data/ami/audio/$m.wav; u=$AMI/$m/audio/$m.Mix-Headset.wav
  for try in $(seq 1 10); do
    want=$(curl -sI "$u" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r')
    if [ -n "$want" ]; then
      if file "$f" 2>/dev/null | grep -q HTML; then rm -f "$f"; fi
      [ "$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo 0)" = "$want" ] && break
      curl -sS -C - -o "$f" "$u" || true
      [ "$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo 0)" = "$want" ] && break
    fi
    sleep 5
  done
done
