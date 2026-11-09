"""Speech pipeline: diarization (who spoke when), speaker embeddings (voice
fingerprints), and transcription. Everything runs locally with sherpa-onnx —
no audio leaves the machine."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx

SAMPLE_RATE = 16_000
ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

SEGMENTATION_MODEL = MODELS / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
# Chosen by benchmark (data/eval/model_comparison.json): NVIDIA TitaNet-small
# separates voices far better than WeSpeaker ResNet34 or 3D-Speaker ERes2Net.
EMBEDDING_MODEL = MODELS / "nemo_en_titanet_small.onnx"
CLUSTER_THRESHOLD = 0.7  # diarization: lower -> more speakers; tuned in evaluate
WHISPER_DIR = MODELS / "sherpa-onnx-whisper-tiny.en"

# Minimum audio we accept for an enrollment. Shorter samples give unstable
# voice fingerprints and push the false-accept rate up.
MIN_ENROLL_SECONDS = 8.0


def load_audio(path: str | Path) -> np.ndarray:
    """Decode any audio format (wav, webm from the browser, mp3, m4a) to
    16 kHz mono float32 via ffmpeg."""
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


@dataclass
class Turn:
    start: float
    end: float
    cluster: int  # anonymous diarization label ("Speaker 1" = cluster 0)


class VoiceEngine:
    def __init__(self, num_threads: int = 2, cluster_threshold: float = CLUSTER_THRESHOLD):
        self._embedder = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(EMBEDDING_MODEL), num_threads=num_threads))
        self._diarizer = sherpa_onnx.OfflineSpeakerDiarization(
            sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                        model=str(SEGMENTATION_MODEL)),
                    num_threads=num_threads),
                embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=str(EMBEDDING_MODEL), num_threads=num_threads),
                clustering=sherpa_onnx.FastClusteringConfig(threshold=cluster_threshold),
                min_duration_on=0.3,
                min_duration_off=0.5))
        self._asr = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(WHISPER_DIR / "tiny.en-encoder.int8.onnx"),
            decoder=str(WHISPER_DIR / "tiny.en-decoder.int8.onnx"),
            tokens=str(WHISPER_DIR / "tiny.en-tokens.txt"),
            num_threads=num_threads)

    def embed(self, samples: np.ndarray) -> np.ndarray:
        """Voice fingerprint: a unit vector (192-dim for TitaNet). Same voice -> vectors point
        the same way, so cosine similarity is a dot product."""
        stream = self._embedder.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        stream.input_finished()
        vec = np.asarray(self._embedder.compute(stream), dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-9)

    def diarize(self, samples: np.ndarray) -> list[Turn]:
        result = self._diarizer.process(samples).sort_by_start_time()
        return [Turn(s.start, s.end, s.speaker) for s in result]

    def transcribe(self, samples: np.ndarray) -> str:
        stream = self._asr.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self._asr.decode_stream(stream)
        # Whisper marks censored words as [beep] / [bleep]; drop the tags.
        return re.sub(r"\s*\[b?l?eep\]", "", stream.result.text).strip()


def cluster_audio(samples: np.ndarray, turns: list[Turn], cluster: int,
                  max_seconds: float = 30.0) -> np.ndarray:
    """All speech of one anonymous speaker, concatenated (capped), so we
    compute one robust fingerprint per speaker instead of one per sentence."""
    parts, total = [], 0.0
    for t in turns:
        if t.cluster != cluster:
            continue
        a, b = int(t.start * SAMPLE_RATE), int(t.end * SAMPLE_RATE)
        parts.append(samples[a:b])
        total += t.end - t.start
        if total >= max_seconds:
            break
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
