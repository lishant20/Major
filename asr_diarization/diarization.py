import numpy as np
import librosa
from resemblyzer import VoiceEncoder
from spectralcluster import SpectralClusterer
from pyannote.core import Annotation, Segment

# ------------ CONFIG ----------------
SAMPLE_RATE = 16000
WINDOW = 1.5     # window size in seconds
STEP = 0.75      # step size in seconds
# ------------------------------------


def extract_embeddings(audio_path):
    wav, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    encoder = VoiceEncoder()

    segments = []
    embeddings = []

    for start in np.arange(0, len(wav) / sr - WINDOW, STEP):
        s = int(start * sr)
        e = int((start + WINDOW) * sr)
        segment_wav = wav[s:e]

        if len(segment_wav) < WINDOW * sr:
            continue

        emb = encoder.embed_utterance(segment_wav)
        embeddings.append(emb)
        segments.append((start, start + WINDOW))

    return np.array(embeddings), segments


def cluster_embeddings(embeddings, min_clusters=2, max_clusters=10):
    clusterer = SpectralClusterer(
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        autotune=None
    )
    labels = clusterer.predict(embeddings)
    return labels


def build_rttm(segments, labels):
    ann = Annotation()

    for (start, end), spk in zip(segments, labels):
        ann[Segment(start, end)] = f"SPEAKER_{spk}"

    return ann


def diarize(audio_path, min_spk=2, max_spk=3):
    print("Extracting embeddings...")
    emb, seg = extract_embeddings(audio_path)

    print("Clustering speakers...")
    labels = cluster_embeddings(emb, min_clusters=min_spk, max_clusters=max_spk)

    print("Building RTTM...")
    ann = build_rttm(seg, labels)
    print("Diarization done.")
    return ann
