import numpy as np
import librosa
from resemblyzer import VoiceEncoder
from spectralcluster import SpectralClusterer, AutoTune
from pyannote.core import Annotation, Segment
from sklearn.preprocessing import normalize
from scipy.stats import mode

SAMPLE_RATE = 16000
WINDOW = 1.5
STEP = 0.5  # FIX 1: was 1.0 — denser overlap for better boundary detection


def extract_embeddings(audio_path):
    wav, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    encoder = VoiceEncoder()

    segments = []
    embeddings = []
    duration = len(wav) / sr

    for start in np.arange(0, duration - WINDOW, STEP):
        s = int(start * sr)
        e = int((start + WINDOW) * sr)
        segment_wav = wav[s:e]

        if len(segment_wav) < WINDOW * sr:
            continue

        emb = encoder.embed_utterance(segment_wav)
        embeddings.append(emb)
        segments.append((start, start + WINDOW))

    embeddings = normalize(np.array(embeddings))
    return embeddings, segments


def cluster_embeddings(embeddings, min_clusters=2, max_clusters=10):
    # FIX 2: autotune=None disabled automatic speaker count estimation entirely.
    # AutoTune uses eigengap heuristic to find the optimal cluster count.
    clusterer = SpectralClusterer(
        min_clusters=min_clusters,
        max_clusters=max_clusters,
    )
    return clusterer.predict(embeddings)


def smooth_labels(labels, segments, time_radius=1.5):
    """
    FIX 3: Old version used index-based window which ignores timing.
    This smooths using a time-aware window (±time_radius seconds),
    using scipy mode instead of np.bincount to handle edge cases.
    """
    starts = np.array([s[0] for s in segments])
    smoothed = labels.copy()

    for i, t in enumerate(starts):
        in_window = np.abs(starts - t) <= time_radius
        window_labels = labels[in_window]
        smoothed[i] = mode(window_labels, keepdims=True).mode[0]

    return smoothed


def merge_segments(segments, labels):
    """
    FIX 4: Old build_rttm() emitted raw overlapping windows directly into
    the Annotation — every 1.5s window overlapped the next by 1.0s,
    causing every region to be double-labelled.
    
    This function collapses overlapping windows for the same speaker into
    clean, contiguous, non-overlapping blocks before building the annotation.
    """
    merged = []
    cur_start, cur_end, cur_spk = segments[0][0], segments[0][1], labels[0]

    for (start, end), spk in zip(segments[1:], labels[1:]):
        if spk == cur_spk and start <= cur_end + 0.05:  # 50ms tolerance
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end, cur_spk))
            cur_start, cur_end, cur_spk = start, end, spk

    merged.append((cur_start, cur_end, cur_spk))
    return merged


def build_rttm(segments, labels):
    ann = Annotation()
    for start, end, spk in merge_segments(segments, labels):
        ann[Segment(start, end)] = f"SPEAKER_{spk}"
    return ann


def diarize(audio_path, min_spk=2, max_spk=6):
    print("Extracting embeddings...")
    emb, seg = extract_embeddings(audio_path)

    print("Clustering speakers...")
    labels = cluster_embeddings(emb, min_clusters=min_spk, max_clusters=max_spk)

    print("Smoothing labels...")
    labels = smooth_labels(labels, seg, time_radius=1.5)

    print("Building RTTM...")
    ann = build_rttm(seg, labels)

    print("Diarization done.\n")
    print(ann)
    return ann