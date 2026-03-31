import librosa
from .asr import transcribe_audio_waveform
from .diarization import diarize
from pyannote.core import Annotation, Segment


def merge_adjacent_segments(ann):
    """
    Merge overlapping or adjacent segments with the same speaker.
    Fixes duplicate transcriptions from overlapping diarization windows.
    """
    merged = Annotation()
    
    tracks = list(ann.itertracks(yield_label=True))
    if not tracks:
        return merged
    
    current_segment, _, current_label = tracks[0]
    current_start = current_segment.start
    current_end = current_segment.end
    
    for segment, _, label in tracks[1:]:
        if label == current_label and segment.start <= current_end + 0.1:  # 100ms tolerance
            # Merge: extend current segment
            current_end = max(current_end, segment.end)
        else:
            # Different speaker or gap detected: save current and start new
            merged[Segment(current_start, current_end)] = current_label
            current_segment = segment
            current_start = segment.start
            current_end = segment.end
            current_label = label
    
    # Save the last segment
    merged[Segment(current_start, current_end)] = current_label
    return merged


def run_asr_diarization(audio_path, min_spk=2, max_spk=3):
    ann = diarize(audio_path, min_spk=min_spk, max_spk=max_spk)
    
    # Merge overlapping segments to avoid duplicate transcriptions
    ann = merge_adjacent_segments(ann)
    
    y, sr = librosa.load(audio_path, sr=16000)

    results = []
    for segment, track, label in ann.itertracks(yield_label=True):
        start_frame = int(segment.start * sr)
        end_frame = int(segment.end * sr)

        if end_frame <= start_frame:
            continue

        chunk = y[start_frame:end_frame]
        text = transcribe_audio_waveform(chunk, sr=sr)

        results.append({
            "speaker": str(label),
            "start": float(segment.start),
            "end": float(segment.end),
            "text": text.strip(),
        })

    full_transcript = " ".join(item["text"] for item in results if item["text"])

    return {
        "audio_path": audio_path,
        "segments": results,
        "full_transcript": full_transcript,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASR + diarization pipeline")
    parser.add_argument("audio_path", type=str, help="Path to audio file")
    parser.add_argument("--min_spk", type=int, default=2)
    parser.add_argument("--max_spk", type=int, default=3)

    args = parser.parse_args()
    outcome = run_asr_diarization(args.audio_path, args.min_spk, args.max_spk)
    print(outcome)
