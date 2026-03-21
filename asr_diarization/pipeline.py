import librosa
from .asr import transcribe_audio_waveform
from .diarization import diarize


def run_asr_diarization(audio_path, min_spk=2, max_spk=3):
    ann = diarize(audio_path, min_spk=min_spk, max_spk=max_spk)
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
