import librosa
from transformers import pipeline

MODEL_NAME = "spktsagar/wav2vec2-large-xls-r-300m-nepali-openslr"
_asr_pipe = None


def get_asr_pipeline(model_name: str = MODEL_NAME):
    global _asr_pipe
    if _asr_pipe is None:
        import torch
        _asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            chunk_length_s=30,
            stride_length_s=(5, 2),
            device=0 if torch.cuda.is_available() else -1,
        )
    return _asr_pipe


def transcribe_audio_waveform(audio, sr: int = 16000, model_name: str = MODEL_NAME):
    if len(audio) == 0:
        return ""

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    pipe = get_asr_pipeline(model_name)
    out = pipe(audio, sampling_rate=sr)
    if isinstance(out, list):
        out = out[0]

    return out.get("text", "")


def transcribe_audio_file(audio_path: str, model_name: str = MODEL_NAME):
    audio, sr = librosa.load(audio_path, sr=16000)
    return transcribe_audio_waveform(audio, sr, model_name)
