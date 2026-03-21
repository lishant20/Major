from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile
import os
from asr_diarization.pipeline import run_asr_diarization

app = FastAPI(title="ASR + Diarization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/transcribe")
async def transcribe(upload_file: UploadFile = File(...), min_spk: int = 2, max_spk: int = 3):
    filename = upload_file.filename
    ext = Path(filename).suffix.lower()
    if ext not in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        data = await upload_file.read()
        tmp_file.write(data)
        tmp_file.close()

        result = run_asr_diarization(tmp_file.name, min_spk=min_spk, max_spk=max_spk)
        result["source_filename"] = filename
        return result
    finally:
        try:
            os.unlink(tmp_file.name)
        except OSError:
            pass


@app.get("/health")
def health():
    return {"status": "ok"}
