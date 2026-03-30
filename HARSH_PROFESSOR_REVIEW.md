# HARSH PROFESSOR QUESTIONS & ANSWERS
## Nepali ASR and Speaker Diarization System

---

## 1. RESEARCH NOVELTY & CONTRIBUTION

### Q1: What is novel about your work? 
**Why would ACL or ICML care about this?**

**A:** 
- **HONEST ANSWER**: Truthfully, not much. We're combining two existing pre-trained models (wav2vec2-nepali ASR + resemblyzer speaker embeddings with spectral clustering). This is primarily an **engineering project**, not a research contribution.
- The wav2vec2-nepali model was created by anish-shilpakar (not us)
- Speaker diarization via spectral clustering has been standard since ~2017
- We simply integrated two components
- **WHAT WE SHOULD HAVE DONE**: 
  - Fine-tune models on Nepali-specific diarization datasets
  - Propose novel speaker segmentation techniques for low-resource languages
  - Create a benchmark dataset for Nepali speech

### Q2: Your diarization window is 1.5 seconds with 0.75 step - why?

**A:** 
- **HONEST ANSWER**: These are arbitrary hyperparameters we didn't optimize. We chose them from resemblyzer documentation.
- **EVIDENCE OF POOR RESEARCH**:
  - No ablation studies on window/step size
  - No justification for these specific values
  - No comparison against other window configurations
  - Original resemblyzer uses 1.5s - we just copied it
- **WHAT WE SHOULD HAVE DONE**:
  - Ablation study: Test {0.5s, 1.0s, 1.5s, 2.0s} windows
  - Test various step sizes
  - Measure DER/WER trade-off for each configuration
  - Document why these are optimal

### Q3: Why use resemblyzer instead of more recent speaker embedding models?

**A:**
- **HONEST ANSWER**: Convenience. Resemblyzer has easy-to-use API.
- **SHORTCOMINGS**:
  - Resemblyzer is from 2019 (5+ years old)
  - Better alternatives exist: ECAPA-TDNN, WavLM, Generalist Speaker Embeddings (2023)
  - No comparison against modern embedding methods
  - WavLM shows 30-40% improvement on speaker verification tasks
- **WHAT WE SHOULD HAVE DONE**:
  - Benchmark resemblyzer vs ECAPA-TDNN vs WavLM
  - Show DER/performance comparison
  - Justify our choice with experimental evidence

---

## 2. EVALUATION & METRICS

### Q4: Where are your experimental results and benchmarks?

**A:**
- **HONEST ANSWER**: We have none documented. The evaluation.py file exists but we haven't actually run comprehensive tests.
- **NO EVIDENCE FOR**:
  - Accuracy metrics (DER, WER)
  - Performance on test datasets
  - Comparison with baseline systems
  - Robustness across audio qualities/accents
- **WHAT WE SHOULD HAVE DONE**:
  - Create or use existing Nepali speech corpora (e.g., OpenSLR, MUCS)
  - Split into train/val/test
  - Report DER, WER, Speaker Confusion Rate
  - Compare against speaker diarization baselines (TDNN, x-vector, PLDA)
  - Report 95% confidence intervals

### Q5: DER benchmark - what's a good DER for Nepali?

**A:**
- **HONEST ANSWER**: We don't know because we haven't evaluated on any test set.
- **INDUSTRY BENCHMARKS** (for reference):
  - Conversational English: 10-15% DER (CALLHOME dataset)
  - Broadcast news: 5-8% DER (TIMIT)
  - Our system: UNKNOWN - never tested
- **WHAT WE SHOULD HAVE DONE**:
  - Define baseline DER from comparable systems
  - Report DER breakdown: Confusion%, Missed Detection%, False Alarm%
  - Test on multiple domains (meetings, calls, interviews, podcasts)
  - Report per-speaker DER (hard speakers vs easy)

### Q6: You claim the system works on "multiple formats" - did you actually test all of them?

**A:**
- **HONEST ANSWER**: No. The backend accepts .wav, .mp3, .flac, .m4a, .ogg, but:
  - Testing was probably done only on .wav
  - librosa.load() handles formats, but encoding issues not tested
  - No validation for corrupted/invalid files
  - No testing on edge cases (mono vs stereo, 8kHz vs 48kHz)
- **SHORTCOMINGS**:
  - No test suite documenting format compatibility
  - No handling of:
    - Mono vs stereo conversion
    - Sample rate mismatches beyond auto-resample
    - File corruption detection
    - Maximum file size limits
- **WHAT WE SHOULD HAVE DONE**:
  - Create test suite with diverse audio formats
  - Test mono, stereo, 22.05kHz, 44.1kHz, 48kHz inputs
  - Document known limitations and edge cases

---

## 3. SYSTEM DESIGN & ARCHITECTURE

### Q7: Your pipeline processes audio twice - librosa.load() in both diarization.py and pipeline.py. Why?

**A:**
- **INEFFICIENCY FOUND**:
  ```python
  # In pipeline.py:
  y, sr = librosa.load(audio_path, sr=16000)  # LOADS AGAIN
  
  # But audio was already loaded in diarization.py:
  wav, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
  ```
- **IMPACT**: 
  - 2x memory usage for large files
  - 2x I/O overhead
  - Possible timing inconsistencies between analyses
- **WHAT WE SHOULD HAVE DONE**:
  - Load once, pass waveform through pipeline
  - Refactor: `diarize(waveform, sr)` instead of `diarize(audio_path)`
  - Reduce memory footprint, improve speed

### Q8: Why no error handling in the diarization module?

**A:**
- **BRITTLE CODE**: Zero try/catch blocks
- **FAILURE MODES**:
  - Empty audio → crashes on embedding extraction
  - Very short audio (< 1.5s) → no embeddings → crashes
  - Non-speech audio → embeddings fail
  - Speaker count exceeds reasonable limits → spectral clustering fails
- **WHAT WE SHOULD HAVE DONE**:
  ```python
  def diarize(audio_path, min_spk=2, max_spk=3):
      try:
          emb, seg = extract_embeddings(audio_path)
          if len(emb) < min_spk:
              raise ValueError(f"Too few speakers detected ({len(emb)} < {min_spk})")
          ...
      except Exception as e:
          logger.error(f"Diarization failed: {e}")
          raise
  ```

### Q9: Hardcoded WINDOW=1.5s and STEP=0.75s - how does this handle variable-length speeches?

**A:**
- **PROBLEM**: 
  - 10-second speech → ~13 embeddings
  - 1-hour speech → ~4,800 embeddings
  - Spectral clustering with 4,800 samples may be slow/unreliable
- **NO SCALING STRATEGY**:
  - No handling of audio length edge cases
  - No memory profiling for long recordings
  - No optimization for batch processing
- **WHAT WE SHOULD HAVE DONE**:
  - Chunk very long audio (e.g., > 30 min)
  - Implement adaptive window sizing based on audio length
  - Profile memory/speed for recordings from 30s to 60min
  - Document maximum recommended audio length

---

## 4. MODEL SELECTION & JUSTIFICATION

### Q10: Why wav2vec2-nepali over other Nepali ASR models?

**A:**
- **HONEST ANSWER**: It's probably the only public Nepali ASR model available.
- **QUESTIONS UNANSWERED**:
  - How does it compare to:
    - wav2vec2-xls-r (multilingual)?
    - Whisper (multilingual, newer)?
    - Other Nepali-specific models?
  - What's the WER on Nepali test sets?
  - Does it handle accents/dialects?
  - What's the training data? Is it representative?
- **WHAT WE SHOULD HAVE DONE**:
  - Benchmark wav2vec2-nepali vs Whisper vs xls-r on Nepali test set
  - Report Word Error Rate (WER) for each
  - Analyze failure modes (proper nouns, numbers, etc.)
  - Compare inference speed and memory requirements

### Q11: You use autotune=None in SpectralClusterer - that's literally disabled autotuning. Why claim it's an improvement?

**A:**
- **CODE BUG ACKNOWLEDGED**: autotune=None causes TypeError in newer versions
- **LAZY FIX**: Changed to autotune=False without understanding impact
- **QUESTIONS NOT ANSWERED**:
  - What does spectral clustering autotuning do?
  - How does autotune=False vs autotune=True compare on DER?
  - Should we use autotuning for better results?
- **WHAT WE SHOULD HAVE DONE**:
  - Understand what autotune parameter does
  - Benchmark: autotune=None (broken), autotune=False, autotune=True
  - Choose based on DER/speed trade-off

---

## 5. DATA & TESTING

### Q12: What datasets did you evaluate on?

**A:**
- **HONEST ANSWER**: None. We have no ground truth annotations.
- **MISSING**:
  - No test set with reference speaker labels (RTTM format)
  - No comparison against baseline diarization systems
  - No per-speaker performance analysis
  - Example file "conv_0.wav" mentioned but not included
- **WHAT WE SHOULD HAVE DONE**:
  - Use MUCS (Multilingual and Code-switching) dataset
  - Use OpenSLR Nepali corpora if available
  - Create annotations using professional annotators
  - Establish ground truth for evaluation
  - Report inter-annotator agreement (IAA)

### Q13: How does system perform on Nepali speech with code-switching (English + Nepali)?

**A:**
- **HONEST ANSWER**: Unknown. Not tested.
- **REAL PROBLEM**: 
  - Modern Nepali conversation often mixes English/Hindi/Nepali
  - Code-switching affects both ASR and speaker segmentation
  - wav2vec2-nepali trained on pure Nepali probably fails on code-switched speech
- **WHAT WE SHOULD HAVE DONE**:
  - Test on code-switched Nepali datasets
  - Measure WER degradation
  - Document performance on pure vs mixed-language speech

---

## 6. REPRODUCIBILITY & DOCUMENTATION

### Q14: Can someone else reproduce your results?

**A:**
- **MISSING DOCUMENTATION**:
  - No README.md with setup instructions
  - No explanation of what wav2vec2-nepali model to download
  - No dataset download links
  - No step-by-step evaluation instructions
  - No results file to show what "success" looks like
- **MISSING ARTIFACTS**:
  - No trained model weights (or pointer to them)
  - No test audio files
  - No ground truth annotations
  - No evaluation scripts that actually work
- **WHAT WE SHOULD HAVE DONE**:
  ```markdown
  # Reproducibility Guide
  
  ## Setup
  1. Clone repo
  2. Create venv
  3. pip install -r requirements.txt
  4. Download wav2vec2-nepali model from [URL]
  5. Download test dataset from [URL]
  
  ## Evaluation
  python evaluate.py --test_audio data/test_audio.wav \
                    --ground_truth data/test_audio.rttm
  
  ## Expected Results
  DER: 12.5% ± 2.1%
  WER: 18.3% ± 3.4%
  ```

### Q15: Your notebook is named "Copy_of_wav2vec2_large_xls_r_300m_nepali_openslr.ipynb" - is this code even yours?

**A:**
- **RED FLAG**: File name suggests copied code
- **QUESTIONS**:
  - Where's the original notebook?
  - What modifications did you make?
  - What's your original contribution in the notebook?
  - Is this just experiment code or part of the final system?
- **WHAT WE SHOULD HAVE DONE**:
  - Proper documentation of code origin
  - Clear separation of experiments vs production code
  - Git history showing your changes
  - Comment blocks explaining your contributions

---

## 7. PERFORMANCE & SCALABILITY

### Q16: What's the inference time for a 1-hour conversation?

**A:**
- **HONEST ANSWER**: Never measured. Likely:
  - Embedding extraction: ~10 minutes (sequentially)
  - Spectral clustering: 1-2 seconds
  - ASR: ~30-60 minutes (depending on GPU)
  - **Total**: >40 minutes for 1 hour audio (2.4x real-time)
- **PRODUCT VIABILITY**: 
  - Too slow for real-time/interactive applications
  - May not scale to large transcription services
- **WHAT WE SHOULD HAVE DONE**:
  - Benchmark inference time vs audio duration
  - Profile where bottlenecks are (likely ASR)
  - Propose parallelization strategies
  - Document latency requirements vs actual performance

### Q17: Memory requirements for long audio files?

**A:**
- **UNKNOWN**:
  - Peak memory usage never measured
  - 2x librosa.load() may cause OOM on large files
  - 4800+ embeddings for 1-hour audio unoptimized
  - Model weights in memory: wav2vec2-nepali (~400MB) + resemblyzer (~100MB)
- **WHAT WE SHOULD HAVE DONE**:
  - Memory profiling for various audio lengths
  - Streaming/chunking strategy for long files
  - Disk space requirements for temporary files
  - GPU memory requirements if CUDA enabled

---

## 8. FRONTEND & USABILITY

### Q18: Your frontend has hardcoded localhost:8000 - not production ready. Why?

**A:**
- **CODE**:
  ```javascript
  const res = await fetch('http://localhost:8000/transcribe', {
  ```
- **ISSUES**:
  - Fixed to localhost - won't work in production
  - No CORS handling (relies on backend allowing all origins)
  - No timeout on upload (could hang forever)
  - No upload progress indicator
  - No validation of audio before sending
  - No WEB ONLY: Can't handle > 2GB files in browser
- **WHAT WE SHOULD HAVE DONE**:
  ```javascript
  // Dynamic backend URL
  const backendUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  
  // File size validation
  if (file.size > 500*1024*1024) throw new Error('Max 500MB');
  
  // Progress tracking
  xhr.upload.addEventListener('progress', e => {
    if (e.lengthComputable) {
      progress = (e.loaded / e.total) * 100;
    }
  });
  
  // Timeout
  const timeout = setTimeout(() => abort(), 300000); // 5 min timeout
  ```

### Q19: No user feedback on long-running operations - what if processing takes 30 minutes?

**A:**
- **CURRENT UX**:
  - "Uploading & processing..." message for 30 minutes
  - No progress indication
  - User doesn't know if system is hung or working
  - No ability to cancel
- **WHAT WE SHOULD HAVE DONE**:
  - Websocket/Server-Sent Events for real-time updates
  - Progress callbacks from backend
  - Estimated time remaining
  - Ability to cancel mid-processing
  - Async job queue with status tracking

---

## 9. SECURITY & ROBUSTNESS

### Q20: Your API allows unlimited file uploads with CORS allow all - security nightmare?

**A:**
- **VULNERABILITIES**:
  1. **No file size limit**
     - User can upload 100GB file
     - DoS attack via disk exhaustion
  2. **CORS allow all**
     ```python
     allow_origins=["*"],  # DANGEROUS
     ```
     - Any website can abuse your API
  3. **No authentication**
     - No rate limiting
     - No user tracking
     - Malicious actor can spam requests
  4. **Temp file cleanup**
     ```python
     try:
         os.unlink(tmp_file.name)  # May fail silently
     except OSError:
         pass  # Disk space leak!
     ```
     - Temp files may accumulate if there's an error
- **WHAT WE SHOULD HAVE DONE**:
  ```python
  from fastapi import UploadFile
  
  MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
  
  @app.post("/transcribe")
  async def transcribe(upload_file: UploadFile = File(...)):
      if upload_file.size and upload_file.size > MAX_FILE_SIZE:
          raise HTTPException(status_code=413, detail="File too large")
      
      try:
          # ... process ...
      finally:
          tmp_file.close()
          os.unlink(tmp_file.name)  # Force clean up
  ```

### Q21: No input validation - what if someone sends corrupted audio?

**A:**
- **CURRENT BEHAVIOR**:
  - Extension check only (insufficient)
  - No validation of actual file format
  - librosa.load() will crash on invalid audio
  - Error not caught - returns 500 error to user
- **WHAT WE SHOULD HAVE DONE**:
  ```python
  try:
      result = run_asr_diarization(tmp_file.name)
  except Exception as e:
      logger.error(f"Processing failed: {e}")
      raise HTTPException(status_code=422, detail="Invalid audio file")
  ```

---

## 10. TECHNICAL DEBT & CODE QUALITY

### Q22: The code has hardcoded model name and parameters - why not configurable?

**A:**
- **HARDCODED VALUES**:
  ```python
  MODEL_NAME = "lishantKarki/wav2vec2-large-xls-r-300m-nepali-openslr2
"  # Hardcoded
  SAMPLE_RATE = 16000  # Hardcoded
  WINDOW = 1.5  # Hardcoded
  STEP = 0.75  # Hardcoded
  DEVICE = 0 if torch.cuda.is_available() else -1  # Magic number
  ```
- **IMPLICATIONS**:
  - Can't switch models without code change
  - Can't test different configurations
  - No environment variable configuration
  - Production deploys require code changes
- **WHAT WE SHOULD HAVE DONE**:
  ```python
  import os
  from dataclasses import dataclass
  
  @dataclass
  class Config:
      MODEL_NAME: str = os.getenv("ASR_MODEL", "anish-shilpakar/wav2vec2-nepali")
      SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))
      WINDOW: float = float(os.getenv("WINDOW", "1.5"))
      STEP: float = float(os.getenv("STEP", "0.75"))
      DEVICE: int = int(os.getenv("DEVICE", "-1"))
  ```

### Q23: No logging anywhere - how do you debug production issues?

**A:**
- **DEBUGGING NIGHTMARE**:
  - print() statements used instead of logging
  - No error tracking
  - No performance metrics
  - No request logging
  - Impossible to diagnose production failures
- **WHAT WE SHOULD HAVE DONE**:
  ```python
  import logging
  
  logger = logging.getLogger(__name__)
  
  @app.post("/transcribe")
  async def transcribe(...):
      logger.info(f"Processing audio: {filename}")
      try:
          result = run_asr_diarization(...)
          logger.info(f"Completed: DER={...}")
          return result
      except Exception as e:
          logger.error(f"Failed: {e}", exc_info=True)
          raise
  ```

---

## 11. METHODOLOGY & SCIENTIFIC RIGOR

### Q24: Where's your experiment design and hypothesis?

**A:**
- **MISSING SCIENTIFIC METHOD**:
  - No clear hypothesis ("System will achieve X% DER")
  - No baseline for comparison
  - No control conditions
  - No statistical significance testing
  - No confidence intervals
- **WHAT WE SHOULD HAVE DONE**:
  ```
  HYPOTHESIS: A wav2vec2-nepali + spectral clustering system 
              achieves < 15% DER on conversational Nepali
  
  BASELINE: TDNN-based diarization achieves 18% DER
  
  EXPERIMENT:
  - Test set: 100 Nepali conversations (10 hours total)
  - Metric: DER with 250ms collar
  - Statistical test: Two-tailed t-test (α=0.05)
  
  EXPECTED RESULTS:
  - Mean DER: 14.2% ± 2.3% (95% CI)
  - Improvement vs baseline: 20% relative reduction
  ```

### Q25: No ablation studies - how do you know each component is necessary?

**A:**
- **MISSING ANALYSIS**:
  - What if we use different speaker embedding model? (impact unknown)
  - What if we use different clustering algorithm? (impact unknown)
  - What if we use different ASR model? (impact unknown)
  - What's the contribution of each component to final DER?
- **WHAT WE SHOULD HAVE DONE**:
  ```
  ABLATION STUDY:
  
  System A: wav2vec2-nepali + resemblyzer + spectral clustering
            DER: 14.2%
  
  System B: wav2vec2-nepali + ECAPA-TDNN + spectral clustering
            DER: 12.8% (Better!)
  
  System C: Whisper + resemblyzer + spectral clustering
            DER: 11.5% (Even better!)
  
  System D: wav2vec2-nepali + resemblyzer + K-means clustering
            DER: 16.1% (Worse)
  
  CONCLUSION: ECAPA-TDNN is critical; spectral clustering better than K-means
  ```

---

## 12. BUSINESS & IMPACT

### Q26: What problem does this solve for Nepali speakers?

**A:**
- **HONEST ANSWER**: We don't have a clear answer.
- **MISSING ANALYSIS**:
  - Who are the users? (Journalists? Researchers? Companies?)
  - What are their pain points?
  - Is this 80% accurate enough for their use case?
  - What's the business model? (Free? Paid API?)
  - How does it compare to hiring human transcribers?
- **WHAT WE SHOULD HAVE DONE**:
  ```
  USER PERSONAS:
  1. Nepali News Organizations
     - Use case: Transcribe interviews for articles
     - Accuracy required: 95%+ (for publishing)
     - Pain point: Manual transcription takes 8+ hours per hour of audio
     - Our solution accuracy: Unknown (likely <85%)
     - VERDICT: Insufficient for this use case
  
  2. Researchers
     - Use case: Analyze Nepali speech corpora
     - Accuracy required: 80%+ (for research)
     - Our solution accuracy: Unknown
     - VERDICT: Might be useful, needs benchmarking
  
  3. Content Creators
     - Use case: Auto-caption YouTube videos
     - Accuracy required: 85%+
     - Our solution accuracy: Unknown
     - VERDICT: Unknown
  ```

### Q27: What's the actual accuracy of your system end-to-end?

**A:**
- **COMPLETELY UNKNOWN**:
  - We never measured WER (Word Error Rate)
  - We never measured DER (Diarization Error Rate)
  - We never measured combined performance
  - Example errors:
    - wav2vec2-nepali WER: Unknown (maybe 15-30%?)
    - Speaker confusion errors: Unknown
    - Combined error rate: Unknown
  - **NO BENCHMARKS ANYWHERE**
- **WHAT WE SHOULD HAVE DONE**:
  ```
  EVALUATION RESULTS:
  
  Component 1: ASR
  - WER on test set: 18.3% ± 2.1%
  - Common errors: Proper nouns, numbers, code-switched words
  
  Component 2: Diarization
  - DER on test set: 12.5% ± 1.8%
  - Breakdown:
    * Confusion: 8.2%
    * Missed Detection: 3.1%
    * False Alarm: 1.2%
  
  Component 3: End-to-End
  - Segment-level accuracy: 82.1%
  - Minimum acceptable WER: 20%
  - Recommended use cases: Research only
  ```

---

## 13. COMPARISON WITH EXISTING SOLUTIONS

### Q28: How does this compare to Google Speech-to-Text + Google Cloud Speech Analytics?

**A:**
- **HONEST ANSWER**: 
  - Google probably 40-50% more accurate
  - Google handles multiple languages
  - Google is production-grade
  - Your system: research prototype
- **MISSING COMPETITIVE ANALYSIS**:
  - No comparison with commercial solutions
  - No comparison with Whisper (OpenAI)
  - No comparison with cloud providers
  - No cost-benefit analysis
  - No discussion of why someone would use yours
- **WHAT WE SHOULD HAVE DONE**:
  ```
  COMPARISON TABLE:
  
  System              | Accuracy | Speed    | Cost      | Nepali Support
  -------------------|----------|----------|-----------|----------------
  Your System         | Unknown  | 2.4x RT  | Free      | Yes (untested)
  Google Cloud STT    | ~95%     | Real-time| $4/hr     | Yes
  Whisper (OpenAI)    | ~85-95%  | 1-2x RT  | Free/API  | Yes
  AWS Transcribe      | ~90%     | Real-time| $0.24/min | No explicit
  
  CONCLUSION: Your system is unclear on all metrics.
  ```

---

## SUMMARY: KEY SHORTCOMINGS

| Category | Issue | Severity |
|----------|-------|----------|
| **Research** | No novel contribution, purely engineering | CRITICAL |
| **Evaluation** | No benchmarks, no test sets, no metrics | CRITICAL |
| **Reproducibility** | Cannot reproduce results (don't exist) | CRITICAL |
| **Documentation** | No README, no setup guide, minimal comments | HIGH |
| **Validation** | No A/B testing, no ablation studies | HIGH |
| **Robustness** | No error handling, brittle code | HIGH |
| **Performance** | Unknown speed, memory, accuracy | HIGH |
| **Security** | Open API, no rate limiting, no auth | MEDIUM |
| **Scalability** | 2x real-time latency, high memory usage | MEDIUM |
| **Code Quality** | Hardcoded values, print() instead of logging | MEDIUM |

---

## WHAT YOU SHOULD PRESENT

Rather than pretending everything is perfect, a strong presentation would:

1. **Be honest about status**: "This is a functional prototype, not production-ready"
2. **Show what works**: Demo on a test audio file
3. **Quantify unknowns**: "WER likely 15-25% based on model specs"
4. **Roadmap improvements**:
   - Evaluation on standard datasets
   - Comparison with baselines
   - Fine-tuning on Nepali-specific data
   - Production hardening
5. **Acknowledge limitations**: Code-switching, background noise, etc.
6. **Next steps**: 
   - Create ground truth dataset
   - Run rigorous evaluation
   - Consider alternative models
   - Optimize for production

---

## FINAL VERDICT

**The system works as a proof-of-concept but lacks**:
- Evidence it actually works well (no metrics)
- Production-ready code
- Research novelty
- Reproducible results
- Clear value proposition

**To make this compelling, you must**: 
1. Evaluate on real data
2. Benchmark against baselines
3. Document limitations honestly
4. Propose concrete improvements
