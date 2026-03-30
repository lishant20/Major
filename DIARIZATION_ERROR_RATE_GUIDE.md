# Diarization Error Rate (DER) Calculation Guide

## Overview

The **Diarization Error Rate (DER)** is the standard metric for evaluating speaker diarization systems. It measures how well a diarization system assigns speaker labels to audio segments compared to a ground truth reference annotation.

## Formula

```
DER = (Confusion + Missed Detection + False Alarm) / Total Duration
```

Where:
- **Total Duration**: Total length of the audio file (in seconds or frames)

## Components of DER

### 1. **Confusion (Speaker Confusion)**
- Occurs when the system assigns the wrong speaker label to a speech segment
- Counts frames where the predicted speaker differs from the reference speaker
- Both speakers must be speaking (not silence)

### 2. **Missed Detection**
- Occurs when the system fails to detect speech that was actually present
- Counts frames where speech should be detected but wasn't
- The reference has a speaker, but the hypothesis doesn't

### 3. **False Alarm**
- Occurs when the system detects speech that wasn't actually present
- Counts frames where the system incorrectly identifies silence as speech
- The hypothesis has a speaker, but the reference doesn't

## Implementation in Your Project

### Using the Evaluation Module

```python
from pyannote.core import Annotation, Segment
from asr_diarization.evaluation import compute_der, plot_der

# Create reference annotation (ground truth)
reference_ann = Annotation()
reference_ann[Segment(0, 5)] = "speaker_1"
reference_ann[Segment(5, 10)] = "speaker_2"
reference_ann[Segment(10, 15)] = "speaker_1"

# Create hypothesis annotation (your model's output)
hypothesis_ann = Annotation()
hypothesis_ann[Segment(0, 5)] = "speaker_1"
hypothesis_ann[Segment(5, 10)] = "speaker_2"
hypothesis_ann[Segment(10, 15)] = "speaker_2"  # Error here

# Compute DER
results = compute_der(reference_ann, hypothesis_ann)
print(results)
# Output:
# {
#     'DER': 0.2,
#     'Confusion': 0.15,
#     'Missed Detection': 0.0,
#     'False Alarm': 0.05
# }

# Plot the breakdown
plot_der(results)
```

## Step-by-Step Calculation Example

### Scenario:
- Audio file: 20 seconds
- Reference: Speaker A (0-10s), Speaker B (10-20s)
- Hypothesis: Speaker A (0-8s), Speaker B (8-20s)

### Analysis:
- **Frames 0-8s**: Correct (both say Speaker A) ✓
- **Frames 8-10s**: Confusion (reference=A, hypothesis=B) ✗
- **Frames 10-20s**: Correct (both say Speaker B) ✓

### Calculation:
```
Confusion = 2 seconds / 20 seconds = 0.10 (10%)
Missed Detection = 0 seconds / 20 seconds = 0.0
False Alarm = 0 seconds / 20 seconds = 0.0

DER = (0.10 + 0.0 + 0.0) = 0.10 (10% error rate)
```

## Key Points

1. **Lower DER is Better**: 0% DER = Perfect diarization, 100% DER = Completely wrong
2. **Threshold for Frame Comparison**: Pyannote typically uses 25ms frames
3. **Speaker ID Alignment**: The exact speaker labels don't matter; only correct/incorrect assignment matters
4. **Silence Handling**: Non-speech frames are not included in the calculation

## Using Pyannote Directly

Your project uses **pyannote.metrics.diarization.DiarizationErrorRate**, which:
- Handles temporal alignment automatically
- Supports overlapping speech
- Provides component breakdown
- Is the industry standard for diarization evaluation

## Improving Your DER

To reduce DER:
1. **Reduce Confusion**: Improve speaker embedding quality and clustering accuracy
2. **Reduce Missed Detection**: Lower VAD thresholds, improve speech detection
3. **Reduce False Alarm**: Increase VAD thresholds, filter background noise better

## Reference

- **Pyannote Documentation**: https://github.com/pyannote/pyannote-metrics
- **Standard Metric**: Used in DIARIZATION track of ICASSP, Interspeech challenges
- **Your Implementation**: See `asr_diarization/evaluation.py`
