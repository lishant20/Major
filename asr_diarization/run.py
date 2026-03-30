from diarization import diarize
from evaluation import compute_der, plot_der
from pyannote.core import Annotation, Segment

# 🔊 your audio file
audio_path = "/home/anmolkhatiwada/Desktop/stuffs/test.mp3"

# ▶️ Run diarization
hypothesis = diarize(audio_path)

# ✅ Ground truth (REFERENCE)
reference = Annotation()

reference[Segment(0.00, 1.50)] = "SPEAKER_1"
reference[Segment(1.50, 3.00)] = "SPEAKER_1"
reference[Segment(3.00, 4.50)] = "SPEAKER_1"
reference[Segment(4.50, 6.00)] = "SPEAKER_1"
reference[Segment(6.00, 7.50)] = "SPEAKER_1"
reference[Segment(7.50, 9.00)] = "SPEAKER_1"
reference[Segment(9.00, 10.50)] = "SPEAKER_0"
reference[Segment(10.50, 12.00)] = "SPEAKER_0"
reference[Segment(12.00, 13.50)] = "SPEAKER_0"
reference[Segment(13.50, 15.00)] = "SPEAKER_0"
reference[Segment(15.00, 16.50)] = "SPEAKER_0"
reference[Segment(16.50, 18.00)] = "SPEAKER_0"
reference[Segment(18.00, 19.50)] = "SPEAKER_0"
reference[Segment(19.50, 21.00)] = "SPEAKER_0"
reference[Segment(21.00, 22.50)] = "SPEAKER_0"
reference[Segment(22.50, 24.00)] = "SPEAKER_0"
reference[Segment(24.00, 25.50)] = "SPEAKER_1"
reference[Segment(25.50, 27.00)] = "SPEAKER_1"  
reference[Segment(27.00, 28.50)] = "SPEAKER_1"
reference[Segment(28.50, 30.00)] = "SPEAKER_1"

# 📊 Compute DER
results = compute_der(reference, hypothesis)

print("\nDER Results:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# 📈 Plot graph
plot_der(results)