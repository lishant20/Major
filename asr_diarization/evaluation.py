import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pyannote.metrics.diarization import DiarizationErrorRate


def compute_der(reference_ann, hypothesis_ann):
    metric = DiarizationErrorRate(collar=0.25, skip_overlap=True)

    der = metric(reference_ann, hypothesis_ann)
    components = metric.compute_components(reference_ann, hypothesis_ann)

    return {
        "DER": der,
        "Confusion": components["confusion"],
        "Missed Detection": components["missed detection"],
        "False Alarm": components["false alarm"]
    }


def plot_der(results):
    labels = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color='skyblue')

    plt.title("Diarization Error Rate Breakdown")
    plt.ylabel("Error Rate")

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center')

    plt.tight_layout()
    plt.savefig("der_plot.png")
    print("Graph saved as der_plot.png")