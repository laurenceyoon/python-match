import json

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretty_midi
import scipy

from config import FRAME_RATE, TOLERANCES


def process_chroma(y, sr, hop_length, n_fft) -> np.ndarray:
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        hop_length=hop_length,
        n_fft=n_fft,
        center=False,
    )
    return chroma.T  # (time, n_chroma)


def process_chroma_decay(y, sr, hop_length, n_fft) -> np.ndarray:
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        hop_length=hop_length,
        n_fft=n_fft,
        center=False,
    )
    diff = np.diff(chroma, axis=0, prepend=chroma[0:1, :])
    half_wave_rectification = np.maximum(diff, 0)
    return half_wave_rectification.T  # (time, n_chroma)


def get_score_features(score_path: str, frame_rate: int = FRAME_RATE) -> np.ndarray:
    mid = pretty_midi.PrettyMIDI(score_path)
    chroma = mid.get_chroma(fs=frame_rate)
    chroma_norm = librosa.util.normalize(chroma)
    return chroma_norm.T  # (time, n_chroma)


def transfer_positions(wp, ref_anns):
    x, y = wp[0], wp[1]
    predicted_targets = [y[np.where(x >= r)[0][0]] for r in ref_anns]
    return predicted_targets


def run_evaluation(
    wp, ref_ann, target_ann, frame_rate=FRAME_RATE, tolerance=TOLERANCES
):
    ref_annots = np.rint(
        pd.read_csv(filepath_or_buffer=ref_ann, delimiter="\t", header=None)[0]
        * frame_rate
    )
    target_annots = np.rint(
        pd.read_csv(filepath_or_buffer=target_ann, delimiter="\t", header=None)[0]
        * frame_rate
    )

    target_annots_predicted = transfer_positions(wp, ref_annots)
    errors_in_delay = (
        (target_annots - target_annots_predicted) / frame_rate * 1000
    )  # in milliseconds

    absolute_errors_in_delay = np.abs(errors_in_delay)
    filtered_abs_errors_in_delay = absolute_errors_in_delay[
        absolute_errors_in_delay <= tolerance[-1]
    ]

    results = {
        "mean": float(f"{np.mean(filtered_abs_errors_in_delay):.4f}"),
        "median": float(f"{np.median(filtered_abs_errors_in_delay):.4f}"),
        "std": float(f"{np.std(filtered_abs_errors_in_delay):.4f}"),
        "skewness": float(f"{scipy.stats.skew(filtered_abs_errors_in_delay):.4f}"),
        "kurtosis": float(f"{scipy.stats.kurtosis(filtered_abs_errors_in_delay):.4f}"),
    }
    for tau in tolerance:
        results[f"{tau}ms"] = float(f"{np.mean(absolute_errors_in_delay <= tau):.4f}")
    print(f"Evaluation Results: {json.dumps(results, indent=4)}")
    return results


def visualize_warping_path(oltw, score_ann, perf_ann, show=True):
    # xmin = 300  # performance range
    # xmax = 900
    # ymin = 300  # score range
    # ymax = 900

    # # Compute the indices of the rows and columns to include
    # x_indices = range(xmin, xmax + 1)
    # y_indices = range(ymin, ymax + 1)

    dist = scipy.spatial.distance.cdist(
        oltw.reference_features,
        oltw.input_features,
        metric=oltw.local_cost_fun,
    )  # [d, wy]
    # dist = dist[y_indices, :][:, x_indices]

    plt.figure(figsize=(10, 10))
    plt.imshow(
        dist,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    plt.title("Accumulated distance matrix with warping path & ground-truth labels")
    plt.xlabel("Performance Features in Time (s)")
    plt.ylabel("Score Features in Time (s)")

    max_x_time = dist.shape[1] / FRAME_RATE
    max_y_time = dist.shape[0] / FRAME_RATE
    x_ticks = np.arange(0, max_x_time, 10)
    y_ticks = np.arange(0, max_y_time, 10)
    x_labels = [f"{x:.0f}" for x in x_ticks]
    y_labels = [f"{y:.0f}" for y in y_ticks]

    plt.xticks(ticks=x_ticks * FRAME_RATE, labels=x_labels)
    plt.yticks(ticks=y_ticks * FRAME_RATE, labels=y_labels)

    # plot warping path
    warping_path = oltw.warping_path
    ref_paths, perf_paths = warping_path[0], warping_path[1]
    # mask_ref = (ymin <= ref_paths) & (ref_paths <= ymax)
    # mask_perf = (xmin <= perf_paths) & (perf_paths <= xmax)
    # ref_paths = ref_paths[mask_ref]
    # perf_paths = perf_paths[mask_perf]
    for n in range(len(ref_paths)):
        plt.plot(
            perf_paths[n],
            ref_paths[n],
            ".",
            color="purple",
            alpha=0.5,
            markersize=3,
            label="Warping Path" if n == 0 else "",
        )
    # cropped_history = [
    #     (perf, ref)
    #     for (perf, ref) in zip(perf_paths, ref_paths)
    #     if xmin <= perf <= xmax and ymin <= ref <= ymax
    # ]
    # for perf, ref in cropped_history:
    #     plt.scatter(perf, ref, color="purple")

    # plot ground-truth labels
    perf_annots = (
        pd.read_csv(filepath_or_buffer=perf_ann, delimiter="\t", header=None)[0]
        * FRAME_RATE
    )
    # mask_perf = (xmin <= perf_annots) & (perf_annots <= xmax)
    # perf_annots = perf_annots[mask_perf]
    ref_annots = (
        pd.read_csv(filepath_or_buffer=score_ann, delimiter="\t", header=None)[0]
        * FRAME_RATE
    )
    # mask_ref = (ymin <= ref_annots) & (ref_annots <= ymax)
    # ref_annots = ref_annots[mask_ref]
    for i, (perf, ref) in enumerate(zip(perf_annots, ref_annots)):
        plt.plot(
            perf,
            ref,
            "x",
            color="r",
            alpha=1,
            markersize=5,
            label="Ground Truth" if i == 0 else "",
        )

    plt.legend()
    plt.savefig("./warping_path_result.png")

    if show:
        plt.show()
