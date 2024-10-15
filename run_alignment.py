from datetime import datetime

import numpy as np

from online_dtw import OLTW
from stream import AudioStream
from utils import get_score_features, run_evaluation, visualize_warping_path


def run_score_following(score_file, performance_file):
    reference_features = get_score_features(score_file)

    with AudioStream(file_path=performance_file) as stream:
        oltw = OLTW(reference_features, stream.queue)
        for current_position in oltw.run():
            print(
                f"[{datetime.now().strftime('%H:%M:%S.%f')}] Current position: {current_position}"
            )

        print(
            f"[Feature extraction] Mean elapsed time: {np.mean(stream.elapsed_times):.5f}, median: {np.median(stream.elapsed_times):.5f}"
        )
        print(
            f"[DTW] Mean elapsed time: {np.mean(oltw.elapsed_times):.5f}, median: {np.median(oltw.elapsed_times):.5f}"
        )

    return oltw, oltw.warping_path


if __name__ == "__main__":
    # Run score following
    score_file = "./resources/ex_score.mid"
    performance_file = "./resources/ex_perf.wav"

    oltw, warping_path = run_score_following(score_file, performance_file)

    # Evaluation
    score_ann = "./resources/ex_score_annotations.txt"
    perf_ann = "./resources/ex_perf_annotations.txt"
    results = run_evaluation(warping_path, score_ann, perf_ann)

    visualize_warping_path(oltw, score_ann, perf_ann)
