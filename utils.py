import json
import threading
import time
from queue import Queue
from typing import Optional

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretty_midi
import pyaudio
import scipy

CHANNELS = 1
TOLERANCES = [50, 100, 200, 300, 500, 1000]
SAMPLE_RATE = 44100
FRAME_RATE = 30
HOP_LENGTH = SAMPLE_RATE // FRAME_RATE
N_FFT = 2 * HOP_LENGTH
FEATURE_TYPE = "chroma"  # option: ["chroma", "chroma_decay"]
FRAME_PER_SEG = 1


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


class AudioStream(threading.Thread):
    """
    A class to process audio stream in real-time

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio stream
    hop_length : int
        Hop length of the audio stream
    queue : Queue
        Queue to store the processed audio
    features : List[str]
        List of features to be processed
    chunk_size : int
        Size of the audio chunk
    """

    def __init__(
        self,
        queue: Queue,
        feature_type: str = FEATURE_TYPE,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        chunk_size: int = FRAME_PER_SEG,
    ):
        threading.Thread.__init__(self)
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.queue = queue
        self.chunk_size = chunk_size * self.hop_length
        self.format = pyaudio.paFloat32
        self.audio_interface = pyaudio.PyAudio()
        self.audio_stream: Optional[pyaudio.Stream] = None
        self.last_chunk = None
        self.init_time = None
        self.listen = False

    def _process_frame(self, data, frame_count, time_info, status_flag):
        target_audio = np.frombuffer(data, dtype=np.float32)  # initial y
        self._process_feature(target_audio)

        return (data, pyaudio.paContinue)

    def _process_feature(self, target_audio):
        if self.last_chunk is None:  # add zero padding at the first block
            target_audio = np.concatenate(
                (np.zeros(self.hop_length, dtype=np.float32), target_audio)
            )
        else:
            # add last chunk for continuity between chunks (overlap)
            target_audio = np.concatenate((self.last_chunk, target_audio))

        features_array = None
        if self.feature_type == "chroma":
            features_array = process_chroma(
                target_audio, self.sample_rate, self.hop_length, 2 * self.hop_length
            )
        elif self.feature_type == "chroma_decay":
            features_array = process_chroma_decay(
                target_audio, self.sample_rate, self.hop_length, 2 * self.hop_length
            )

        self.queue.put(features_array)
        self.last_chunk = target_audio[-self.hop_length :]

    @property
    def current_time(self):
        """
        Get current time since starting to listen
        """
        return time.time() - self.init_time if self.init_time else None

    def start_listening(self):
        self.audio_stream.start_stream()
        print("* Start listening to audio stream....")
        self.listen = True
        self.init_time = self.audio_stream.get_time()

    def stop_listening(self):
        print("* Stop listening to audio stream....")
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.audio_interface.terminate()
        self.listen = False

    def run(self):
        self.audio_stream = self.audio_interface.open(
            format=self.format,
            channels=CHANNELS,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._process_frame,
        )
        self.start_listening()

    def stop(self):
        self.stop_listening()


class MockAudioStream(AudioStream):
    """
    A class to process audio stream from a file

    Parameters
    ----------
    file_path : str
        Path to the audio file
    """

    def __init__(
        self,
        queue: Queue,
        file_path: str = "",
        feature_type: str = FEATURE_TYPE,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        chunk_size: int = FRAME_PER_SEG,
    ):
        super().__init__(
            sample_rate=sample_rate,
            hop_length=hop_length,
            queue=queue,
            feature_type=feature_type,
            chunk_size=chunk_size,
        )
        self.file_path = file_path

    def start_listening(self):
        self.listen = True
        self.init_time = time.time()

    def stop_listening(self):
        self.listen = False

    def mock_stream(self):
        duration = int(librosa.get_duration(path=self.file_path))
        audio_y, _ = librosa.load(self.file_path, sr=self.sample_rate)
        padded_audio = np.concatenate(  # zero padding at the end
            (audio_y, np.zeros(duration * 2 * self.sample_rate, dtype=np.float32))
        )
        trimmed_audio = padded_audio[  # trim to multiple of chunk_size
            : len(padded_audio) - (len(padded_audio) % self.chunk_size)
        ]
        self.start_listening()
        while self.listen and trimmed_audio.any():
            target_audio = trimmed_audio[: self.chunk_size]
            self._process_feature(target_audio)
            trimmed_audio = trimmed_audio[self.chunk_size :]

    def run(self):
        if self.queue.not_empty:
            self.queue.queue.clear()
        print(f"* [Mocking] Loading existing audio file({self.file_path})....")
        self.mock_stream()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


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
    dist = scipy.spatial.distance.cdist(
        oltw.reference_features,
        oltw.input_features,
        metric=oltw.local_cost_fun,
    )  # [d, wy]
    plt.figure(figsize=(10, 10))
    plt.imshow(dist, aspect="auto", origin="lower", interpolation="nearest")
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

    # plot ground-truth labels
    ref_annots = pd.read_csv(filepath_or_buffer=score_ann, delimiter="\t", header=None)[
        0
    ]
    target_annots = pd.read_csv(
        filepath_or_buffer=perf_ann, delimiter="\t", header=None
    )[0]
    for i, (ref, target) in enumerate(zip(ref_annots, target_annots)):
        plt.plot(
            target * FRAME_RATE,
            ref * FRAME_RATE,
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
