import threading
import time
from queue import Queue
from typing import Optional

import librosa
import numpy as np
import pyaudio

CHANNELS = 1


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
        feature_type: str,
        sample_rate: int,
        hop_length: int,
        chunk_size: int,
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
        feature_type: str,
        sample_rate: int,
        hop_length: int,
        chunk_size: int,
        file_path: str = "",
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
