"""
Módulo de Gravação de Vídeo + Áudio
===================================
Grava frames de vídeo processados + áudio (com voice changer) num ficheiro MP4.
Usa FFmpeg para multiplexar vídeo e áudio no final.
"""

import os
import sys
import subprocess
import threading
import time
import tempfile
import wave
import queue
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
import pyaudio


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
AUDIO_RATE = 44100
AUDIO_CHUNK = 1024


class Recorder:
    """
    Gravador de vídeo + áudio.
    
    Uso:
        recorder = Recorder("output.mp4", fps=30, width=640, height=480)
        recorder.start()
        
        while capturando:
            recorder.write_frame(frame_bgr)
        
        recorder.stop()
    """

    def __init__(
        self,
        output_path: str,
        fps: float = 30.0,
        width: int = 640,
        height: int = 480,
        input_device_index: int | None = None,
    ):
        self.output_path = Path(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        self.input_device_index = input_device_index

        # Estado
        self._recording = False
        self._video_writer: cv2.VideoWriter | None = None
        self._audio: pyaudio.PyAudio | None = None
        self._audio_stream = None
        self._audio_thread: threading.Thread | None = None
        self._audio_frames: list[bytes] = []
        self._video_ready = threading.Event()

        # Ficheiros temporários
        self._temp_dir = tempfile.mkdtemp(prefix="recorder_")
        self._temp_video = os.path.join(self._temp_dir, "video_temp.avi")
        self._temp_audio = os.path.join(self._temp_dir, "audio_temp.wav")

        # Callback de áudio externo (voice changer)
        self._audio_processor: Callable | None = None

    # ------------------------------------------------------------------
    # API Pública
    # ------------------------------------------------------------------
    def set_audio_processor(self, processor: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Define um processador de áudio.
        processor recebe um numpy array int16 e deve retornar um numpy array int16.
        """
        self._audio_processor = processor

    def start(self) -> bool:
        """Inicia a gravação."""
        if self._recording:
            return False

        try:
            # Inicializar VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self._video_writer = cv2.VideoWriter(
                self._temp_video, fourcc, self.fps, (self.width, self.height)
            )
            if not self._video_writer.isOpened():
                raise RuntimeError("Falha ao abrir VideoWriter")

            # Inicializar PyAudio
            self._audio = pyaudio.PyAudio()

            # Iniciar captura de áudio
            self._audio_frames = []
            self._video_ready.clear()
            self._recording = True
            self._audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
            self._audio_thread.start()

            return True

        except Exception as e:
            print(f"[Recorder] Erro ao iniciar: {e}")
            self._cleanup_partial()
            return False

    def write_frame(self, frame: np.ndarray) -> None:
        """Escreve um frame BGR no vídeo."""
        if self._recording and self._video_writer:
            # Garantir que o frame tem o tamanho correto
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            self._video_writer.write(frame)
            self._video_ready.set()

    def stop(self) -> bool:
        """Para a gravação e finaliza o ficheiro."""
        if not self._recording:
            return False

        self._recording = False

        # Esperar thread de áudio
        if self._audio_thread:
            self._audio_thread.join(timeout=3.0)
            self._audio_thread = None

        # Fechar VideoWriter
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None

        # Fechar áudio
        if self._audio_stream:
            try:
                self._audio_stream.stop_stream()
                self._audio_stream.close()
            except Exception:
                pass
            self._audio_stream = None

        if self._audio:
            try:
                self._audio.terminate()
            except Exception:
                pass
            self._audio = None

        # Salvar áudio WAV
        if self._audio_frames:
            self._save_audio_wav()

        # Multiplexar com ffmpeg
        success = self._mux_with_ffmpeg()

        # Limpar temporários
        self._cleanup_temp()

        return success

    def is_recording(self) -> bool:
        return self._recording

    @property
    def recorded_duration(self) -> float:
        """Duração estimada em segundos."""
        if self._audio_frames:
            return len(self._audio_frames) * AUDIO_CHUNK / AUDIO_RATE
        return 0.0

    # ------------------------------------------------------------------
    # Loop de áudio
    # ------------------------------------------------------------------
    def _audio_loop(self) -> None:
        """Thread de captura de áudio."""
        try:
            self._audio_stream = self._audio.open(
                format=AUDIO_FORMAT,
                channels=AUDIO_CHANNELS,
                rate=AUDIO_RATE,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=AUDIO_CHUNK,
            )
        except Exception as e:
            print(f"[Recorder] Erro ao abrir microfone: {e}")
            return

        while self._recording:
            try:
                data = self._audio_stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            except Exception:
                continue

            # Aplicar processador de áudio (voice changer)
            if self._audio_processor:
                try:
                    audio_np = np.frombuffer(data, dtype=np.int16)
                    processed = self._audio_processor(audio_np)
                    data = processed.tobytes()
                except Exception:
                    pass

            self._audio_frames.append(data)

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------
    def _save_audio_wav(self) -> None:
        """Salva os frames de áudio num ficheiro WAV temporário."""
        try:
            with wave.open(self._temp_audio, "wb") as wf:
                wf.setnchannels(AUDIO_CHANNELS)
                wf.setsampwidth(self._audio.get_sample_size(AUDIO_FORMAT))
                wf.setframerate(AUDIO_RATE)
                wf.writeframes(b"".join(self._audio_frames))
        except Exception as e:
            print(f"[Recorder] Erro ao salvar WAV: {e}")

    def _mux_with_ffmpeg(self) -> bool:
        """
        Usa FFmpeg para combinar vídeo e áudio.
        """
        # Verificar se há vídeo e áudio
        if not os.path.exists(self._temp_video):
            print("[Recorder] Nenhum vídeo para salvar.")
            return False

        if not os.path.exists(self._temp_audio):
            # Não há áudio — copiar só o vídeo
            print("[Recorder] Sem áudio, a copiar apenas o vídeo...")
            return self._copy_video_only()

        # Criar diretório de saída se necessário
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",  # Sobrescrever
            "-i", self._temp_video,
            "-i", self._temp_audio,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            str(self.output_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                print(f"[Recorder] Gravado: {self.output_path}")
                return True
            else:
                print(f"[Recorder] Erro FFmpeg:\n{result.stderr}")
                # Fallback: copiar só vídeo
                return self._copy_video_only()
        except subprocess.TimeoutExpired:
            print("[Recorder] FFmpeg excedeu o tempo limite.")
            return self._copy_video_only()
        except FileNotFoundError:
            print("[Recorder] FFmpeg não encontrado. A instalar ffmpeg?")
            return self._copy_video_only()

    def _copy_video_only(self) -> bool:
        """Copia apenas o vídeo (sem áudio) para o destino."""
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            # Renomear temp_video para output
            os.rename(self._temp_video, str(self.output_path))
            print(f"[Recorder] Vídeo (sem áudio) salvo: {self.output_path}")
            return True
        except Exception as e:
            print(f"[Recorder] Erro ao salvar vídeo: {e}")
            return False

    def _cleanup_partial(self) -> None:
        """Liberta recursos parciais em caso de erro no start."""
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
        if self._audio:
            self._audio.terminate()
            self._audio = None
        self._recording = False

    def _cleanup_temp(self) -> None:
        """Remove ficheiros temporários."""
        for f in [self._temp_video, self._temp_audio]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass
        try:
            os.rmdir(self._temp_dir)
        except Exception:
            pass
