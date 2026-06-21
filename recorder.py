"""Gravação simples de câmera, microfone e trilha de áudio."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pyaudio


ROOT = Path(__file__).resolve().parent


class Recorder:
    """Grava frames e microfone, finalizando um MP4 com FFmpeg."""

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        input_device_index: int | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.fps = max(1.0, float(fps))
        self.width = int(width)
        self.height = int(height)
        self.input_device_index = input_device_index

        self._recording = False
        self._started_at = 0.0
        self._written_frames = 0
        self._last_frame: np.ndarray | None = None
        self._lock = threading.Lock()

        self._temp_dir = Path(tempfile.mkdtemp(prefix="voice_avatar_"))
        self._video_path = self._temp_dir / "video.avi"
        self._audio_path = self._temp_dir / "microfone.wav"

        self._writer: cv2.VideoWriter | None = None
        self._audio: pyaudio.PyAudio | None = None
        self._audio_stream = None
        self._audio_thread: threading.Thread | None = None
        self._audio_frames: list[bytes] = []
        self._audio_rate = 44100
        self._sample_width = 2
        self._microphone_error: str | None = None

    def start(self) -> bool:
        if self._recording:
            return False

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._writer = cv2.VideoWriter(
            str(self._video_path),
            fourcc,
            self.fps,
            (self.width, self.height),
        )
        if not self._writer.isOpened():
            self._writer.release()
            self._writer = None
            return False

        self._recording = True
        self._started_at = time.perf_counter()
        self._start_microphone()
        return True

    def _start_microphone(self) -> None:
        if self.input_device_index == -1:
            return
        try:
            self._audio = pyaudio.PyAudio()
            if self.input_device_index is not None:
                info = self._audio.get_device_info_by_index(self.input_device_index)
                self._audio_rate = int(info.get("defaultSampleRate", 44100))
            self._sample_width = self._audio.get_sample_size(pyaudio.paInt16)
            self._audio_stream = self._audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self._audio_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=1024,
            )
            self._audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
            self._audio_thread.start()
        except Exception as exc:
            self._microphone_error = str(exc)
            self._close_audio()

    def _audio_loop(self) -> None:
        while self._recording and self._audio_stream is not None:
            try:
                data = self._audio_stream.read(1024, exception_on_overflow=False)
                self._audio_frames.append(data)
            except Exception:
                break

    def write_frame(self, frame: np.ndarray) -> None:
        if not self._recording:
            return

        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        # Mantém a duração correta mesmo quando o face swap reduz o FPS.
        target_count = max(
            1,
            int((time.perf_counter() - self._started_at) * self.fps) + 1,
        )
        with self._lock:
            if self._writer is None:
                return
            while self._written_frames < target_count:
                self._writer.write(frame)
                self._written_frames += 1
            self._last_frame = frame.copy()

    def stop(
        self,
        soundtrack_path: str | None = None,
        voice_model_path: str | None = None,
        voice_index_path: str | None = None,
        voice_pitch: int = 0,
        status_callback: Callable[[str], None] | None = None,
    ) -> tuple[bool, str]:
        if not self._recording:
            return False, "A gravação não estava ativa."

        self._recording = False
        if self._audio_thread is not None:
            self._audio_thread.join(timeout=2.0)
            self._audio_thread = None

        with self._lock:
            if self._writer is not None:
                self._writer.release()
                self._writer = None

        self._close_audio()
        self._save_microphone()

        try:
            success, message = self._encode(
                soundtrack_path,
                voice_model_path,
                voice_index_path,
                voice_pitch,
                status_callback,
            )
        finally:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        return success, message

    def _close_audio(self) -> None:
        if self._audio_stream is not None:
            try:
                self._audio_stream.stop_stream()
                self._audio_stream.close()
            except Exception:
                pass
            self._audio_stream = None
        if self._audio is not None:
            try:
                self._audio.terminate()
            except Exception:
                pass
            self._audio = None

    def _save_microphone(self) -> None:
        if not self._audio_frames:
            return
        with wave.open(str(self._audio_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(self._sample_width)
            wav.setframerate(self._audio_rate)
            wav.writeframes(b"".join(self._audio_frames))

    def _convert_voice(
        self,
        model_path: str,
        index_path: str | None,
        pitch: int,
    ) -> tuple[Path | None, str | None]:
        runtime_python = ROOT / ".rvc-venv" / "Scripts" / "python.exe"
        worker = ROOT / "rvc_worker.py"
        clean_input = self._temp_dir / "microfone_limpo.wav"
        converted_output = self._temp_dir / "voz_rvc_convertida.wav"
        final_output = self._temp_dir / "voz_rvc.wav"

        if not runtime_python.is_file():
            return None, "Runtime RVC ausente. Execute setup_rvc_runtime.ps1."

        # Remove graves mecânicos, chiado e grandes variações de volume antes
        # da extração HuBERT/F0. Isso evita que ruído seja interpretado como
        # fonema e melhora especialmente consoantes do português.
        prepare_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(self._audio_path),
            "-af",
            "highpass=f=70,lowpass=f=12000,"
            "afftdn=nf=-28:nr=10:tn=1,"
            "acompressor=threshold=-18dB:ratio=2:attack=10:release=100,"
            "alimiter=limit=0.95",
            "-ac", "1",
            "-ar", "44100",
            str(clean_input),
        ]
        try:
            prepared = subprocess.run(
                prepare_cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, f"Falha ao preparar o microfone: {exc}"
        if prepared.returncode != 0 or not clean_input.is_file():
            return None, prepared.stderr.strip() or "Falha ao limpar o microfone."

        cmd = [
            str(runtime_python),
            str(worker),
            "--input", str(clean_input),
            "--output", str(converted_output),
            "--model", str(Path(model_path).resolve()),
            "--pitch", str(int(pitch)),
        ]
        if index_path:
            cmd += ["--index", str(Path(index_path).resolve())]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=1800,
            )
        except subprocess.TimeoutExpired:
            return None, "A conversão RVC excedeu 30 minutos."
        except OSError as exc:
            return None, f"Não foi possível iniciar o RVC: {exc}"

        if result.returncode != 0 or not converted_output.is_file():
            details = (result.stderr or result.stdout).strip()
            if len(details) > 800:
                details = details[-800:]
            return None, f"Falha no RVC: {details or 'erro desconhecido'}"

        # 82% da voz convertida mantém o timbre; 18% do microfone limpo
        # recupera articulação e reduz o efeito de "outro idioma".
        blend_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(converted_output),
            "-i", str(clean_input),
            "-filter_complex",
            "[0:a]volume=0.82[converted];"
            "[1:a]volume=0.18[dry];"
            "[converted][dry]amix=inputs=2:duration=longest:"
            "dropout_transition=0,alimiter=limit=0.95[out]",
            "-map", "[out]",
            "-ac", "1",
            "-ar", "44100",
            str(final_output),
        ]
        try:
            blended = subprocess.run(
                blend_cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return converted_output, f"Não foi possível aplicar clareza extra: {exc}"
        if blended.returncode != 0 or not final_output.is_file():
            return converted_output, None
        return final_output, None

    def _encode(
        self,
        soundtrack_path: str | None,
        voice_model_path: str | None,
        voice_index_path: str | None,
        voice_pitch: int,
        status_callback: Callable[[str], None] | None,
    ) -> tuple[bool, str]:
        soundtrack = Path(soundtrack_path) if soundtrack_path else None
        has_mic = self._audio_path.exists() and self._audio_path.stat().st_size > 44
        has_soundtrack = bool(soundtrack and soundtrack.is_file())
        microphone_path = self._audio_path
        voice_warning: str | None = None

        if has_mic and voice_model_path:
            if status_callback:
                status_callback("Convertendo o microfone com o modelo RVC...")
            converted, voice_warning = self._convert_voice(
                voice_model_path,
                voice_index_path,
                voice_pitch,
            )
            if converted is not None:
                microphone_path = converted

        if status_callback:
            status_callback("Gerando o arquivo MP4...")

        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(self._video_path)]
        if has_mic:
            cmd += ["-i", str(microphone_path)]
        if has_soundtrack:
            cmd += ["-stream_loop", "-1", "-i", str(soundtrack)]

        if has_mic and has_soundtrack:
            soundtrack_index = 2
            cmd += [
                "-filter_complex",
                f"[1:a]aresample=async=1:first_pts=0,volume=1.0[mic];"
                f"[{soundtrack_index}:a]volume=0.30[bg];"
                "[mic][bg]amix=inputs=2:duration=first:dropout_transition=2[a]",
                "-map", "0:v:0", "-map", "[a]",
            ]
        elif has_mic:
            cmd += ["-map", "0:v:0", "-map", "1:a:0"]
        elif has_soundtrack:
            cmd += ["-map", "0:v:0", "-map", "1:a:0"]
        else:
            cmd += ["-map", "0:v:0", "-an"]

        cmd += [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
        ]
        if has_mic or has_soundtrack:
            cmd += ["-c:a", "aac", "-b:a", "160k", "-shortest"]
        cmd.append(str(self.output_path))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except FileNotFoundError:
            return False, "FFmpeg não foi encontrado no PATH."
        except subprocess.TimeoutExpired:
            return False, "O FFmpeg excedeu o tempo limite."

        if result.returncode != 0:
            return False, result.stderr.strip() or "Falha ao gerar o MP4."

        extra = ""
        if self._microphone_error:
            extra = " O microfone não abriu; o vídeo foi salvo sem ele."
        elif voice_warning:
            extra = f" Aviso: {voice_warning} O áudio original foi mantido."
        return True, f"Vídeo salvo em: {self.output_path}{extra}"

    def is_recording(self) -> bool:
        return self._recording

    @property
    def recorded_duration(self) -> float:
        if not self._recording:
            return 0.0
        return time.perf_counter() - self._started_at
