"""
Módulo de Voice Changer em Tempo Real
=====================================
Suporta múltiplos efeitos: pitch shift, robot, eco, formant shift, etc.
Usa PyAudio para captura e reprodução de áudio em tempo real.
"""

import numpy as np
import pyaudio
import threading
import time
from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Constantes de áudio
# ---------------------------------------------------------------------------
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100


class VoiceEffect(Enum):
    NONE = "none"
    PITCH_UP = "pitch_up"           # Voz mais aguda
    PITCH_DOWN = "pitch_down"       # Voz mais grave
    ROBOT = "robot"                 # Voz robótica
    ECHO = "echo"                   # Eco
    CHIPMUNK = "chipmunk"           # Voz de esquilo
    DEEP = "deep"                   # Voz profunda (Darth Vader)
    WHISPER = "whisper"             # Sussurro
    CHORUS = "chorus"               # Efeito coral


@dataclass
class VoiceConfig:
    effect: VoiceEffect = VoiceEffect.NONE
    pitch_semitones: float = 0.0           # -12 a +12 semitons
    robot_phases: int = 4                  # Mais fases = mais robótico
    echo_delay_seconds: float = 0.3
    echo_decay: float = 0.4
    chorus_voices: int = 3
    volume: float = 1.0
    monitor: bool = True                   # Ouvir a própria voz


class VoiceChanger:
    """
    Voice changer em tempo real.
    Captura do microfone -> aplica efeito -> reproduz nos alto-falantes (monitor)
    ou envia para um stream virtual (VB-Cable).
    """

    def __init__(self, config: VoiceConfig | None = None):
        self.config = config or VoiceConfig()
        self._audio = pyaudio.PyAudio()
        self._input_stream = None
        self._output_stream = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # Buffers para efeitos
        self._echo_buffer = np.zeros(0, dtype=np.float32)
        self._chorus_phase = 0.0

        # Callback para gravação externa
        self._audio_callback = None

    # ------------------------------------------------------------------
    # Listar dispositivos
    # ------------------------------------------------------------------
    @staticmethod
    def list_input_devices() -> list[dict]:
        """Lista dispositivos de entrada (microfones)."""
        p = pyaudio.PyAudio()
        devices = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                devices.append({
                    "index": i,
                    "name": info["name"],
                    "channels": info["maxInputChannels"],
                    "sample_rate": int(info["defaultSampleRate"]),
                })
        p.terminate()
        return devices

    @staticmethod
    def list_output_devices() -> list[dict]:
        """Lista dispositivos de saída (alto-falantes)."""
        p = pyaudio.PyAudio()
        devices = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["maxOutputChannels"] > 0:
                devices.append({
                    "index": i,
                    "name": info["name"],
                    "channels": info["maxOutputChannels"],
                    "sample_rate": int(info["defaultSampleRate"]),
                })
        p.terminate()
        return devices

    # ------------------------------------------------------------------
    # Efeitos de áudio
    # ------------------------------------------------------------------
    def _pitch_shift(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        """Pitch shift usando stretching + resampling."""
        if semitones == 0:
            return audio

        factor = 2.0 ** (semitones / 12.0)
        indices = np.arange(0, len(audio), factor)
        indices = indices[indices < len(audio)].astype(np.int32)

        stretched = audio[indices]

        # Resample de volta ao tamanho original
        if len(stretched) > 0:
            old_indices = np.linspace(0, len(stretched) - 1, len(stretched))
            new_indices = np.linspace(0, len(stretched) - 1, len(audio))
            result = np.interp(new_indices, old_indices, stretched).astype(np.float32)
        else:
            result = np.zeros_like(audio, dtype=np.float32)

        return result

    def _robot_effect(self, audio: np.ndarray, phases: int) -> np.ndarray:
        """Efeito robótico: quantização + sample & hold."""
        if phases <= 1:
            return audio

        # Reduzir resolução temporal
        step = max(1, len(audio) // (phases * 10))
        result = np.zeros_like(audio, dtype=np.float32)

        for i in range(0, len(audio), step):
            end = min(i + step, len(audio))
            result[i:end] = np.mean(audio[i:end])

        return result

    def _echo_effect(self, audio: np.ndarray, delay_s: float, decay: float) -> np.ndarray:
        """Adiciona eco ao áudio."""
        delay_samples = int(delay_s * RATE)

        # Atualizar buffer de eco
        needed = len(audio) + delay_samples
        if len(self._echo_buffer) < needed:
            self._echo_buffer = np.pad(self._echo_buffer, (0, needed - len(self._echo_buffer)))

        # Deslocar buffer
        self._echo_buffer[delay_samples:delay_samples + len(audio)] += audio * decay
        result = audio + self._echo_buffer[:len(audio)]

        # Rotacionar buffer
        self._echo_buffer = np.roll(self._echo_buffer, -len(audio))
        self._echo_buffer[-len(audio):] = 0

        return np.clip(result, -1.0, 1.0)

    def _chorus_effect(self, audio: np.ndarray, voices: int) -> np.ndarray:
        """Efeito chorus com múltiplas vozes desafinadas."""
        result = audio.copy()
        for v in range(1, voices):
            # Pequena modulação de delay
            modulation = np.sin(2 * np.pi * 0.5 * np.arange(len(audio)) / RATE + v) * 0.003
            delay_samples = (v * 10 + modulation * RATE).astype(np.int32)

            # Aplicar delay variável
            delayed = np.zeros_like(audio)
            for i in range(len(audio)):
                src = i - max(0, delay_samples[i])
                if src >= 0 and src < len(audio):
                    delayed[i] = audio[int(src)]

            result += delayed * (0.3 / voices)

        return np.clip(result, -1.0, 1.0)

    def _whisper_effect(self, audio: np.ndarray) -> np.ndarray:
        """Simula sussurro adicionando ruído branco modulado."""
        noise = np.random.normal(0, 0.3, len(audio)).astype(np.float32)
        envelope = np.abs(audio) / (np.max(np.abs(audio)) + 1e-8)
        return audio * 0.3 + noise * envelope

    # ------------------------------------------------------------------
    # Pipeline de processamento
    # ------------------------------------------------------------------
    def process_audio(self, audio_int16: np.ndarray) -> np.ndarray:
        """
        Processa um chunk de áudio aplicando o efeito configurado.
        Entrada/Saída: int16 numpy array.
        """
        # Converter para float32 [-1, 1]
        audio = audio_int16.astype(np.float32) / 32768.0

        with self._lock:
            cfg = self.config

        effect = cfg.effect

        if effect == VoiceEffect.PITCH_UP:
            audio = self._pitch_shift(audio, 4.0)
        elif effect == VoiceEffect.PITCH_DOWN:
            audio = self._pitch_shift(audio, -4.0)
        elif effect == VoiceEffect.CHIPMUNK:
            audio = self._pitch_shift(audio, 12.0)
        elif effect == VoiceEffect.DEEP:
            audio = self._pitch_shift(audio, -12.0)
        elif effect == VoiceEffect.ROBOT:
            audio = self._robot_effect(audio, cfg.robot_phases)
        elif effect == VoiceEffect.ECHO:
            audio = self._echo_effect(audio, cfg.echo_delay_seconds, cfg.echo_decay)
        elif effect == VoiceEffect.WHISPER:
            audio = self._whisper_effect(audio)
        elif effect == VoiceEffect.CHORUS:
            audio = self._chorus_effect(audio, cfg.chorus_voices)

        # Aplicar pitch customizado (sobrepõe com efeito se ambos especificados)
        if cfg.pitch_semitones != 0:
            audio = self._pitch_shift(audio, cfg.pitch_semitones)

        # Volume
        audio *= cfg.volume

        # Clip
        audio = np.clip(audio, -1.0, 1.0)

        # Voltar para int16
        result = (audio * 32767).astype(np.int16)

        # Callback externo (para gravação)
        if self._audio_callback:
            try:
                self._audio_callback(result)
            except Exception:
                pass

        return result

    # ------------------------------------------------------------------
    # Iniciar / Parar
    # ------------------------------------------------------------------
    def start(
        self,
        input_device_index: int | None = None,
        output_device_index: int | None = None,
    ) -> bool:
        """Inicia o voice changer em tempo real."""
        if self._running:
            return False

        try:
            self._input_stream = self._audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=CHUNK,
            )
        except Exception as e:
            print(f"[VoiceChanger] Erro ao abrir microfone: {e}")
            return False

        if self.config.monitor:
            try:
                self._output_stream = self._audio.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True,
                    output_device_index=output_device_index,
                    frames_per_buffer=CHUNK,
                )
            except Exception as e:
                print(f"[VoiceChanger] Erro ao abrir saída de áudio: {e}")
                self._input_stream.close()
                self._input_stream = None
                return False

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self) -> None:
        """Loop principal de processamento de áudio."""
        while self._running:
            try:
                data = self._input_stream.read(CHUNK, exception_on_overflow=False)
            except Exception:
                continue

            audio_in = np.frombuffer(data, dtype=np.int16)
            audio_out = self.process_audio(audio_in)

            if self._output_stream:
                try:
                    self._output_stream.write(audio_out.tobytes())
                except Exception:
                    pass

    def stop(self) -> None:
        """Para o voice changer."""
        self._running = False

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._output_stream:
            try:
                self._output_stream.stop_stream()
                self._output_stream.close()
            except Exception:
                pass
            self._output_stream = None

        if self._input_stream:
            try:
                self._input_stream.stop_stream()
                self._input_stream.close()
            except Exception:
                pass
            self._input_stream = None

    def set_audio_callback(self, callback) -> None:
        """Define callback que recebe cada chunk processado (int16 numpy array)."""
        self._audio_callback = callback

    def update_config(self, config: VoiceConfig) -> None:
        """Atualiza a configuração em tempo real (thread-safe)."""
        with self._lock:
            self.config = config

    def close(self) -> None:
        """Libera recursos."""
        self.stop()
        try:
            self._audio.terminate()
        except Exception:
            pass


# ===================================================================
# Teste rápido
# ===================================================================
if __name__ == "__main__":
    print("=== Voice Changer - Teste Rápido ===")
    print("Dispositivos de entrada:")
    for d in VoiceChanger.list_input_devices():
        print(f"  [{d['index']}] {d['name']}")

    print("\nDispositivos de saída:")
    for d in VoiceChanger.list_output_devices():
        print(f"  [{d['index']}] {d['name']}")

    config = VoiceConfig(
        effect=VoiceEffect.PITCH_UP,
        monitor=True,
    )
    vc = VoiceChanger(config)

    print("\nIniciando voice changer (PITCH_UP)... Pressione Ctrl+C para parar.")
    if vc.start():
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nParando...")
        finally:
            vc.close()
    else:
        print("Falha ao iniciar!")
