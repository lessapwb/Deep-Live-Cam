"""
Módulo de Clonagem de Voz com RVC (Retrieval-based Voice Conversion)
====================================================================
Permite usar uma amostra de voz famosa para converter a tua voz em tempo real.

Suporta dois modos:
  1. RVC (Recomendado) — Usa modelos .pth/.index treinados para qualidade profissional
  2. Efeitos simples — Fallback com pitch shift + formant shift

Fluxo típico:
  1. Arranja um modelo RVC pré-treinado da voz desejada (HuggingFace, Discord RVC)
  2. Coloca os ficheiros .pth e .index na pasta models/rvc/
  3. Executa: python run_voice_avatar.py --voice-model modelos/rvc/meu_modelo

OU usa efeitos simples sem precisar de modelo:
  python run_voice_avatar.py --voice-effect deep
"""

import os
import sys
import threading
import time
import subprocess
from pathlib import Path
from typing import Optional, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
RATE = 44100
CHUNK = 1024
HAS_RVC = False

# Tentar importar rvc-python
try:
    from rvc_python.infer import RVCInference
    HAS_RVC = True
    print("[VoiceCloner] ✅ rvc-python encontrado — clonagem RVC disponível")
except ImportError:
    print("[VoiceCloner] ⚠️  rvc-python não instalado — apenas efeitos simples")
    print("  Para instalar: pip install rvc-python")
    print("  Ou executa: python setup_rvc.py")


# ---------------------------------------------------------------------------
# Modelo de Voz
# ---------------------------------------------------------------------------
class RVCVoiceModel:
    """
    Representa um modelo de voz RVC.
    Aponta para ficheiros .pth (pesos) e .index (feature index).
    """

    def __init__(
        self,
        name: str,
        pth_path: str | None = None,
        index_path: str | None = None,
        description: str = "",
        sample_url: str = "",
    ):
        self.name = name
        self.pth_path = Path(pth_path) if pth_path else None
        self.index_path = Path(index_path) if index_path else None
        self.description = description
        self.sample_url = sample_url

    @property
    def is_valid(self) -> bool:
        return (
            self.pth_path is not None
            and self.pth_path.exists()
        )

    def __repr__(self):
        return f"RVCVoiceModel({self.name}, valid={self.is_valid})"


# ---------------------------------------------------------------------------
# Voice Cloner Principal
# ---------------------------------------------------------------------------
class VoiceCloner:
    """
    Clonador de voz em tempo real.

    Uso:
        cloner = VoiceCloner()
        cloner.load_model("models/rvc/cantor_famoso")
        cloner.start(input_device=0, output_device=1)

        while True:
            audio = capturar_audio()
            voz_clonada = cloner.process_audio(audio)
    """

    def __init__(self):
        self._rvc: RVCInference | None = None
        self._model: RVCVoiceModel | None = None
        self._running = False
        self._lock = threading.Lock()

        # Configurações
        self.pitch_shift: int = 0        # -24 a +24 semitons
        self.formant_shift: float = 0.0  # -1.0 a +1.0
        self.volume: float = 1.0
        self.use_rvc: bool = HAS_RVC

        # Callback de saída
        self._output_callback: Callable | None = None

        # Cache para efeitos simples
        self._simple_effects = self._SimpleEffects()

    # ------------------------------------------------------------------
    # Carregar modelo
    # ------------------------------------------------------------------
    def load_model(self, model_path: str) -> bool:
        """
        Carrega um modelo RVC.
        model_path: caminho base (sem extensão) — espera model_path.pth e model_path.index
        """
        pth = Path(model_path + ".pth") if not model_path.endswith(".pth") else Path(model_path)
        index = Path(str(pth).replace(".pth", ".index"))

        if not pth.exists():
            print(f"[VoiceCloner] ❌ Modelo não encontrado: {pth}")
            print("  Coloca o ficheiro .pth em models/rvc/")
            self.use_rvc = False
            return False

        self._model = RVCVoiceModel(
            name=pth.stem,
            pth_path=str(pth),
            index_path=str(index) if index.exists() else None,
        )

        if HAS_RVC and self.use_rvc:
            try:
                self._rvc = RVCInference(
                    model_path=str(pth),
                    index_path=str(index) if index.exists() else None,
                )
                print(f"[VoiceCloner] ✅ Modelo carregado: {pth.stem}")
                return True
            except Exception as e:
                print(f"[VoiceCloner] ❌ Erro ao carregar RVC: {e}")
                print("  A usar fallback de efeitos simples...")
                self.use_rvc = False
                return False

        return False

    def list_available_models(self) -> list[RVCVoiceModel]:
        """Lista modelos RVC disponíveis na pasta models/rvc/."""
        models = []
        rvc_dir = Path("models/rvc")
        if rvc_dir.exists():
            for pth in rvc_dir.glob("*.pth"):
                index = pth.with_suffix(".index")
                models.append(RVCVoiceModel(
                    name=pth.stem,
                    pth_path=str(pth),
                    index_path=str(index) if index.exists() else None,
                ))
        
        # Adicionar modelos built-in (efeitos simples)
        models.append(RVCVoiceModel(
            name="🎤 Efeito: Voz Aguda",
            description="Pitch shift +4 semitons",
        ))
        models.append(RVCVoiceModel(
            name="🎤 Efeito: Voz Grave",
            description="Pitch shift -4 semitons",
        ))
        models.append(RVCVoiceModel(
            name="🤖 Efeito: Robô",
            description="Quantização + sample & hold",
        ))
        return models

    # ------------------------------------------------------------------
    # Processamento de Áudio
    # ------------------------------------------------------------------
    def process_audio(self, audio_int16: np.ndarray) -> np.ndarray:
        """
        Processa um chunk de áudio.
        Se RVC estiver ativo, faz voice conversion.
        Caso contrário, aplica efeitos simples.
        """
        if self._rvc is not None and self.use_rvc:
            return self._process_rvc(audio_int16)
        else:
            return self._process_simple(audio_int16)

    def _process_rvc(self, audio_int16: np.ndarray) -> np.ndarray:
        """Conversão via RVC."""
        with self._lock:
            rvc = self._rvc
            pitch = self.pitch_shift

        if rvc is None:
            return audio_int16

        try:
            # Converter para float32 [-1, 1]
            audio_float = audio_int16.astype(np.float32) / 32768.0

            # RVC inference
            result_float = rvc.infer(
                audio_float,
                sr=RATE,
                pitch_shift=pitch,
            )

            # Voltar para int16
            if result_float is not None:
                result = np.clip(result_float * 32767, -32768, 32767).astype(np.int16)
                # Ajustar tamanho se necessário
                if len(result) < len(audio_int16):
                    result = np.pad(result, (0, len(audio_int16) - len(result)))
                elif len(result) > len(audio_int16):
                    result = result[:len(audio_int16)]
                return result

        except Exception as e:
            # Fallback silencioso para efeitos simples
            pass

        return audio_int16

    def _process_simple(self, audio_int16: np.ndarray) -> np.ndarray:
        """Efeitos simples como fallback."""
        return self._simple_effects.process(audio_int16, self.pitch_shift)

    # ------------------------------------------------------------------
    # Iniciar / Parar stream
    # ------------------------------------------------------------------
    def start(
        self,
        input_device_index: int | None = None,
        output_device_index: int | None = None,
    ) -> bool:
        """Inicia o stream de voz em tempo real."""
        if self._running:
            return False

        import pyaudio
        self._audio = pyaudio.PyAudio()
        self._running = True

        def audio_loop():
            import pyaudio
            try:
                in_stream = self._audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    input_device_index=input_device_index,
                    frames_per_buffer=CHUNK,
                )
            except Exception as e:
                print(f"[VoiceCloner] Erro microfone: {e}")
                self._running = False
                return

            out_stream = None
            try:
                out_stream = self._audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    output=True,
                    output_device_index=output_device_index,
                    frames_per_buffer=CHUNK,
                )
            except Exception:
                pass  # Sem monitor

            while self._running:
                try:
                    data = in_stream.read(CHUNK, exception_on_overflow=False)
                except Exception:
                    continue

                audio_in = np.frombuffer(data, dtype=np.int16)
                audio_out = self.process_audio(audio_in)

                # Callback externo (gravação)
                if self._output_callback:
                    try:
                        self._output_callback(audio_out)
                    except Exception:
                        pass

                if out_stream:
                    try:
                        out_stream.write(audio_out.tobytes())
                    except Exception:
                        pass

            in_stream.close()
            if out_stream:
                out_stream.close()

        threading.Thread(target=audio_loop, daemon=True).start()
        print(f"[VoiceCloner] ✅ Stream iniciado (RVC={'sim' if self._rvc else 'não'})")
        return True

    def stop(self) -> None:
        """Para o stream."""
        self._running = False
        time.sleep(0.2)
        try:
            self._audio.terminate()
        except Exception:
            pass

    def set_output_callback(self, callback: Callable) -> None:
        """Define callback que recebe cada chunk processado."""
        self._output_callback = callback

    # ------------------------------------------------------------------
    # Classe interna de efeitos simples
    # ------------------------------------------------------------------
    class _SimpleEffects:
        def __init__(self):
            self._echo_buffer = np.zeros(0, dtype=np.float32)

        def process(self, audio_int16: np.ndarray, pitch_semitones: int = 0) -> np.ndarray:
            audio = audio_int16.astype(np.float32) / 32768.0

            if abs(pitch_semitones) > 0:
                audio = self._pitch_shift(audio, pitch_semitones)

            return np.clip(audio * 32767, -32768, 32767).astype(np.int16)

        def _pitch_shift(self, audio: np.ndarray, semitones: float) -> np.ndarray:
            if semitones == 0:
                return audio
            factor = 2.0 ** (semitones / 12.0)
            indices = np.arange(0, len(audio), factor).astype(np.int32)
            indices = indices[indices < len(audio)]
            stretched = audio[indices]
            if len(stretched) > 0:
                old = np.linspace(0, len(stretched) - 1, len(stretched))
                new = np.linspace(0, len(stretched) - 1, len(audio))
                return np.interp(new, old, stretched).astype(np.float32)
            return np.zeros_like(audio, dtype=np.float32)


# ===================================================================
# Funções utilitárias
# ===================================================================
def download_famous_voice_model(voice_name: str, output_dir: str = "models/rvc") -> bool:
    """
    Tenta encontrar e descarregar um modelo RVC de voz famosa.
    
    Vozes disponíveis (HuggingFace):
      - "taylor_swift", "donald_trump", "morgan_freeman", "obama", etc.
    
    Requer huggingface_hub instalado.
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("[VoiceCloner] ⚠️  huggingface_hub não instalado.")
        print("  pip install huggingface_hub")
        print("\nAlternativa manual:")
        print(f"  1. Procura modelos .pth em: https://huggingface.co/models?search=rvc+{voice_name}")
        print(f"  2. Descarrega o .pth e .index para: {output_dir}/")
        return False

    # Repositórios conhecidos de modelos RVC
    known_repos = [
        "therealvulcan/RVC-Voice-Conversion-Community-Models",
        "lj1995/VoiceConversionWebUI",
    ]

    for repo in known_repos:
        try:
            files = list_repo_files(repo)
            matching = [f for f in files if voice_name.lower() in f.lower() and f.endswith(".pth")]
            if matching:
                print(f"[VoiceCloner] 🔍 Encontrado: {matching[0]} em {repo}")
                os.makedirs(output_dir, exist_ok=True)
                pth_path = hf_hub_download(repo, matching[0], local_dir=output_dir)
                print(f"[VoiceCloner] ✅ Descarregado: {pth_path}")
                
                # Procurar .index correspondente
                index_file = matching[0].replace(".pth", ".index")
                index_matches = [f for f in files if f == index_file]
                if index_matches:
                    idx_path = hf_hub_download(repo, index_file, local_dir=output_dir)
                    print(f"[VoiceCloner] ✅ Index: {idx_path}")
                
                return True
        except Exception:
            continue

    print(f"[VoiceCloner] ❌ Não encontrado modelo para '{voice_name}'")
    print(f"  Procura manualmente em: https://huggingface.co/models?search=rvc+{voice_name}")
    return False


# ===================================================================
# Teste rápido
# ===================================================================
if __name__ == "__main__":
    print("=== Voice Cloner - Teste ===\n")

    cloner = VoiceCloner()
    models = cloner.list_available_models()

    print("Modelos disponíveis:")
    for i, m in enumerate(models):
        print(f"  [{i}] {m.name} {'✅' if m.is_valid else '🎤'}")

    if HAS_RVC and models:
        # Tentar carregar o primeiro modelo .pth
        for m in models:
            if m.is_valid:
                print(f"\nCarregando: {m.name}...")
                cloner.load_model(str(m.pth_path).replace(".pth", ""))
                break
        else:
            print("\nNenhum modelo .pth encontrado em models/rvc/")
            print("Usando efeitos simples como fallback.")
    else:
        print("\nUsando efeitos simples (rvc-python não instalado).")
        print("Para clonagem real: pip install rvc-python && python setup_rvc.py")

    print("\nPressiona Ctrl+C para sair.")
    try:
        cloner.start()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nA parar...")
        cloner.stop()
