"""
Voice Avatar — Rosto + Voz em Tempo Real com Gravação
======================================================
Interface unificada que junta:
  🎭 Face Swap (Deep-LiveCam)
  🎤 Voice Cloning (RVC + efeitos)
  🎬 Gravação de vídeo + áudio processado

Uso:
  python run_voice_avatar.py                          # Interface gráfica
  python run_voice_avatar.py --source minha_foto.jpg  # Com source image
  python run_voice_avatar.py --voice-model models/rvc/minha_voz  # Com modelo RVC
  python run_voice_avatar.py --voice-effect deep      # Efeito simples
"""

import os
import sys
import time
import threading
import argparse
from pathlib import Path

# ── Adicionar project root ao PATH (necessário para ffmpeg) ──
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] = project_root + os.pathsep + os.environ.get("PATH", "")

# ── DLL registration (Windows CUDA) ──
if sys.platform == "win32":
    for sp in [
        os.path.join(sys.prefix, "Lib", "site-packages"),
        os.path.join(project_root, "venv", "Lib", "site-packages"),
    ]:
        if not os.path.isdir(sp):
            continue
        for root_dir in [os.path.join(sp, "torch", "lib"), os.path.join(sp, "nvidia")]:
            if not os.path.isdir(root_dir):
                continue
            if root_dir.endswith("nvidia"):
                for pkg in os.listdir(root_dir):
                    bin_dir = os.path.join(root_dir, pkg, "bin")
                    if os.path.isdir(bin_dir):
                        os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
                        try:
                            os.add_dll_directory(bin_dir)
                        except (OSError, AttributeError):
                            pass
            else:
                os.environ["PATH"] = root_dir + os.pathsep + os.environ["PATH"]
                try:
                    os.add_dll_directory(root_dir)
                except (OSError, AttributeError):
                    pass

# ── Reduzir logs ──
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "4"

import cv2
import numpy as np

# ── Nossos módulos ──
from voice_changer import VoiceChanger, VoiceConfig, VoiceEffect
from voice_cloner import VoiceCloner, RVCVoiceModel, HAS_RVC, download_famous_voice_model
from recorder import Recorder

# ── Módulos do Deep-LiveCam ──
sys.path.insert(0, project_root)
import modules.globals as dlc_globals
from modules.face_analyser import get_face_analyser, get_one_face
from modules.processors.frame.face_swapper import get_face_swapper
from modules import imread_unicode
from modules.video_capture import VideoCapturer


# ===================================================================
# Constantes
# ===================================================================
WINDOW_TITLE = "Voice Avatar — Face + Voice"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480


# ===================================================================
# Pipeline de Processamento
# ===================================================================
class ProcessingPipeline:
    """
    Pipeline que junta face swap + voice cloning.
    Corre em threads separadas para não bloquear a UI.
    """

    def __init__(self):
        # Face swap
        self._face_swapper = None
        self._source_face = None
        self._source_path: str | None = None

        # Voice
        self._voice_cloner = VoiceCloner()
        self._voice_config = VoiceConfig(effect=VoiceEffect.NONE)

        # Recorder
        self._recorder: Recorder | None = None

        # Estado
        self._running = False
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._fps = 0.0

        # Callback para UI
        self._frame_callback = None

    # ------------------------------------------------------------------
    # Inicialização
    # ------------------------------------------------------------------
    def init_face_swap(self, source_path: str) -> bool:
        """Inicializa o face swapper com a imagem source."""
        if not source_path or not os.path.exists(source_path):
            print("[Pipeline] ❌ Imagem source não encontrada.")
            return False

        try:
            print("[Pipeline] A carregar face analyser...")
            get_face_analyser()
            print("[Pipeline] A carregar face swapper...")
            self._face_swapper = get_face_swapper()
            print("[Pipeline] A extrair face da source...")
            self._source_face = get_one_face(imread_unicode(source_path))
            self._source_path = source_path

            if self._source_face is None:
                print("[Pipeline] ❌ Nenhuma face encontrada na imagem source.")
                return False

            print("[Pipeline] ✅ Face swap pronto!")
            return True
        except Exception as e:
            print(f"[Pipeline] ❌ Erro ao iniciar face swap: {e}")
            import traceback
            traceback.print_exc()
            return False

    def init_voice(self, model_path: str | None = None, effect: str | None = None) -> bool:
        """Inicializa o voice cloner."""
        if model_path:
            print(f"[Pipeline] A carregar modelo de voz: {model_path}")
            return self._voice_cloner.load_model(model_path)
        elif effect:
            # Mapear nome do efeito
            effect_map = {
                "none": VoiceEffect.NONE,
                "pitch_up": VoiceEffect.PITCH_UP,
                "pitch_down": VoiceEffect.PITCH_DOWN,
                "robot": VoiceEffect.ROBOT,
                "echo": VoiceEffect.ECHO,
                "chipmunk": VoiceEffect.CHIPMUNK,
                "deep": VoiceEffect.DEEP,
                "whisper": VoiceEffect.WHISPER,
                "chorus": VoiceEffect.CHORUS,
            }
            self._voice_config.effect = effect_map.get(effect, VoiceEffect.NONE)
            print(f"[Pipeline] ✅ Efeito de voz: {self._voice_config.effect.value}")
            return True
        else:
            print("[Pipeline] ✅ Voz sem alterações (pass-through)")
            return True

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------
    def run(self, camera_index: int = 0) -> None:
        """Executa o pipeline de processamento em tempo real."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()

        # Inicializar câmera
        cap = VideoCapturer(camera_index)
        if not cap.start(PREVIEW_WIDTH, PREVIEW_HEIGHT, 30):
            print("[Pipeline] ❌ Falha ao abrir câmera.")
            self._running = False
            return

        self._cap = cap

        # Inicializar voice cloner (stream)
        self._voice_cloner.start()

        # Thread de processamento
        def process_loop():
            prev_time = time.time()
            frame_count = 0

            while self._running and not self._stop_event.is_set():
                ret, frame = cap.cap.read()
                if not ret:
                    time.sleep(0.001)
                    continue

                # Face swap
                if self._face_swapper and self._source_face:
                    try:
                        # Detetar face no frame
                        from modules.face_analyser import detect_one_face_fast
                        target_face = detect_one_face_fast(frame)
                        if target_face is not None:
                            frame = self._face_swapper.swap_face(
                                self._source_face, target_face, frame
                            )
                            # Post-processing
                            if hasattr(target_face, "bbox") and target_face.bbox is not None:
                                bbox = target_face.bbox.astype(int)
                                frame = self._face_swapper.apply_post_processing(frame, [bbox])
                    except Exception as e:
                        pass  # Ignorar erros de frame individuais

                # FPS
                frame_count += 1
                now = time.time()
                if now - prev_time >= 0.5:
                    self._fps = frame_count / (now - prev_time)
                    frame_count = 0
                    prev_time = now

                # FPS no canto
                if self._fps > 0:
                    cv2.putText(
                        frame, f"FPS: {self._fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2,
                    )

                # Guardar frame
                with self._frame_lock:
                    self._latest_frame = frame.copy()

                # Gravar frame
                if self._recorder and self._recorder.is_recording():
                    self._recorder.write_frame(frame)

                # Callback UI
                if self._frame_callback:
                    try:
                        self._frame_callback(frame)
                    except Exception:
                        pass

            # Cleanup
            try:
                cap.release()
            except Exception:
                pass

        threading.Thread(target=process_loop, daemon=True).start()
        print("[Pipeline] ✅ Pipeline em execução!")

    def stop(self) -> None:
        """Para o pipeline."""
        self._running = False
        self._stop_event.set()

        self._voice_cloner.stop()

        if self._recorder and self._recorder.is_recording():
            self._recorder.stop()

        time.sleep(0.3)
        print("[Pipeline] Parado.")

    # ------------------------------------------------------------------
    # Gravação
    # ------------------------------------------------------------------
    def start_recording(self, output_path: str) -> bool:
        """Inicia a gravação de vídeo + áudio."""
        if self._recorder and self._recorder.is_recording():
            return False

        self._recorder = Recorder(
            output_path=output_path,
            fps=30.0,
            width=PREVIEW_WIDTH,
            height=PREVIEW_HEIGHT,
        )

        # Ligar processador de áudio
        self._recorder.set_audio_processor(self._voice_cloner.process_audio)

        # Ligar callback de saída do voice cloner ao recorder
        self._voice_cloner.set_output_callback(
            lambda audio: None  # O recorder já processa via set_audio_processor
        )

        return self._recorder.start()

    def stop_recording(self) -> bool:
        """Para a gravação e finaliza o ficheiro."""
        if not self._recorder:
            return False

        result = self._recorder.stop()
        self._recorder = None
        return result

    def is_recording(self) -> bool:
        return self._recorder is not None and self._recorder.is_recording()

    @property
    def recording_duration(self) -> float:
        if self._recorder:
            return self._recorder.recorded_duration
        return 0.0

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------
    @property
    def latest_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    @property
    def voice_cloner(self) -> VoiceCloner:
        return self._voice_cloner

    @property
    def voice_config(self) -> VoiceConfig:
        return self._voice_config

    def set_frame_callback(self, callback):
        self._frame_callback = callback


# ===================================================================
# GUI Simples (OpenCV)
# ===================================================================
class SimpleGUI:
    """
    Interface gráfica mínima usando OpenCV.
    Tem trackbars para controlar efeitos e teclas para gravar.
    """

    def __init__(self, pipeline: ProcessingPipeline):
        self._pipeline = pipeline
        self._recording = False
        self._output_dir = Path("recordings")
        self._output_dir.mkdir(exist_ok=True)

    def run(self):
        """Loop principal da UI."""
        print("\n" + "=" * 60)
        print("  🎭 VOICE AVATAR — Rosto + Voz em Tempo Real")
        print("=" * 60)
        print("\n  🖱️  Comandos:")
        print("    ESPAÇO  — Iniciar/Parar gravação")
        print("    1-9     — Efeitos de voz")
        print("    R       — Recarregar source")
        print("    M       — Espelhar câmera")
        print("    Q / ESC — Sair")
        print("\n  📁 Gravações em:", self._output_dir.absolute())
        print()

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, DEFAULT_WIDTH, DEFAULT_HEIGHT)

        mirror = False

        while True:
            frame = self._pipeline.latest_frame
            if frame is None:
                # Mostrar tela de espera
                frame = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8)
                cv2.putText(
                    frame, "A iniciar...", (PREVIEW_WIDTH // 2 - 100, PREVIEW_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3,
                )

            # Adicionar info
            self._draw_overlay(frame, mirror)

            if mirror:
                frame = cv2.flip(frame, 1)

            cv2.imshow(WINDOW_TITLE, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # Q ou ESC
                break
            elif key == ord(" "):  # ESPAÇO
                self._toggle_recording()
            elif key == ord("m"):
                mirror = not mirror
            elif key == ord("r"):
                self._reload_source()
            elif key == ord("1"):
                self._set_effect(VoiceEffect.NONE)
            elif key == ord("2"):
                self._set_effect(VoiceEffect.PITCH_UP)
            elif key == ord("3"):
                self._set_effect(VoiceEffect.PITCH_DOWN)
            elif key == ord("4"):
                self._set_effect(VoiceEffect.DEEP)
            elif key == ord("5"):
                self._set_effect(VoiceEffect.ROBOT)
            elif key == ord("6"):
                self._set_effect(VoiceEffect.CHIPMUNK)
            elif key == ord("7"):
                self._set_effect(VoiceEffect.ECHO)
            elif key == ord("8"):
                self._set_effect(VoiceEffect.WHISPER)
            elif key == ord("9"):
                self._set_effect(VoiceEffect.CHORUS)

        cv2.destroyAllWindows()

    def _draw_overlay(self, frame: np.ndarray, mirror: bool):
        """Desenha informações no frame."""
        h, w = frame.shape[:2]

        # Barra de status no fundo
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 70), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, dst=frame)

        # Indicador de gravação
        secs = self._pipeline.recording_duration
        if self._pipeline.is_recording():
            # Círculo vermelho piscante
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(frame, (30, h - 40), 10, (0, 0, 255), -1)
            cv2.putText(
                frame, f"🔴 GRAVANDO {secs:.0f}s",
                (50, h - 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2,
            )
        else:
            cv2.putText(
                frame, f"⏸️  Pronto  |  Duração: {secs:.0f}s",
                (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1,
            )

        # Efeito de voz atual
        effect_name = self._pipeline.voice_config.effect.value.upper()
        cv2.putText(
            frame, f"Voz: {effect_name}",
            (w - 250, h - 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 200, 0), 2,
        )

    def _toggle_recording(self):
        if self._pipeline.is_recording():
            print("\n⏹️  A parar gravação...")
            self._pipeline.stop_recording()
            print("✅ Gravação salva!")
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = str(self._output_dir / f"avatar_{ts}.mp4")
            print(f"\n🔴 A gravar para: {path}")
            if self._pipeline.start_recording(path):
                print("✅ Gravação iniciada! Pressiona ESPAÇO para parar.")
            else:
                print("❌ Falha ao iniciar gravação!")

    def _set_effect(self, effect: VoiceEffect):
        config = self._pipeline.voice_config
        config.effect = effect
        print(f"\n🎤 Voz alterada: {effect.value.upper()}")

    def _reload_source(self):
        print("\n🔄 Recarregar source não implementado na GUI simples.")
        print("  Reinicia o programa com --source <imagem>")


# ===================================================================
# CLI / Entry Point
# ===================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Voice Avatar — Face Swap + Voice Cloning + Gravação",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python run_voice_avatar.py --source minha_foto.jpg
  python run_voice_avatar.py --source face.jpg --voice-effect deep
  python run_voice_avatar.py --source face.jpg --voice-model models/rvc/taylor_swift
  python run_voice_avatar.py --voice-effect robot  (sem face swap)
        """,
    )

    parser.add_argument(
        "-s", "--source",
        help="Imagem source com o rosto a usar no face swap",
        default=None,
    )
    parser.add_argument(
        "-c", "--camera",
        help="Índice da câmera (default: 0)",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--voice-model",
        help="Caminho para modelo RVC (.pth) — ex: models/rvc/minha_voz",
        default=None,
    )
    parser.add_argument(
        "--voice-effect",
        help="Efeito de voz simples (sem RVC)",
        choices=["none", "pitch_up", "pitch_down", "robot", "echo", "chipmunk", "deep", "whisper", "chorus"],
        default=None,
    )
    parser.add_argument(
        "--download-voice",
        help="Descarrega modelo de voz famosa (ex: taylor_swift, morgan_freeman)",
        default=None,
    )
    parser.add_argument(
        "--no-face-swap",
        help="Desativa face swap (apenas voz)",
        action="store_true",
        default=False,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Descarregar modelo se pedido
    if args.download_voice:
        print(f"\n📥 A descarregar modelo para: {args.download_voice}")
        result = download_famous_voice_model(args.download_voice)
        if result:
            print(f"✅ Modelo descarregado! Usa --voice-model models/rvc/{args.download_voice}")
        else:
            print("❌ Falha ao descarregar. Procura manualmente no HuggingFace.")
        if not args.source and not args.voice_model and not args.voice_effect:
            return

    # Criar pipeline
    pipeline = ProcessingPipeline()

    # Inicializar face swap
    if args.source and not args.no_face_swap:
        print(f"\n🎭 A iniciar Face Swap com: {args.source}")
        if not pipeline.init_face_swap(args.source):
            print("⚠️  Face swap indisponível. Continuando apenas com voz...")
    else:
        print("\n🎭 Face swap desativado.")

    # Inicializar voz
    if args.voice_model:
        print(f"\n🎤 A carregar modelo RVC: {args.voice_model}")
        pipeline.init_voice(model_path=args.voice_model)
    elif args.voice_effect:
        print(f"\n🎤 A usar efeito: {args.voice_effect}")
        pipeline.init_voice(effect=args.voice_effect)
    else:
        print("\n🎤 Voz pass-through (sem alterações)")

    # Iniciar pipeline
    print("\n🚀 A iniciar pipeline...")
    pipeline.run(camera_index=args.camera)

    # Iniciar GUI
    gui = SimpleGUI(pipeline)
    try:
        gui.run()
    except KeyboardInterrupt:
        print("\n⏹️  Interrompido pelo utilizador.")
    finally:
        pipeline.stop()
        print("👋 Até logo!")


if __name__ == "__main__":
    main()
