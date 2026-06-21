"""Interface simples: foto + câmera + microfone + áudio + gravação."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ["PATH"] = str(ROOT) + os.pathsep + os.environ.get("PATH", "")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def _register_windows_dlls() -> None:
    if sys.platform != "win32":
        return
    candidates = [
        Path(sys.prefix) / "Lib" / "site-packages",
        ROOT / "venv" / "Lib" / "site-packages",
    ]
    for site_packages in candidates:
        paths = [site_packages / "torch" / "lib"]
        nvidia = site_packages / "nvidia"
        if nvidia.is_dir():
            paths.extend(path / "bin" for path in nvidia.iterdir())
        for path in paths:
            if not path.is_dir():
                continue
            os.environ["PATH"] = str(path) + os.pathsep + os.environ["PATH"]
            try:
                os.add_dll_directory(str(path))
            except (AttributeError, OSError):
                pass


_register_windows_dlls()

import cv2
import numpy as np
import onnxruntime
import pyaudio
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import modules.globals as globals_
from modules import imread_unicode
from modules.face_analyser import (
    detect_one_face_fast,
    ensure_landmarks,
    get_face_analyser,
    get_one_face,
)
from modules.processors.frame.face_swapper import (
    apply_post_processing,
    get_face_swapper,
    swap_face,
)
from modules.video_capture import VideoCapturer
from recorder import Recorder


APP_TITLE = "Avatar simples"
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
RECORD_FPS = 30.0
RVC_MODELS_DIR = ROOT / "models" / "rvc"
RVC_PYTHON = ROOT / ".rvc-venv" / "Scripts" / "python.exe"
RVC_PACKAGE = ROOT / ".rvc-venv" / "Lib" / "site-packages" / "rvc_python"


def configure_runtime() -> None:
    onnxruntime.set_default_logger_severity(3)
    available = onnxruntime.get_available_providers()
    preferred = [
        provider
        for provider in (
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        )
        if provider in available
    ]
    globals_.execution_providers = preferred or ["CPUExecutionProvider"]
    globals_.execution_threads = 2 if "CUDAExecutionProvider" in preferred else max(2, (os.cpu_count() or 4) - 2)
    globals_.frame_processors = ["face_swapper"]
    globals_.many_faces = False
    globals_.mouth_mask = False
    globals_.mouth_mask_size = 0.0
    globals_.opacity = 1.0
    globals_.sharpness = 0.0
    globals_.poisson_blend = False
    globals_.enable_interpolation = False


def _find_index(model_path: Path) -> Path | None:
    direct = model_path.with_suffix(".index")
    if direct.is_file():
        return direct

    sibling_indexes = sorted(model_path.parent.glob("*.index"))
    if len(sibling_indexes) == 1:
        return sibling_indexes[0]
    for index in sibling_indexes:
        if model_path.stem.lower() in index.stem.lower():
            return index

    if RVC_MODELS_DIR.is_dir():
        all_indexes = sorted(RVC_MODELS_DIR.rglob("*.index"))
        for index in all_indexes:
            if model_path.stem.lower() in index.stem.lower():
                return index
    return None


def discover_voice_models() -> list[dict[str, str | None]]:
    if not RVC_MODELS_DIR.is_dir():
        return []

    # Modelos duplicados na raiz e em subpastas aparecem apenas uma vez.
    selected: dict[str, Path] = {}
    for model in sorted(
        RVC_MODELS_DIR.rglob("*.pth"),
        key=lambda path: (len(path.relative_to(RVC_MODELS_DIR).parts), str(path).lower()),
    ):
        selected.setdefault(model.stem.casefold(), model)

    models = []
    for model in selected.values():
        index = _find_index(model)
        models.append(
            {
                "name": model.stem,
                "model": str(model.resolve()),
                "index": str(index.resolve()) if index else None,
            }
        )
    return models


def rvc_runtime_ready() -> bool:
    return RVC_PYTHON.is_file() and RVC_PACKAGE.is_dir()


class AppSignals(QObject):
    frame = Signal(QImage)
    status = Signal(str)
    camera_state = Signal(bool)
    recording_state = Signal(bool)
    face_state = Signal(bool, str)
    saved = Signal(bool, str)


class PreviewWindow(QMainWindow):
    closed_by_user = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Câmera — Avatar")
        self.resize(1000, 620)
        self._programmatic_close = False

        self.preview = QLabel("Abrindo câmera...")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(640, 360)
        self.preview.setStyleSheet("background:#111; color:#bbb;")
        self.setCentralWidget(self.preview)

    def show_frame(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        self.preview.setPixmap(
            pixmap.scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def close_from_app(self) -> None:
        self._programmatic_close = True
        self.close()
        self._programmatic_close = False

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._programmatic_close:
            self.closed_by_user.emit()
        event.accept()


class CameraPipeline:
    def __init__(self, signals: AppSignals) -> None:
        self.signals = signals
        self.running = False
        self.mirror = True
        self._thread: threading.Thread | None = None
        self._capture: VideoCapturer | None = None
        self._source_face = None
        self._face_lock = threading.Lock()
        self._load_generation = 0
        self._enhancer_generation = 0
        self._enhancer_module = None
        self._enhancer_lock = threading.Lock()
        self._recorder: Recorder | None = None
        self._recorder_lock = threading.Lock()
        self._soundtrack: str | None = None
        self._voice_model: dict[str, str | None] | None = None
        self._voice_pitch = 0
        self._last_swap_error = 0.0

    def load_face(self, image_path: str) -> None:
        self._load_generation += 1
        generation = self._load_generation
        self.signals.status.emit("Carregando modelos e analisando a foto...")

        def work() -> None:
            try:
                image = imread_unicode(image_path)
                if image is None:
                    raise RuntimeError("Não foi possível abrir a imagem.")
                get_face_analyser()
                source_face = get_one_face(image)
                if source_face is None:
                    raise RuntimeError("Nenhum rosto foi encontrado na foto.")
                if get_face_swapper() is None:
                    raise RuntimeError("O modelo inswapper não pôde ser carregado.")
                if generation != self._load_generation:
                    return
                with self._face_lock:
                    self._source_face = source_face
                self.signals.face_state.emit(True, Path(image_path).name)
                self.signals.status.emit("Foto pronta. O rosto será aplicado à câmera.")
            except Exception as exc:
                if generation == self._load_generation:
                    with self._face_lock:
                        self._source_face = None
                    self.signals.face_state.emit(False, str(exc))
                    self.signals.status.emit(f"Erro na foto: {exc}")

        threading.Thread(target=work, daemon=True).start()

    def set_enhancer(self, enhancer_name: str) -> None:
        self._enhancer_generation += 1
        generation = self._enhancer_generation
        if enhancer_name == "none":
            with self._enhancer_lock:
                self._enhancer_module = None
            self.signals.status.emit("Enhancer desativado.")
            return

        self.signals.status.emit(f"Carregando {enhancer_name.upper()}...")

        def work() -> None:
            try:
                if enhancer_name == "gpen256":
                    from modules.processors.frame import face_enhancer_gpen256 as module
                elif enhancer_name == "gpen512":
                    from modules.processors.frame import face_enhancer_gpen512 as module
                else:
                    raise ValueError(f"Enhancer desconhecido: {enhancer_name}")
                module.get_enhancer()
                if generation != self._enhancer_generation:
                    return
                with self._enhancer_lock:
                    self._enhancer_module = module
                self.signals.status.emit(f"{enhancer_name.upper()} pronto.")
            except Exception as exc:
                if generation == self._enhancer_generation:
                    with self._enhancer_lock:
                        self._enhancer_module = None
                    self.signals.status.emit(f"Erro ao carregar enhancer: {exc}")

        threading.Thread(target=work, daemon=True).start()

    def start_camera(self, camera_index: int) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(
            target=self._camera_loop,
            args=(camera_index,),
            daemon=True,
        )
        self._thread.start()

    def _camera_loop(self, camera_index: int) -> None:
        try:
            self.signals.status.emit("Abrindo câmera...")
            self._capture = VideoCapturer(camera_index)
            if not self._capture.start(FRAME_WIDTH, FRAME_HEIGHT, 30):
                raise RuntimeError("Não foi possível abrir a câmera selecionada.")
            self.signals.camera_state.emit(True)
            self.signals.status.emit("Câmera ativa.")

            while self.running:
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue

                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                with self._face_lock:
                    source_face = self._source_face
                with self._enhancer_lock:
                    enhancer_module = self._enhancer_module

                if source_face is not None or enhancer_module is not None:
                    try:
                        target_face = detect_one_face_fast(frame)
                        if target_face is not None:
                            if source_face is not None:
                                if globals_.mouth_mask:
                                    ensure_landmarks(frame, target_face)
                                frame = swap_face(source_face, target_face, frame)
                            if enhancer_module is not None:
                                frame = enhancer_module.process_frame(
                                    None,
                                    frame,
                                    detected_faces=[target_face],
                                )
                            if source_face is not None:
                                frame = apply_post_processing(
                                    frame,
                                    [target_face.bbox.astype(int)],
                                )
                    except Exception as exc:
                        now = time.monotonic()
                        if now - self._last_swap_error > 5:
                            self.signals.status.emit(f"Face swap temporariamente indisponível: {exc}")
                            self._last_swap_error = now

                if self.mirror:
                    frame = cv2.flip(frame, 1)

                with self._recorder_lock:
                    recorder = self._recorder
                if recorder is not None:
                    recorder.write_frame(frame)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = QImage(
                    rgb.data,
                    rgb.shape[1],
                    rgb.shape[0],
                    rgb.strides[0],
                    QImage.Format.Format_RGB888,
                ).copy()
                self.signals.frame.emit(image)
        except Exception as exc:
            self.signals.status.emit(f"Erro da câmera: {exc}")
        finally:
            if self._capture is not None:
                self._capture.release()
                self._capture = None
            self.running = False
            self.signals.camera_state.emit(False)

    def stop_camera(self) -> None:
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def start_recording(
        self,
        output_path: str,
        microphone_index: int | None,
        soundtrack: str | None,
        voice_model: dict[str, str | None] | None,
        voice_pitch: int,
    ) -> bool:
        if not self.running:
            self.signals.status.emit("Abra a câmera antes de gravar.")
            return False
        with self._recorder_lock:
            if self._recorder is not None:
                return False
            recorder = Recorder(
                output_path=output_path,
                fps=RECORD_FPS,
                width=FRAME_WIDTH,
                height=FRAME_HEIGHT,
                input_device_index=microphone_index,
            )
            if not recorder.start():
                self.signals.status.emit("Não foi possível iniciar a gravação.")
                return False
            self._recorder = recorder
            self._soundtrack = soundtrack
            self._voice_model = voice_model
            self._voice_pitch = voice_pitch
        self.signals.recording_state.emit(True)
        if voice_model:
            self.signals.status.emit(
                f"Gravando... A voz {voice_model['name']} será aplicada ao salvar."
            )
        else:
            self.signals.status.emit("Gravando...")
        return True

    def stop_recording(self) -> None:
        with self._recorder_lock:
            recorder = self._recorder
            soundtrack = self._soundtrack
            voice_model = self._voice_model
            voice_pitch = self._voice_pitch
            self._recorder = None
            self._soundtrack = None
            self._voice_model = None
            self._voice_pitch = 0
        if recorder is None:
            return
        self.signals.recording_state.emit(False)
        self.signals.status.emit("Finalizando o MP4...")

        def finalize() -> None:
            success, message = recorder.stop(
                soundtrack_path=soundtrack,
                voice_model_path=str(voice_model["model"]) if voice_model else None,
                voice_index_path=str(voice_model["index"]) if voice_model and voice_model.get("index") else None,
                voice_pitch=voice_pitch,
                status_callback=self.signals.status.emit,
            )
            self.signals.saved.emit(success, message)
            self.signals.status.emit(message)

        # Não daemon: ao fechar a janela durante a finalização, o Python
        # aguarda o FFmpeg terminar e não deixa um MP4 corrompido.
        threading.Thread(target=finalize, daemon=False).start()

    def close(self) -> None:
        self.stop_recording()
        self.stop_camera()


def list_cameras() -> list[tuple[int, str]]:
    if sys.platform == "win32":
        try:
            from pygrabber.dshow_graph import FilterGraph

            return list(enumerate(FilterGraph().get_input_devices()))
        except Exception:
            pass
    cameras: list[tuple[int, str]] = []
    for index in range(6):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cameras.append((index, f"Câmera {index}"))
        cap.release()
    return cameras


def list_microphones() -> list[tuple[int, str]]:
    audio = pyaudio.PyAudio()
    try:
        host_names = {
            index: str(audio.get_host_api_info_by_index(index).get("name", ""))
            for index in range(audio.get_host_api_count())
        }
        candidates: list[tuple[int, int, str]] = []
        for index in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(index)
            if int(info.get("maxInputChannels", 0)) <= 0:
                continue

            name = str(info.get("name", f"Microfone {index}")).strip()
            host_name = host_names.get(int(info.get("hostApi", -1)), "")
            lowered = name.casefold()

            # WDM-KS expõe pinos internos, saídas, MIDI e várias cópias do
            # mesmo hardware. WASAPI fornece a lista mais limpa no Windows.
            if host_name == "Windows WDM-KS":
                continue
            if "sound mapper" in lowered or "primary sound capture" in lowered:
                continue

            priority = {
                "Windows WASAPI": 0,
                "Windows DirectSound": 1,
                "MME": 2,
            }.get(host_name, 3)
            candidates.append((priority, index, name))

        # Se WASAPI estiver disponível, usar somente essa API evita que cada
        # microfone apareça novamente via DirectSound e MME.
        if any(priority == 0 for priority, _, _ in candidates):
            candidates = [item for item in candidates if item[0] == 0]

        devices: list[tuple[int, str]] = []
        seen: set[str] = set()
        for _, index, name in sorted(candidates):
            key = " ".join(name.casefold().split())
            if key in seen:
                continue
            seen.add(key)
            devices.append((index, name))
        return devices
    finally:
        audio.terminate()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(900, 620)

        self.signals = AppSignals()
        self.pipeline = CameraPipeline(self.signals)
        self.preview_window = PreviewWindow()
        self.photo_path: str | None = None
        self.audio_path: str | None = None
        self.recording = False

        self.photo_label = QLabel("Nenhuma foto selecionada")
        self.audio_label = QLabel("Nenhum áudio selecionado")
        self.status_label = QLabel("Pronto.")
        self.status_label.setWordWrap(True)

        self.camera_combo = QComboBox()
        for index, name in list_cameras():
            self.camera_combo.addItem(name, index)

        self.microphone_combo = QComboBox()
        self.microphone_combo.addItem("Sem microfone", -1)
        for index, name in list_microphones():
            self.microphone_combo.addItem(name, index)

        self.voice_combo = QComboBox()
        self.voice_refresh_button = QPushButton("Atualizar vozes")
        self.voice_pitch = QSpinBox()
        self.voice_pitch.setRange(-24, 24)
        self.voice_pitch.setValue(0)
        self.voice_pitch.setSuffix(" semitons")
        self.voice_runtime_label = QLabel()
        self.refresh_voice_models()

        self.enhancer_combo = QComboBox()
        self.enhancer_combo.addItem("Sem enhancer", "none")
        self.enhancer_combo.addItem("GPEN 256 (mais rápido)", "gpen256")
        self.enhancer_combo.addItem("GPEN 512 (mais qualidade)", "gpen512")

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_value = QLabel("100%")

        self.sharpness_slider = QSlider(Qt.Orientation.Horizontal)
        self.sharpness_slider.setRange(0, 50)
        self.sharpness_slider.setValue(0)
        self.sharpness_value = QLabel("0.0")

        self.mouth_slider = QSlider(Qt.Orientation.Horizontal)
        self.mouth_slider.setRange(0, 100)
        self.mouth_slider.setValue(0)
        self.mouth_value = QLabel("0%")

        self.poisson_check = QCheckBox("Poisson blend")

        self.photo_button = QPushButton("1. Escolher foto")
        self.audio_button = QPushButton("2. Escolher áudio (opcional)")
        self.camera_button = QPushButton("3. Abrir câmera")
        self.record_button = QPushButton("4. Iniciar gravação")
        self.record_button.setEnabled(False)

        self._build_layout()
        self._connect_signals()

    def _build_layout(self) -> None:
        source_box = QGroupBox("Arquivos")
        source_layout = QGridLayout(source_box)
        source_layout.addWidget(self.photo_button, 0, 0)
        source_layout.addWidget(self.photo_label, 0, 1)
        source_layout.addWidget(self.audio_button, 1, 0)
        source_layout.addWidget(self.audio_label, 1, 1)

        device_box = QGroupBox("Dispositivos")
        device_layout = QGridLayout(device_box)
        device_layout.addWidget(QLabel("Câmera:"), 0, 0)
        device_layout.addWidget(self.camera_combo, 0, 1)
        device_layout.addWidget(QLabel("Microfone:"), 1, 0)
        device_layout.addWidget(self.microphone_combo, 1, 1)
        device_layout.addWidget(QLabel("Modelo de voz:"), 2, 0)
        device_layout.addWidget(self.voice_combo, 2, 1)
        device_layout.addWidget(self.voice_refresh_button, 2, 2)
        device_layout.addWidget(QLabel("Pitch RVC:"), 3, 0)
        device_layout.addWidget(self.voice_pitch, 3, 1)
        device_layout.addWidget(self.voice_runtime_label, 4, 0, 1, 3)

        face_box = QGroupBox("Ajustes do rosto")
        face_layout = QGridLayout(face_box)
        face_layout.addWidget(QLabel("Enhancer:"), 0, 0)
        face_layout.addWidget(self.enhancer_combo, 0, 1, 1, 2)
        face_layout.addWidget(QLabel("Transparência:"), 1, 0)
        face_layout.addWidget(self.opacity_slider, 1, 1)
        face_layout.addWidget(self.opacity_value, 1, 2)
        face_layout.addWidget(QLabel("Sharpness:"), 2, 0)
        face_layout.addWidget(self.sharpness_slider, 2, 1)
        face_layout.addWidget(self.sharpness_value, 2, 2)
        face_layout.addWidget(QLabel("Mouth mask:"), 3, 0)
        face_layout.addWidget(self.mouth_slider, 3, 1)
        face_layout.addWidget(self.mouth_value, 3, 2)
        face_layout.addWidget(self.poisson_check, 4, 1)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.camera_button)
        button_layout.addWidget(self.record_button)

        layout = QVBoxLayout()
        layout.addWidget(source_box)
        layout.addWidget(device_box)
        layout.addWidget(face_box)
        layout.addLayout(button_layout)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def _connect_signals(self) -> None:
        self.photo_button.clicked.connect(self.choose_photo)
        self.audio_button.clicked.connect(self.choose_audio)
        self.voice_refresh_button.clicked.connect(self.refresh_voice_models)
        self.camera_button.clicked.connect(self.toggle_camera)
        self.record_button.clicked.connect(self.toggle_recording)
        self.enhancer_combo.currentIndexChanged.connect(self.change_enhancer)
        self.opacity_slider.valueChanged.connect(self.change_opacity)
        self.sharpness_slider.valueChanged.connect(self.change_sharpness)
        self.mouth_slider.valueChanged.connect(self.change_mouth)
        self.mouth_slider.sliderPressed.connect(self.show_mouth_box)
        self.mouth_slider.sliderReleased.connect(self.hide_mouth_box)
        self.poisson_check.toggled.connect(self.change_poisson)
        self.signals.frame.connect(self.preview_window.show_frame)
        self.signals.status.connect(self.status_label.setText)
        self.signals.camera_state.connect(self.set_camera_state)
        self.signals.recording_state.connect(self.set_recording_state)
        self.signals.face_state.connect(self.set_face_state)
        self.signals.saved.connect(self.recording_finished)
        self.preview_window.closed_by_user.connect(self.close_camera_from_preview)

    def refresh_voice_models(self, _checked: bool = False) -> None:
        current = self.voice_combo.currentText() if self.voice_combo.count() else ""
        self.voice_combo.clear()
        self.voice_combo.addItem("Sem conversão de voz", None)
        for model in discover_voice_models():
            index_note = " + index" if model.get("index") else " (sem index)"
            self.voice_combo.addItem(f"{model['name']}{index_note}", model)

        match = self.voice_combo.findText(current)
        if match >= 0:
            self.voice_combo.setCurrentIndex(match)

        if rvc_runtime_ready():
            self.voice_runtime_label.setText(
                f"RVC pronto — {self.voice_combo.count() - 1} modelo(s). "
                "A conversão é aplicada depois que a gravação parar."
            )
        else:
            self.voice_runtime_label.setText(
                "RVC não instalado. Execute: "
                "powershell -ExecutionPolicy Bypass -File .\\setup_rvc_runtime.ps1"
            )

    def change_enhancer(self, _index: int = 0) -> None:
        self.pipeline.set_enhancer(str(self.enhancer_combo.currentData()))

    def change_opacity(self, value: int) -> None:
        globals_.opacity = value / 100.0
        self.opacity_value.setText(f"{value}%")

    def change_sharpness(self, value: int) -> None:
        globals_.sharpness = value / 10.0
        self.sharpness_value.setText(f"{value / 10.0:.1f}")

    def change_mouth(self, value: int) -> None:
        globals_.mouth_mask_size = float(value)
        globals_.mouth_mask = value > 0
        if value == 0:
            globals_.show_mouth_mask_box = False
        self.mouth_value.setText(f"{value}%")

    def show_mouth_box(self) -> None:
        globals_.show_mouth_mask_box = globals_.mouth_mask_size > 0

    def hide_mouth_box(self) -> None:
        globals_.show_mouth_mask_box = False

    def change_poisson(self, checked: bool) -> None:
        globals_.poisson_blend = checked

    def choose_photo(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Escolher foto",
            "",
            "Imagens (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if path:
            self.photo_path = path
            self.photo_label.setText(f"Analisando: {Path(path).name}")
            self.pipeline.load_face(path)

    def choose_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Escolher áudio",
            "",
            "Áudio (*.mp3 *.wav *.m4a *.aac *.ogg *.flac)",
        )
        if path:
            self.audio_path = path
            self.audio_label.setText(f"{Path(path).name} (trilha a 30%)")

    def toggle_camera(self) -> None:
        if self.pipeline.running:
            if self.recording:
                self.pipeline.stop_recording()
            self.pipeline.stop_camera()
            return
        if self.camera_combo.currentIndex() < 0:
            QMessageBox.warning(self, APP_TITLE, "Nenhuma câmera foi encontrada.")
            return
        camera_index = int(self.camera_combo.currentData())
        self.pipeline.start_camera(camera_index)

    def toggle_recording(self) -> None:
        if self.recording:
            self.record_button.setEnabled(False)
            self.pipeline.stop_recording()
            return
        output_dir = ROOT / "recordings"
        output_dir.mkdir(exist_ok=True)
        output = output_dir / time.strftime("avatar_%Y%m%d_%H%M%S.mp4")
        microphone = int(self.microphone_combo.currentData())
        voice_model = self.voice_combo.currentData()
        if voice_model is not None and microphone == -1:
            QMessageBox.warning(
                self,
                APP_TITLE,
                "Selecione um microfone para usar o modelo de voz.",
            )
            return
        if voice_model is not None and not rvc_runtime_ready():
            QMessageBox.critical(
                self,
                APP_TITLE,
                "Runtime RVC ausente. Execute no PowerShell:\n\n"
                "powershell -ExecutionPolicy Bypass -File .\\setup_rvc_runtime.ps1",
            )
            return
        self.pipeline.start_recording(
            str(output),
            microphone,
            self.audio_path,
            voice_model,
            self.voice_pitch.value(),
        )

    def close_camera_from_preview(self) -> None:
        if self.recording:
            self.pipeline.stop_recording()
        if self.pipeline.running:
            self.pipeline.stop_camera()

    def set_camera_state(self, active: bool) -> None:
        self.camera_button.setText("Fechar câmera" if active else "3. Abrir câmera")
        self.camera_combo.setEnabled(not active)
        self.record_button.setEnabled(active)
        if active:
            self.preview_window.show()
            self.preview_window.raise_()
            self.preview_window.activateWindow()
        elif self.preview_window.isVisible():
            self.preview_window.close_from_app()

    def set_recording_state(self, active: bool) -> None:
        self.recording = active
        self.record_button.setText("Parar e salvar" if active else "4. Iniciar gravação")
        self.camera_button.setEnabled(not active)
        self.microphone_combo.setEnabled(not active)
        self.audio_button.setEnabled(not active)
        self.voice_combo.setEnabled(not active)
        self.voice_refresh_button.setEnabled(not active)
        self.voice_pitch.setEnabled(not active)

    def set_face_state(self, success: bool, message: str) -> None:
        if success:
            self.photo_label.setText(f"{message} — rosto pronto")
        else:
            self.photo_label.setText("Foto inválida")
            QMessageBox.warning(self, APP_TITLE, message)

    def recording_finished(self, success: bool, message: str) -> None:
        self.record_button.setEnabled(self.pipeline.running)
        if success:
            QMessageBox.information(self, APP_TITLE, message)
        else:
            QMessageBox.critical(self, APP_TITLE, message)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.preview_window.close_from_app()
        self.pipeline.close()
        event.accept()


def main() -> int:
    configure_runtime()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
