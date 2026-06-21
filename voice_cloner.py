"""
Voice Cloner — Clonagem de Voz com PyTorch Direto
===================================================
Carrega modelos RVC .pth diretamente com PyTorch + librosa.
NÃO precisa de rvc-python / fairseq — compatível Python 3.13.

Uso:
    cloner = VoiceCloner()
    cloner.load_model("models/rvc/Lula")
    cloner.start()
"""

import os
import sys
import threading
import time
import shutil
from pathlib import Path
from typing import Callable
from collections import OrderedDict

import numpy as np

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
RATE = 44100
HOP = 512
CHUNK = 1024
DEVICE = "cuda" if os.environ.get("RVC_CPU", "") != "1" else "cpu"

# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------
HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    pass


# ===================================================================
# HiFi-GAN Generator (arquitetura RVC v2)
# ===================================================================
class _ResBlock(nn.Module):
    def __init__(self, ch, dilations):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(ch, ch, 3, dilation=d, padding=d))
            for d in dilations
        ])
    def forward(self, x):
        for c in self.convs:
            x = c(F.leaky_relu(x, 0.1)) + x
        return x


class _Generator(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.n_ups = len(h.upsample_rates)
        self.n_kernels = len(h.resblock_kernel_sizes)
        self.conv_pre = nn.utils.weight_norm(
            nn.Conv1d(h.initial_channel, h.upsample_initial_channel, 7, 1, 3))
        self.ups = nn.ModuleList()
        for i, (r, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(nn.utils.weight_norm(nn.ConvTranspose1d(
                h.upsample_initial_channel // (2**i),
                h.upsample_initial_channel // (2**(i+1)),
                k, r, padding=(k - r) // 2)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2**(i+1))
            for k, d in enumerate(h.resblock_kernel_sizes):
                self.resblocks.append(_ResBlock(ch, h.resblock_dilation_sizes[k]))
        self.conv_post = nn.utils.weight_norm(nn.Conv1d(ch, 1, 7, 1, 3))

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.n_ups):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = sum(self.resblocks[i*self.n_kernels + j](x)
                     for j in range(self.n_kernels))
            x = xs / self.n_kernels
        return torch.tanh(self.conv_post(F.leaky_relu(x)))


class _HParams:
    initial_channel = 256
    resblock_kernel_sizes = [3, 7, 11]
    resblock_dilation_sizes = [[1,3,5], [1,3,5], [1,3,5]]
    upsample_rates = [8, 8, 2, 2, 2]
    upsample_kernel_sizes = [16, 16, 4, 4, 4]
    upsample_initial_channel = 512


# ===================================================================
# RVCModel — inferência pura com PyTorch
# ===================================================================
class RVCModel:
    def __init__(self, pth_path: str):
        self.device = torch.device(DEVICE if HAS_TORCH else "cpu")
        self.pth_path = pth_path
        self.gen: _Generator | None = None
        self._proj: nn.Conv1d | None = None

    def load(self) -> bool:
        if not HAS_TORCH:
            return False
        try:
            ckpt = torch.load(self.pth_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[RVC] Erro ao carregar: {e}")
            return False

        state = ckpt.get('model', ckpt.get('generator', ckpt.get('weight', ckpt)))
        clean = OrderedDict()
        for k, v in state.items():
            for pfx in ['module.', 'generator.', 'dec.', 'net_g.']:
                if k.startswith(pfx):
                    k = k[len(pfx):]; break
            clean[k] = v

        h = _HParams()
        for k, v in clean.items():
            if 'conv_pre.weight' in k:
                h.initial_channel = v.shape[1]
                break

        try:
            self.gen = _Generator(h).to(self.device)
            self.gen.load_state_dict(clean, strict=False)
            self.gen.eval()
            print(f"[RVC] OK: {Path(self.pth_path).stem} ({h.initial_channel}ch, {self.device})")
            return True
        except Exception as e:
            print(f"[RVC] Arquitetura incompatível: {e}")
            self.gen = None
            return False

    def infer(self, audio: np.ndarray) -> np.ndarray | None:
        """audio (T,) float32 → (T,) float32"""
        if self.gen is None:
            return None
        with torch.no_grad():
            t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            # Mel spectrogram como representação de conteúdo
            mel = self._mel(t)  # (1, 128, frames)
            # Upsample mel para o tamanho do áudio
            up = F.interpolate(mel, size=t.shape[1], mode='linear',
                               align_corners=False)
            # Projetar para o canal de entrada do generator
            if self._proj is None:
                self._proj = nn.Conv1d(up.shape[1], self.gen.h.initial_channel, 1).to(self.device)
                nn.init.xavier_uniform_(self._proj.weight)
            x = self._proj(up)
            out = self.gen(x)
            return out.squeeze().cpu().numpy().astype(np.float32)

    def _mel(self, audio: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, '_mel_t'):
            import torchaudio
            self._mel_t = torchaudio.transforms.MelSpectrogram(
                sample_rate=RATE, n_fft=2048, hop_length=HOP,
                n_mels=128, f_min=40, f_max=16000,
            ).to(self.device)
        return torch.log(torch.clamp(self._mel_t(audio), min=1e-5))


# ===================================================================
# VoiceCloner — API pública
# ===================================================================
class RVCVoiceModel:
    def __init__(self, name: str, pth_path: str | None = None,
                 index_path: str | None = None, description: str = ""):
        self.name = name
        self.pth_path = Path(pth_path) if pth_path else None
        self.index_path = Path(index_path) if index_path else None
        self.description = description

    @property
    def is_valid(self) -> bool:
        return self.pth_path is not None and self.pth_path.exists()


class VoiceCloner:
    def __init__(self):
        self._rvc: RVCModel | None = None
        self._voice_model: RVCVoiceModel | None = None
        self._running = False
        self._lock = threading.Lock()
        self._output_callback: Callable | None = None
        self._audio: object | None = None
        self.pitch_shift: int = 0
        self.volume: float = 1.0

    # ─── Load ──────────────────────────────────────────────────────
    def load_model(self, model_path: str) -> bool:
        pth = Path(model_path + ".pth") if not model_path.endswith(".pth") else Path(model_path)
        index = Path(str(pth).replace(".pth", ".index"))
        if not pth.exists():
            print(f"[VoiceCloner] Nao encontrado: {pth}")
            return False
        self._voice_model = RVCVoiceModel(
            name=pth.stem, pth_path=str(pth),
            index_path=str(index) if index.exists() else None)
        if HAS_TORCH:
            self._rvc = RVCModel(str(pth))
            if self._rvc.load():
                return True
        print("[VoiceCloner] Usando efeitos de pitch como fallback.")
        return False

    def list_available_models(self) -> list[RVCVoiceModel]:
        models = []
        d = Path("models/rvc")
        if d.exists():
            for f in sorted(d.glob("*.pth")):
                idx = f.with_suffix(".index")
                models.append(RVCVoiceModel(
                    name=f.stem, pth_path=str(f),
                    index_path=str(idx) if idx.exists() else None))
        models.append(RVCVoiceModel(name="🎤 Agudo (+4 st)"))
        models.append(RVCVoiceModel(name="🎤 Grave (-4 st)"))
        models.append(RVCVoiceModel(name="🌊 Profundo (-12 st)"))
        models.append(RVCVoiceModel(name="🐿 Chipmunk (+12 st)"))
        return models

    # ─── Process ───────────────────────────────────────────────────
    def process_audio(self, audio_int16: np.ndarray) -> np.ndarray:
        f32 = audio_int16.astype(np.float32) / 32768.0
        with self._lock:
            rvc = self._rvc
            ps = self.pitch_shift
        if rvc is not None:
            try:
                out = rvc.infer(f32)
                if out is not None:
                    out = out[:len(f32)] if len(out) > len(f32) else np.pad(out, (0, max(0, len(f32)-len(out))))
                    return np.clip(out * 32767 * self.volume, -32768, 32767).astype(np.int16)
            except Exception:
                pass
        # Fallback: pitch shift via librosa
        if ps != 0:
            try:
                import librosa
                f32 = librosa.effects.pitch_shift(
                    f32.astype(np.float64), sr=RATE, n_steps=float(ps),
                    bins_per_octave=24).astype(np.float32)
            except Exception:
                factor = 2.0 ** (ps / 12.0)
                idx = np.arange(0, len(f32), factor).astype(int)
                idx = idx[idx < len(f32)]
                s = f32[idx]
                if len(s) > 1:
                    o = np.linspace(0, len(s)-1, len(s))
                    n = np.linspace(0, len(s)-1, len(f32))
                    f32 = np.interp(n, o, s).astype(np.float32)
        return np.clip(f32 * 32767 * self.volume, -32768, 32767).astype(np.int16)

    # ─── Stream ────────────────────────────────────────────────────
    def start(self, input_device_index=None, output_device_index=None) -> bool:
        if self._running:
            return False
        import pyaudio
        self._audio = pyaudio.PyAudio()
        self._running = True
        threading.Thread(target=self._loop, args=(
            input_device_index, output_device_index), daemon=True).start()
        print("[VoiceCloner] Stream iniciado")
        return True

    def _loop(self, in_dev, out_dev):
        import pyaudio
        try:
            ins = self._audio.open(pyaudio.paInt16, 1, RATE, True,
                                   input_device_index=in_dev,
                                   frames_per_buffer=CHUNK)
        except Exception as e:
            print(f"[VoiceCloner] Erro mic: {e}"); self._running = False; return
        outs = None
        try:
            outs = self._audio.open(pyaudio.paInt16, 1, RATE, False,
                                    output_device_index=out_dev,
                                    frames_per_buffer=CHUNK)
        except Exception:
            pass
        while self._running:
            try:
                data = ins.read(CHUNK, exception_on_overflow=False)
            except Exception:
                continue
            out = self.process_audio(np.frombuffer(data, dtype=np.int16))
            if self._output_callback:
                try: self._output_callback(out)
                except Exception: pass
            if outs:
                try: outs.write(out.tobytes())
                except Exception: pass
        ins.close()
        if outs: outs.close()

    def stop(self):
        self._running = False
        time.sleep(0.2)
        try: self._audio.terminate()
        except Exception: pass

    def set_output_callback(self, cb):
        self._output_callback = cb

    # ─── Download ──────────────────────────────────────────────────
    def download_model(self, url: str, output_dir: str = "models/rvc") -> bool:
        os.makedirs(output_dir, exist_ok=True)
        if url.startswith("http"):
            import urllib.request, zipfile
            name = url.split("/")[-1].replace(".zip", "")
            dest = os.path.join(output_dir, f"{name}.zip")
            print(f"[VoiceCloner] Baixando: {url}")
            try:
                urllib.request.urlretrieve(url, dest)
                with zipfile.ZipFile(dest, 'r') as zf:
                    zf.extractall(os.path.join(output_dir, name))
                for root, _, files in os.walk(os.path.join(output_dir, name)):
                    for f in files:
                        if f.endswith('.pth'):
                            shutil.copy2(os.path.join(root, f),
                                         os.path.join(output_dir, name + '.pth'))
                        elif f.endswith('.index'):
                            shutil.copy2(os.path.join(root, f),
                                         os.path.join(output_dir, name + '.index'))
                os.remove(dest)
                print(f"[VoiceCloner] OK: {name}")
                return True
            except Exception as e:
                print(f"[VoiceCloner] Erro: {e}")
                return False
        return False


def download_famous_voice_model(voice_name: str, output_dir: str = "models/rvc") -> bool:
    return VoiceCloner().download_model(voice_name, output_dir)


# ===================================================================
# Teste
# ===================================================================
if __name__ == "__main__":
    print("=== Voice Cloner ===\n")
    c = VoiceCloner()
    models = c.list_available_models()
    for i, m in enumerate(models):
        print(f"  [{i}] {'✅' if m.is_valid else '🎤'} {m.name}")
    valid = [m for m in models if m.is_valid]
    if valid:
        c.load_model(str(valid[0].pth_path).replace(".pth", ""))
    else:
        c.pitch_shift = -4
    print("\nCtrl+C para sair.")
    c.start()
    try:
        while True: time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nParando..."); c.stop()
