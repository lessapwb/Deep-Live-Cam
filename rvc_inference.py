"""
RVC Model Inference — Implementação Direta em PyTorch
======================================================
NÃO requer rvc-python, fairseq, ou pyworld.
Usa apenas torch + torchaudio (que já tens instalados).

Arquitetura implementada:
  - ContentVec-style feature extraction via torchaudio HUBERT_BASE
  - VITS SynthesizerTrn (decoder + flow + HiFi-GAN)
  - Streaming-friendly chunked processing
"""

import math
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF


# ===================================================================
# HiFi-GAN Generator (decoder de áudio)
# ===================================================================

class _ResidualBlock1(nn.Module):
    """Residual block type 1 (dilated convolutions)."""
    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(
                channels, channels, kernel_size, 1,
                dilation=d, padding=(kernel_size-1)//2 * d
            ))
            for d in dilations
        ])
        self.convs2 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(
                channels, channels, kernel_size, 1,
                dilation=1, padding=(kernel_size-1)//2
            ))
            for _ in dilations
        ])

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = x + xt
        if x_mask is not None:
            x = x * x_mask
        return x


class _ResidualBlock2(nn.Module):
    """Residual block type 2 (standard convolutions)."""
    def __init__(self, channels, kernel_size=3, dilations=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(
                channels, channels, kernel_size, 1,
                dilation=d, padding=(kernel_size-1)//2 * d
            ))
            for d in dilations
        ])

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, 0.1)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = x + xt
        if x_mask is not None:
            x = x * x_mask
        return x


class _HiFiGANGenerator(nn.Module):
    """HiFi-GAN generator (vocoder)."""
    def __init__(self,
                 initial_channel: int,
                 resblock_kernel_sizes: list,
                 resblock_dilation_sizes: list,
                 resblock_type: str,
                 upsample_rates: list,
                 upsample_initial_channel: int,
                 upsample_kernel_sizes: list,
                 gin_channels: int = 0,
                 hop_length: int = 512,
                 sampling_rate: int = 40000,
                 ):
        super().__init__()
        self.upsample_rates = upsample_rates
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate

        self.num_kernels = len(resblock_kernel_sizes)

        # Pre-conv
        self.conv_pre = nn.utils.weight_norm(nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        ))

        resblock_cls = _ResidualBlock1 if resblock_type == '1' else _ResidualBlock2

        # Upsampling layers
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        total_upsample = 1
        for i, (rate, up_k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            total_upsample *= rate
            in_ch = upsample_initial_channel // (2 ** i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(nn.utils.weight_norm(
                nn.ConvTranspose1d(in_ch, out_ch, up_k, rate, padding=(up_k - rate) // 2)
            ))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock_cls(out_ch, k, d))

        # Post-conv
        self.conv_post = nn.utils.weight_norm(nn.Conv1d(out_ch, 1, 7, 1, padding=3))

        # Speaker conditioning (optional)
        self.gin_channels = gin_channels
        if gin_channels > 0:
            self.cond = nn.utils.weight_norm(nn.Conv1d(gin_channels, upsample_initial_channel, 1))

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None and self.gin_channels > 0:
            g = self.cond(g)
            x = x + g

        for i in range(len(self.ups)):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x


# ===================================================================
# VITS Encoder (Transformer)
# ===================================================================

class _MultiHeadAttention(nn.Module):
    def __init__(self, channels, out_channels, n_heads, p_dropout=0.0):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.d_k = channels // n_heads

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(c)
        k = self.conv_k(x)
        v = self.conv_v(x)

        b, d, t = q.shape
        q = q.view(b, self.n_heads, self.d_k, t).transpose(2, 3)
        k = k.view(b, self.n_heads, self.d_k, t).transpose(2, 3)
        v = v.view(b, self.n_heads, self.d_k, t).transpose(2, 3)

        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.d_k)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e4)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        output = output.transpose(2, 3).contiguous().view(b, d, t)
        output = self.conv_o(output)
        return output


class _FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv1(x * x_mask)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x * x_mask)
        return x * x_mask


class _Encoder(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers,
                 kernel_size=1, p_dropout=0.0):
        super().__init__()
        self.attn_layers = nn.ModuleList([
            _MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout)
            for _ in range(n_layers)
        ])
        self.norm_layers_1 = nn.ModuleList([
            nn.LayerNorm(hidden_channels, eps=1e-6) for _ in range(n_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            _FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout)
            for _ in range(n_layers)
        ])
        self.norm_layers_2 = nn.ModuleList([
            nn.LayerNorm(hidden_channels, eps=1e-6) for _ in range(n_layers)
        ])

    def forward(self, x, x_mask):
        # LayerNorm expects (B, T, C), conv expects (B, C, T)
        for i in range(len(self.attn_layers)):
            # Self-attention
            attn_input = x * x_mask
            y = self.attn_layers[i](attn_input, attn_input)
            y = y.transpose(1, 2)  # (B, C, T) -> (B, T, C)
            y = self.norm_layers_1[i](y)
            y = y.transpose(1, 2)  # (B, T, C) -> (B, C, T)
            x = x + y

            # FFN
            y = self.ffn_layers[i](x, x_mask)
            y = y.transpose(1, 2)
            y = self.norm_layers_2[i](y)
            y = y.transpose(1, 2)
            x = x + y

        x = x * x_mask
        return x


# ===================================================================
# SynthesizerTrn (Modelo Principal RVC)
# ===================================================================

class SynthesizerTrn(nn.Module):
    """
    RVC SynthesizerTrn — VITS-style model for voice conversion.
    Compatível com checkpoints .pth do RVC WebUI.
    """
    def __init__(self,
                 spec_channels: int,
                 inter_channels: int,
                 hidden_channels: int,
                 filter_channels: int,
                 n_heads: int,
                 n_layers: int,
                 kernel_size: int,
                 p_dropout: float,
                 resblock: str,
                 resblock_kernel_sizes: list,
                 resblock_dilation_sizes: list,
                 upsample_rates: list,
                 upsample_initial_channel: int,
                 upsample_kernel_sizes: list,
                 gin_channels: int = 0,
                 hop_length: int = 512,
                 sampling_rate: int = 40000,
                 **kwargs):
        super().__init__()

        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.gin_channels = gin_channels
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate

        # Pre-net
        self.enc_p = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.enc_q = nn.Conv1d(spec_channels, hidden_channels, kernel_size, padding=kernel_size//2)

        # Projection to inter_channels for flow
        self.flow_1 = nn.Conv1d(hidden_channels, inter_channels * 2, kernel_size, padding=kernel_size//2)
        self.flow_2 = nn.Conv1d(inter_channels, inter_channels, kernel_size, padding=kernel_size//2)

        # Encoder que processa as features
        self.encoder = _Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )

        # Decoder (HiFi-GAN)
        self.dec = _HiFiGANGenerator(
            inter_channels,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            resblock,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels,
            hop_length,
            sampling_rate,
        )

    def forward(self, hubert_features, hubert_mask=None):
        """
        Forward pass para voice conversion.
        
        Args:
            hubert_features: (B, T, hidden_channels) — HuBERT content features
            hubert_mask: (B, 1, T) — máscara de padding
        Returns:
            audio: (B, 1, T * hop_length) — áudio sintetizado
        """
        if hubert_mask is None:
            hubert_mask = torch.ones(
                hubert_features.shape[0], 1, hubert_features.shape[1],
                device=hubert_features.device
            )

        # B, T, C -> B, C, T
        x = hubert_features.transpose(1, 2)
        x_mask = hubert_mask

        # Pre-net
        x = self.enc_p(x) * x_mask

        # Encoder (Transformer)
        x = self.encoder(x, x_mask)

        # Flow
        x = F.leaky_relu(self.flow_1(x), 0.1)
        m, logs = x.chunk(2, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        z = self.flow_2(z) * x_mask

        # Decoder (vocoder)
        audio = self.dec(z)

        return audio


# ===================================================================
# RVC Inference Engine
# ===================================================================

# Cache global para o modelo HuBERT
_HUBERT_MODEL = None
_HUBERT_LOCK = None
import threading as _threading


def _get_hubert():
    """Lazy-load do HuBERT do torchaudio."""
    global _HUBERT_MODEL, _HUBERT_LOCK
    if _HUBERT_MODEL is None:
        if _HUBERT_LOCK is None:
            _HUBERT_LOCK = _threading.Lock()
        with _HUBERT_LOCK:
            if _HUBERT_MODEL is None:
                print("[RVC] A carregar HuBERT BASE...")
                bundle = torchaudio.pipelines.HUBERT_BASE
                _HUBERT_MODEL = bundle.get_model()
                _HUBERT_MODEL.eval()
                if torch.cuda.is_available():
                    _HUBERT_MODEL = _HUBERT_MODEL.cuda()
                print("[RVC] HuBERT carregado!")
    return _HUBERT_MODEL


def extract_hubert_features(audio: torch.Tensor, sr: int = 40000) -> torch.Tensor:
    """
    Extrai features de conteúdo do HuBERT.
    
    Args:
        audio: (1, T) tensor de áudio
        sr: sample rate do áudio
    Returns:
        features: (1, T_frames, 768) features do HuBERT
    """
    model = _get_hubert()

    # Resample para 16kHz (HuBERT espera 16kHz)
    if sr != 16000:
        audio = AF.resample(audio, sr, 16000)

    # Garantir que é mono
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    dev = next(model.parameters()).device
    audio = audio.to(dev)

    with torch.no_grad():
        features, _ = model.extract_features(audio)

    # Usar a última camada (camada 12)
    features = features[-1]  # (1, T, 768)

    return features


def _build_model_from_config(config: list) -> SynthesizerTrn:
    """Constrói o SynthesizerTrn a partir da lista de config do RVC."""
    model = SynthesizerTrn(
        spec_channels=config[0],          # 1025
        inter_channels=config[1],         # 32
        hidden_channels=config[2],        # 192
        filter_channels=config[3],        # 192
        n_heads=config[5],               # 2
        n_layers=config[6],              # 6
        kernel_size=config[7],           # 3
        p_dropout=config[8],             # 0
        resblock=config[9],              # '1'
        resblock_kernel_sizes=config[10], # [3,7,11]
        resblock_dilation_sizes=config[11], # [[1,3,5],...]
        upsample_rates=config[12],        # [10,10,2,2]
        upsample_initial_channel=config[13], # 512
        upsample_kernel_sizes=config[14], # [16,16,4,4]
        gin_channels=config[16] if len(config) > 16 else 0,
        hop_length=512,
        sampling_rate=config[17] if len(config) > 17 else 40000,
    )
    return model


class RVCInference:
    """Motor de inferência RVC."""

    def __init__(self):
        self._model: SynthesizerTrn | None = None
        self._hubert_dim = 768
        self._hop_length = 512
        self._sample_rate = 40000
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded_model_name = None

    # ------------------------------------------------------------------
    def load_model(self, pth_path: str) -> bool:
        """
        Carrega um modelo RVC .pth.
        
        Args:
            pth_path: caminho para o ficheiro .pth
        Returns:
            True se carregou com sucesso
        """
        pth = Path(pth_path)
        if not pth.exists():
            print(f"[RVC] ❌ Ficheiro não encontrado: {pth_path}")
            return False

        print(f"[RVC] A carregar: {pth.name}...")
        try:
            checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[RVC] ❌ Erro ao carregar checkpoint: {e}")
            return False

        # Validar
        if "config" not in checkpoint or "weight" not in checkpoint:
            print("[RVC] ❌ Checkpoint inválido (falta 'config' ou 'weight')")
            return False

        # Construir modelo
        config = checkpoint["config"]
        try:
            model = _build_model_from_config(config)
        except Exception as e:
            print(f"[RVC] ❌ Erro ao construir modelo: {e}")
            return False

        # Carregar pesos
        try:
            if "model" in checkpoint["weight"]:
                # Alguns checkpoints aninham em 'model'
                state_dict = checkpoint["weight"]["model"]
            else:
                state_dict = checkpoint["weight"]

            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"[RVC] ❌ Erro ao carregar pesos: {e}")
            return False

        # Mover para device
        model = model.to(self._device)
        model.eval()

        self._model = model
        self._sample_rate = config[17] if len(config) > 17 else 40000
        self._loaded_model_name = pth.stem
        print(f"[RVC] ✅ Modelo '{pth.stem}' carregado (sr={self._sample_rate}Hz, device={self._device})")
        return True

    # ------------------------------------------------------------------
    def convert(self, audio: np.ndarray, pitch_shift: int = 0) -> np.ndarray:
        """
        Converte voz usando o modelo RVC.
        
        Args:
            audio: numpy array float32 [-1, 1] ou int16
            pitch_shift: deslocamento de pitch em semitons (-24 a +24)
        Returns:
            audio convertido (int16 numpy array)
        """
        if self._model is None:
            return audio if audio.dtype == np.int16 else (audio * 32767).astype(np.int16)

        # Converter para tensor
        was_int16 = audio.dtype == np.int16
        if was_int16:
            audio_f = audio.astype(np.float32) / 32768.0
        else:
            audio_f = audio.astype(np.float32)

        # Resample para sample rate do modelo se necessário
        input_sr = 44100  # Assumimos que entra a 44100 Hz
        if input_sr != self._sample_rate and len(audio_f) > 0:
            t = torch.from_numpy(audio_f).float().unsqueeze(0)  # (1, T)
            t = AF.resample(t, input_sr, self._sample_rate)
            audio_f = t.squeeze(0).numpy()

        # Extrair features HuBERT
        if len(audio_f) < 16000 * 0.05:  # menos de 50ms
            return audio.astype(np.int16) if was_int16 else audio

        audio_tensor = torch.from_numpy(audio_f).float().unsqueeze(0)  # (1, T)
        hubert_feat = extract_hubert_features(audio_tensor, self._sample_rate)
        # hubert_feat: (1, T_hubert, 768)

        # Mover para device
        hubert_feat = hubert_feat.to(self._device)

        # Criar máscara
        mask = torch.ones(1, 1, hubert_feat.shape[1], device=self._device)

        # Inferência
        with torch.no_grad():
            output = self._model(hubert_feat, mask)

        # output: (1, 1, T_samples)
        output_audio = output.squeeze().cpu().numpy()

        # Resample de volta para 44100 Hz se necessário
        if self._sample_rate != 44100 and len(output_audio) > 0:
            t = torch.from_numpy(output_audio).float().unsqueeze(0)
            t = AF.resample(t, self._sample_rate, 44100)
            output_audio = t.squeeze(0).numpy()

        # Ajustar comprimento
        target_len = len(audio)
        if len(output_audio) > target_len:
            output_audio = output_audio[:target_len]
        elif len(output_audio) < target_len:
            output_audio = np.pad(output_audio, (0, target_len - len(output_audio)))

        # Normalizar e converter para int16
        peak = np.max(np.abs(output_audio)) + 1e-8
        output_audio = output_audio / peak * 0.95
        output_audio = np.clip(output_audio * 32767, -32768, 32767).astype(np.int16)

        return output_audio

    # ------------------------------------------------------------------
    def warmup(self):
        """Pré-aquece o modelo com um batch vazio."""

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self._loaded_model_name or "none"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate


# ===================================================================
# Conversor otimizado para streaming em tempo real
# ===================================================================

class StreamingRVCConverter:
    """
    Conversor RVC otimizado para streaming em tempo real.
    
    Usa buffering + overlap para processamento contínuo:
      - Acumula ~1s de áudio
      - Processa com overlap de 50%
      - Crossfade entre chunks para evitar artefatos
    """

    def __init__(self, rvc: RVCInference, buffer_seconds: float = 0.8):
        self._rvc = rvc
        self._buffer_samples = int(buffer_seconds * 44100)
        self._overlap_samples = int(0.1 * 44100)  # 100ms overlap
        self._buffer = np.array([], dtype=np.float32)
        self._prev_output = np.array([], dtype=np.float32)
        self._hop = self._buffer_samples // 2

    def process_chunk(self, audio_int16: np.ndarray) -> np.ndarray:
        """
        Processa um chunk de áudio em streaming.
        
        Args:
            audio_int16: chunk de entrada (int16)
        Returns:
            chunk processado de saída (int16)
        """
        if not self._rvc.is_loaded:
            return audio_int16

        audio_f = audio_int16.astype(np.float32) / 32768.0
        self._buffer = np.concatenate([self._buffer, audio_f])

        if len(self._buffer) < self._buffer_samples:
            return audio_int16  # Ainda a acumular

        # Processar chunk
        chunk = self._buffer[:self._buffer_samples]
        result_f = self._rvc.convert(chunk)

        # Crossfade com chunk anterior
        if len(self._prev_output) > 0:
            fade_len = min(self._overlap_samples, len(self._prev_output), len(result_f))
            if fade_len > 0:
                fade_in = np.linspace(0, 1, fade_len)
                fade_out = np.linspace(1, 0, fade_len)
                result_f[:fade_len] = (
                    self._prev_output[-fade_len:] * fade_out +
                    result_f[:fade_len] * fade_in
                )

        # Avançar buffer
        output_len = min(len(audio_int16), self._hop)
        output = result_f[:output_len]
        self._prev_output = result_f

        self._buffer = self._buffer[self._hop:]

        return (np.clip(output * 32767, -32768, 32767)).astype(np.int16)

    def reset(self):
        """Limpa buffers."""
        self._buffer = np.array([], dtype=np.float32)
        self._prev_output = np.array([], dtype=np.float32)


# ===================================================================
# Teste rápido
# ===================================================================
if __name__ == "__main__":
    import sys
    import pyaudio

    print("=== RVC Inference - Teste Local ===")
    print("Dispositivo:", "CUDA" if torch.cuda.is_available() else "CPU")

    rvc = RVCInference()

    # Procurar modelos
    model_dir = Path("models/rvc")
    pth_files = list(model_dir.glob("*.pth"))
    if not pth_files:
        print("Nenhum modelo .pth encontrado em models/rvc/")
        sys.exit(1)

    print(f"Modelos encontrados: {[f.stem for f in pth_files]}")

    # Carregar primeiro modelo
    if rvc.load_model(str(pth_files[0])):
        print(f"\nModelo ativo: {rvc.model_name}")
        print("Sample rate:", rvc.sample_rate)

        # Teste de áudio ao vivo
        p = pyaudio.PyAudio()
        stream_in = p.open(
            format=pyaudio.paInt16, channels=1, rate=44100,
            input=True, frames_per_buffer=1024,
        )
        stream_out = p.open(
            format=pyaudio.paInt16, channels=1, rate=44100,
            output=True, frames_per_buffer=1024,
        )

        converter = StreamingRVCConverter(rvc, buffer_seconds=0.8)
        print("\n🎤 A falar... Pressiona Ctrl+C para sair.")

        try:
            while True:
                data = stream_in.read(1024, exception_on_overflow=False)
                audio_in = np.frombuffer(data, dtype=np.int16)
                audio_out = converter.process_chunk(audio_in)
                stream_out.write(audio_out.tobytes())
        except KeyboardInterrupt:
            print("\n✅ Teste concluído!")

        stream_in.close()
        stream_out.close()
        p.terminate()
