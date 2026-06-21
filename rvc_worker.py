"""Executa uma conversão RVC dentro do ambiente Python 3.10 isolado."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from rvc_python.infer import RVCInference


def model_version(model_path: Path) -> str:
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    version = str(checkpoint.get("version", "v2")).lower()
    return version if version in {"v1", "v2"} else "v2"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", default="")
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--device", default="cpu:0")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    model_path = Path(args.model).resolve()
    index_path = Path(args.index).resolve() if args.index else None

    if not input_path.is_file():
        raise FileNotFoundError(f"Áudio não encontrado: {input_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    if index_path is not None and not index_path.is_file():
        raise FileNotFoundError(f"Índice não encontrado: {index_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rvc = RVCInference(device=args.device)
    rvc.load_model(
        str(model_path),
        version=model_version(model_path),
        index_path=str(index_path) if index_path else "",
    )
    rvc.set_params(
        f0method="rmvpe",
        f0up_key=args.pitch,
        # Valores conservadores priorizam a inteligibilidade. Um index_rate
        # alto aproxima mais o timbre, mas pode substituir fonemas e fazer a
        # fala parecer outro idioma.
        index_rate=0.20 if index_path else 0.0,
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=0.90,
        protect=0.05,
    )
    rvc.infer_file(str(input_path), str(output_path))
    print(f"RVC_OK={output_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
