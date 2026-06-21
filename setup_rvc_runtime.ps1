$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$MainPython = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $MainPython)) {
    throw "Ambiente principal não encontrado: $MainPython"
}

$Uv = Join-Path $Root "venv\Scripts\uv.exe"
if (-not (Test-Path -LiteralPath $Uv)) {
    & $MainPython -m pip install uv
}

& $Uv python install 3.10
& $Uv venv ".rvc-venv" --python 3.10
& $Uv pip install --python ".rvc-venv\Scripts\python.exe" "rvc-python==0.1.5"
& $Uv pip install --python ".rvc-venv\Scripts\python.exe" --reinstall "torch==2.1.1" "torchaudio==2.1.1"

& ".rvc-venv\Scripts\python.exe" -c "from rvc_python.infer import RVCInference; RVCInference(device='cpu:0'); print('Runtime RVC pronto.')"
