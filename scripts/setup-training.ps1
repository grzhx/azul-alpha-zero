$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    if (!(Test-Path '.venv/Scripts/python.exe')) { python -m venv .venv }
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { throw 'venv failed' }
    & ./.venv/Scripts/python.exe -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed' }
    & ./.venv/Scripts/python.exe -m pip install numpy
    if ($LASTEXITCODE -ne 0) { throw 'NumPy installation failed' }
    & ./.venv/Scripts/python.exe -c "import torch; assert torch.cuda.is_available(); x=torch.randn(512,256,device='cuda',dtype=torch.bfloat16); y=x@x.T; torch.cuda.synchronize(); assert y.isfinite().all(); print(torch.__version__,torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { throw 'CUDA validation failed' }
} finally { Pop-Location }
