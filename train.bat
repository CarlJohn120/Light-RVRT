@echo off
REM Set CUDA environment FIRST - Use CUDA 12.1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
set CUDA_PATH=%CUDA_HOME%
set TORCH_CUDA_ARCH_LIST=7.0
set TORCH_NVCC_FLAGS=-allow-unsupported-compiler
set CUDAFLAGS=-allow-unsupported-compiler

REM Add CUDA runtime DLLs to PATH at the very beginning
set PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%

REM Initialize Visual Studio Build Tools environment BEFORE anything else
REM Use MSVC 14.44 toolset for CUDA 12.1 compatibility (avoid 14.50's CUDA 12.4 requirement)
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44

REM Re-add CUDA to PATH after vcvars (vcvars may modify PATH)
set PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%

REM Add venv to PATH without activation (to preserve vcvars environment)
set PATH=C:\Users\VM03\Desktop\Attention\KAIR\.venv\Scripts;%PATH%
set VIRTUAL_ENV=C:\Users\VM03\Desktop\Attention\KAIR\.venv

REM Change to KAIR directory
cd C:\Users\VM03\Desktop\Attention\KAIR

REM Verify environment
echo CUDA_HOME: %CUDA_HOME%
echo Checking cl.exe...
where cl.exe
echo Checking ninja...
where ninja.exe
echo Python check:
python -c "import os; print('Python CUDA_HOME:', os.environ.get('CUDA_HOME', 'NOT SET')); import torch; print('PyTorch CUDA:', torch.cuda.is_available())"

REM Run training
python main_train_psnr.py --opt options/debug_cdnet.json
