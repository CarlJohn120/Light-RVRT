@echo off
REM Test trained model on TRAINING SET with comparisons

REM Set CUDA environment FIRST - Use CUDA 12.1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
set CUDA_PATH=%CUDA_HOME%
set TORCH_CUDA_ARCH_LIST=7.0
set TORCH_NVCC_FLAGS=-allow-unsupported-compiler
set CUDAFLAGS=-allow-unsupported-compiler

REM Add CUDA runtime DLLs to PATH at the very beginning
set PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%

REM Initialize Visual Studio Build Tools environment BEFORE anything else
REM Use MSVC 14.44 toolset for CUDA 12.1 compatibility
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44

REM Re-add CUDA to PATH after vcvars (vcvars may modify PATH)
set PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%

REM Add venv to PATH without activation (to preserve vcvars environment)
set PATH=C:\Users\VM03\Desktop\Attention\KAIR\.venv\Scripts;%PATH%
set VIRTUAL_ENV=C:\Users\VM03\Desktop\Attention\KAIR\.venv

REM Change to KAIR directory
cd C:\Users\VM03\Desktop\Attention\KAIR

echo ========================================
echo Testing Trained Model on Training Set
echo ========================================
echo.
echo This will:
echo  1. Load your trained model
echo  2. Test on the TRAINING data
echo  3. Compare: Input vs Model Output vs Ground Truth
echo  4. Calculate PSNR/SSIM improvements
echo  5. Save side-by-side comparison images
echo.

REM Run test with comparison
python main_test_cdnet.py --opt options/test_cdnet.json --save_comparison

echo.
echo ========================================
echo Testing Complete!
echo ========================================
echo Results saved in: results\TEST_CDNET\
echo   - output\       : Model output videos
echo   - comparison\   : Side-by-side comparison images
echo   - test.log      : Full metrics log
echo ========================================
pause
