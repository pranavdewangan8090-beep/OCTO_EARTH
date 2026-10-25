@echo off
REM ==========================================================================
REM  OCTO-EARTH prototype launcher (Windows). Double-click to run.
REM  1. finds Python 3.10-3.13   2. creates .venv next to this file (no admin)
REM  3. installs missing packages from requirements.txt (internet needed once)
REM  4. runs octoearth_rl.py: physics checks -> viewer in browser -> PPO training
REM  Extra arguments are passed through, e.g.  RUN_OCTOEARTH.bat --smoke
REM ==========================================================================
setlocal
cd /d "%~dp0"
title OCTO-EARTH prototype

set "PYEXE="
where py >nul 2>nul && (
  for %%V in (3.13 3.12 3.11 3.10) do (
    if not defined PYEXE ( py -%%V -c "import sys" >nul 2>nul && set "PYEXE=py -%%V" )
  )
)
if not defined PYEXE (
  where python >nul 2>nul && (
    python -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)" >nul 2>nul && set "PYEXE=python"
  )
)
if not defined PYEXE (
  echo.
  echo [OCTO-EARTH] Python 3.10 - 3.13 ^(64-bit^) was not found.
  echo Install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^) and run this file again.
  pause
  exit /b 1
)
echo [OCTO-EARTH] using %PYEXE%

if not exist ".venv\Scripts\python.exe" (
  echo [OCTO-EARTH] creating virtual environment .venv ...
  %PYEXE% -m venv .venv
  if errorlevel 1 ( echo [OCTO-EARTH] could not create .venv & pause & exit /b 1 )
)
set "VPY=.venv\Scripts\python.exe"

"%VPY%" -c "import numpy, numba, gymnasium, stable_baselines3, torch" >nul 2>nul
if errorlevel 1 (
  echo [OCTO-EARTH] installing / repairing packages ^(first run downloads ~300 MB, needs internet^) ...
  "%VPY%" -m pip install --upgrade pip
  "%VPY%" -m pip install -r requirements.txt
  if errorlevel 1 ( echo [OCTO-EARTH] package installation failed - check the internet connection. & pause & exit /b 1 )
  "%VPY%" -c "import numpy, numba, gymnasium, stable_baselines3, torch" || ( echo [OCTO-EARTH] packages installed but cannot be imported. & pause & exit /b 1 )
)

REM ---- optional: NVIDIA GPU for the PPO network (training falls back to the CPU automatically) ----
REM      set OCTO_NO_GPU=1     before running this file to skip the GPU check.
REM      set OCTO_GPU_RETRY=1  to try the CUDA PyTorch installation again after a failed attempt.
if defined OCTO_NO_GPU goto nogpu
set "HASNV="
where nvidia-smi >nul 2>nul && set "HASNV=1"
if exist "%SystemRoot%\System32\nvidia-smi.exe" set "HASNV=1"
if exist "%ProgramFiles%\NVIDIA Corporation\NVSMI\nvidia-smi.exe" set "HASNV=1"
if not defined HASNV (
  powershell -NoProfile -Command "if (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' }) { exit 0 } else { exit 1 }" >nul 2>nul && set "HASNV=1"
)
if not defined HASNV goto nonv
"%VPY%" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul && goto gpuok
if defined OCTO_GPU_RETRY if exist ".venv\gpu_install_tried.txt" del ".venv\gpu_install_tried.txt"
if exist ".venv\gpu_install_tried.txt" goto gputried
echo [OCTO-EARTH] NVIDIA graphics card found but PyTorch is the CPU-only build.
echo [OCTO-EARTH] installing the CUDA build of PyTorch once ^(about 2.5 GB, needs internet^) ...
echo tried > ".venv\gpu_install_tried.txt"
for %%C in (cu126 cu130) do (
  "%VPY%" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul || (
    echo [OCTO-EARTH] trying PyTorch %%C ...
    "%VPY%" -m pip install --upgrade --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/%%C
  )
)
"%VPY%" -c "import torch" >nul 2>nul && goto gpucheck
echo [OCTO-EARTH] CUDA PyTorch could not be imported - restoring the CPU build. Training will use the CPU.
"%VPY%" -m pip install --force-reinstall --no-deps torch
goto gpufail
:gpucheck
"%VPY%" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul && goto gpuok
goto gpufail
:gputried
echo [OCTO-EARTH] an earlier CUDA PyTorch installation did not give a usable GPU.
echo [OCTO-EARTH] to try again: run  set OCTO_GPU_RETRY=1  in this window and start this file again
echo [OCTO-EARTH] ^(or delete .venv\gpu_install_tried.txt^). Updating the NVIDIA driver first often helps.
:gpufail
"%VPY%" -c "import torch; print('[OCTO-EARTH] PyTorch', torch.__version__, '| CUDA build:', torch.version.cuda, '| GPU usable:', torch.cuda.is_available())" 2>nul
echo [OCTO-EARTH] training device: CPU. The viewer and the physics are not affected.
goto gpudone
:gpuok
"%VPY%" -c "import torch; print('[OCTO-EARTH] GPU:', torch.cuda.get_device_name(0), '| PyTorch', torch.__version__, '| CUDA', torch.version.cuda)" 2>nul
echo [OCTO-EARTH] training device Auto: CPU and GPU are measured when training starts and the faster one is used
echo [OCTO-EARTH] ^(for this small network usually the CPU^). Choose CPU / GPU in the Mission panel. Physics always runs on the CPU.
echo [OCTO-EARTH] In Task Manager choose the "Cuda" or "Compute" graph of the GPU - the default "3D" graph does not show it.
goto gpudone
:nonv
echo [OCTO-EARTH] no NVIDIA graphics card found: training device CPU ^(PyTorch training needs an NVIDIA card with CUDA^).
goto gpudone
:nogpu
echo [OCTO-EARTH] training device: CPU ^(OCTO_NO_GPU is set^).
:gpudone

echo [OCTO-EARTH] starting ... the browser opens automatically. Close this window or press Ctrl+C to stop.
"%VPY%" octoearth_rl.py %*
set "RC=%errorlevel%"
if not "%RC%"=="0" (
  echo.
  echo [OCTO-EARTH] exited with code %RC%. See the messages above.
  pause
)
endlocal
