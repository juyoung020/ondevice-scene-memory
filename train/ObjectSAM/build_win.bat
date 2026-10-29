@echo off
rem Windows build (speed/parity tests only). Real training runs on Linux.
cd /d "%~dp0"
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >/dev/null || exit /b 1
if "%CUDNN_ROOT%"=="" set CUDNN_ROOT=C:/tools/cudnn
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDNN_ROOT=%CUDNN_ROOT% || exit /b 1
if "%~1"=="" (cmake --build build) else (cmake --build build --target %*)
if errorlevel 1 exit /b 1
