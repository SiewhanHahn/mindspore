@echo off
chcp 65001 >nul
echo ========================================================
echo   DefectGAN - Statistical Guided Real Face Generation
echo   Windows Quick Start
echo ========================================================
echo.

REM 检查 conda 环境
echo [1/4] Checking conda environment...
call conda activate mindspore_env 2>nul
if errorlevel 1 (
    echo [WARN] Could not activate mindspore_env
    echo        Trying with system Python...
    where python >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python not found. Install Python and MindSpore first.
        pause
        exit /b 1
    )
)

echo [OK] Environment ready
echo.

REM 切换到项目目录
cd /d "%~dp0"

REM 运行验证
echo [2/4] Running validation...
python run_defectgan.py test --resolution 512
if errorlevel 1 (
    echo.
    echo [WARN] Some tests failed. See above for details.
    echo        To skip checkpoint check: python run_defectgan.py test --skip-checkpoint
    echo.
)

echo.
echo [3/4] Running demo inference (5 images)...
python run_defectgan.py infer --resolution 512 --seeds 42,123,456,789,1024 --device Ascend
if errorlevel 1 (
    echo.
    echo [WARN] Ascend inference failed, trying CPU...
    python run_defectgan.py infer --resolution 512 --seeds 42,123,456 --device CPU
)

echo.
echo [4/4] Done!
echo ========================================================
echo   DefectGAN demo complete!
echo.
echo   Available commands:
echo     python run_defectgan.py test   - Validate setup
echo     python run_defectgan.py train   - Fine-tune StyleGAN2
echo     python run_defectgan.py infer   - Generate images
echo     python run_defectgan.py build   - Build dataset
echo     python run_defectgan.py all     - Full pipeline
echo.
echo   For help: python run_defectgan.py --help
echo ========================================================
echo.
pause
