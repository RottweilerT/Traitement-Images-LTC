@echo off
cd /d "%~dp0"

echo ==========================================
echo        BILAN DU CONTROLE MANUEL
echo ==========================================
echo.

rem Choix de Python : environnement du projet (.venv), sinon Python 3.12, sinon python.
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
    py -3.12 -c "import numpy, scipy, cv2, PIL" >nul 2>&1
    if not errorlevel 1 set "PY=py -3.12"
)
if not defined PY set "PY=python"

%PY% run_pipeline.py --bilan-manuel
if errorlevel 1 (
    echo.
    echo Une erreur est survenue ^(voir ci-dessus^).
    echo Si un module est introuvable, lancez INSTALLER.bat.
)

echo.
echo ==========================================
echo Bilan termine.
echo Vous pouvez fermer cette fenetre.
echo ==========================================
echo.
pause
