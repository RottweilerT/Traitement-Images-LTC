@echo off
cd /d "%~dp0"

echo ==========================================
echo        TRAITEMENT DES IMAGES
echo ==========================================
echo.

rem Choix de Python : environnement du projet (.venv), sinon Python 3.12, sinon python.
set "PY="
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import numpy, scipy, cv2, PIL" >nul 2>&1
    if not errorlevel 1 set "PY=.venv\Scripts\python.exe"
)
if not defined PY (
    py -3.12 -c "import numpy, scipy, cv2, PIL" >nul 2>&1
    if not errorlevel 1 set "PY=py -3.12"
)
if not defined PY set "PY=python"

%PY% run_pipeline.py %*
if errorlevel 1 (
    echo.
    echo ==========================================
    echo Une erreur est survenue ^(voir ci-dessus^).
    echo Si le message parle d'un module introuvable
    echo ^(numpy, scipy, cv2...^), double-cliquez une
    echo fois sur INSTALLER.bat puis relancez.
    echo ==========================================
) else (
    echo.
    echo ==========================================
    echo Traitement termine. Resultats dans output.
    echo ==========================================
)

echo.
echo Vous pouvez fermer cette fenetre.
echo.
pause
