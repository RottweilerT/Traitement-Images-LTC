@echo off
cd /d "%~dp0"

echo ==========================================
echo    INSTALLATION (a faire une seule fois)
echo ==========================================
echo.
echo Cree un environnement Python 3.12 dans le dossier .venv
echo et y installe numpy, Pillow, OpenCV et SciPy.
echo Python 3.14 n'est pas utilise : Windows y bloque SciPy.
echo.

py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR : Python 3.12 est introuvable.
    echo Installez-le depuis https://www.python.org/downloads/
    echo ^(version 3.12.x, "Windows installer 64-bit"^) puis relancez.
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creation de l'environnement .venv ...
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo ERREUR : impossible de creer l'environnement .venv
        pause
        exit /b 1
    )
)

echo Installation des dependances ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERREUR : l'installation des dependances a echoue ^(voir ci-dessus^).
    pause
    exit /b 1
)

echo.
echo Verification :
".venv\Scripts\python.exe" -c "import numpy, scipy, cv2, PIL; print('Bibliotheques OK')"
if errorlevel 1 (
    echo ERREUR : une bibliotheque ne se charge pas ^(voir ci-dessus^).
    pause
    exit /b 1
)
".venv\Scripts\python.exe" run_pipeline.py --version

echo.
echo ==========================================
echo Installation terminee.
echo Lancez maintenant LANCER_TRAITEMENT.bat
echo ==========================================
echo.
pause
