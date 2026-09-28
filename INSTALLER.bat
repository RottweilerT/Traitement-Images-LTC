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

for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)"') do set "PY312=%%P"
echo Python 3.12 utilise : %PY312%
echo.

rem Un environnement .venv existant peut etre casse si Python 3.12 a ete
rem reinstalle, repare ou mis a jour : il est alors recree automatiquement.
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo L'environnement .venv existant ne fonctionne plus : il est recree.
        rmdir /s /q ".venv"
        if exist ".venv" (
            echo ERREUR : impossible de supprimer l'ancien dossier .venv.
            echo Fermez les fenetres du programme, supprimez .venv a la main, puis relancez.
            pause
            exit /b 1
        )
    )
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
