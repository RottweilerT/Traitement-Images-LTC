@echo off
cd /d "%~dp0"

echo ==========================================
echo        TRAITEMENT DES IMAGES
echo ==========================================
echo.

python run_pipeline.py
if errorlevel 1 (
    echo.
    echo ==========================================
    echo Une erreur est survenue ^(voir ci-dessus^).
    echo Verifiez que Python est installe et que les
    echo dependances le sont aussi :
    echo   python -m pip install -r requirements.txt
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
