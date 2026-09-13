@echo off
cd /d "%~dp0"

echo ==========================================
echo        BILAN DU CONTROLE MANUEL
echo ==========================================
echo.

python run_pipeline.py --bilan-manuel

echo.
echo ==========================================
echo Bilan termine.
echo Vous pouvez fermer cette fenetre.
echo ==========================================
echo.
pause