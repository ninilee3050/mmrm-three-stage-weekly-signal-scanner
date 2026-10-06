@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo MMRM 스캐너 실행 파일을 만드는 중입니다. 몇 분 걸립니다...
python -m pip install --quiet --upgrade pyinstaller
python -m PyInstaller --noconfirm --clean --windowed --name MMRM-Scanner ^
  --add-data "resources;resources" --collect-data tzdata ^
  --exclude-module matplotlib --exclude-module yfinance --exclude-module pytest ^
  app.py
if errorlevel 1 (
  echo 만들기에 실패했습니다. 위의 오류 내용을 확인해 주세요.
  pause
  exit /b 1
)
echo.
echo 완료: dist\MMRM-Scanner 폴더를 통째로 복사해서 사용하세요.
echo 실행 파일은 그 안의 MMRM-Scanner.exe 입니다.
pause
