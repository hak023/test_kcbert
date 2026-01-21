# KcBERT 욕설/폭언 감지 시스템 - 설치 스크립트

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "    KcBERT 욕설/폭언 감지 시스템 설치" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Python 버전 확인
Write-Host "[1/4] Python 확인 중..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python이 설치되어 있지 않습니다." -ForegroundColor Red
    Write-Host "   Python 3.8 이상을 설치해주세요: https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}
Write-Host "✓ $pythonVersion 확인됨" -ForegroundColor Green
Write-Host ""

# 가상환경 생성
Write-Host "[2/4] 가상환경 생성 중..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "   기존 가상환경 제거 중..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force venv
}
python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 가상환경 생성 실패" -ForegroundColor Red
    exit 1
}
Write-Host "✓ 가상환경 생성 완료" -ForegroundColor Green
Write-Host ""

# 가상환경 활성화
Write-Host "[3/4] 가상환경 활성화 중..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 가상환경 활성화 실패" -ForegroundColor Red
    Write-Host "   PowerShell 실행 정책을 변경해야 할 수 있습니다:" -ForegroundColor Yellow
    Write-Host "   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ 가상환경 활성화 완료" -ForegroundColor Green
Write-Host ""

# pip 업그레이드
Write-Host "   pip 업그레이드 중..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# 의존성 설치
Write-Host "[4/4] 의존성 패키지 설치 중..." -ForegroundColor Yellow
Write-Host "   (시간이 걸릴 수 있습니다. 약 5-10분 소요)" -ForegroundColor Yellow
Write-Host ""

pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ 패키지 설치 실패" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✓ 모든 패키지 설치 완료" -ForegroundColor Green
Write-Host ""

# 디렉토리 생성
Write-Host "필요한 디렉토리 생성 중..." -ForegroundColor Yellow
$directories = @("data/samples", "data/results", "models/kcbert", "docs/guides")
foreach ($dir in $directories) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}
Write-Host "✓ 디렉토리 생성 완료" -ForegroundColor Green
Write-Host ""

# 완료 메시지
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "✅ 설치가 완료되었습니다!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "사용 방법:" -ForegroundColor White
Write-Host "  1. 대화형 실행:     .\run.ps1" -ForegroundColor Yellow
Write-Host "  2. 직접 실행:       .\run_simple.ps1 data\samples\normal_call.txt" -ForegroundColor Yellow
Write-Host "  3. Python 직접:     python main.py --input <파일경로>" -ForegroundColor Yellow
Write-Host ""
Write-Host "예제 파일:" -ForegroundColor White
Write-Host "  - data\samples\normal_call.txt    (정상 통화)" -ForegroundColor Green
Write-Host "  - data\samples\abusive_call.txt   (욕설 포함)" -ForegroundColor Red
Write-Host "  - data\samples\mixed_call.txt     (혼합)" -ForegroundColor Yellow
Write-Host ""
