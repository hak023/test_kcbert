# KcBERT 욕설/폭언 감지 시스템 - PowerShell 실행 스크립트

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "    KcBERT 욕설/폭언 감지 시스템 실행" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Python 버전 확인
Write-Host "[1/6] Python 버전 확인 중..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python이 설치되어 있지 않습니다." -ForegroundColor Red
    Write-Host "   Python 3.8 이상을 설치해주세요: https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}
Write-Host "✓ $pythonVersion" -ForegroundColor Green
Write-Host ""

# 2. 가상환경 확인 및 생성
Write-Host "[2/6] 가상환경 확인 중..." -ForegroundColor Yellow
if (-Not (Test-Path "venv")) {
    Write-Host "   가상환경이 없습니다. 새로 생성합니다..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ 가상환경 생성 실패" -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ 가상환경 생성 완료" -ForegroundColor Green
} else {
    Write-Host "✓ 가상환경 존재" -ForegroundColor Green
}
Write-Host ""

# 3. 가상환경 활성화
Write-Host "[3/6] 가상환경 활성화 중..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 가상환경 활성화 실패" -ForegroundColor Red
    Write-Host "   PowerShell 실행 정책을 변경해야 할 수 있습니다:" -ForegroundColor Yellow
    Write-Host "   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ 가상환경 활성화 완료" -ForegroundColor Green
Write-Host ""

# 4. 의존성 설치 확인
Write-Host "[4/6] 의존성 패키지 확인 중..." -ForegroundColor Yellow
$pipList = pip list 2>&1
if ($pipList -notmatch "transformers") {
    Write-Host "   필요한 패키지를 설치합니다..." -ForegroundColor Yellow
    Write-Host "   (최초 실행 시 시간이 걸릴 수 있습니다. 약 5-10분 소요)" -ForegroundColor Yellow
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ 패키지 설치 실패" -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ 패키지 설치 완료" -ForegroundColor Green
} else {
    Write-Host "✓ 필요한 패키지가 이미 설치되어 있습니다" -ForegroundColor Green
}
Write-Host ""

# 5. 필요한 디렉토리 생성
Write-Host "[5/6] 디렉토리 구조 확인 중..." -ForegroundColor Yellow
$directories = @("data/samples", "data/results", "models/kcbert")
foreach ($dir in $directories) {
    if (-Not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
}
Write-Host "✓ 디렉토리 구조 확인 완료" -ForegroundColor Green
Write-Host ""

# 6. 예제 파일 선택 및 실행
Write-Host "[6/6] 분석할 통화 내용 선택" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "예제 파일 목록:" -ForegroundColor White
Write-Host "  1. normal_call.txt    - 정상 통화 (욕설 없음)" -ForegroundColor Green
Write-Host "  2. abusive_call.txt   - 욕설 포함 통화 (욕설 다수)" -ForegroundColor Red
Write-Host "  3. mixed_call.txt     - 혼합 통화 (불만 표현)" -ForegroundColor Yellow
Write-Host "  4. complaint_call.txt - 불만 통화 (경미한 불만)" -ForegroundColor Yellow
Write-Host "  5. 직접 파일 경로 입력" -ForegroundColor Cyan
Write-Host ""

$choice = Read-Host "선택 (1-5)"

switch ($choice) {
    "1" { $inputFile = "data/samples/normal_call.txt" }
    "2" { $inputFile = "data/samples/abusive_call.txt" }
    "3" { $inputFile = "data/samples/mixed_call.txt" }
    "4" { $inputFile = "data/samples/complaint_call.txt" }
    "5" { 
        $inputFile = Read-Host "파일 경로를 입력하세요"
        if (-Not (Test-Path $inputFile)) {
            Write-Host "❌ 파일을 찾을 수 없습니다: $inputFile" -ForegroundColor Red
            exit 1
        }
    }
    default {
        Write-Host "❌ 잘못된 선택입니다" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "🔍 분석 시작: $inputFile" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 메인 스크립트 실행
python main.py --input $inputFile

# 실행 결과 저장
$exitCode = $LASTEXITCODE

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan

if ($exitCode -eq 0) {
    Write-Host "✅ 분석 완료: 정상 통화" -ForegroundColor Green
} elseif ($exitCode -eq 1) {
    Write-Host "⚠️  분석 완료: 욕설/폭언 감지됨" -ForegroundColor Red
} else {
    Write-Host "❌ 오류 발생" -ForegroundColor Red
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 결과 파일 위치 안내
Write-Host "📁 결과 파일은 data/results/ 디렉토리에 저장되었습니다." -ForegroundColor Yellow
Write-Host ""

# 추가 실행 여부 확인
$continue = Read-Host "다른 파일을 분석하시겠습니까? (Y/N)"
if ($continue -eq "Y" -or $continue -eq "y") {
    Write-Host ""
    & $MyInvocation.MyCommand.Path
} else {
    Write-Host ""
    Write-Host "프로그램을 종료합니다." -ForegroundColor Cyan
}
