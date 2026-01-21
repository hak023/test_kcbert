# KcBERT 욕설/폭언 감지 시스템 - 간편 실행 스크립트

Write-Host "🚀 KcBERT 욕설/폭언 감지 시스템" -ForegroundColor Cyan
Write-Host ""

# 가상환경 활성화
if (Test-Path "venv\Scripts\Activate.ps1") {
    & .\venv\Scripts\Activate.ps1
}

# 인자가 없으면 대화형 모드
if ($args.Count -eq 0) {
    Write-Host "사용법: .\run_simple.ps1 <파일경로>" -ForegroundColor Yellow
    Write-Host "예시: .\run_simple.ps1 data\samples\normal_call.txt" -ForegroundColor Yellow
    Write-Host ""
    
    # 예제 파일 목록 표시
    Write-Host "예제 파일:" -ForegroundColor White
    Get-ChildItem "data\samples\*.txt" | ForEach-Object {
        Write-Host "  - $($_.Name)" -ForegroundColor Green
    }
    Write-Host ""
    
    $file = Read-Host "분석할 파일명을 입력하세요"
    $inputFile = "data\samples\$file"
} else {
    $inputFile = $args[0]
}

# 파일 존재 확인
if (-Not (Test-Path $inputFile)) {
    Write-Host "❌ 파일을 찾을 수 없습니다: $inputFile" -ForegroundColor Red
    exit 1
}

# 실행
Write-Host ""
Write-Host "분석 중: $inputFile" -ForegroundColor Cyan
Write-Host ""

python main.py --input $inputFile
