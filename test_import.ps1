# 빠른 Import 테스트 스크립트

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "    KcBERT Import 테스트" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "⚠️  주의: 처음 실행 시 2-3분 소요될 수 있습니다." -ForegroundColor Yellow
Write-Host "         PyTorch/Transformers 모듈을 로딩하는 동안 기다려주세요." -ForegroundColor Yellow
Write-Host ""

# 가상환경 활성화
if (Test-Path "venv\Scripts\Activate.ps1") {
    & .\venv\Scripts\Activate.ps1
}

# 테스트 실행
python test_import.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "✅ Import 테스트 성공!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "이제 메인 프로그램을 실행하시겠습니까? (Y/N)" -ForegroundColor Yellow
    $response = Read-Host
    
    if ($response -eq "Y" -or $response -eq "y") {
        Write-Host ""
        & .\run.ps1
    }
} else {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host "❌ Import 테스트 실패" -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Red
}
