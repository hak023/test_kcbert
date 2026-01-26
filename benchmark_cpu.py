# -*- coding: utf-8 -*-
"""
KcBERT CPU 성능 벤치마크
노트북 CPU vs 서버 CPU 비교
"""

import sys
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import logging
logging.getLogger('transformers').setLevel(logging.ERROR)

import time
import platform
import psutil
import torch
from pathlib import Path


def get_system_info():
    """시스템 정보 수집"""
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "processor": platform.processor(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "cpu_freq_current": psutil.cpu_freq().current if psutil.cpu_freq() else "N/A",
        "cpu_freq_max": psutil.cpu_freq().max if psutil.cpu_freq() else "N/A",
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
    }
    return info


def benchmark_kcbert():
    """KcBERT 성능 벤치마크"""
    print("\n" + "=" * 70)
    print("⚡ KcBERT CPU 성능 벤치마크")
    print("=" * 70 + "\n")
    
    # 시스템 정보
    print("📊 현재 시스템 정보")
    print("─" * 70)
    sys_info = get_system_info()
    print(f"  • OS: {sys_info['os']} {sys_info['os_version']}")
    print(f"  • CPU: {sys_info['processor']}")
    print(f"  • 물리 코어: {sys_info['cpu_cores_physical']}개")
    print(f"  • 논리 코어: {sys_info['cpu_cores_logical']}개")
    print(f"  • CPU 현재 클럭: {sys_info['cpu_freq_current']} MHz")
    print(f"  • CPU 최대 클럭: {sys_info['cpu_freq_max']} MHz")
    print(f"  • 전체 RAM: {sys_info['ram_total_gb']} GB")
    print(f"  • 사용 가능 RAM: {sys_info['ram_available_gb']} GB")
    print(f"  • PyTorch: {torch.__version__}")
    print(f"  • 디바이스: CPU (GPU 미사용)")
    print()
    
    # 모델 로드
    print("📥 KcBERT 모델 로딩 중...")
    load_start = time.time()
    
    # stderr 억제
    class SuppressStderr:
        def __enter__(self):
            self.original_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stderr.close()
            sys.stderr = self.original_stderr
    
    with SuppressStderr():
        from src.detector import AbusiveDetector
        detector = AbusiveDetector()
        detector.load_model()
    
    load_time = time.time() - load_start
    print(f"✅ 로딩 완료 (소요 시간: {load_time:.2f}초)")
    print()
    
    # 테스트 데이터
    samples_dir = Path("data/samples")
    test_files = list(samples_dir.glob("*.txt"))
    
    print(f"📝 테스트 데이터: {len(test_files)}개 파일")
    print("─" * 70)
    
    # 테스트 텍스트 준비
    test_cases = []
    for file_path in test_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            test_cases.append({
                "name": file_path.stem,
                "text": text,
                "length": len(text)
            })
    
    for case in test_cases:
        print(f"  • {case['name']:20s}: {case['length']:4d}자")
    print()
    
    # 워밍업 (첫 실행은 느릴 수 있음)
    print("🔥 워밍업 중...")
    _ = detector.predict(test_cases[0]['text'])
    print("✅ 워밍업 완료")
    print()
    
    # 단일 처리 벤치마크
    print("─" * 70)
    print("⏱️  1. 단일 처리 벤치마크 (각 파일 1회)")
    print("─" * 70)
    
    single_times = []
    for case in test_cases:
        start = time.time()
        result = detector.predict(case['text'])
        elapsed = time.time() - start
        single_times.append(elapsed)
        
        print(f"  • {case['name']:20s}: {elapsed*1000:6.2f}ms "
              f"(점수: {result['abusive_score']:.3f})")
    
    avg_single = sum(single_times) / len(single_times)
    print(f"\n  평균 처리 시간: {avg_single*1000:.2f}ms")
    print()
    
    # 반복 처리 벤치마크
    print("─" * 70)
    print("⏱️  2. 반복 처리 벤치마크 (100회)")
    print("─" * 70)
    
    test_text = test_cases[0]['text']
    iterations = 100
    
    print(f"  테스트 텍스트: {test_cases[0]['name']}")
    print(f"  반복 횟수: {iterations}회")
    print()
    
    # CPU 사용률 측정 시작
    cpu_before = psutil.cpu_percent(interval=0.1)
    
    start = time.time()
    for i in range(iterations):
        _ = detector.predict(test_text)
        if (i + 1) % 20 == 0:
            print(f"  진행: {i+1}/{iterations}회...")
    elapsed = time.time() - start
    
    # CPU 사용률 측정
    cpu_after = psutil.cpu_percent(interval=0.1)
    
    avg_iter = elapsed / iterations
    throughput = iterations / elapsed
    
    print()
    print(f"  ✓ 총 소요 시간: {elapsed:.2f}초")
    print(f"  ✓ 평균 처리 시간: {avg_iter*1000:.2f}ms")
    print(f"  ✓ 처리량 (TPS): {throughput:.2f}건/초")
    print(f"  ✓ CPU 사용률: {cpu_after:.1f}%")
    print()
    
    # 배치 크기별 처리량
    print("─" * 70)
    print("⏱️  3. 배치 크기별 처리량")
    print("─" * 70)
    
    batch_sizes = [1, 5, 10, 20]
    batch_results = []
    
    for batch_size in batch_sizes:
        texts = [test_text] * batch_size
        
        start = time.time()
        for text in texts:
            _ = detector.predict(text)
        elapsed = time.time() - start
        
        avg_per_item = elapsed / batch_size
        tps = batch_size / elapsed
        
        batch_results.append({
            "size": batch_size,
            "total_time": elapsed,
            "avg_time": avg_per_item,
            "tps": tps
        })
        
        print(f"  • 배치 크기 {batch_size:2d}: "
              f"총 {elapsed:.3f}초, "
              f"평균 {avg_per_item*1000:.2f}ms, "
              f"{tps:.2f}건/초")
    
    print()
    
    # 결과 요약
    print("=" * 70)
    print("📊 벤치마크 결과 요약")
    print("=" * 70)
    print()
    
    print(f"  🖥️  현재 노트북 CPU")
    print(f"  ├─ 프로세서: {sys_info['processor']}")
    print(f"  ├─ 코어: {sys_info['cpu_cores_physical']}개 (논리 {sys_info['cpu_cores_logical']}개)")
    print(f"  ├─ 클럭: {sys_info['cpu_freq_max']} MHz")
    print(f"  ├─ 평균 처리 시간: {avg_single*1000:.2f}ms")
    print(f"  └─ 처리량: {throughput:.2f}건/초")
    print()
    
    # 서버 CPU 비교 추정
    print("─" * 70)
    print("💻 서버 CPU와의 비교 (추정)")
    print("─" * 70)
    print()
    
    # 일반적인 서버 CPU 성능 배수
    server_cpus = [
        {
            "name": "Intel Xeon Gold 6248R",
            "cores": 24,
            "threads": 48,
            "base_clock": 3.0,
            "turbo_clock": 4.0,
            "year": 2020,
            "performance_factor": 2.5,  # 노트북 대비
            "notes": "중급 서버 CPU"
        },
        {
            "name": "AMD EPYC 7543",
            "cores": 32,
            "threads": 64,
            "base_clock": 2.8,
            "turbo_clock": 3.7,
            "year": 2021,
            "performance_factor": 3.0,  # 노트북 대비
            "notes": "고급 서버 CPU"
        },
        {
            "name": "Intel Xeon E5-2680 v4",
            "cores": 14,
            "threads": 28,
            "base_clock": 2.4,
            "turbo_clock": 3.3,
            "year": 2016,
            "performance_factor": 1.8,  # 노트북 대비
            "notes": "구형 서버 CPU"
        },
        {
            "name": "AMD EPYC 9654",
            "cores": 96,
            "threads": 192,
            "base_clock": 2.4,
            "turbo_clock": 3.7,
            "year": 2022,
            "performance_factor": 4.0,  # 노트북 대비
            "notes": "최신 고성능 서버 CPU"
        }
    ]
    
    for cpu in server_cpus:
        est_time = avg_single / cpu['performance_factor']
        est_tps = throughput * cpu['performance_factor']
        speedup = cpu['performance_factor']
        
        print(f"  🖥️  {cpu['name']}")
        print(f"  ├─ 사양: {cpu['cores']}코어/{cpu['threads']}쓰레드, "
              f"{cpu['base_clock']}GHz (최대 {cpu['turbo_clock']}GHz)")
        print(f"  ├─ 성능: 노트북 대비 약 {speedup:.1f}배")
        print(f"  ├─ 예상 처리 시간: {est_time*1000:.2f}ms (현재: {avg_single*1000:.2f}ms)")
        print(f"  ├─ 예상 처리량: {est_tps:.2f}건/초 (현재: {throughput:.2f}건/초)")
        print(f"  └─ {cpu['notes']} ({cpu['year']}년)")
        print()
    
    print("─" * 70)
    print()
    
    # 실제 사용 시나리오
    print("💡 실제 사용 시나리오 비교")
    print("─" * 70)
    print()
    
    scenarios = [
        {"name": "1,000건 일괄 처리", "count": 1000},
        {"name": "10,000건 일괄 처리", "count": 10000},
        {"name": "100,000건 일괄 처리", "count": 100000},
        {"name": "실시간 처리 (초당 10건)", "count": 10, "unit": "초"},
        {"name": "실시간 처리 (초당 100건)", "count": 100, "unit": "초"},
    ]
    
    print(f"{'시나리오':<25s} {'현재 노트북':<15s} {'서버 (2.5배)':<15s} {'서버 (3배)':<15s}")
    print("─" * 70)
    
    for scenario in scenarios:
        count = scenario['count']
        
        if scenario.get('unit') == '초':
            # 실시간 처리 - TPS 기준
            current = "가능" if throughput >= count else "불가능"
            server_25x = "가능" if throughput * 2.5 >= count else "불가능"
            server_3x = "가능" if throughput * 3.0 >= count else "불가능"
            
            print(f"{scenario['name']:<25s} {current:<15s} {server_25x:<15s} {server_3x:<15s}")
        else:
            # 배치 처리 - 시간 기준
            current_time = count * avg_single
            server_25x_time = count * avg_single / 2.5
            server_3x_time = count * avg_single / 3.0
            
            def format_time(seconds):
                if seconds < 60:
                    return f"{seconds:.1f}초"
                elif seconds < 3600:
                    return f"{seconds/60:.1f}분"
                else:
                    return f"{seconds/3600:.1f}시간"
            
            print(f"{scenario['name']:<25s} "
                  f"{format_time(current_time):<15s} "
                  f"{format_time(server_25x_time):<15s} "
                  f"{format_time(server_3x_time):<15s}")
    
    print()
    
    # 성능 향상 팁
    print("=" * 70)
    print("🚀 성능 향상 팁")
    print("=" * 70)
    print()
    
    tips = [
        ("서버 CPU 선택", "코어 수보다 단일 코어 성능이 중요 (BERT는 단일 스레드)"),
        ("배치 처리", "가능하면 여러 건을 모아서 처리 (오버헤드 감소)"),
        ("멀티 프로세스", "여러 프로세스로 병렬 처리 (코어 수만큼 향상)"),
        ("모델 최적화", "ONNX Runtime 사용 시 1.5~2배 빨라짐"),
        ("양자화", "INT8 양자화 시 2~4배 빨라지고 메모리 절약"),
        ("GPU 사용", "서버에 GPU 있으면 10~20배 빨라짐"),
    ]
    
    for i, (title, desc) in enumerate(tips, 1):
        print(f"  {i}. {title}")
        print(f"     → {desc}")
        print()
    
    print("=" * 70)
    
    # 최종 결론
    print()
    print("📌 결론")
    print("─" * 70)
    print()
    print(f"  현재 노트북 CPU: {avg_single*1000:.2f}ms/건, {throughput:.2f}건/초")
    print(f"  중급 서버 CPU (예상): {avg_single/2.5*1000:.2f}ms/건, {throughput*2.5:.2f}건/초")
    print(f"  고급 서버 CPU (예상): {avg_single/3.0*1000:.2f}ms/건, {throughput*3.0:.2f}건/초")
    print()
    print(f"  → 서버 CPU 사용 시 약 2~4배 빠른 처리 가능")
    print(f"  → GPU 사용 시 10~20배 더 빠른 처리 가능")
    print()
    print("=" * 70)
    
    return {
        "system_info": sys_info,
        "avg_time_ms": avg_single * 1000,
        "throughput_tps": throughput,
        "load_time": load_time
    }


if __name__ == "__main__":
    try:
        results = benchmark_kcbert()
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
