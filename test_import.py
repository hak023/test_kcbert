# -*- coding: utf-8 -*-
"""
Import 테스트 스크립트
PyTorch와 Transformers가 정상적으로 로드되는지 확인
"""

import sys
import time
import warnings
import os

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# UTF-8 출력 설정
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 60)
print("KcBERT 의존성 로딩 테스트")
print("=" * 60)
print()

# 1단계: PyTorch
print("[1/3] PyTorch 로딩 중...")
print("      (처음 실행 시 1-2분 소요될 수 있습니다)")
start = time.time()
try:
    import torch
    elapsed = time.time() - start
    print(f"      OK PyTorch 로딩 완료 ({elapsed:.2f}초)")
    print(f"      버전: {torch.__version__}")
    print(f"      CUDA 사용 가능: {torch.cuda.is_available()}")
except Exception as e:
    print(f"      FAIL PyTorch 로딩 실패: {e}")
    exit(1)

print()

# 2단계: Transformers
print("[2/3] Transformers 로딩 중...")
print("      (처음 실행 시 1-2분 소요될 수 있습니다)")
start = time.time()
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    elapsed = time.time() - start
    print(f"      OK Transformers 로딩 완료 ({elapsed:.2f}초)")
    import transformers
    print(f"      버전: {transformers.__version__}")
except Exception as e:
    print(f"      FAIL Transformers 로딩 실패: {e}")
    exit(1)

print()

# 3단계: 전체 모듈
print("[3/3] KcBERT 모듈 로딩 중...")
start = time.time()
try:
    from src.detector import AbusiveDetector
    elapsed = time.time() - start
    print(f"      OK KcBERT 모듈 로딩 완료 ({elapsed:.2f}초)")
except Exception as e:
    print(f"      FAIL KcBERT 모듈 로딩 실패: {e}")
    exit(1)

print()
print("=" * 60)
print("SUCCESS 모든 의존성 로딩 성공!")
print("=" * 60)
print()
print("이제 메인 프로그램을 실행할 수 있습니다:")
print("  python main.py -i data/samples/normal_call.txt")
