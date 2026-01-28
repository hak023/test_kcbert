# -*- coding: utf-8 -*-
"""
KcBERT 배치 처리 스크립트
samples 디렉토리의 모든 txt 파일을 순차적으로 처리
"""

import os
import sys
import time
import glob
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# UTF-8 출력 설정 (Windows 호환)
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Transformers 로깅 레벨 조정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow 경고 숨기기
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Transformers 경고 숨기기
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)

# ⚡ Lazy import: 필요한 시점에만 로드
# from src.detector import AbusiveDetector  # 주석 처리
from src.utils import load_config, save_result, create_output_filename


def print_header():
    """헤더 출력"""
    print("\n" + "=" * 70)
    print("🚀 " * 10)
    print("         KcBERT 욕설/폭언 감지 시스템 - 배치 처리")
    print("🚀 " * 10)
    print("=" * 70 + "\n")


def print_result_summary(results):
    """결과 요약 출력"""
    print("\n" + "=" * 70)
    print("📊 전체 처리 결과 요약")
    print("=" * 70 + "\n")
    
    total_files = len(results)
    abusive_count = sum(1 for r in results if r['is_abusive'])
    normal_count = total_files - abusive_count
    total_time = sum(r['processing_time'] for r in results)
    avg_time = total_time / total_files if total_files > 0 else 0
    
    print(f"📁 처리된 파일: {total_files}개")
    print(f"⚠️  욕설 감지: {abusive_count}개")
    print(f"✅ 정상 통화: {normal_count}개")
    print(f"⏱️  총 처리 시간: {total_time:.3f}초")
    print(f"📈 평균 처리 시간: {avg_time:.3f}초/파일")
    print()
    
    # 개별 결과 테이블
    print("─" * 70)
    print(f"{'파일명':<25} {'결과':<15} {'점수':<10} {'시간(초)':<10}")
    print("─" * 70)
    
    for result in results:
        filename = os.path.basename(result['source_file'])
        status = "⚠️  욕설 감지" if result['is_abusive'] else "✅ 정상"
        score = f"{result['abusive_score']:.4f}"
        proc_time = f"{result['processing_time']:.3f}"
        
        print(f"{filename:<25} {status:<15} {score:<10} {proc_time:<10}")
    
    print("─" * 70)
    print()


def main():
    """메인 실행 함수"""
    
    print_header()
    
    # 설정 로드
    try:
        config = load_config('config.yaml')
    except FileNotFoundError:
        print("⚠️  설정 파일을 찾을 수 없습니다. 기본 설정을 사용합니다.")
        config = {
            'model': {
                'name': 'beomi/kcbert-base',
                'cache_dir': './models/kcbert',
                'max_length': 300
            },
            'detection': {
                'threshold': 0.5
            },
            'output': {
                'save_results': True,
                'results_dir': './data/results'
            }
        }
    
    # samples 디렉토리의 모든 txt 파일 찾기
    samples_dir = 'data/samples'
    txt_files = sorted(glob.glob(os.path.join(samples_dir, '*.txt')))
    
    if not txt_files:
        print(f"❌ {samples_dir}에 txt 파일이 없습니다.")
        sys.exit(1)
    
    print(f"📂 샘플 디렉토리: {samples_dir}")
    print(f"📄 발견된 파일: {len(txt_files)}개")
    print()
    
    for i, filepath in enumerate(txt_files, 1):
        print(f"   {i}. {os.path.basename(filepath)}")
    
    print()
    print("─" * 70)
    print()
    
    # ⚡ Lazy import: 실제 필요한 시점에 로드
    print("📥 모델 모듈 로딩 중... (최초 1회, 약 40초 소요)")
    from src.detector import AbusiveDetector
    print("✅ 모듈 로딩 완료!")
    print()
    
    # 감지 엔진 초기화 (한 번만)
    print("🤖 KcBERT 모델 초기화 중...")
    print()
    
    init_start = time.time()
    
    # stderr 숨기기 클래스
    class SuppressStderr:
        def __enter__(self):
            self._stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            return self
        def __exit__(self, *args):
            sys.stderr.close()
            sys.stderr = self._stderr
    
    # 모델 초기화 시 경고 메시지 숨기기
    with SuppressStderr():
        detector = AbusiveDetector(
            model_name=config['model']['name'],
            cache_dir=config['model']['cache_dir'],
            threshold=config['detection']['threshold'],
            max_length=config['model']['max_length']
        )
    
    init_time = time.time() - init_start
    
    print(f"✅ 모델 로딩 완료! ({init_time:.2f}초)")
    print()
    
    # 워밍업 실행 (모델 초기화 시간 제외)
    print("🔥 모델 워밍업 중... (첫 케이스 처리 시간 보정)")
    warmup_text = "안녕하세요. 테스트입니다."
    _ = detector.predict(warmup_text)
    print("✅ 워밍업 완료!")
    print()
    print("=" * 70)
    print()
    
    # 각 파일 처리
    results = []
    
    for i, filepath in enumerate(txt_files, 1):
        filename = os.path.basename(filepath)
        print(f"[{i}/{len(txt_files)}] 처리 중: {filename}")
        print("─" * 70)
        
        try:
            # 파일 분석
            result = detector.predict_file(filepath)
            results.append(result)
            
            # 결과 출력
            status = "⚠️  욕설/폭언 감지됨" if result['is_abusive'] else "✅ 정상 통화"
            print(f"   결과: {status}")
            print(f"   공격성 점수: {result['abusive_score']:.4f}")
            print(f"   신뢰도: {result['confidence']:.4f}")
            print(f"   처리 시간: {result['processing_time']:.3f}초")
            
            # 결과 저장
            if config['output']['save_results']:
                output_path = create_output_filename(filepath, config['output']['results_dir'])
                save_result(result, output_path)
            
        except Exception as e:
            print(f"   ❌ 오류 발생: {e}")
        
        print()
    
    # 전체 결과 요약
    if results:
        print_result_summary(results)
    
    print("=" * 70)
    print("🎉 배치 처리 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
