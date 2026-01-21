# -*- coding: utf-8 -*-
"""
KcBERT 욕설/폭언 감지 시스템 - 메인 실행 스크립트
"""

import os
import sys
import argparse

# UTF-8 출력 설정 (Windows 호환)
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from src.detector import AbusiveDetector
from src.utils import load_config, save_result, format_result_text, create_output_filename


def main():
    """메인 실행 함수"""
    
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(
        description="KcBERT 기반 통화 내용 욕설/폭언 감지 시스템"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='입력 텍스트 파일 경로'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='결과 저장 파일 경로 (미지정시 자동 생성)'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=None,
        help='감지 임계값 (0.0 ~ 1.0, 기본값은 config.yaml 참조)'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='설정 파일 경로'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='결과 저장 안함'
    )
    
    args = parser.parse_args()
    
    # 입력 파일 확인
    if not os.path.exists(args.input):
        print(f"❌ 오류: 입력 파일을 찾을 수 없습니다: {args.input}")
        sys.exit(1)
    
    # 설정 로드
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"⚠️  경고: 설정 파일을 찾을 수 없습니다. 기본 설정을 사용합니다.")
        config = {
            'model': {
                'name': 'beomi/kcbert-base',
                'cache_dir': './models/kcbert',
                'max_length': 512
            },
            'detection': {
                'threshold': 0.5
            },
            'output': {
                'save_results': True,
                'results_dir': './data/results'
            }
        }
    
    # 임계값 설정 (명령행 인자가 우선)
    threshold = args.threshold if args.threshold is not None else config['detection']['threshold']
    
    print("\n" + "🚀 " * 20)
    print("    KcBERT 욕설/폭언 감지 시스템")
    print("🚀 " * 20 + "\n")
    
    print(f"📄 입력 파일: {args.input}")
    print(f"🎚️  감지 임계값: {threshold}")
    print(f"🤖 모델: {config['model']['name']}")
    
    # 감지 엔진 초기화
    detector = AbusiveDetector(
        model_name=config['model']['name'],
        cache_dir=config['model']['cache_dir'],
        threshold=threshold,
        max_length=config['model']['max_length']
    )
    
    # 예측 실행
    print(f"\n{'='*60}")
    print("🔍 분석 시작...")
    print(f"{'='*60}\n")
    
    result = detector.predict_file(args.input)
    
    # 결과 출력
    print("\n" + format_result_text(result))
    
    # 결과 저장
    if not args.no_save:
        # 출력 경로 결정
        if args.output:
            output_path = args.output
        else:
            results_dir = config['output']['results_dir']
            output_path = create_output_filename(args.input, results_dir)
        
        # 저장
        save_result(result, output_path)
    
    # 종료 코드 반환 (욕설 감지 시 1, 정상 시 0)
    sys.exit(1 if result['is_abusive'] else 0)


if __name__ == "__main__":
    main()
