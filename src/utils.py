"""
유틸리티 함수 모음
"""

import os
import json
import yaml
import logging
from datetime import datetime
from typing import Dict, Any


def setup_logging(level: str = "INFO") -> logging.Logger:
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """설정 파일 로드"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def save_result(result: Dict[str, Any], output_path: str):
    """결과를 JSON 파일로 저장"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 결과 저장 완료: {output_path}")


def format_result_text(result: Dict[str, Any]) -> str:
    """결과를 텍스트 형식으로 포맷팅"""
    lines = [
        "=" * 60,
        "KcBERT 욕설/폭언 감지 결과",
        "=" * 60,
        "",
        f"📄 입력 텍스트:",
        f"  {result['text'][:100]}..." if len(result['text']) > 100 else f"  {result['text']}",
        "",
        f"🎯 감지 결과: {'⚠️  욕설/폭언 감지됨' if result['is_abusive'] else '✓ 정상'}",
        f"📊 공격성 점수: {result['abusive_score']:.4f}",
        f"📈 신뢰도: {result['confidence']:.4f}",
        f"🎚️  임계값: {result['threshold']:.2f}",
        f"⏱️  처리 시간: {result['processing_time']:.3f}초",
        "",
        "=" * 60,
    ]
    
    return "\n".join(lines)


def get_timestamp() -> str:
    """현재 타임스탬프 반환"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_output_filename(input_filename: str, results_dir: str) -> str:
    """입력 파일명 기반으로 출력 파일명 생성"""
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    timestamp = get_timestamp()
    output_filename = f"{base_name}_result_{timestamp}.json"
    return os.path.join(results_dir, output_filename)
