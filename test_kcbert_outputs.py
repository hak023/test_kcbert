# -*- coding: utf-8 -*-
"""
KcBERT 출력 필드 분석 테스트
모델에서 얻을 수 있는 모든 정보 확인
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

import torch
import numpy as np


def test_kcbert_outputs():
    """KcBERT 모델 출력 분석"""
    print("\n" + "=" * 70)
    print("🔬 KcBERT 모델 출력 필드 분석")
    print("=" * 70 + "\n")
    
    # 모델 로드
    print("📥 KcBERT 모델 로딩 중...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    tokenizer = AutoTokenizer.from_pretrained(
        "beomi/kcbert-base",
        cache_dir="./models/kcbert"
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "beomi/kcbert-base",
        cache_dir="./models/kcbert",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    
    model.eval()
    print("✅ 로딩 완료\n")
    
    # 테스트 텍스트
    test_text = "진짜 너무 화나네요. 이게 뭐하는 짓이야!"
    
    print("─" * 70)
    print(f"📝 테스트 텍스트: \"{test_text}\"")
    print("─" * 70)
    print()
    
    # 토큰화
    inputs = tokenizer(
        test_text,
        return_tensors="pt",
        max_length=300,
        padding="max_length",
        truncation=True
    )
    
    print("📊 1. 기본 출력 (현재 사용 중)")
    print("─" * 70)
    
    with torch.no_grad():
        # 기본 출력
        outputs = model(**inputs)
        
        # Logits (원시 출력값)
        logits = outputs.logits
        print(f"  • logits: {logits}")
        print(f"    - shape: {logits.shape}")
        print(f"    - 정상 클래스 스코어: {logits[0][0].item():.4f}")
        print(f"    - 욕설 클래스 스코어: {logits[0][1].item():.4f}")
        print()
        
        # Softmax 확률
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        normal_prob = probabilities[0][0].item()
        abusive_prob = probabilities[0][1].item()
        confidence = torch.max(probabilities).item()
        
        print(f"  • probabilities (softmax 적용 후):")
        print(f"    - 정상 확률: {normal_prob:.4f} ({normal_prob*100:.2f}%)")
        print(f"    - 욕설 확률: {abusive_prob:.4f} ({abusive_prob*100:.2f}%)")
        print(f"    - 신뢰도 (최대값): {confidence:.4f} ({confidence*100:.2f}%)")
        print()
        
        # 예측 클래스
        predicted_class = torch.argmax(logits, dim=-1).item()
        print(f"  • predicted_class: {predicted_class}")
        print(f"    - 0: 정상, 1: 욕설")
        print()
    
    print()
    print("📊 2. 추가 출력 (output_hidden_states=True)")
    print("─" * 70)
    
    with torch.no_grad():
        # Hidden states 포함
        outputs = model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states
        print(f"  • hidden_states: 전체 레이어의 은닉 상태")
        print(f"    - 레이어 수: {len(hidden_states)}개")
        print(f"    - 각 레이어 shape: {hidden_states[0].shape}")
        print(f"    - (batch_size, sequence_length, hidden_size)")
        print()
        
        # 마지막 레이어의 [CLS] 토큰 임베딩
        last_hidden_state = hidden_states[-1]
        cls_embedding = last_hidden_state[0][0]  # [CLS] 토큰
        
        print(f"  • [CLS] 토큰 임베딩 (문장 전체 표현):")
        print(f"    - shape: {cls_embedding.shape}")
        print(f"    - 예시 값 (처음 5개): {cls_embedding[:5].tolist()}")
        print(f"    - 이 벡터로 문장 유사도 계산 가능")
        print()
    
    print()
    print("📊 3. 추가 출력 (output_attentions=True)")
    print("─" * 70)
    
    with torch.no_grad():
        # Attention weights 포함
        outputs = model(**inputs, output_attentions=True)
        
        attentions = outputs.attentions
        print(f"  • attentions: 각 레이어의 어텐션 가중치")
        print(f"    - 레이어 수: {len(attentions)}개")
        print(f"    - 각 레이어 shape: {attentions[0].shape}")
        print(f"    - (batch_size, num_heads, seq_length, seq_length)")
        print()
        
        # 마지막 레이어의 평균 어텐션
        last_attention = attentions[-1]
        avg_attention = last_attention.mean(dim=1)[0]  # 헤드들의 평균
        
        print(f"  • 마지막 레이어 평균 어텐션:")
        print(f"    - shape: {avg_attention.shape}")
        print(f"    - [CLS] 토큰이 다른 토큰에 주목하는 정도")
        print()
        
        # 토큰별 어텐션 점수
        tokens = tokenizer.tokenize(test_text)
        cls_attention_to_tokens = avg_attention[0][1:len(tokens)+1]
        
        print(f"  • 토큰별 어텐션 점수 (중요도):")
        for token, attn_score in zip(tokens[:10], cls_attention_to_tokens[:10]):
            print(f"    - '{token}': {attn_score.item():.4f}")
        print()
    
    print()
    print("📊 4. 현재 반환 중인 필드")
    print("─" * 70)
    
    current_fields = {
        "text": "입력 텍스트",
        "is_abusive": "욕설 여부 (불린)",
        "confidence": "신뢰도 (0~1)",
        "abusive_score": "최종 공격성 점수 (0~1)",
        "model_score": "모델 원시 점수 (0~1)",
        "rule_score": "규칙 기반 점수 (0~1)",
        "threshold": "감지 임계값",
        "processing_time": "처리 시간 (초)"
    }
    
    for field, desc in current_fields.items():
        print(f"  ✓ {field:20s}: {desc}")
    
    print()
    print()
    print("📊 5. 추가 가능한 필드")
    print("─" * 70)
    
    additional_fields = {
        # 기본 모델 출력
        "logits": {
            "설명": "원시 출력값 (softmax 전)",
            "용도": "디버깅, 커스텀 후처리",
            "크기": "작음",
            "추천": "⭐"
        },
        "class_probabilities": {
            "설명": "각 클래스별 확률 [정상, 욕설]",
            "용도": "상세 확률 분포 확인",
            "크기": "작음",
            "추천": "⭐⭐⭐"
        },
        "predicted_class": {
            "설명": "예측된 클래스 (0 또는 1)",
            "용도": "간단한 분류 결과",
            "크기": "작음",
            "추천": "⭐⭐"
        },
        
        # 토큰 정보
        "token_count": {
            "설명": "입력 토큰 개수",
            "용도": "길이 체크, 잘림 감지",
            "크기": "작음",
            "추천": "⭐⭐⭐"
        },
        "tokens": {
            "설명": "토큰화된 결과",
            "용도": "디버깅, 분석",
            "크기": "중간",
            "추천": "⭐⭐"
        },
        "is_truncated": {
            "설명": "300 토큰 초과로 잘렸는지 여부",
            "용도": "데이터 손실 감지",
            "크기": "작음",
            "추천": "⭐⭐⭐"
        },
        
        # 고급 기능
        "sentence_embedding": {
            "설명": "[CLS] 토큰 벡터 (768차원)",
            "용도": "문장 유사도, 클러스터링",
            "크기": "중간",
            "추천": "⭐"
        },
        "token_attentions": {
            "설명": "각 토큰의 중요도 점수",
            "용도": "어떤 단어가 중요했는지 분석",
            "크기": "중간",
            "추천": "⭐⭐"
        },
        "hidden_states": {
            "설명": "전체 레이어 은닉 상태",
            "용도": "고급 NLP 연구",
            "크기": "매우 큼",
            "추천": ""
        },
        "attention_weights": {
            "설명": "전체 어텐션 가중치",
            "용도": "어텐션 시각화, 연구",
            "크기": "매우 큼",
            "추천": ""
        },
        
        # 분석 정보
        "abusive_words_found": {
            "설명": "감지된 욕설 단어 리스트",
            "용도": "구체적인 문제 단어 확인",
            "크기": "작음",
            "추천": "⭐⭐⭐"
        },
        "severity_level": {
            "설명": "심각도 (낮음/중간/높음/매우높음)",
            "용도": "등급별 분류",
            "크기": "작음",
            "추천": "⭐⭐⭐"
        },
        "detection_method": {
            "설명": "감지 방법 (모델/규칙/혼합)",
            "용도": "감지 근거 추적",
            "크기": "작음",
            "추천": "⭐⭐"
        }
    }
    
    for field, info in additional_fields.items():
        print(f"\n  • {field}")
        print(f"    - 설명: {info['설명']}")
        print(f"    - 용도: {info['용도']}")
        print(f"    - 데이터 크기: {info['크기']}")
        print(f"    - 추천도: {info['추천']}")
    
    print()
    print()
    print("💡 추천 추가 필드 (실용적)")
    print("─" * 70)
    
    recommended = [
        "class_probabilities - 정상/욕설 각각의 확률",
        "token_count - 입력 토큰 개수",
        "is_truncated - 300 토큰 초과 여부",
        "abusive_words_found - 감지된 욕설 단어 목록",
        "severity_level - 심각도 등급 (낮음/중간/높음)"
    ]
    
    for i, rec in enumerate(recommended, 1):
        print(f"  {i}. {rec}")
    
    print()
    print()
    print("⚡ 성능 고려사항")
    print("─" * 70)
    print("  • hidden_states, attention_weights는 매우 큰 데이터")
    print("  • 일반 사용에는 필요 없음 (연구/시각화 목적)")
    print("  • 기본 필드 + 추천 필드만으로도 충분")
    print("  • 필요시 output_hidden_states=True로 활성화 가능")
    print()
    
    print("=" * 70)
    
    # 실제 예제
    print()
    print("📝 실제 활용 예제")
    print("─" * 70)
    print()
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        tokens = tokenizer.tokenize(test_text)
        token_count = len(tokens)
        is_truncated = token_count > 300
        
        # 예제 결과
        example_result = {
            # 기존 필드
            "text": test_text,
            "is_abusive": True,
            "confidence": 0.89,
            "abusive_score": 0.85,
            "model_score": 0.78,
            "rule_score": 0.95,
            "threshold": 0.5,
            "processing_time": 0.035,
            
            # 추가 가능한 필드 (추천)
            "class_probabilities": {
                "normal": probabilities[0][0].item(),
                "abusive": probabilities[0][1].item()
            },
            "token_count": token_count,
            "is_truncated": is_truncated,
            "abusive_words_found": ["화나네요", "뭐하는 짓"],
            "severity_level": "높음",
            "detection_method": "혼합"
        }
        
        import json
        print(json.dumps(example_result, ensure_ascii=False, indent=2))
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    test_kcbert_outputs()
