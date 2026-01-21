"""
테스트용 빠른 실행 스크립트
"""

from src.detector import AbusiveDetector

# 간단한 테스트
if __name__ == "__main__":
    print("KcBERT 욕설 감지 시스템 테스트")
    print("=" * 60)
    
    detector = AbusiveDetector(threshold=0.5)
    
    test_texts = [
        "안녕하세요. 문의 드립니다.",
        "야 이 병신아. 빨리 해.",
        "정말 답답하네요. 빨리 처리 부탁드립니다.",
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n[테스트 {i}] {text}")
        result = detector.predict(text)
        print(f"  → 욕설 감지: {result['is_abusive']}")
        print(f"  → 공격성 점수: {result['abusive_score']:.3f}")
