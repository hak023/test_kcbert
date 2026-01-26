# -*- coding: utf-8 -*-
"""
다중 카테고리 감지기
욕설/폭언, 성희롱을 동시에 판단
"""

import re
import time
from typing import Dict, Any, List
from .detector import AbusiveDetector


class MultiCategoryDetector(AbusiveDetector):
    """
    다중 카테고리 감지기 (욕설/폭언 + 성희롱)
    기존 AbusiveDetector를 확장하여 성희롱 탐지 추가
    """
    
    def __init__(self,
                 model_name: str = "beomi/kcbert-base",
                 cache_dir: str = "./models/kcbert",
                 threshold: float = 0.5,
                 max_length: int = 300):
        """초기화"""
        super().__init__(
            model_name=model_name,
            cache_dir=cache_dir,
            threshold=threshold,
            max_length=max_length
        )
        
        # 성희롱 패턴 정의
        self._init_harassment_patterns()
    
    def _init_harassment_patterns(self):
        """성희롱 관련 패턴 초기화"""
        
        # 심각한 성희롱 (즉시 높은 점수)
        self.severe_harassment_patterns = [
            r'강간', r'성폭행', r'성관계', r'성행위',
            r'몸\s*만지', r'몸\s*[봐보]', r'옷\s*벗',
            r'가슴\s*만지', r'엉덩이\s*만지',
            r'키스\s*[하해]', r'포옹\s*[하해]'
        ]
        
        # 중간 수준 성희롱
        self.moderate_harassment_patterns = [
            # 성적 제안/암시
            r'같이\s*자[자요자요]', r'호텔\s*가[자요]', r'모텔',
            r'원나잇', r'섹스', r'잠자리', r'밤\s*같이',
            
            # 신체 언급
            r'가슴', r'엉덩이', r'몸매', r'바디',
            r'섹시하?[네다요]', r'색시하?[네다요]',
            r'스타일\s*좋', r'몸\s*좋',
            
            # 외모 평가 (과도한)
            r'이쁘?[다네요].*같이', r'예쁘?[다네요].*같이',
            r'귀엽?[다네요].*같이',
            
            # 개인적 질문 (복합)
            r'남자친구.*있[어냐니]', r'여자친구.*있[어냐니]',
            r'혼자.*[사살]?[냐니].*같이',
            
            # 은어/비속어
            r'꼬시', r'작업\s*걸', r'헌팅', r'픽업',
            r'[따]?먹[어을]'  # 성적 의미
        ]
        
        # 경미한 성희롱 (단독으로는 낮은 점수, 복합 시 상승)
        self.minor_harassment_patterns = [
            # 외모 평가 (단독)
            r'예쁘?[네다요]', r'이쁘?[네다요]', r'귀엽?[네다요]',
            r'잘\s*생[겼기]', r'멋[있지]',
            
            # 개인적 질문 (단독)
            r'결혼.*했?[어니냐]', r'나이.*몇',
            r'사[는니].*어디', r'집.*어디',
            
            # 칭찬 (과도한)
            r'매력[적있]', r'멋[지있]', r'끌[려리]'
        ]
        
        # 화이트리스트 (오탐 방지)
        self.harassment_whitelist = [
            r'섹시한?\s*디자인',  # 제품 설명
            r'섹시한?\s*이미지',
            r'섹시한?\s*컨셉',
            r'예쁘?게\s*포장',  # 서비스 표현
            r'예쁘?게\s*만[들들]',
            r'예쁘?[다네]\s*제품',
            r'예쁘?[다네]\s*상품',
            r'귀엽?[다네]\s*디자인',
            r'스타일\s*좋[은은].*제품',  # 제품 평가
            r'몸매\s*좋[은은].*디자인'   # 디자인 표현
        ]
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        다중 카테고리 예측
        
        Args:
            text: 입력 텍스트
            
        Returns:
            욕설/폭언 + 성희롱 감지 결과
        """
        start_time = time.time()
        
        # 기존 욕설/폭언 감지
        abusive_result = super().predict(text)
        
        # 성희롱 감지
        harassment_result = self._detect_sexual_harassment(text)
        
        # 전체 처리 시간
        total_time = time.time() - start_time
        
        # 결과 통합
        result = {
            # 원본 텍스트
            "text": text,
            
            # 욕설/폭언
            "is_abusive": abusive_result['is_abusive'],
            "abusive_score": abusive_result['abusive_score'],
            "abusive_confidence": abusive_result['confidence'],
            
            # 성희롱
            "is_sexual_harassment": harassment_result['is_harassment'],
            "harassment_score": harassment_result['harassment_score'],
            "harassment_level": harassment_result['level'],
            
            # 전체 부적절성
            "is_inappropriate": (
                abusive_result['is_abusive'] or 
                harassment_result['is_harassment']
            ),
            "max_severity": max(
                abusive_result['abusive_score'],
                harassment_result['harassment_score']
            ),
            
            # 카테고리
            "categories": self._categorize_issues(
                abusive_result['abusive_score'],
                harassment_result['harassment_score']
            ),
            
            # 상세 정보
            "details": {
                "abusive_words": abusive_result.get('matched_patterns', []),
                "harassment_words": harassment_result['matched_words'],
                "model_score": abusive_result.get('model_score', 0),
                "rule_score": abusive_result.get('rule_score', 0)
            },
            
            # 기타 정보
            "threshold": self.threshold,
            "processing_time": total_time
        }
        
        return result
    
    def _detect_sexual_harassment(self, text: str) -> Dict[str, Any]:
        """성희롱 감지"""
        
        # 화이트리스트 체크 (정상적인 표현)
        for pattern in self.harassment_whitelist:
            if re.search(pattern, text, re.IGNORECASE):
                return {
                    "is_harassment": False,
                    "harassment_score": 0.0,
                    "level": "정상",
                    "matched_words": []
                }
        
        # 패턴 매칭
        severe_matches = []
        moderate_matches = []
        minor_matches = []
        
        # 심각한 성희롱
        for pattern in self.severe_harassment_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                severe_matches.extend(matches)
        
        # 중간 수준
        for pattern in self.moderate_harassment_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                moderate_matches.extend(matches)
        
        # 경미한 수준
        for pattern in self.minor_harassment_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                minor_matches.extend(matches)
        
        # 점수 계산
        score, level = self._calculate_harassment_score(
            len(severe_matches),
            len(moderate_matches),
            len(minor_matches)
        )
        
        # 매칭된 단어
        all_matches = severe_matches + moderate_matches + minor_matches
        
        return {
            "is_harassment": score >= 0.5,
            "harassment_score": score,
            "level": level,
            "matched_words": list(set(all_matches))  # 중복 제거
        }
    
    def _calculate_harassment_score(
        self, 
        severe_count: int, 
        moderate_count: int, 
        minor_count: int
    ) -> tuple:
        """
        성희롱 점수 계산
        
        Returns:
            (score, level) 튜플
        """
        
        # 심각한 성희롱 1개라도 있으면 높은 점수
        if severe_count > 0:
            return 0.95, "매우 심각"
        
        # 중간 수준 복합
        if moderate_count >= 3:
            return 0.85, "심각"
        elif moderate_count == 2:
            return 0.7, "경고"
        elif moderate_count == 1:
            # 경미한 것과 함께 있으면 점수 상승
            if minor_count >= 2:
                return 0.65, "경고"
            else:
                return 0.5, "주의"
        
        # 경미한 수준만
        if minor_count >= 3:
            return 0.6, "주의"
        elif minor_count >= 2:
            return 0.45, "의심"
        elif minor_count == 1:
            return 0.3, "의심"
        
        # 매칭 없음
        return 0.0, "정상"
    
    def _categorize_issues(
        self, 
        abusive_score: float, 
        harassment_score: float
    ) -> List[str]:
        """이슈 카테고리 분류"""
        categories = []
        
        if abusive_score >= 0.5:
            categories.append("욕설/폭언")
        
        if harassment_score >= 0.5:
            categories.append("성희롱")
        
        if not categories:
            categories.append("정상")
        
        return categories
    
    def get_severity_description(self, result: Dict[str, Any]) -> str:
        """심각도 설명 텍스트 생성"""
        
        if not result['is_inappropriate']:
            return "✅ 정상적인 대화입니다."
        
        issues = []
        
        if result['is_abusive']:
            issues.append(f"욕설/폭언 (점수: {result['abusive_score']:.2f})")
        
        if result['is_sexual_harassment']:
            issues.append(
                f"성희롱 [{result['harassment_level']}] "
                f"(점수: {result['harassment_score']:.2f})"
            )
        
        desc = "⚠️ 부적절한 발언이 감지되었습니다.\n"
        desc += "\n".join(f"  - {issue}" for issue in issues)
        
        return desc
