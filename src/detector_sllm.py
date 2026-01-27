"""
sLLM 기반 욕설/폭언 감지 엔진
GGUF 형식 모델 사용 (llama.cpp)
"""

import time
import os
from typing import Dict, Any, List


class SLLMAbusiveDetector:
    """
    sLLM 기반 욕설/폭언 감지 엔진
    Midm-2.0-Mini-Instruct 4B 모델 사용
    """
    
    def __init__(self,
                 model_path: str = "./models/Midm-2.0-Mini-Instruct-Q4_K_M.gguf",
                 threshold: float = 0.5,
                 n_ctx: int = 2048,
                 n_threads: int = None,
                 verbose: bool = False):
        """
        Args:
            model_path: GGUF 모델 파일 경로
            threshold: 감지 임계값
            n_ctx: 컨텍스트 길이
            n_threads: CPU 스레드 수 (None=자동)
            verbose: 상세 로그 출력
        """
        self.model_path = model_path
        self.threshold = threshold
        self.n_ctx = n_ctx
        self.n_threads = n_threads or os.cpu_count()
        self.verbose = verbose
        
        self.llm = None
        
        # 프롬프트 템플릿
        self.system_prompt = """당신은 고객 서비스 품질 관리 및 통화 분석 전문가입니다.
통화 내용을 분석하여 부적절한 발언이 있는지 종합적으로 평가해주세요.

【평가 대상】
1. 욕설/비속어: 명시적인 욕설 및 비속어 사용
2. 모욕/폭언: 상대방을 깎아내리거나 공격하는 표현
3. 성희롱: 외모 평가, 성적 제안, 개인적 질문 등
4. 위협: 협박, 위협적 표현

【평가 방법】
- 명확한 표현뿐만 아니라 문맥과 의도를 종합적으로 판단
- 간접적이거나 우회적인 표현도 의도가 명확하면 감지
- 단어만 보지 말고 전체 맥락에서 발화자의 의도 파악
- 정상적인 불만이나 건의는 부적절하지 않음

【정상 vs 부적절 구분】
정상적 표현:
- "불편합니다", "개선이 필요합니다", "답답하네요"
- "빠른 처리 부탁드립니다", "실망스럽습니다"

부적절한 표현:
- 직접적 욕설: "씨발", "병신", "개새끼" 등
- 모욕적 표현: "쓰레기", "인간 말종", "미친놈" 등
- 성희롱: "예쁘네요", "남자친구 있어요?", "몸매 좋으시네요" 등
- 위협: "죽이고 싶다", "가만 안 둔다" 등

【채점 기준】
0.0-0.2: 전혀 부적절하지 않음 (정상적인 대화)
0.3-0.4: 약간 거친 표현이나 불만 (경계선)
0.5-0.6: 부적절한 의도가 있으나 경미함 (주의 필요)
0.7-0.8: 명백히 부적절함 (경고)
0.9-1.0: 매우 심각한 부적절함 (즉시 조치 필요)

반드시 아래 형식으로만 응답하세요:
점수: [0.0~1.0 사이 숫자]
판단: [부적절/정상]
카테고리: [욕설/모욕/성희롱/위협/복합/없음]
이유: [문맥과 의도를 고려한 구체적 판단 근거]"""
        
    def load_model(self):
        """모델 로드"""
        if self.llm is None:
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python이 설치되지 않았습니다.\n"
                    "설치: pip install llama-cpp-python"
                )
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"모델 파일을 찾을 수 없습니다: {self.model_path}\n"
                    "models/ 폴더에 GGUF 모델을 배치하세요."
                )
            
            print(f"\n{'='*60}")
            print("🤖 sLLM 모델 로딩 중...")
            print(f"{'='*60}\n")
            print(f"📦 모델: {os.path.basename(self.model_path)}")
            print(f"🧵 스레드: {self.n_threads}")
            print(f"📝 컨텍스트: {self.n_ctx}")
            print()
            
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=self.verbose,
                n_gpu_layers=0  # CPU only (GPU 사용 시 값 조정)
            )
            
            print(f"\n{'='*60}")
            print("✅ sLLM 모델 로딩 완료!")
            print(f"{'='*60}\n")
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        LLM 응답 파싱
        
        예상 형식:
        점수: 0.85
        판단: 부적절
        카테고리: 욕설
        이유: "씨발", "병신" 등의 욕설 포함
        """
        lines = response.strip().split('\n')
        
        score = 0.5
        is_abusive = False
        category = "없음"
        reason = ""
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('점수:'):
                try:
                    score_str = line.split(':')[1].strip()
                    score = float(score_str)
                    # 점수가 0.5 이상이면 부적절로 판단
                    is_abusive = score >= self.threshold
                except:
                    pass
            
            elif line.startswith('판단:'):
                judgment = line.split(':')[1].strip()
                # "부적절" 또는 "욕설 있음" 등의 표현 감지
                is_abusive = any(keyword in judgment for keyword in ['부적절', '있음', '감지', '발견'])
            
            elif line.startswith('카테고리:'):
                category = line.split(':', 1)[1].strip()
            
            elif line.startswith('이유:'):
                reason = line.split(':', 1)[1].strip()
        
        return {
            'score': score,
            'is_abusive': is_abusive,
            'category': category,
            'reason': reason
        }
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        텍스트 분석
        """
        if self.llm is None:
            self.load_model()
        
        start_time = time.time()
        
        # 프롬프트 구성
        prompt = f"""<|im_start|>system
{self.system_prompt}<|im_end|>
<|im_start|>user
다음 통화 내용을 종합적으로 분석해주세요.
명시적 표현뿐만 아니라 문맥과 의도를 깊이 파악하여 점수를 매겨주세요.

통화 내용:
"{text}"

위 내용에서:
1. 욕설, 모욕, 폭언, 성희롱 등의 부적절한 표현이 있나요?
2. 직접적 표현이 없더라도 그러한 의도가 담겨있나요?
3. 전체 맥락에서 발화자의 태도와 의도는 어떤가요?
<|im_end|>
<|im_start|>assistant
"""
        
        # LLM 추론
        response = self.llm(
            prompt,
            max_tokens=256,
            temperature=0.1,  # 낮은 온도로 일관성 확보
            top_p=0.9,
            stop=["<|im_end|>", "\n\n\n"],
            echo=False
        )
        
        response_text = response['choices'][0]['text'].strip()
        
        # 응답 파싱
        parsed = self._parse_response(response_text)
        
        processing_time = time.time() - start_time
        
        # 결과 구성
        result = {
            "text": text,
            "is_abusive": parsed['is_abusive'] or parsed['score'] >= self.threshold,
            "confidence": 1.0 - abs(parsed['score'] - 0.5) * 2,  # 0.5에서 멀수록 확신
            "abusive_score": parsed['score'],
            "category": parsed.get('category', '없음'),  # 카테고리 추가
            "threshold": self.threshold,
            "processing_time": processing_time,
            "model_type": "sLLM",
            "model_name": os.path.basename(self.model_path),
            "reason": parsed['reason'],
            "raw_response": response_text
        }
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """배치 예측"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
    
    def predict_file(self, filepath: str) -> Dict[str, Any]:
        """파일에서 읽어서 예측"""
        from .preprocessor import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        text = preprocessor.preprocess_file(filepath)
        
        result = self.predict(text)
        result["source_file"] = filepath
        
        return result
    
    def __del__(self):
        """소멸자 - 모델 정리"""
        if self.llm is not None:
            del self.llm
