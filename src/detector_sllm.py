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
        self.system_prompt = """당신은 고객 서비스 품질 관리 전문가입니다.
통화 내용을 분석하여 욕설, 폭언, 공격적인 언어가 있는지 판단해주세요.

평가 기준:
- 욕설/비속어 사용 (씨발, 병신, 개새끼 등)
- 위협적 표현 (죽이고 싶다, 때리고 싶다 등)
- 심한 모욕 (쓰레기, 인간말종 등)

정상적인 불만 표현은 욕설이 아닙니다:
- "답답하네요", "불편합니다", "개선 필요합니다" 등

반드시 다음 형식으로만 응답하세요:
점수: [0.0에서 1.0 사이의 숫자]
판단: [욕설 있음/욕설 없음]
이유: [간단한 이유]"""
        
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
        판단: 욕설 있음
        이유: "씨발", "병신" 등의 욕설 포함
        """
        lines = response.strip().split('\n')
        
        score = 0.5
        is_abusive = False
        reason = ""
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('점수:'):
                try:
                    score_str = line.split(':')[1].strip()
                    score = float(score_str)
                except:
                    pass
            
            elif line.startswith('판단:'):
                judgment = line.split(':')[1].strip()
                is_abusive = '욕설 있음' in judgment or '있음' in judgment
            
            elif line.startswith('이유:'):
                reason = line.split(':', 1)[1].strip()
        
        return {
            'score': score,
            'is_abusive': is_abusive,
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
다음 통화 내용을 분석해주세요:

"{text}"
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
