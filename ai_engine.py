"""
AI 추천 시스템 모듈
개선된 Gemini API 연동 및 팩터 추천 기능 제공
"""

import requests
import json
from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv
import time
import openai

# 환경 변수 로드
load_dotenv()

class AIRecommendationEngine:
  """AI 기반 팩터 추천 엔진"""
  
  PROMPT_TEMPLATE: str = (
    """
    당신은 퀀트 투자 전문가입니다. 다음 투자 아이디어를 분석하여 구체적인 알파 팩터를 추천해주세요.
    
    **투자 아이디어:**
    {user_idea}
    
    **위험 선호도:** {risk_description}
    
    **시장 상황:** {market_context}
    
    다음 형식으로 JSON 응답을 제공해주세요:
    ... (이하 기존 프롬프트 내용 동일)
    """
  )

  def __init__(self, api_key: Optional[str] = None) -> None:
    self.api_key: Optional[str] = api_key or os.getenv('OPENAI_API_KEY')
    self.model: str = "gpt-4.1-mini"
    self.client = openai.OpenAI(api_key=self.api_key)
    self.factor_categories: Dict[str, List[str]] = {
      'momentum': ['수익률', '가격 모멘텀', 'RSI', 'MACD', '상대강도'],
      'value': ['PER', 'PBR', 'ROE', 'ROA', '배당수익률', '장부가치'],
      'quality': ['부채비율', '유동비율', '매출성장률', '이익성장률', '안정성'],
      'volatility': ['변동성', 'VIX', '베타', '표준편차', '샤프비율'],
      'volume': ['거래량', '회전율', '거래대금', 'OBV', '거래량가중평균가격'],
      'technical': ['이동평균', '볼린저밴드', 'ATR', '스토캐스틱', '윌리엄스%R'],
      'macro': ['금리', '환율', '유가', '인플레이션', '경제지표']
    }
    self.formula_templates: List[str] = [
      "({feature1} / {feature2})",
      "({feature1} - {feature2}) / {feature2}",
      "rolling_mean({feature1}, {window})",
      "({feature1} - rolling_mean({feature1}, {window})) / rolling_std({feature1}, {window})",
      "rank({feature1}) / len({feature1})",
      "log({feature1} / shift({feature1}, 1))",
      "({feature1} * {feature2}) / ({feature3} + 1e-6)"
    ]

  def recommend_factors(self, prompt: str, max_tokens: int = 512) -> str:
    """
    OpenAI GPT-4.1-mini API를 사용하여 팩터 추천 결과를 반환합니다.
    """
    if not self.api_key:
      raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
    try:
      response = self.client.chat.completions.create(
        model=self.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7
      )
      return response.choices[0].message.content.strip()
    except Exception as e:
      raise RuntimeError(f"OpenAI API 호출 실패: {e}")

  def _create_enhanced_prompt(
    self, 
    user_idea: str,
    market_context: Optional[str],
    risk_preference: str
  ) -> str:
    """
    프롬프트 생성 (템플릿 활용)
    """
    risk_descriptions = {
      'low': '안정적이고 보수적인 투자 접근',
      'medium': '균형잡힌 위험-수익 접근',
      'high': '공격적이고 고수익 추구 접근'
    }
    prompt = self.PROMPT_TEMPLATE.format(
      user_idea=user_idea,
      risk_description=risk_descriptions.get(risk_preference, '균형잡힌'),
      market_context=market_context or '일반적인 시장 환경'
    )
    return prompt

  def _call_gemini_api(self, prompt: str, max_retries: int = 3) -> str:
    """
    Gemini API 호출
    """
    if not self.api_key:
      raise ValueError("Gemini API 키가 설정되지 않았습니다.")
    headers = {"Content-Type": "application/json"}
    data = {
      "contents": [{"parts": [{"text": prompt}]}],
      "generationConfig": {
        "temperature": 0.7,
        "maxOutputTokens": 2048,
        "topP": 0.8,
        "topK": 40
      }
    }
    params = {"key": self.api_key}
    for attempt in range(max_retries):
      try:
        response = requests.post(
          self.base_url, 
          headers=headers, 
          params=params, 
          json=data,
          timeout=30
        )
        response.raise_for_status()
        result = response.json()
        if 'candidates' in result and len(result['candidates']) > 0:
          content = result['candidates'][0]['content']['parts'][0]['text']
          return content
        else:
          raise ValueError("API 응답에서 콘텐츠를 찾을 수 없습니다.")
      except requests.exceptions.RequestException as e:
        if attempt < max_retries - 1:
          time.sleep(2 ** attempt)
          continue
        else:
          raise RuntimeError(f"Gemini API 호출 실패: {e}")

  def _parse_recommendations(self, response_text: str) -> Dict[str, Any]:
    """
    API 응답 파싱
    - JSON 파싱 실패 시 부분적 정보라도 최대한 추출
    - 주니어 개발자: 실제 서비스에서는 예외 발생 시 fallback 추천을 제공해야 함
    """
    text = response_text.strip()
    if text.startswith("```json"):
      text = text[7:]
    if text.startswith("```"):
      text = text[3:]
    if text.endswith("```"):
      text = text[:-3]
    try:
      recommendations = json.loads(text)
      required_fields = ['features', 'formulas', 'ml_methods']
      for field in required_fields:
        if field not in recommendations:
          recommendations[field] = []
      return recommendations
    except json.JSONDecodeError:
      return self._extract_partial_recommendations(response_text)

  def _extract_partial_recommendations(self, text: str) -> Dict[str, Any]:
    """
    부분적 추천 정보 추출
    """
    recommendations: Dict[str, Any] = {
      'analysis': '텍스트에서 추천 정보를 추출했습니다.',
      'features': [],
      'formulas': [],
      'ml_methods': [],
      'risk_considerations': [],
      'implementation_tips': []
    }
    lines = text.split('\n')
    for line in lines:
      line = line.strip()
      for category, features in self.factor_categories.items():
        for feature in features:
          if feature.lower() in line.lower():
            recommendations['features'].append({
              'name': feature,
              'category': category,
              'description': f'{feature} 관련 팩터',
              'importance': 'medium'
            })
    if not recommendations['formulas']:
      recommendations['formulas'] = [
        {
          'expression': 'Close / SMA_20',
          'description': '현재가 대비 20일 이동평균 비율',
          'rationale': '가격 모멘텀 측정',
          'risk_level': 'medium'
        }
      ]
    if not recommendations['ml_methods']:
      recommendations['ml_methods'] = [
        {
          'method': 'RandomForest',
          'description': '랜덤포레스트 기반 피처 중요도 분석',
          'features_needed': ['Close', 'Volume', 'RSI'],
          'complexity': 'medium'
        }
      ]
    return recommendations

  def _enhance_recommendations(
    self, 
    recommendations: Dict[str, Any],
    user_idea: str
  ) -> Dict[str, Any]:
    """
    추천 결과 보완 및 검증
    - 수식/피처/ML 방법의 유효성 robust하게 검증 및 보완
    - 디폴트 값 추가 시 중복/불필요 항목 제거
    - 주니어 개발자: 추천 결과가 부족할 때는 디폴트 예시를 추가해 사용자가 혼란스럽지 않게 해야 함
    """
    if 'features' in recommendations:
      seen_features = set()
      unique_features = []
      for feature in recommendations['features']:
        if isinstance(feature, dict) and 'name' in feature:
          if feature['name'] not in seen_features:
            seen_features.add(feature['name'])
            unique_features.append(feature)
      recommendations['features'] = unique_features
    if 'formulas' in recommendations:
      valid_formulas = []
      for formula in recommendations['formulas']:
        if isinstance(formula, dict) and 'expression' in formula:
          if self._validate_formula(formula['expression']):
            valid_formulas.append(formula)
      recommendations['formulas'] = valid_formulas
    # 디폴트 값 추가(중복 방지)
    if len(recommendations.get('features', [])) < 3:
      defaults = self._get_default_features()
      exist_names = {f['name'] for f in recommendations['features']}
      recommendations['features'].extend([f for f in defaults if f['name'] not in exist_names])
    if len(recommendations.get('formulas', [])) < 2:
      defaults = self._get_default_formulas()
      exist_expr = {f['expression'] for f in recommendations['formulas']}
      recommendations['formulas'].extend([f for f in defaults if f['expression'] not in exist_expr])
    if len(recommendations.get('ml_methods', [])) < 2:
      defaults = self._get_default_ml_methods()
      exist_methods = {m['method'] for m in recommendations['ml_methods']}
      recommendations['ml_methods'].extend([m for m in defaults if m['method'] not in exist_methods])
    return recommendations

  def _validate_formula(self, formula: str) -> bool:
    """
    수식 유효성 검증
    - 위험한 키워드, 허용 문자 robust하게 검증
    - 주니어 개발자: eval, exec, import 등 위험 키워드는 반드시 차단해야 함
    """
    try:
      allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-*/().(), ')
      if not all(c in allowed_chars for c in formula.replace(' ', '')):
        return False
      dangerous_keywords = ['import', 'exec', 'eval', '__', 'open', 'file']
      formula_lower = formula.lower()
      if any(keyword in formula_lower for keyword in dangerous_keywords):
        return False
      return True
    except Exception:
      return False

  def _get_fallback_recommendations(self, user_idea: str) -> Dict[str, Any]:
    """
    기본 추천 결과 반환
    """
    return {
      'analysis': f'"{user_idea}"에 대한 기본 팩터 추천을 제공합니다.',
      'recommended_categories': ['momentum', 'value', 'quality'],
      'features': self._get_default_features(),
      'formulas': self._get_default_formulas(),
      'ml_methods': self._get_default_ml_methods(),
      'risk_considerations': ['시장 변동성', '유동성 위험', '모델 위험'],
      'implementation_tips': ['충분한 백테스팅 수행', '리스크 관리 적용', '정기적 모델 업데이트']
    }

  def _get_default_features(self) -> List[Dict[str, str]]:
    """
    기본 피처 목록
    - 실제 사용에 적합한 예시/설명/리스크 등 보강
    """
    return [
      {'name': 'RSI', 'category': 'momentum', 'description': '상대강도지수', 'importance': 'high'},
      {'name': 'PBR', 'category': 'value', 'description': '주가순자산비율', 'importance': 'high'},
      {'name': 'ROE', 'category': 'quality', 'description': '자기자본이익률', 'importance': 'medium'},
      {'name': 'Volume', 'category': 'volume', 'description': '거래량', 'importance': 'medium'},
      {'name': 'ATR', 'category': 'volatility', 'description': '평균진폭', 'importance': 'low'}
    ]

  def _get_default_formulas(self) -> List[Dict[str, str]]:
    """
    기본 수식 목록
    - 실제 사용에 적합한 예시/설명/리스크 등 보강
    """
    return [
      {
        'expression': 'Close / SMA_20',
        'description': '현재가 대비 20일 이동평균 비율',
        'rationale': '단기 가격 모멘텀 측정',
        'risk_level': 'medium'
      },
      {
        'expression': 'ROE / PBR',
        'description': 'ROE 대비 PBR 비율',
        'rationale': '수익성 대비 밸류에이션 효율성',
        'risk_level': 'low'
      },
      {
        'expression': '(Close - SMA_50) / ATR',
        'description': '50일 이동평균 대비 편차의 ATR 정규화',
        'rationale': '변동성 조정 모멘텀',
        'risk_level': 'medium'
      }
    ]

  def _get_default_ml_methods(self) -> List[Dict[str, Any]]:
    """
    기본 ML 방법 목록
    - 실제 사용에 적합한 예시/설명/리스크 등 보강
    """
    return [
      {
        'method': 'RandomForest',
        'description': '랜덤포레스트 기반 피처 중요도 분석',
        'features_needed': ['Close', 'Volume', 'RSI', 'PBR'],
        'complexity': 'medium'
      },
      {
        'method': 'PCA',
        'description': '주성분 분석을 통한 차원 축소',
        'features_needed': ['Close', 'Volume', 'RSI', 'MACD', 'ATR'],
        'complexity': 'low'
      },
      {
        'method': 'MLP',
        'description': '다층 퍼셉트론 신경망',
        'features_needed': ['Close', 'Volume', 'RSI', 'PBR', 'ROE'],
        'complexity': 'high'
      }
    ]

# 전역 AI 추천 엔진
# (Streamlit 캐시 함수는 app.py 등 UI 계층에서만 사용)

