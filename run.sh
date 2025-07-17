#!/bin/bash

# 알파 팩터 생성기 v2.0 실행 스크립트

echo "🚀 알파 팩터 생성기 v2.0 시작"
echo "=================================="

# 가상환경 확인
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ 가상환경 활성화됨: $VIRTUAL_ENV"
else
    echo "⚠️  가상환경이 활성화되지 않았습니다."
    echo "   권장: python -m venv venv && source venv/bin/activate"
fi

# 의존성 확인
echo "📦 의존성 확인 중..."
if python -c "import streamlit, pandas, numpy, yfinance, plotly" 2>/dev/null; then
    echo "✅ 필수 패키지 설치됨"
else
    echo "❌ 필수 패키지 누락. 설치 중..."
    pip install -r requirements.txt
fi

# 환경 변수 파일 확인
if [ ! -f .env ]; then
    echo "⚠️  .env 파일이 없습니다. 예시 파일을 복사합니다."
    cp .env.example .env
    echo "📝 .env 파일을 편집하여 API 키를 설정하세요."
fi

# 데이터 디렉토리 생성
mkdir -p data/cache
echo "📁 데이터 디렉토리 생성 완료"

# Streamlit 앱 실행
echo "🌐 웹 애플리케이션 시작..."
echo "   브라우저에서 http://localhost:8501 을 열어주세요."
echo ""

streamlit run app.py

