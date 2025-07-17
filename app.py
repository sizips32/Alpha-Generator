import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from safe_expression_evaluator import SafeExpressionEvaluator
import ta  # Technical Analysis Library
import os
from dotenv import load_dotenv
import datetime
import time
import ast  # For SafeExpressionEvaluator
import operator  # For SafeExpressionEvaluator
import openai
import json
import re

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- 1. 기본 설정 및 전역 변수 ---
st.set_page_config(layout="wide", page_title="AlphaForge v2.0")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- 2. 핵심 기능 클래스 (모듈화된 아키텍처) ---
# 기존 직접 구현된 클래스/함수 삭제 및 외부 모듈 import로 대체
from data_manager import DataManager
from factor_engine import FactorGenerator, FactorAnalyzer
from backtest_engine import BacktestEngine
from ai_engine import AIRecommendationEngine

data_manager = DataManager()
factor_engine = FactorGenerator()
factor_analyzer = FactorAnalyzer()
backtest_engine = BacktestEngine()
ai_engine = AIRecommendationEngine(OPENAI_API_KEY)

# --- 세션 상태 관리 ---
if 'financial_data' not in st.session_state: st.session_state.financial_data = pd.DataFrame()
if 'factor_zoo' not in st.session_state: st.session_state.factor_zoo = {}

# --- 4. 페이지 렌더링 함수 ---
def render_page_data():
    st.header("📊 1단계: 데이터 관리")
    symbols = st.text_input("분석할 주식 티커 입력 (쉼표로 구분)", "NVDA, MSFT, AAPL, GOOGL, META, AMZN, TSLA").split(',')
    start, end = st.date_input("분석 기간", [datetime.date.today() - datetime.timedelta(days=730), datetime.date.today()])
    if st.button("데이터 로드 및 처리"):
        with st.spinner("데이터 로드 및 지표 계산 중..."):
            # 데이터 수집 및 전처리 전체 파이프라인 실행
            df = data_manager.get_market_data([s.strip() for s in symbols], period="2y", future_n=5)
            st.session_state.financial_data = df
    if not st.session_state.financial_data.empty:
        st.success("데이터 준비 완료! 총 {:,}개 행".format(len(st.session_state.financial_data)))
        st.dataframe(st.session_state.financial_data.head())

def render_page_ai():
    st.header("🤖 2단계: AI 팩터 추천")
    user_idea = st.text_area("투자 아이디어를 입력하세요:", "성장주이면서 변동성이 낮은 종목에 투자하고 싶습니다.")
    market = st.selectbox("시장 선택", ["코스피", "코스닥", "미국", "중국", "글로벌"])
    sector = st.multiselect("관심 섹터", ["IT", "헬스케어", "금융", "소비재", "에너지", "기타"])
    style = st.multiselect("선호 스타일", ["가치", "모멘텀", "퀄리티", "저변동성", "성장"])
    risk_pref = st.selectbox("위험 선호도:", ["low", "medium", "high"])
    target_metric = st.selectbox("중시하는 성과 지표", ["수익률", "샤프 비율", "IC"])
    backtest_period = st.slider("백테스트 기간(년)", 1, 10, 3)
    if st.button("AI 추천 받기"):
        if not OPENAI_API_KEY:
            st.warning("설정 페이지에서 OpenAI API 키를 입력하세요.")
        else:
            with st.spinner("Gemini AI를 통해 팩터 추천을 생성 중입니다..."):
                # AI 추천 프롬프트 생성 부분
                prompt = (
                    f"아이디어: {user_idea}\n"
                    f"시장 상황: {market}\n"
                    f"위험 성향: {risk_pref}\n"
                    "아래와 같은 JSON 형식으로만 답변하세요. 예시: {\"formulas\": [{\"description\": \"설명\", \"expression\": \"수식\", \"rationale\": \"근거\"}]}"
                )
                reco = ai_engine.recommend_factors(prompt)
                reco = extract_json_from_response(reco)
                try:
                    reco_dict = json.loads(reco)
                except Exception:
                    reco_dict = None

                if reco_dict and 'formulas' in reco_dict:
                    for formula in reco_dict['formulas']:
                        st.subheader("AI 분석 요약")
                        st.write(f"시장: {market}, 섹터: {', '.join(sector) if sector else '전체'}, 스타일: {', '.join(style) if style else '전체'}, 위험: {risk_pref}, 목표지표: {target_metric}, 백테스트 기간: {backtest_period}년")
                        st.write(f"'{user_idea}' 아이디어와 선택 옵션을 바탕으로 아래와 같은 팩터 수식을 추천합니다.")
                        st.subheader("추천 수식 및 설명")
                        st.markdown(f"**{formula.get('description', formula.get('expression'))}**")
                        st.code(formula['expression'], language='python')
                        st.caption(formula.get('rationale', ''))
                else:
                    st.warning("AI 추천 결과를 파싱할 수 없습니다. 원문을 그대로 출력합니다.")
                    st.write(reco)

def render_formula_factor_creation():
    """
    수식 기반 팩터 생성 UI 및 로직.
    - 컬럼 카테고리 안내
    - 템플릿 버튼
    - 수식 입력 및 실시간 미리보기
    - 팩터 생성
    """
    st.markdown("**가격 관련**: open, high, low, close")
    st.markdown("**거래량 관련**: volume")
    st.markdown("**기술적 지표**: momentum_rsi, trend_macd_diff, volatility_kc, trend_psar_up, SMA_20, ...")
    st.info("사용 가능 컬럼: " + ", ".join(st.session_state.financial_data.columns))

    # 1. 템플릿 버튼
    col1, col2, col3 = st.columns(3)
    if 'formula_input' not in st.session_state:
        st.session_state['formula_input'] = "log(volume + 1) * (close - open)"
    with col1:
        if st.button("모멘텀 팩터 예시"):
            st.session_state['formula_input'] = "close / SMA_20"
    with col2:
        if st.button("가치 팩터 예시"):
            st.session_state['formula_input'] = "roe / pbr"
    with col3:
        if st.button("변동성 팩터 예시"):
            st.session_state['formula_input'] = "1 / volatility_kc"

    # 2. 수식 입력
    formula = st.text_input("수식 입력", st.session_state['formula_input'], key="formula_input_box")
    name = st.text_input("팩터 이름", "거래량가중_가격변동")

    # 3. 실시간 미리보기 및 오류 안내
    preview_placeholder = st.empty()
    error_placeholder = st.empty()
    if formula:
        try:
            preview = factor_engine.create_formula_factor(st.session_state.financial_data, formula, name)
            preview_placeholder.line_chart(preview['data'].head(100))
            error_placeholder.empty()
        except Exception as e:
            preview_placeholder.empty()
            error_placeholder.error(f"수식 오류: {e}")

    # 4. 팩터 생성 버튼
    if st.button("수식 팩터 생성"):
        try:
            factor_result = factor_engine.create_formula_factor(st.session_state.financial_data, formula, name)
            factor = factor_result['data']
            meta = factor_result['meta']
            stats = factor_analyzer.calculate_factor_stats(factor, st.session_state.financial_data['future_return'])
            st.session_state.factor_zoo[name] = {
                'data': factor, 'meta': meta, 'ic': stats['ic'], 'icir': stats['icir'],
                'type': 'Formulaic', 'expression': formula
            }
            st.success(f"'{name}' 생성 완료! IC: {stats['ic']:.4f}, ICIR: {stats['icir']:.2f}")
        except Exception as e:
            st.error(f"팩터 생성 중 오류: {e}")

def train_random_forest(X_train, y_train, n_estimators):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, n_estimators, learning_rate):
    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train, n_estimators, learning_rate):
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def render_ml_factor_creation():
    st.markdown("**기술적 지표**: momentum_rsi, trend_macd_diff, volatility_kc, ...")
    st.markdown("**가격/거래량**: open, high, low, close, volume")

    algo = st.selectbox("ML 알고리즘", ["RandomForest", "XGBoost", "LightGBM"])
    # ML 피처 선택 안전 보정
    available_cols = list(st.session_state.financial_data.columns)
    default_features = [col for col in ['trend_macd_diff', 'momentum_rsi'] if col in available_cols]
    features = st.multiselect("ML 모델에 사용할 피처 선택", options=available_cols, default=default_features)
    test_ratio = st.slider("테스트 데이터 비율(%)", 10, 50, 20, step=5)
    name = st.text_input("ML 팩터 이름", f"{algo}_Factor_1")

    # 알고리즘별 하이퍼파라미터 UI
    if algo == "RandomForest":
        n_estimators = st.slider("트리 개수 (RandomForest)", 10, 200, 50, step=10)
    elif algo in ["XGBoost", "LightGBM"]:
        n_estimators = st.slider("트리 개수", 10, 200, 50, step=10)
        learning_rate = st.number_input("러닝레이트", 0.001, 1.0, 0.1, step=0.01)

    if st.button("ML 팩터 생성"):
        df_ml = st.session_state.financial_data[features + ['future_return']].dropna()
        if df_ml.empty:
            st.warning("ML 모델 학습을 위한 데이터가 부족합니다.")
        else:
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import r2_score
            X, y = df_ml[features], df_ml['future_return']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio/100, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            # 알고리즘별 학습
            if algo == "RandomForest":
                model = train_random_forest(X_train_scaled, y_train, n_estimators)
            elif algo == "XGBoost":
                try:
                    model = train_xgboost(X_train_scaled, y_train, n_estimators, learning_rate)
                except ImportError:
                    st.error("XGBoost 라이브러리가 설치되어 있지 않습니다. pip install xgboost")
                    return
            elif algo == "LightGBM":
                try:
                    model = train_lightgbm(X_train_scaled, y_train, n_estimators, learning_rate)
                except ImportError:
                    st.error("LightGBM 라이브러리가 설치되어 있지 않습니다. pip install lightgbm")
                    return
            else:
                st.error("지원하지 않는 알고리즘입니다.")
                return
            y_pred = model.predict(X_test_scaled)
            # 피처 중요도 시각화
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                imp_df = pd.DataFrame({'feature': features, 'importance': importances}).sort_values(by='importance', ascending=False)
                st.bar_chart(imp_df.set_index('feature'))
            # 예측력(설명력) 표시
            r2 = r2_score(y_test, y_pred)
            st.info(f"테스트 데이터 R2(설명력): {r2:.4f}")
            # 전체 데이터에 대해 예측값 생성
            X_all_scaled = scaler.transform(X)
            factor = pd.Series(model.predict(X_all_scaled), index=df_ml.index)
            stats = factor_analyzer.calculate_factor_stats(factor, y)
            ic, icir = stats['ic'], stats['icir']
            st.session_state.factor_zoo[name] = {
                'data': factor, 'meta': {'algorithm': algo, 'features': features, 'test_ratio': test_ratio}, 'ic': ic, 'icir': icir,
                'type': 'ML', 'expression': f"{algo}({n_estimators}) on {', '.join(features)}"
            }
            st.success(f"'{name}' 생성 완료! IC: {ic:.4f}, ICIR: {icir:.2f}")

def render_page_creation():
    st.header("✨ 3단계: 팩터 생성")
    if st.session_state.financial_data.empty:
        st.warning("1단계에서 데이터를 먼저 로드해주세요.")
        return
    tab1, tab2 = st.tabs(["수식 기반", "ML 기반"])
    with tab1:
        render_formula_factor_creation()
    with tab2:
        render_ml_factor_creation()

def render_page_backtest():
    st.header("📈 4단계: 백테스팅")
    if not st.session_state.factor_zoo: st.warning("3단계에서 팩터를 먼저 생성해주세요."); return
    factor_name = st.selectbox("백테스팅할 팩터 선택", list(st.session_state.factor_zoo.keys()))
    if st.button("백테스팅 실행"):
        try:
            results = backtest_engine.run_backtest(st.session_state.factor_zoo[factor_name]['data'], st.session_state.financial_data)
            st.session_state.factor_zoo[factor_name]['backtest'] = results # 결과 저장
            st.success("백테스팅 완료!")
        except Exception as e:
            st.error(f"백테스팅 중 오류 발생: {e}")
    if 'backtest' in st.session_state.factor_zoo.get(factor_name, {}):
        results = st.session_state.factor_zoo[factor_name]['backtest']
        metrics_df = pd.DataFrame([results['metrics']]).T.applymap(lambda x: f"{x:.2%}" if isinstance(x, (float, np.number)) else x)
        st.dataframe(metrics_df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results['cumulative_returns'].index, y=results['cumulative_returns'], name='My Strategy'))
        fig.add_trace(go.Scatter(x=results['benchmark_cumulative'].index, y=results['benchmark_cumulative'], name='Benchmark'))
        st.plotly_chart(fig, use_container_width=True)

def render_page_zoo():
    st.header("🐒 5단계: 팩터 동물원")
    if not st.session_state.factor_zoo: st.warning("아직 생성된 팩터가 없습니다."); return
    zoo_df = pd.DataFrame([
        {'팩터명': name, '유형': info['type'], 'IC': info['ic'], 'ICIR': info['icir'], '수식/설명': info['expression']}
        for name, info in st.session_state.factor_zoo.items()
    ]).sort_values(by='IC', ascending=False).set_index('팩터명')
    st.dataframe(zoo_df)

def render_page_mega_alpha():
    st.header("🏆 6단계: 메가 알파 생성")
    if len(st.session_state.factor_zoo) < 2: st.warning("메가 알파를 만들려면 팩터가 2개 이상 필요합니다."); return
    zoo_df = pd.DataFrame(st.session_state.factor_zoo).T.sort_values(by='ic', ascending=False)
    st.write("IC 기준 상위 팩터:")
    st.dataframe(zoo_df[['type', 'ic', 'icir', 'expression']])

    factor_count = len(zoo_df)
    if factor_count == 2:
        st.info("팩터가 2개뿐이므로 자동으로 2개를 조합합니다.")
        top_n = 2
    else:
        top_n = st.slider("상위 몇 개 팩터를 조합할까요?", 2, factor_count, min(5, factor_count))

    if st.button("메가 알파 생성 및 백테스팅"):
        top_factors = zoo_df.head(top_n)
        all_ranks = [info['data'].groupby('Date').rank(pct=True) for name, info in top_factors.iterrows()]
        mega_alpha = pd.concat(all_ranks, axis=1).mean(axis=1)
        mega_alpha.name = "Mega_Alpha"
        st.success(f"메가 알파 생성 완료! (조합된 팩터: {', '.join(top_factors.index)})")
        
        # 메가 알파 백테스팅
        try:
            results = backtest_engine.run_backtest(mega_alpha, st.session_state.financial_data)
            st.session_state.factor_zoo['Mega_Alpha'] = {'data': mega_alpha, 'meta': {'top_factors': top_factors.index.tolist()}, 'ic': np.nan, 'icir': np.nan, 'type': 'Combined', 'expression': f"Top {top_n} factors", 'backtest': results}
            st.experimental_rerun()
        except Exception as e:
            st.error(f"메가 알파 백테스팅 중 오류 발생: {e}")

def render_page_settings():
    st.header("⚙️ 설정")
    st.text_input("OpenAI API 키", value=OPENAI_API_KEY, key="api_key_input", type="password")
    if st.button("API 키 저장"):
        # 실제 앱에서는 이 키를 안전하게 저장/사용해야 합니다.
        os.environ["OPENAI_API_KEY"] = st.session_state.api_key_input
        st.success("API 키가 세션에 임시 저장되었습니다.")

def extract_json_from_response(response: str) -> str:
    match = re.search(r"```(?:json)?\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()

# --- 5. 사이드바 및 메인 라우팅 ---
# 단계별 사용법/해석법 안내 (expander)
with st.sidebar.expander("1. 데이터 관리 - 사용법/해석"):
    st.markdown("""
    **목적**  
    - 분석에 사용할 주식 데이터를 수집하고, 기술적 지표를 자동으로 계산합니다.

    **사용법**  
    - 티커(종목코드)와 기간을 입력 후 '데이터 로드 및 처리' 버튼을 누르세요.

    **해석법**  
    - 데이터프레임에 각 날짜별로 다양한 지표가 추가된 것을 확인할 수 있습니다.
    - 결측치/이상치가 자동으로 정제됩니다.
    """)

with st.sidebar.expander("2. AI 팩터 추천 - 사용법/해석"):
    st.markdown("""
    **목적**  
    - AI가 투자 아이디어와 옵션을 바탕으로 팩터 수식을 추천합니다.

    **사용법**  
    - 투자 아이디어, 시장, 섹터, 스타일, 목표지표 등을 선택 후 'AI 추천 받기'를 누르세요.

    **해석법**  
    - 추천 수식과 설명을 참고해 직접 팩터를 만들어볼 수 있습니다.
    """)

with st.sidebar.expander("3. 팩터 생성 - 사용법/해석"):
    st.markdown("""
    **목적**  
    - 직접 수식 또는 머신러닝으로 나만의 팩터를 만듭니다.

    **사용법**  
    - 수식 기반: 템플릿 버튼/직접 입력 후 미리보기로 확인, '수식 팩터 생성'  
    - ML 기반: 피처/알고리즘/옵션 선택 후 'ML 팩터 생성'

    **해석법**  
    - IC, ICIR 등 성과지표로 팩터의 예측력을 평가할 수 있습니다.
    """)

with st.sidebar.expander("4. 백테스팅 - 사용법/해석"):
    st.markdown("""
    **목적**  
    - 만든 팩터로 과거 데이터를 시뮬레이션하여 실제 투자 성과를 검증합니다.

    **사용법**  
    - 팩터를 선택 후 '백테스팅 실행' 버튼을 누르세요.

    **해석법**  
    - 수익률 곡선, 벤치마크 대비 성과, Sharpe, MDD 등 다양한 지표로 전략을 평가합니다.
    """)

with st.sidebar.expander("5. 팩터 동물원 - 사용법/해석"):
    st.markdown("""
    **목적**  
    - 생성된 모든 팩터를 한눈에 비교/관리합니다.

    **사용법**  
    - 팩터별 IC, ICIR, 수식, 유형 등을 표로 확인할 수 있습니다.

    **해석법**  
    - 성과가 좋은 팩터를 선별해 조합에 활용할 수 있습니다.
    """)

with st.sidebar.expander("6. 메가 알파 생성 - 사용법/해석"):
    st.markdown("""
    **목적**  
    - 여러 팩터를 조합해 시장에 적응하는 최적의 투자 시그널을 만듭니다.

    **사용법**  
    - 상위 팩터 개수 선택 후 '메가 알파 생성 및 백테스팅'을 누르세요.

    **해석법**  
    - 조합된 팩터의 성과를 백테스트로 확인할 수 있습니다.
    """)

with st.sidebar.expander("7. 설정 - 사용법"):
    st.markdown("""
    - Gemini API 키 등 환경설정을 할 수 있습니다.
    """)

st.sidebar.title("AlphaForge v2.0 🚀")
PAGES = {
    "1. 데이터 관리": render_page_data,
    "2. AI 팩터 추천": render_page_ai,
    "3. 팩터 생성": render_page_creation,
    "4. 백테스팅": render_page_backtest,
    "5. 팩터 동물원": render_page_zoo,
    "6. 메가 알파 생성": render_page_mega_alpha,
    "7. 설정": render_page_settings
}
selection = st.sidebar.radio("메뉴", list(PAGES.keys()))
page_func = PAGES[selection]
page_func()

st.sidebar.info("© 2025 Alpha Forge 연구 개발. 모든 권리 보유.")
