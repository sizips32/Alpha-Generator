"""
백테스트 엔진 모듈
- 팩터 기반 포트폴리오 전략의 성과를 robust하게 평가
- 인덱스/NaN/inf/극단값/랭킹/포지션/누적수익률/성과지표/동적 결합/에러 robust 처리
- 주니어 개발자용 상세 주석/설명 포함
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

class BacktestEngine:
    """
    팩터 기반 포트폴리오 백테스트 엔진
    - 팩터, 가격데이터, future_return을 받아 전략 성과를 robust하게 평가
    - 동적 결합(메가-알파) 등 확장성 구조
    """
    def __init__(self):
        pass

    def run_backtest(
        self,
        factor: pd.Series,
        price_data: pd.DataFrame,
        transaction_cost: float = 0.001,
        long_pct: float = 0.2,
        short_pct: float = 0.2
    ) -> Dict[str, Any]:
        """
        백테스트 실행
        - 인덱스/NaN/inf/극단값/랭킹/포지션/누적수익률/성과 robust 처리
        - 결과 dict 반환
        """
        # 1. 입력값 체크 및 인덱스 일치
        if factor is None or factor.empty:
            raise ValueError("팩터 시리즈가 비어있습니다.")
        if price_data is None or price_data.empty:
            raise ValueError("가격 데이터가 비어있습니다.")
        if 'future_return' not in price_data.columns:
            raise ValueError("'future_return' 컬럼이 price_data에 없습니다. 데이터 준비 단계를 확인하세요.")
        # 인덱스 일치
        if not factor.index.equals(price_data.index):
            factor = factor.reindex(price_data.index)
        # NaN/inf/극단값 보정
        factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
        price_data = price_data.replace([np.inf, -np.inf], np.nan).fillna(0)
        # 2. future_return 값 검증
        future_return = price_data['future_return']
        if future_return.isnull().all() or (future_return == 0).all():
            raise ValueError("'future_return' 값이 모두 NaN 또는 0입니다. 데이터 준비 단계를 확인하세요.")
        # 3. 일별 팩터 랭킹 계산
        # --- 인덱스 구조 robust 보정 ---
        if isinstance(price_data.index, pd.MultiIndex):
            # 인덱스에 'Date'와 'Stock'이 모두 있는지 확인
            if set(['Date', 'Stock']).issubset(price_data.index.names):
                # 인덱스 순서가 ('Stock', 'Date')가 아니면 맞춤
                if price_data.index.names != ['Stock', 'Date']:
                    price_data = price_data.reorder_levels(['Stock', 'Date'])
                    factor = factor.reorder_levels(['Stock', 'Date'])
            else:
                raise ValueError(f"price_data 인덱스에 'Stock', 'Date'가 모두 필요합니다. 현재: {price_data.index.names}")
            date_level = price_data.index.names.index('Date')
            ranked_factor = factor.groupby(price_data.index.get_level_values(date_level)).rank(pct=True)
            date_index = price_data.index.get_level_values(date_level)
        else:
            ranked_factor = factor.rank(pct=True)
            date_index = price_data.index
        # 4. 롱온리 포지션 수익률 계산
        mask_longs = (ranked_factor >= 1 - long_pct)
        mask_longs = mask_longs.reindex(future_return.index, fill_value=False)
        if isinstance(price_data.index, pd.MultiIndex):
            longs = future_return[mask_longs].groupby(level='Date').mean()
        else:
            longs = future_return[mask_longs].groupby(price_data.index).mean()
        daily_returns = longs - transaction_cost
        shorts = None  # 롱온리 전략이므로 숏은 사용하지 않음
        # 5. 벤치마크(시장 평균)
        benchmark_returns = future_return.groupby(date_index).mean()
        # 6. 누적수익률, 성과지표 robust 계산
        cumulative_returns = (1 + daily_returns.fillna(0)).cumprod()
        benchmark_cumulative = (1 + benchmark_returns.fillna(0)).cumprod()
        # 7. 결과값 검증 및 경고
        if cumulative_returns.isnull().all() or (cumulative_returns == 1).all():
            raise ValueError(
                f"누적수익률이 비정상적입니다.\n"
                f"longs 샘플: {longs.head(5)}\n"
                f"daily_returns 샘플: {daily_returns.head(5)}\n"
                f"future_return 샘플: {future_return.head(5)}\n"
                f"팩터 샘플: {factor.head(5)}\n"
                f"인덱스: longs={longs.index}, daily_returns={daily_returns.index}, future_return={future_return.index}, factor={factor.index}\n"
                f"long_pct={long_pct}, 데이터 길이={len(future_return)}"
            )
        if np.abs(cumulative_returns).max() > 1e6:
            raise Warning("누적수익률 값이 비정상적으로 큽니다. 팩터, 데이터, future_return을 다시 확인하세요.")
        # 8. 성과지표 계산
        annual_return = daily_returns.mean() * 252
        annual_volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        # 9. 결과 dict 반환
        return {
            'cumulative_returns': cumulative_returns,
            'benchmark_cumulative': benchmark_cumulative,
            'metrics': {
                "최종 수익률": cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else np.nan,
                "연평균 수익률": annual_return,
                "연 변동성": annual_volatility,
                "샤프 비율": sharpe_ratio,
                "최대 낙폭 (MDD)": max_drawdown
            },
            'daily_returns': daily_returns,
            'longs': longs,
            'shorts': shorts,
            'benchmark_returns': benchmark_returns
        }

    # (확장) 여러 팩터 조합(메가-알파) 지원 구조 예시
    def combine_factors(self, factors: list, method: str = 'mean') -> pd.Series:
        """
        여러 팩터의 랭킹을 조합하여 메가-알파 시리즈 생성
        - method: 'mean'(평균), 'median'(중앙값) 등
        """
        if not factors:
            raise ValueError("팩터 리스트가 비어있습니다.")
        df = pd.concat(factors, axis=1)
        if method == 'mean':
            return df.mean(axis=1)
        elif method == 'median':
            return df.median(axis=1)
        else:
            raise ValueError(f"지원하지 않는 조합 방법: {method}")
