"""
데이터 관리 모듈
실제 주식 데이터 수집, 처리, 캐싱 기능 제공
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
from typing import List, Dict, Optional, Tuple
import ta
from scipy import stats

class DataManager:
    """
    주식 데이터 관리 클래스
    - 데이터 수집, 기술/기본적 지표 추가, 데이터 정제 기능 제공
    """
    def __init__(self, cache_dir: str = "data/cache") -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_stock_data(
        self, 
        symbols: List[str], 
        start: Optional[str] = None, 
        end: Optional[str] = None, 
        period: str = "2y"
    ) -> pd.DataFrame:
        """
        yfinance 최신 API 기반 주식 데이터 수집
        - 복수 종목 동시 다운로드, 멀티인덱스 구조 표준화
        """
        if not isinstance(symbols, (list, tuple)) or not symbols:
            raise ValueError("종목 리스트가 비어있거나 잘못되었습니다.")
        # period와 start/end 동시 사용 불가
        yf_kwargs = {
            "tickers": symbols,
            "progress": False,
            "group_by": "ticker",
            "auto_adjust": True,
            "multi_level_index": True
        }
        if start and end:
            yf_kwargs["start"] = start
            yf_kwargs["end"] = end
        else:
            yf_kwargs["period"] = period

        df = yf.download(**yf_kwargs)
        if df.empty:
            raise RuntimeError("데이터를 로드할 수 없습니다. 심볼명, 기간을 확인하세요.")

        # 멀티인덱스: (Ticker, Date) → (Stock, Date)로 표준화
        if isinstance(df.columns, pd.MultiIndex):
            # OHLCV만 추출
            df = df.stack(level=0).rename_axis(['Date', 'Stock']).reset_index()
            df.set_index(['Stock', 'Date'], inplace=True)
        else:
            # 단일 종목일 때
            df['Stock'] = symbols[0]
            df.reset_index(inplace=True)
            df.set_index(['Stock', 'Date'], inplace=True)

        # 컬럼명 표준화
        rename_map = {col: col.capitalize() for col in ['open','high','low','close','volume']}
        df.rename(columns=rename_map, inplace=True)

        # 필수 컬럼 체크
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            raise RuntimeError("필수 컬럼이 누락되었습니다.")

        return df

    def add_future_return(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """
        종목별 미래수익률 컬럼(future_return) 생성
        - n일 후 종가 대비 수익률
        - NaN/inf/극단값 처리
        - 인덱스/컬럼명 일관성 보장
        """
        if df.empty or 'Close' not in df.columns:
            raise ValueError("입력 데이터가 비어있거나 'Close' 컬럼이 없습니다.")
        df = df.copy()
        # future_return 계산: (n일 뒤 종가 / 오늘 종가) - 1
        df['future_return'] = (
            df.groupby(level='Stock')['Close']
            .transform(lambda x: x.shift(-n) / x - 1)
        )
        # inf, -inf, 극단값 처리
        df['future_return'] = df['future_return'].replace([np.inf, -np.inf], np.nan)
        # 너무 큰 값(예: 1000% 이상)도 NaN 처리
        df.loc[df['future_return'].abs() > 10, 'future_return'] = np.nan
        # 결측치 보간
        df['future_return'] = df['future_return'].fillna(0)
        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        기술적 지표 추가
        - ta 패키지 기반 주요 지표
        - 사용자 정의/ta-lib 지표 확장 구조(가이드)
        - 컬럼명 일관성(snake_case)
        """
        result_dfs: List[pd.DataFrame] = []
        for symbol in df.index.get_level_values('Stock').unique():
            symbol_df = df.xs(symbol, level='Stock').copy()
            # 주요 기술적 지표
            close = symbol_df['Close']
            # Series 타입 체크
            if isinstance(close, pd.DataFrame):
                raise ValueError(
                    f"'Close' 컬럼이 DataFrame입니다. (컬럼: {list(close.columns)})\n"
                    f"단일 Series만 허용됩니다. 데이터 구조를 확인하세요."
                )
            returns = close.pct_change()
            if not isinstance(returns, pd.Series):
                raise ValueError(
                    f"returns 계산 결과가 Series가 아닙니다. (type: {type(returns)})"
                )
            symbol_df['returns'] = returns
            symbol_df['log_returns'] = np.log(symbol_df['Close'] / symbol_df['Close'].shift(1))
            symbol_df['sma_20'] = ta.trend.sma_indicator(symbol_df['Close'], window=20)
            symbol_df['sma_50'] = ta.trend.sma_indicator(symbol_df['Close'], window=50)
            symbol_df['ema_12'] = ta.trend.ema_indicator(symbol_df['Close'], window=12)
            symbol_df['ema_26'] = ta.trend.ema_indicator(symbol_df['Close'], window=26)
            symbol_df['rsi'] = ta.momentum.rsi(symbol_df['Close'], window=14)
            symbol_df['macd'] = ta.trend.macd_diff(symbol_df['Close'])
            symbol_df['stoch_k'] = ta.momentum.stoch(symbol_df['High'], symbol_df['Low'], symbol_df['Close'])
            symbol_df['bb_upper'] = ta.volatility.bollinger_hband(symbol_df['Close'])
            symbol_df['bb_middle'] = ta.volatility.bollinger_mavg(symbol_df['Close'])
            symbol_df['bb_lower'] = ta.volatility.bollinger_lband(symbol_df['Close'])
            symbol_df['atr'] = ta.volatility.average_true_range(
                symbol_df['High'], symbol_df['Low'], symbol_df['Close']
            )
            symbol_df['volume_sma'] = symbol_df['Volume'].rolling(window=20).mean()
            symbol_df['obv'] = ta.volume.on_balance_volume(symbol_df['Close'], symbol_df['Volume'])
            symbol_df['high_low_ratio'] = symbol_df['High'] / symbol_df['Low']
            symbol_df['close_open_ratio'] = symbol_df['Close'] / symbol_df['Open']
            # 사용자 정의/ta-lib 지표 확장 예시(주석)
            # if hasattr(ta, 'custom_indicator'):
            #     symbol_df['custom'] = ta.custom_indicator(...)
            symbol_df['Stock'] = symbol
            symbol_df.reset_index(inplace=True)
            symbol_df.set_index(['Date', 'Stock'], inplace=True)
            result_dfs.append(symbol_df)
        return pd.concat(result_dfs)

    def add_fundamental_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        기본적 분석 데이터 추가 (시뮬레이션)
        실제 구현시에는 외부 재무 API 연동 필요 (현재는 랜덤값)
        """
        np.random.seed(42)
        result_dfs: List[pd.DataFrame] = []
        for symbol in df.index.get_level_values('Stock').unique():
            symbol_df = df.xs(symbol, level='Stock').copy()
            n_days = len(symbol_df)
            # 실제 서비스에서는 아래 부분을 외부 재무 API 연동으로 대체하세요.
            symbol_df['PER'] = np.random.uniform(5, 30, n_days)
            symbol_df['PBR'] = np.random.uniform(0.5, 5.0, n_days)
            symbol_df['ROE'] = np.random.uniform(0.05, 0.25, n_days)
            symbol_df['ROA'] = np.random.uniform(0.02, 0.15, n_days)
            symbol_df['Debt_Ratio'] = np.random.uniform(0.1, 0.8, n_days)
            symbol_df['Current_Ratio'] = np.random.uniform(1.0, 3.0, n_days)
            symbol_df['Stock'] = symbol
            symbol_df.reset_index(inplace=True)
            symbol_df.set_index(['Date', 'Stock'], inplace=True)
            result_dfs.append(symbol_df)
        return pd.concat(result_dfs)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터 정제 (결측치/이상치/inf/음수/극단값 처리)
        - z-score, clip, fillna 등 활용
        """
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in df.columns:
                # z-score 3 이상은 NaN 처리(극단값)
                z_scores = np.abs(stats.zscore(df[col].fillna(0)))
                df.loc[z_scores > 3, col] = np.nan
                # 음수 불가 컬럼(예: volume 등)은 0으로 보정
                if col.lower() in ['volume', 'obv']:
                    df[col] = df[col].clip(lower=0)
        # 결측치 보간
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df

    def get_market_data(self, symbols: List[str], period: str = "2y", future_n: int = 5) -> pd.DataFrame:
        """
        완전한 시장 데이터 수집 (가격 + 기술적 지표 + 기본적 지표 + future_return)
        """
        df = self.get_stock_data(symbols, period=period)
        if df.empty:
            return df
        df = self.add_technical_indicators(df)
        df = self.add_fundamental_data(df)
        df = self.add_future_return(df, n=future_n)
        df = self.clean_data(df)
        return df

