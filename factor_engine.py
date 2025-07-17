"""
팩터 생성 및 분석 모듈
안전한 수식 파서, ML 기반 팩터 생성, 성과 분석 기능 제공
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from safe_expression_evaluator import SafeExpressionEvaluator  # 별도 파일로 분리했다고 가정
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def extract_column_names(formula: str) -> set:
    """
    수식에서 컬럼명 후보 추출 (알파벳, 숫자, _로 시작)
    모두 소문자로 변환하여 반환
    """
    return set(token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula))

class FactorGenerator:
    """
    팩터 생성기
    - 수식 기반, ML 기반 팩터 생성 지원
    """
    def __init__(self) -> None:
        pass

    def create_formula_factor(
        self, 
        data: pd.DataFrame, 
        formula: str, 
        factor_name: str
    ) -> Dict[str, Any]:
        """
        수식 기반 팩터 생성 (단계별 에러 안전 처리)
        - 컬럼명 긴 것부터 치환(겹침 방지)
        - 치환 전/후 로그(주석)
        - 인덱스 일치 robust하게 처리
        - NaN/inf/극단값 보정
        - 반환값: dict(시리즈+메타데이터)
        """
        # 1. 입력값 체크
        if not formula or not isinstance(formula, str):
            raise ValueError("수식을 입력하세요.")
        if data is None or data.empty:
            raise ValueError("입력 데이터가 비어있습니다.")
        # 2. 수식 파싱 및 컬럼명 추출
        try:
            needed_cols = extract_column_names(formula)
        except Exception as e:
            raise ValueError(f"수식 파싱 오류: {e}")
        # 3. 컬럼명 유효성 검사 (대소문자 무시)
        col_map = {col.lower(): col for col in data.columns}
        missing_cols = [col for col in needed_cols if col not in col_map]
        if missing_cols:
            raise ValueError(f"수식에 존재하지 않는 컬럼명(대소문자 무시): {', '.join(missing_cols)}\n사용 가능한 컬럼: {', '.join(data.columns)}")
        # 4. 수식 내 컬럼명 치환 (긴 컬럼명부터)
        # (주니어 개발자: 긴 컬럼명부터 치환해야 겹치는 이름이 있을 때 안전합니다)
        for lower_col in sorted(col_map.keys(), key=lambda x: -len(x)):
            real_col = col_map[lower_col]
            formula = re.sub(rf'\b{lower_col}\b', real_col, formula)
        # print(f"[DEBUG] 치환된 수식: {formula}")
        # 5. 수식 평가
        try:
            factor_series = data.groupby(level='Stock').apply(
                lambda x: SafeExpressionEvaluator(x).evaluate(formula)
            )
            # groupby.apply 결과가 DataFrame이면 Series로 변환
            if isinstance(factor_series, pd.DataFrame):
                factor_series = factor_series.iloc[:, 0]
            # 인덱스 레벨이 3 이상이면 마지막 레벨을 제거
            if isinstance(factor_series.index, pd.MultiIndex) and factor_series.index.nlevels > 2:
                factor_series.index = factor_series.index.droplevel(-1)
        except Exception as e:
            raise ValueError(f"수식 평가 오류: {e}\n수식: {formula}\n데이터 샘플: {data.head(2)}")
        # 6. 인덱스 일치 보정
        if not factor_series.index.equals(data.index):
            factor_series = factor_series.reindex(data.index)
        # 7. 결과값 검증 및 보정
        factor_series.name = factor_name
        factor_series = factor_series.replace([np.inf, -np.inf], np.nan)
        # 극단값(예: ±1e6 이상)은 NaN 처리
        factor_series.loc[factor_series.abs() > 1e6] = np.nan
        factor_series = factor_series.fillna(method='ffill').fillna(method='bfill')
        if factor_series.isnull().all():
            raise ValueError("수식 결과가 모두 NaN입니다. 수식 또는 데이터를 확인하세요.")
        # 8. 메타데이터 생성
        meta = {
            'name': factor_name,
            'expression': formula,
            'type': 'Formulaic',
            'n_missing': factor_series.isna().sum(),
            'max': factor_series.max(),
            'min': factor_series.min(),
            'n_unique': factor_series.nunique()
        }
        return {'data': factor_series, 'meta': meta}

    def create_ml_factor(
        self, 
        data: pd.DataFrame, 
        target_col: str,
        method: str,
        factor_name: str,
        feature_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        ML 기반 팩터 생성
        - feature/target 체크, 데이터 충분성, NaN/inf/극단값 보정
        - 반환값: dict(시리즈+메타데이터)
        """
        if data is None or data.empty:
            raise ValueError("입력 데이터가 비어있습니다.")
        if feature_cols is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            feature_cols = numeric_cols[:10]
        # 1. 컬럼 존재 체크
        for col in feature_cols + [target_col]:
            if col not in data.columns:
                raise ValueError(f"존재하지 않는 컬럼명: {col}")
        # 2. 데이터 충분성 체크
        df_ml = data[feature_cols + [target_col]].dropna()
        if df_ml.empty:
            raise ValueError("ML 학습을 위한 데이터가 부족합니다.")
        X = df_ml[feature_cols]
        y = df_ml[target_col]
        try:
            if method.lower() == 'rf':
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X, y)
                importances = model.feature_importances_
                factor_values = np.dot(X.values, importances)
            elif method.lower() == 'mlp':
                model = MLPRegressor(
                    hidden_layer_sizes=(64, 32), 
                    max_iter=200, 
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1
                )
                model.fit(X, y)
                factor_values = model.predict(X)
            elif method.lower() == 'pca':
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                pca = PCA(n_components=1, random_state=42)
                factor_values = pca.fit_transform(X_scaled).flatten()
            elif method.lower() == 'xgb':
                try:
                    from xgboost import XGBRegressor
                    model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    model.fit(X, y)
                    importances = model.feature_importances_
                    factor_values = np.dot(X.values, importances)
                except ImportError:
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    model.fit(X, y)
                    importances = model.feature_importances_
                    factor_values = np.dot(X.values, importances)
            else:
                raise ValueError(f"지원되지 않는 ML 방법: {method}")
            factor_series = pd.Series(factor_values, index=df_ml.index, name=factor_name)
            factor_series = factor_series.replace([np.inf, -np.inf], np.nan)
            # 극단값(예: ±1e6 이상)은 NaN 처리
            factor_series.loc[factor_series.abs() > 1e6] = np.nan
            factor_series = factor_series.fillna(method='ffill').fillna(method='bfill')
            # 3. 결과값 검증
            if factor_series.isnull().all():
                raise ValueError("ML 팩터 결과가 모두 NaN입니다. 수식, 피처, 데이터를 확인하세요.")
            meta = {
                'name': factor_name,
                'type': f'ML_{method}',
                'features': feature_cols,
                'target': target_col,
                'n_missing': factor_series.isna().sum(),
                'max': factor_series.max(),
                'min': factor_series.min(),
                'n_unique': factor_series.nunique()
            }
            return {'data': factor_series, 'meta': meta}
        except Exception as e:
            raise ValueError(f"ML 팩터 생성 실패: {e}")

class FactorAnalyzer:
    """
    팩터 분석기
    - IC, ICIR, 통계 등 팩터 성과 분석 지원
    """
    def __init__(self) -> None:
        pass

    def calculate_ic(
        self, 
        factor: pd.Series, 
        returns: pd.Series, 
        method: str = 'pearson'
    ) -> float:
        """
        Information Coefficient (IC) 계산
        """
        try:
            common_idx = factor.index.intersection(returns.index)
            if len(common_idx) == 0:
                return 0.0
            factor_aligned = factor.loc[common_idx]
            returns_aligned = returns.loc[common_idx]
            valid_mask = factor_aligned.notna() & returns_aligned.notna()
            if valid_mask.sum() < 10:
                return 0.0
            factor_clean = factor_aligned[valid_mask]
            returns_clean = returns_aligned[valid_mask]
            if method == 'pearson':
                ic = factor_clean.corr(returns_clean)
            elif method == 'spearman':
                ic = factor_clean.corr(returns_clean, method='spearman')
            else:
                raise ValueError(f"지원되지 않는 방법: {method}")
            return ic if not np.isnan(ic) else 0.0
        except Exception:
            return 0.0

    def calculate_icir(
        self, 
        factor: pd.Series, 
        returns: pd.Series, 
        window: int = 20
    ) -> float:
        """
        Information Coefficient Information Ratio (ICIR) 계산
        """
        try:
            rolling_ics: List[float] = []
            if isinstance(factor.index, pd.MultiIndex):
                dates = factor.index.get_level_values(0).unique()
            else:
                dates = factor.index.unique()
            for i in range(window, len(dates)):
                date_range = dates[i-window:i]
                if isinstance(factor.index, pd.MultiIndex):
                    factor_window = factor[factor.index.get_level_values(0).isin(date_range)]
                    returns_window = returns[returns.index.get_level_values(0).isin(date_range)]
                else:
                    factor_window = factor[factor.index.isin(date_range)]
                    returns_window = returns[returns.index.isin(date_range)]
                ic = self.calculate_ic(factor_window, returns_window)
                rolling_ics.append(ic)
            if len(rolling_ics) == 0:
                return 0.0
            ic_mean = np.mean(rolling_ics)
            ic_std = np.std(rolling_ics)
            if ic_std == 0:
                return 0.0
            return ic_mean / ic_std
        except Exception:
            return 0.0

    def calculate_factor_stats(
        self, 
        factor: pd.Series, 
        returns: pd.Series
    ) -> Dict[str, float]:
        """
        팩터 통계 계산 (평균, 표준편차, 왜도, 첨도, IC, ICIR 등)
        """
        stats: Dict[str, float] = {}
        try:
            stats['mean'] = factor.mean()
            stats['std'] = factor.std()
            stats['skew'] = factor.skew()
            stats['kurt'] = factor.kurtosis()
            stats['ic'] = self.calculate_ic(factor, returns, 'pearson')
            stats['ic_spearman'] = self.calculate_ic(factor, returns, 'spearman')
            stats['icir'] = self.calculate_icir(factor, returns)
            stats['min'] = factor.min()
            stats['max'] = factor.max()
            stats['q25'] = factor.quantile(0.25)
            stats['q50'] = factor.quantile(0.50)
            stats['q75'] = factor.quantile(0.75)
            stats['missing_ratio'] = factor.isna().mean()
        except Exception:
            pass
        return stats

