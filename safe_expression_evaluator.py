import ast
import operator
import numpy as np
import pandas as pd

class SafeExpressionEvaluator:
    """
    AST 기반의 안전한 수식 실행기
    - eval() 대신 안전하게 수식 평가
    - 지원 함수: log, abs, sqrt, rank, rolling_mean, rolling_std, shift 등
    """
    ALLOWED_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.allowed_funcs = {
            'log': np.log,
            'abs': np.abs,
            'sqrt': np.sqrt,
            'rank': lambda x: x.rank(pct=True),
            'rolling_mean': lambda x, w: x.rolling(window=w).mean(),
            'rolling_std': lambda x, w: x.rolling(window=w).std(),
            'shift': lambda x, p: x.shift(p)
        }

    def evaluate(self, expression: str) -> pd.Series:
        try:
            tree = ast.parse(expression, mode='eval')
            result = self._evaluate_node(tree.body)
            if isinstance(result, pd.DataFrame):
                raise ValueError(
                    f"수식 평가 결과가 DataFrame입니다. (컬럼: {list(result.columns)})\n"
                    f"단일 Series(컬럼)만 반환해야 합니다. 수식을 확인하세요."
                )
            if not isinstance(result, pd.Series):
                raise ValueError(
                    f"수식 평가 결과가 Series가 아닙니다. (type: {type(result)})\n"
                    f"수식: {expression}\n반환값: {result}"
                )
            return result
        except Exception as e:
            raise ValueError(f"수식 평가 중 예외 발생: {e}\n수식: {expression}")

    def _evaluate_node(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            col_map = {col.lower(): col for col in self.df.columns}
            col_id = node.id.lower()
            if col_id in col_map:
                return self.df[col_map[col_id]]
            else:
                raise NameError(f"존재하지 않는 컬럼명(대소문자 무시): {node.id} (사용 가능: {list(self.df.columns)})")
        elif isinstance(node, ast.BinOp):
            return self.ALLOWED_OPS[type(node.op)](
                self._evaluate_node(node.left),
                self._evaluate_node(node.right)
            )
        elif isinstance(node, ast.UnaryOp):
            return self.ALLOWED_OPS[type(node.op)](self._evaluate_node(node.operand))
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name in self.allowed_funcs:
                return self.allowed_funcs[func_name](*[self._evaluate_node(arg) for arg in node.args])
            else:
                raise NameError(f"허용되지 않는 함수: {func_name}")
        else:
            raise TypeError(f"허용되지 않는 노드 타입: {type(node).__name__}")
