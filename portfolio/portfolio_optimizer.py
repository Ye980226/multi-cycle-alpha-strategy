#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
组合优化模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf, OAS
from utils.logger import Logger


class BaseOptimizer(ABC):
    """优化器基类"""
    
    def __init__(self, name: str):
        """初始化优化器"""
        self.name = name
        self.logger = Logger().get_logger(f"optimizer_{name}")
    
    @abstractmethod
    def optimize(self, expected_returns: pd.Series, 
                covariance_matrix: pd.DataFrame,
                constraints: Dict = None) -> pd.Series:
        """优化组合权重"""
        pass
    
    def validate_inputs(self, expected_returns: pd.Series,
                       covariance_matrix: pd.DataFrame) -> bool:
        """验证输入数据"""
        try:
            # 检查数据类型
            if not isinstance(expected_returns, pd.Series):
                self.logger.error("预期收益率必须是pandas Series")
                return False
                
            if not isinstance(covariance_matrix, pd.DataFrame):
                self.logger.error("协方差矩阵必须是pandas DataFrame")
                return False
            
            # 检查维度匹配
            if len(expected_returns) != len(covariance_matrix):
                self.logger.error("预期收益率和协方差矩阵维度不匹配")
                return False
            
            # 检查协方差矩阵是否为方阵
            if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
                self.logger.error("协方差矩阵必须是方阵")
                return False
            
            # 检查协方差矩阵是否对称
            if not np.allclose(covariance_matrix.values, covariance_matrix.values.T):
                self.logger.error("协方差矩阵必须是对称的")
                return False
            
            # 检查协方差矩阵是否正定
            eigenvalues = np.linalg.eigvals(covariance_matrix.values)
            if np.any(eigenvalues <= 0):
                self.logger.warning("协方差矩阵不是正定的")
            
            return True
            
        except Exception as e:
            self.logger.error(f"输入验证失败: {e}")
            return False


class MeanVarianceOptimizer(BaseOptimizer):
    """均值方差优化器"""
    
    def __init__(self, risk_aversion: float = 1.0):
        """初始化均值方差优化器"""
        super().__init__("mean_variance")
        self.risk_aversion = risk_aversion
    
    def optimize(self, expected_returns: pd.Series,
                covariance_matrix: pd.DataFrame,
                constraints: Dict = None,
                risk_aversion: float = None) -> pd.Series:
        """均值方差优化"""
        try:
            # 验证输入
            if not self.validate_inputs(expected_returns, covariance_matrix):
                return pd.Series(index=expected_returns.index, dtype=float)
            
            n_assets = len(expected_returns)
            
            # 设置默认约束
            if constraints is None:
                constraints = {}
            
            # 使用传入的风险厌恶系数，如果没有则使用默认值
            effective_risk_aversion = risk_aversion if risk_aversion is not None else self.risk_aversion
            
            # 定义目标函数
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
                # 最大化 return - risk_aversion * variance
                return -(portfolio_return - effective_risk_aversion * portfolio_variance)
            
            # 约束条件
            constraints_list = []
            
            # 权重和为1
            constraints_list.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            # 仓位约束
            max_weight = constraints.get('max_weight', 0.1)
            min_weight = constraints.get('min_weight', 0.0)
            bounds = [(min_weight, max_weight) for _ in range(n_assets)]
            
            # 初始权重
            initial_weights = np.ones(n_assets) / n_assets
            
            # 优化
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = pd.Series(result.x, index=expected_returns.index)
                self.logger.info(f"均值方差优化成功，目标函数值: {-result.fun:.6f}")
                return optimal_weights
            else:
                self.logger.error(f"均值方差优化失败: {result.message}")
                return pd.Series(index=expected_returns.index, dtype=float)
                
        except Exception as e:
            self.logger.error(f"均值方差优化失败: {e}")
            return pd.Series(index=expected_returns.index, dtype=float)
    
    def efficient_frontier(self, expected_returns: pd.Series,
                          covariance_matrix: pd.DataFrame,
                          n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """有效前沿"""
        try:
            returns = []
            risks = []
            
            # 计算最小方差组合
            min_var_weights = self._minimum_variance_portfolio(covariance_matrix)
            min_return = np.dot(min_var_weights, expected_returns)
            
            # 计算最大收益组合
            max_return = expected_returns.max()
            
            # 在收益率范围内生成有效前沿
            target_returns = np.linspace(min_return, max_return, n_points)
            
            for target_return in target_returns:
                weights = self._optimize_for_target_return(
                    expected_returns, covariance_matrix, target_return
                )
                
                if weights is not None:
                    portfolio_return = np.dot(weights, expected_returns)
                    portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
                    
                    returns.append(portfolio_return)
                    risks.append(portfolio_risk)
            
            return np.array(returns), np.array(risks)
            
        except Exception as e:
            self.logger.error(f"有效前沿计算失败: {e}")
            return np.array([]), np.array([])
    
    def _minimum_variance_portfolio(self, covariance_matrix: pd.DataFrame) -> np.ndarray:
        """计算最小方差组合"""
        try:
            n_assets = len(covariance_matrix)
            
            # 目标函数：最小化组合方差
            def objective(weights):
                return np.dot(weights, np.dot(covariance_matrix, weights))
            
            # 约束条件
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            bounds = [(0, 1) for _ in range(n_assets)]
            
            # 初始权重
            initial_weights = np.ones(n_assets) / n_assets
            
            # 优化
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            return result.x if result.success else initial_weights
            
        except Exception as e:
            self.logger.error(f"最小方差组合计算失败: {e}")
            return np.ones(len(covariance_matrix)) / len(covariance_matrix)
    
    def _optimize_for_target_return(self, expected_returns: pd.Series,
                                   covariance_matrix: pd.DataFrame,
                                   target_return: float) -> Optional[np.ndarray]:
        """针对目标收益率进行优化"""
        try:
            n_assets = len(expected_returns)
            
            # 目标函数：最小化组合方差
            def objective(weights):
                return np.dot(weights, np.dot(covariance_matrix, weights))
            
            # 约束条件
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - target_return}
            ]
            bounds = [(0, 1) for _ in range(n_assets)]
            
            # 初始权重
            initial_weights = np.ones(n_assets) / n_assets
            
            # 优化
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            return result.x if result.success else None
            
        except Exception as e:
            self.logger.error(f"目标收益率优化失败: {e}")
            return None


class BlackLittermanOptimizer(BaseOptimizer):
    """Black-Litterman优化器"""
    
    def __init__(self, tau: float = 0.05, risk_aversion: float = 3.0):
        """初始化BL优化器"""
        pass
    
    def optimize(self, expected_returns: pd.Series,
                covariance_matrix: pd.DataFrame,
                constraints: Dict = None) -> pd.Series:
        """BL优化"""
        pass
    
    def incorporate_views(self, picking_matrix: pd.DataFrame,
                         views: pd.Series, omega: pd.DataFrame):
        """融入投资者观点"""
        pass


class RiskParityOptimizer(BaseOptimizer):
    """风险平价优化器"""
    
    def __init__(self):
        """初始化风险平价优化器"""
        pass
    
    def optimize(self, expected_returns: pd.Series,
                covariance_matrix: pd.DataFrame,
                constraints: Dict = None) -> pd.Series:
        """风险平价优化"""
        pass
    
    def equal_risk_contribution(self, covariance_matrix: pd.DataFrame) -> pd.Series:
        """等风险贡献"""
        pass


class HierarchicalRiskParityOptimizer(BaseOptimizer):
    """层次风险平价优化器"""
    
    def __init__(self, clustering_method: str = "single"):
        """初始化HRP优化器"""
        pass
    
    def optimize(self, expected_returns: pd.Series,
                covariance_matrix: pd.DataFrame,
                constraints: Dict = None) -> pd.Series:
        """HRP优化"""
        pass
    
    def build_cluster_tree(self, covariance_matrix: pd.DataFrame):
        """构建聚类树"""
        pass
    
    def allocate_weights_recursive(self, cluster_items: List,
                                  covariance_matrix: pd.DataFrame) -> pd.Series:
        """递归分配权重"""
        pass


class MultiCycleOptimizer:
    """多周期优化器"""
    
    def __init__(self, cycles: List[str] = ["intraday", "daily", "weekly"]):
        """初始化多周期优化器"""
        pass
    
    def optimize_multi_cycle(self, signals_dict: Dict[str, pd.DataFrame],
                           returns_dict: Dict[str, pd.DataFrame],
                           cycle_weights: Dict[str, float] = None) -> pd.DataFrame:
        """多周期优化"""
        pass
    
    def align_cycles(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """对齐不同周期"""
        pass
    
    def calculate_cycle_weights(self, performance_dict: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """计算周期权重"""
        pass
    
    def dynamic_cycle_allocation(self, market_regime: pd.Series) -> Dict[str, float]:
        """动态周期配置"""
        pass


class ConstraintManager:
    """约束管理器"""
    
    def __init__(self):
        """初始化约束管理器"""
        pass
    
    def add_position_constraints(self, n_assets: int, max_weight: float = 0.05,
                               min_weight: float = 0.0) -> List:
        """添加仓位约束"""
        pass
    
    def add_sector_constraints(self, sector_exposure: pd.DataFrame,
                             max_sector_weight: float = 0.3) -> List:
        """添加行业约束"""
        pass
    
    def add_turnover_constraints(self, current_weights: pd.Series,
                               max_turnover: float = 0.5) -> List:
        """添加换手率约束"""
        pass
    
    def add_tracking_error_constraint(self, benchmark_weights: pd.Series,
                                    max_tracking_error: float = 0.05) -> List:
        """添加跟踪误差约束"""
        pass
    
    def add_leverage_constraint(self, max_leverage: float = 1.0) -> List:
        """添加杠杆约束"""
        pass


class TransactionCostModel:
    """交易成本模型"""
    
    def __init__(self, linear_cost: float = 0.001, 
                 market_impact_model: str = "linear"):
        """初始化交易成本模型"""
        pass
    
    def calculate_transaction_costs(self, current_weights: pd.Series,
                                  target_weights: pd.Series,
                                  volume_data: pd.DataFrame = None) -> float:
        """计算交易成本"""
        pass
    
    def linear_market_impact(self, trade_size: pd.Series,
                           volume: pd.Series) -> pd.Series:
        """线性市场冲击模型"""
        pass
    
    def square_root_market_impact(self, trade_size: pd.Series,
                                volume: pd.Series) -> pd.Series:
        """平方根市场冲击模型"""
        pass


class RiskBudgetingOptimizer:
    """风险预算优化器"""
    
    def __init__(self):
        """初始化风险预算优化器"""
        pass
    
    def optimize_with_risk_budget(self, expected_returns: pd.Series,
                                 covariance_matrix: pd.DataFrame,
                                 risk_budgets: pd.Series) -> pd.Series:
        """基于风险预算优化"""
        pass
    
    def calculate_risk_contributions(self, weights: pd.Series,
                                   covariance_matrix: pd.DataFrame) -> pd.Series:
        """计算风险贡献"""
        pass
    
    def optimize_equal_risk_budget(self, covariance_matrix: pd.DataFrame) -> pd.Series:
        """等风险预算优化"""
        pass


class PortfolioRebalancer:
    """组合再平衡器"""
    
    def __init__(self, rebalance_frequency: str = "daily",
                 rebalance_threshold: float = 0.05):
        """初始化再平衡器"""
        pass
    
    def should_rebalance(self, current_weights: pd.Series,
                        target_weights: pd.Series) -> bool:
        """判断是否需要再平衡"""
        pass
    
    def calculate_trades(self, current_weights: pd.Series,
                        target_weights: pd.Series) -> pd.Series:
        """计算交易量"""
        pass
    
    def optimize_rebalancing_timing(self, signal_strength: pd.Series,
                                  transaction_costs: pd.Series) -> pd.Series:
        """优化再平衡时机"""
        pass


class PortfolioOptimizer:
    """组合优化主类"""
    
    def __init__(self, config: Dict = None):
        """初始化组合优化器"""
        self.config = config or {}
        self.logger = Logger().get_logger("portfolio_optimizer")
        
        # 初始化各种优化器
        self.mean_variance_optimizer = MeanVarianceOptimizer()
        
        # 配置参数
        self.risk_aversion = self.config.get('risk_aversion', 1.0)
        self.lookback_window = self.config.get('lookback_window', 252)
        self.min_weight = self.config.get('min_weight', 0.0)
        self.max_weight = self.config.get('max_weight', 0.1)
    
    def initialize(self, config: Dict):
        """初始化配置"""
        try:
            self.config = config
            
            # 更新配置参数
            self.risk_aversion = config.get('risk_aversion', 1.0)
            self.lookback_window = config.get('lookback_window', 252)
            self.min_weight = config.get('min_weight', 0.0)
            self.max_weight = config.get('max_weight', 0.1)
            
            # 重新初始化优化器
            self.mean_variance_optimizer = MeanVarianceOptimizer(self.risk_aversion)
            
            self.logger.info("组合优化器初始化完成")
            
        except Exception as e:
            self.logger.error(f"组合优化器初始化失败: {e}")
            raise
    
    def optimize_portfolio(self, signals: pd.DataFrame,
                         returns_data: pd.DataFrame = None,
                         method: str = "mean_variance",
                         constraints: Dict = None,
                         current_weights: pd.Series = None) -> pd.Series:
        """优化组合"""
        try:
            if signals.empty:
                self.logger.warning("信号数据为空")
                return pd.Series(dtype=float)
            
            # 获取最新信号
            latest_signals = signals.iloc[-1]
            latest_signals = latest_signals.dropna()
            
            if len(latest_signals) == 0:
                self.logger.warning("没有有效信号")
                return pd.Series(dtype=float)
            
            # 估计预期收益率
            if returns_data is not None:
                expected_returns = self.estimate_expected_returns(
                    signals, returns_data, method="signal_weighted"
                )
                # 获取最新预期收益率
                latest_expected_returns = expected_returns.iloc[-1].dropna()
                
                # 估计协方差矩阵
                covariance_matrix = self.estimate_covariance_matrix(
                    returns_data, method="ledoit_wolf"
                )
            else:
                # 如果没有收益率数据，直接使用信号作为预期收益率
                latest_expected_returns = latest_signals
                # 使用对角协方差矩阵
                covariance_matrix = pd.DataFrame(
                    np.eye(len(latest_signals)), 
                    index=latest_signals.index, 
                    columns=latest_signals.index
                )
            
            # 确保数据一致性
            common_assets = latest_expected_returns.index.intersection(covariance_matrix.index)
            if len(common_assets) == 0:
                self.logger.warning("没有共同的资产")
                return pd.Series(dtype=float)
            
            expected_returns_aligned = latest_expected_returns[common_assets]
            covariance_matrix_aligned = covariance_matrix.loc[common_assets, common_assets]
            
            # 设置约束条件
            if constraints is None:
                constraints = {}
            
            constraints.update({
                'min_weight': self.min_weight,
                'max_weight': self.max_weight
            })
            
            # 进行优化
            if method == "mean_variance":
                optimal_weights = self.mean_variance_optimizer.optimize(
                    expected_returns_aligned, covariance_matrix_aligned, constraints
                )
            elif method == "equal_weight":
                optimal_weights = self._equal_weight_portfolio(expected_returns_aligned)
            elif method == "signal_weight":
                optimal_weights = self._signal_weighted_portfolio(expected_returns_aligned)
            else:
                self.logger.error(f"不支持的优化方法: {method}")
                return pd.Series(dtype=float)
            
            # 后处理权重
            if not optimal_weights.empty:
                optimal_weights = self._post_process_weights(optimal_weights, constraints)
            
            self.logger.info(f"组合优化完成，使用方法: {method}")
            return optimal_weights
            
        except Exception as e:
            self.logger.error(f"组合优化失败: {e}")
            return pd.Series(dtype=float)
    
    def estimate_expected_returns(self, signals: pd.DataFrame,
                                historical_returns: pd.DataFrame,
                                method: str = "signal_weighted") -> pd.DataFrame:
        """估计预期收益"""
        try:
            if method == "signal_weighted":
                # 使用信号加权历史收益率
                expected_returns = pd.DataFrame(
                    index=signals.index, 
                    columns=signals.columns
                )
                
                # 使用过去收益率的信号加权平均
                for timestamp in signals.index:
                    if timestamp in historical_returns.index:
                        current_signals = signals.loc[timestamp].dropna()
                        
                        # 获取过去的收益率
                        past_returns = historical_returns.loc[:timestamp].tail(self.lookback_window)
                        
                        if not past_returns.empty and len(current_signals) > 0:
                            # 计算每个资产的信号加权预期收益率
                            for asset in current_signals.index:
                                if asset in past_returns.columns:
                                    # 使用信号强度加权过去的收益率
                                    signal_strength = abs(current_signals[asset])
                                    if signal_strength > 0:
                                        # 简单使用历史收益率均值乘以信号强度
                                        expected_returns.loc[timestamp, asset] = (
                                            past_returns[asset].mean() * current_signals[asset]
                                        )
                
            elif method == "historical_mean":
                # 使用历史收益率均值
                expected_returns = pd.DataFrame(
                    index=signals.index, 
                    columns=signals.columns
                )
                
                for timestamp in signals.index:
                    past_returns = historical_returns.loc[:timestamp].tail(self.lookback_window)
                    if not past_returns.empty:
                        expected_returns.loc[timestamp] = past_returns.mean()
                        
            else:
                raise ValueError(f"不支持的预期收益率估计方法: {method}")
            
            return expected_returns.fillna(0)
            
        except Exception as e:
            self.logger.error(f"预期收益率估计失败: {e}")
            return pd.DataFrame()
    
    def estimate_covariance_matrix(self, returns: pd.DataFrame,
                                 method: str = "ledoit_wolf",
                                 lookback: int = None) -> pd.DataFrame:
        """估计协方差矩阵"""
        try:
            if lookback is None:
                lookback = self.lookback_window
            
            # 使用最近的数据
            recent_returns = returns.tail(lookback)
            
            if recent_returns.empty:
                self.logger.warning("没有足够的历史数据")
                return pd.DataFrame()
            
            # 去除空值
            clean_returns = recent_returns.dropna()
            
            if clean_returns.empty:
                self.logger.warning("清洗后数据为空")
                return pd.DataFrame()
            
            if method == "ledoit_wolf":
                # 使用Ledoit-Wolf收缩估计
                lw = LedoitWolf()
                covariance_matrix = lw.fit(clean_returns).covariance_
                
            elif method == "oas":
                # 使用Oracle近似收缩估计
                oas = OAS()
                covariance_matrix = oas.fit(clean_returns).covariance_
                
            elif method == "sample":
                # 使用样本协方差矩阵
                covariance_matrix = clean_returns.cov().values
                
            else:
                raise ValueError(f"不支持的协方差矩阵估计方法: {method}")
            
            # 转换为DataFrame
            covariance_df = pd.DataFrame(
                covariance_matrix,
                index=clean_returns.columns,
                columns=clean_returns.columns
            )
            
            return covariance_df
            
        except Exception as e:
            self.logger.error(f"协方差矩阵估计失败: {e}")
            return pd.DataFrame()
    
    def _equal_weight_portfolio(self, assets: pd.Series) -> pd.Series:
        """等权重组合"""
        try:
            n_assets = len(assets)
            if n_assets == 0:
                return pd.Series(dtype=float)
            
            equal_weights = pd.Series(1.0 / n_assets, index=assets.index)
            return equal_weights
            
        except Exception as e:
            self.logger.error(f"等权重组合计算失败: {e}")
            return pd.Series(dtype=float)
    
    def _signal_weighted_portfolio(self, signals: pd.Series) -> pd.Series:
        """信号加权组合"""
        try:
            # 使用信号强度进行加权
            abs_signals = signals.abs()
            
            if abs_signals.sum() == 0:
                return self._equal_weight_portfolio(signals)
            
            # 归一化权重
            signal_weights = abs_signals / abs_signals.sum()
            
            # 考虑信号方向
            signal_weights *= np.sign(signals)
            
            return signal_weights
            
        except Exception as e:
            self.logger.error(f"信号加权组合计算失败: {e}")
            return pd.Series(dtype=float)
    
    def _post_process_weights(self, weights: pd.Series, constraints: Dict) -> pd.Series:
        """后处理权重"""
        try:
            processed_weights = weights.copy()
            
            # 应用权重约束
            min_weight = constraints.get('min_weight', 0.0)
            max_weight = constraints.get('max_weight', 1.0)
            
            # 裁剪权重
            processed_weights = processed_weights.clip(lower=min_weight, upper=max_weight)
            
            # 重新归一化
            if processed_weights.sum() != 0:
                processed_weights = processed_weights / processed_weights.sum()
            
            return processed_weights
            
        except Exception as e:
            self.logger.error(f"权重后处理失败: {e}")
            return weights
    
    def multi_cycle_optimization(self, signals_dict: Dict[str, pd.DataFrame],
                               returns_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """多周期优化"""
        pass
    
    def dynamic_optimization(self, signals: pd.DataFrame,
                           returns: pd.DataFrame,
                           market_regime: pd.Series) -> pd.DataFrame:
        """动态优化"""
        pass
    
    def risk_budgeting_optimization(self, signals: pd.DataFrame,
                                  returns: pd.DataFrame,
                                  risk_budgets: Dict[str, float]) -> pd.DataFrame:
        """风险预算优化"""
        pass
    
    def factor_risk_model_optimization(self, signals: pd.DataFrame,
                                     factor_returns: pd.DataFrame,
                                     factor_exposures: pd.DataFrame) -> pd.DataFrame:
        """因子风险模型优化"""
        pass
    
    def validate_portfolio(self, weights: pd.DataFrame,
                         constraints: Dict) -> Dict[str, bool]:
        """验证组合"""
        try:
            validation_results = {}
            
            # 基本验证
            validation_results['not_empty'] = not weights.empty
            validation_results['no_nan'] = not weights.isnull().any().any()
            validation_results['finite_values'] = np.isfinite(weights).all().all()
            
            if weights.empty:
                return validation_results
            
            # 权重和验证
            weight_sums = weights.sum(axis=1)
            validation_results['weights_sum_to_one'] = np.allclose(weight_sums, 1.0, atol=1e-6)
            
            # 约束验证
            if constraints:
                # 最小权重约束
                if 'min_weight' in constraints:
                    min_weight = constraints['min_weight']
                    validation_results['min_weight_satisfied'] = (weights >= min_weight - 1e-6).all().all()
                
                # 最大权重约束
                if 'max_weight' in constraints:
                    max_weight = constraints['max_weight']
                    validation_results['max_weight_satisfied'] = (weights <= max_weight + 1e-6).all().all()
                
                # 最大杠杆约束
                if 'max_leverage' in constraints:
                    max_leverage = constraints['max_leverage']
                    leverage = weights.abs().sum(axis=1)
                    validation_results['leverage_satisfied'] = (leverage <= max_leverage + 1e-6).all()
                
                # 最大换手率约束
                if 'max_turnover' in constraints and len(weights) > 1:
                    max_turnover = constraints['max_turnover']
                    turnover = weights.diff().abs().sum(axis=1)
                    validation_results['turnover_satisfied'] = (turnover <= max_turnover + 1e-6).all()
                
                # 最大集中度约束
                if 'max_concentration' in constraints:
                    max_concentration = constraints['max_concentration']
                    concentration = (weights ** 2).sum(axis=1)
                    validation_results['concentration_satisfied'] = (concentration <= max_concentration + 1e-6).all()
                
                # 行业约束
                if 'sector_constraints' in constraints and 'sector_mapping' in constraints:
                    sector_mapping = constraints['sector_mapping']
                    sector_limits = constraints['sector_constraints']
                    
                    for sector, limit in sector_limits.items():
                        sector_stocks = [stock for stock, s in sector_mapping.items() if s == sector]
                        if sector_stocks:
                            sector_weights = weights[sector_stocks].sum(axis=1)
                            validation_results[f'sector_{sector}_satisfied'] = (sector_weights <= limit + 1e-6).all()
            
            # 计算总体验证得分
            validation_scores = [v for v in validation_results.values() if isinstance(v, bool)]
            validation_results['overall_valid'] = all(validation_scores) if validation_scores else False
            
            self.logger.info(f"组合验证完成，总体有效性: {validation_results['overall_valid']}")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"组合验证失败: {e}")
            return {'error': True}
    
    def calculate_portfolio_metrics(self, weights: pd.DataFrame,
                                  returns: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算组合指标"""
        try:
            metrics = {}
            
            # 计算组合收益率
            portfolio_returns = (weights.shift(1) * returns).sum(axis=1)
            metrics['portfolio_returns'] = portfolio_returns
            
            # 计算累积收益率
            cumulative_returns = (1 + portfolio_returns).cumprod()
            metrics['cumulative_returns'] = cumulative_returns
            
            # 计算夏普比率（滚动）
            rolling_sharpe = portfolio_returns.rolling(window=252).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
            )
            metrics['rolling_sharpe'] = rolling_sharpe
            
            # 计算最大回撤
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            metrics['drawdown'] = drawdown
            metrics['max_drawdown'] = drawdown.expanding().min()
            
            # 计算波动率
            rolling_volatility = portfolio_returns.rolling(window=252).std() * np.sqrt(252)
            metrics['volatility'] = rolling_volatility
            
            # 计算权重集中度（Herfindahl指数）
            concentration = (weights ** 2).sum(axis=1)
            metrics['concentration'] = concentration
            
            # 计算换手率
            weight_changes = weights.diff().abs().sum(axis=1)
            metrics['turnover'] = weight_changes
            
            # 计算有效股票数量
            effective_stocks = 1 / concentration
            metrics['effective_num_stocks'] = effective_stocks
            
            # 计算胜率
            win_rate = (portfolio_returns > 0).rolling(window=252).mean()
            metrics['win_rate'] = win_rate
            
            self.logger.info("组合指标计算完成")
            return metrics
            
        except Exception as e:
            self.logger.error(f"计算组合指标失败: {e}")
            return {}
    
    def optimize(self, signals: pd.DataFrame,
                returns_data: pd.DataFrame = None,
                method: str = "mean_variance",
                constraints: Dict = None,
                current_weights: pd.Series = None) -> pd.Series:
        """优化组合权重 - 兼容性方法"""
        try:
            # 调用主要的优化方法
            result = self.optimize_portfolio(
                signals=signals,
                returns_data=returns_data,
                method=method,
                constraints=constraints,
                current_weights=current_weights
            )
            
            # 如果结果是DataFrame，取最后一行
            if isinstance(result, pd.DataFrame):
                return result.iloc[-1]
            else:
                return result
                
        except Exception as e:
            self.logger.error(f"组合优化失败: {e}")
            return pd.Series(index=signals.columns if hasattr(signals, 'columns') else [], dtype=float)
    
    def backtest_optimization(self, signals: pd.DataFrame,
                            returns: pd.DataFrame,
                            start_date: str, end_date: str) -> Dict:
        """回测优化策略"""
        try:
            # 筛选回测期间数据
            mask = (signals.index >= start_date) & (signals.index <= end_date)
            signals_bt = signals.loc[mask]
            returns_bt = returns.loc[mask]
            
            if signals_bt.empty or returns_bt.empty:
                self.logger.warning("回测期间数据为空")
                return {}
            
            # 执行优化
            optimized_weights = self.optimize_portfolio(
                signals_bt, 
                returns_bt,
                method=self.optimization_method
            )
            
            if optimized_weights.empty:
                self.logger.warning("优化权重为空")
                return {}
            
            # 计算组合指标
            portfolio_metrics = self.calculate_portfolio_metrics(optimized_weights, returns_bt)
            
            # 计算总体表现指标
            portfolio_returns = portfolio_metrics.get('portfolio_returns', pd.Series())
            
            backtest_results = {
                'weights': optimized_weights,
                'portfolio_returns': portfolio_returns,
                'metrics': portfolio_metrics
            }
            
            if not portfolio_returns.empty:
                # 总收益率
                total_return = portfolio_returns.sum()
                backtest_results['total_return'] = total_return
                
                # 年化收益率
                years = len(portfolio_returns) / 252
                annualized_return = (1 + portfolio_returns).prod() ** (1/years) - 1 if years > 0 else 0
                backtest_results['annualized_return'] = annualized_return
                
                # 年化波动率
                annualized_volatility = portfolio_returns.std() * np.sqrt(252)
                backtest_results['annualized_volatility'] = annualized_volatility
                
                # 夏普比率
                sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
                backtest_results['sharpe_ratio'] = sharpe_ratio
                
                # 最大回撤
                cumulative = (1 + portfolio_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min()
                backtest_results['max_drawdown'] = max_drawdown
                
                # 胜率
                win_rate = (portfolio_returns > 0).mean()
                backtest_results['win_rate'] = win_rate
                
                # 平均换手率
                if 'turnover' in portfolio_metrics:
                    avg_turnover = portfolio_metrics['turnover'].mean()
                    backtest_results['avg_turnover'] = avg_turnover
                
                # 信息比率（相对基准）
                if hasattr(self, 'benchmark_returns') and self.benchmark_returns is not None:
                    excess_returns = portfolio_returns - self.benchmark_returns.reindex(portfolio_returns.index)
                    information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
                    backtest_results['information_ratio'] = information_ratio
            
            self.logger.info(f"回测完成: {start_date} 到 {end_date}")
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"回测优化失败: {e}")
            return {} 