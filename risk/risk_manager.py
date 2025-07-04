#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风险管理模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from abc import ABC, abstractmethod
from scipy import stats
from sklearn.decomposition import PCA


class BaseRiskModel(ABC):
    """风险模型基类"""
    
    def __init__(self, name: str):
        """初始化风险模型"""
        self.name = name
        self.fitted = False
    
    @abstractmethod
    def calculate_portfolio_risk(self, weights: pd.Series,
                               returns: pd.DataFrame) -> float:
        """计算组合风险"""
        pass
    
    @abstractmethod
    def calculate_var(self, portfolio_returns: pd.Series,
                     confidence: float = 0.95) -> float:
        """计算VaR"""
        pass


class HistoricalRiskModel(BaseRiskModel):
    """历史风险模型"""
    
    def __init__(self, lookback_period: int = 252):
        """初始化历史风险模型"""
        super().__init__("historical")
        self.lookback_period = lookback_period
        self.covariance_matrix = None
        self.historical_returns = None
    
    def calculate_portfolio_risk(self, weights: pd.Series,
                               returns: pd.DataFrame) -> float:
        """基于历史数据计算组合风险"""
        # 取最近lookback_period期数据
        recent_returns = returns.tail(self.lookback_period)
        
        # 计算协方差矩阵
        cov_matrix = recent_returns.cov()
        
        # 对齐权重和协方差矩阵
        common_assets = weights.index.intersection(cov_matrix.index)
        aligned_weights = weights.reindex(common_assets, fill_value=0)
        aligned_cov = cov_matrix.reindex(common_assets).reindex(common_assets, axis=1)
        
        # 计算组合风险
        portfolio_variance = aligned_weights.T @ aligned_cov @ aligned_weights
        portfolio_risk = np.sqrt(portfolio_variance) * np.sqrt(252)  # 年化
        
        return portfolio_risk
    
    def calculate_var(self, portfolio_returns: pd.Series,
                     confidence: float = 0.95) -> float:
        """历史VaR"""
        if len(portfolio_returns) == 0:
            return 0.0
        
        # 取最近lookback_period期数据
        recent_returns = portfolio_returns.tail(self.lookback_period)
        
        # 计算VaR
        var = -recent_returns.quantile(1 - confidence)
        
        return var
    
    def calculate_expected_shortfall(self, portfolio_returns: pd.Series,
                                   confidence: float = 0.95) -> float:
        """条件VaR"""
        if len(portfolio_returns) == 0:
            return 0.0
        
        # 取最近lookback_period期数据
        recent_returns = portfolio_returns.tail(self.lookback_period)
        
        # 计算VaR
        var = -recent_returns.quantile(1 - confidence)
        
        # 计算条件VaR（期望损失）
        tail_losses = recent_returns[recent_returns <= -var]
        if len(tail_losses) > 0:
            expected_shortfall = -tail_losses.mean()
        else:
            expected_shortfall = var
            
        return expected_shortfall


class ParametricRiskModel(BaseRiskModel):
    """参数化风险模型"""
    
    def __init__(self):
        """初始化参数化风险模型"""
        super().__init__("parametric")
        self.distribution_params = {}
    
    def calculate_portfolio_risk(self, weights: pd.Series,
                               returns: pd.DataFrame) -> float:
        """参数化风险计算"""
        # 假设正态分布
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # 对齐权重和数据
        common_assets = weights.index.intersection(cov_matrix.index)
        aligned_weights = weights.reindex(common_assets, fill_value=0)
        aligned_cov = cov_matrix.reindex(common_assets).reindex(common_assets, axis=1)
        
        # 计算组合风险
        portfolio_variance = aligned_weights.T @ aligned_cov @ aligned_weights
        portfolio_risk = np.sqrt(portfolio_variance) * np.sqrt(252)  # 年化
        
        return portfolio_risk
    
    def calculate_var(self, portfolio_returns: pd.Series,
                     confidence: float = 0.95) -> float:
        """参数化VaR"""
        if len(portfolio_returns) == 0:
            return 0.0
        
        # 拟合正态分布
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # 计算VaR
        var = -(mean_return + std_return * stats.norm.ppf(1 - confidence))
        
        return var
    
    def fit_distribution(self, returns: pd.Series,
                        distribution: str = "normal") -> Dict:
        """拟合分布"""
        if distribution == "normal":
            mean = returns.mean()
            std = returns.std()
            
            # 正态性检验
            shapiro_stat, shapiro_p = stats.shapiro(returns.sample(min(len(returns), 5000)))
            
            params = {
                'mean': mean,
                'std': std,
                'distribution': 'normal',
                'normality_test': {'statistic': shapiro_stat, 'p_value': shapiro_p}
            }
            
        elif distribution == "t":
            # 拟合t分布
            params_fitted = stats.t.fit(returns)
            params = {
                'df': params_fitted[0],
                'loc': params_fitted[1],
                'scale': params_fitted[2],
                'distribution': 't'
            }
            
        else:
            # 默认正态分布
            mean = returns.mean()
            std = returns.std()
            params = {
                'mean': mean,
                'std': std,
                'distribution': 'normal'
            }
        
        self.distribution_params = params
        return params


class MonteCarloRiskModel(BaseRiskModel):
    """蒙特卡洛风险模型"""
    
    def __init__(self, n_simulations: int = 10000):
        """初始化蒙特卡洛风险模型"""
        super().__init__("monte_carlo")
        self.n_simulations = n_simulations
        self.simulated_returns = None
    
    def calculate_portfolio_risk(self, weights: pd.Series,
                               returns: pd.DataFrame) -> float:
        """蒙特卡洛风险计算"""
        # 模拟组合收益
        simulated_returns = self.simulate_portfolio_returns(weights, returns)
        
        # 计算风险（标准差）
        portfolio_risk = np.std(simulated_returns) * np.sqrt(252)  # 年化
        
        return portfolio_risk
    
    def calculate_var(self, portfolio_returns: pd.Series,
                     confidence: float = 0.95) -> float:
        """蒙特卡洛VaR"""
        if len(portfolio_returns) == 0:
            return 0.0
        
        # 基于历史数据的参数估计
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # 蒙特卡洛模拟
        simulated_returns = np.random.normal(mean_return, std_return, self.n_simulations)
        
        # 计算VaR
        var = -np.percentile(simulated_returns, (1 - confidence) * 100)
        
        return var
    
    def simulate_portfolio_returns(self, weights: pd.Series,
                                 returns: pd.DataFrame,
                                 time_horizon: int = 1) -> np.ndarray:
        """模拟组合收益"""
        # 估计参数
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # 对齐权重和数据
        common_assets = weights.index.intersection(cov_matrix.index)
        aligned_weights = weights.reindex(common_assets, fill_value=0)
        aligned_mean = mean_returns.reindex(common_assets, fill_value=0)
        aligned_cov = cov_matrix.reindex(common_assets).reindex(common_assets, axis=1)
        
        # 蒙特卡洛模拟
        simulated_portfolio_returns = []
        
        for _ in range(self.n_simulations):
            # 生成随机收益
            random_returns = np.random.multivariate_normal(aligned_mean, aligned_cov, time_horizon)
            
            # 计算组合收益
            portfolio_returns = []
            for t in range(time_horizon):
                portfolio_return = aligned_weights.T @ random_returns[t]
                portfolio_returns.append(portfolio_return)
            
            # 计算累积收益
            cumulative_return = np.prod(1 + np.array(portfolio_returns)) - 1
            simulated_portfolio_returns.append(cumulative_return)
        
        self.simulated_returns = np.array(simulated_portfolio_returns)
        return self.simulated_returns


class FactorRiskModel:
    """因子风险模型"""
    
    def __init__(self, factor_model_type: str = "pca"):
        """初始化因子风险模型"""
        self.factor_model_type = factor_model_type
        self.factor_loadings = None
        self.factor_returns = None
        self.specific_risk = None
        self.factor_covariance = None
    
    def fit_factor_model(self, returns: pd.DataFrame,
                        n_factors: int = 10) -> Dict:
        """拟合因子模型"""
        if self.factor_model_type == "pca":
            # PCA因子模型
            pca = PCA(n_components=n_factors)
            factor_scores = pca.fit_transform(returns.fillna(0))
            
            # 因子收益
            factor_returns = pd.DataFrame(factor_scores, 
                                        index=returns.index,
                                        columns=[f'Factor_{i+1}' for i in range(n_factors)])
            
            # 因子载荷
            factor_loadings = pd.DataFrame(pca.components_.T,
                                         index=returns.columns,
                                         columns=factor_returns.columns)
            
            # 解释方差比例
            explained_variance = pca.explained_variance_ratio_
            
            self.factor_returns = factor_returns
            self.factor_loadings = factor_loadings
            
            return {
                'factor_returns': factor_returns,
                'factor_loadings': factor_loadings,
                'explained_variance': explained_variance,
                'cumulative_variance': explained_variance.cumsum()
            }
        
        else:
            # 简化实现：使用前n_factors个资产作为因子
            factor_returns = returns.iloc[:, :n_factors]
            factor_loadings = pd.DataFrame(np.eye(len(returns.columns), n_factors),
                                         index=returns.columns,
                                         columns=factor_returns.columns)
            
            self.factor_returns = factor_returns
            self.factor_loadings = factor_loadings
            
            return {
                'factor_returns': factor_returns,
                'factor_loadings': factor_loadings
            }
    
    def calculate_factor_exposures(self, returns: pd.DataFrame) -> pd.DataFrame:
        """计算因子暴露"""
        if self.factor_returns is None:
            self.fit_factor_model(returns)
        
        exposures = []
        
        for asset in returns.columns:
            asset_returns = returns[asset].dropna()
            
            # 对因子进行回归
            from sklearn.linear_model import LinearRegression
            
            # 对齐数据
            common_dates = asset_returns.index.intersection(self.factor_returns.index)
            y = asset_returns.loc[common_dates].values.reshape(-1, 1)
            X = self.factor_returns.loc[common_dates].values
            
            if len(y) > 0 and X.shape[1] > 0:
                model = LinearRegression()
                model.fit(X, y)
                exposures.append(model.coef_[0])
            else:
                exposures.append(np.zeros(self.factor_returns.shape[1]))
        
        exposure_df = pd.DataFrame(exposures,
                                 index=returns.columns,
                                 columns=self.factor_returns.columns)
        
        return exposure_df
    
    def calculate_specific_risk(self, returns: pd.DataFrame,
                              factor_returns: pd.DataFrame,
                              exposures: pd.DataFrame) -> pd.Series:
        """计算特异风险"""
        specific_risks = []
        
        for asset in returns.columns:
            if asset not in exposures.index:
                specific_risks.append(returns[asset].std())
                continue
            
            asset_returns = returns[asset].dropna()
            asset_exposures = exposures.loc[asset]
            
            # 计算系统性风险
            common_dates = asset_returns.index.intersection(factor_returns.index)
            if len(common_dates) > 0:
                systematic_returns = factor_returns.loc[common_dates] @ asset_exposures
                residual_returns = asset_returns.loc[common_dates] - systematic_returns
                specific_risk = residual_returns.std()
            else:
                specific_risk = asset_returns.std()
            
            specific_risks.append(specific_risk)
        
        return pd.Series(specific_risks, index=returns.columns)
    
    def calculate_portfolio_risk_decomposition(self, weights: pd.Series,
                                             factor_cov: pd.DataFrame,
                                             exposures: pd.DataFrame,
                                             specific_risk: pd.Series) -> Dict:
        """组合风险分解"""
        # 对齐数据
        common_assets = weights.index.intersection(exposures.index)
        aligned_weights = weights.reindex(common_assets, fill_value=0)
        aligned_exposures = exposures.reindex(common_assets)
        aligned_specific_risk = specific_risk.reindex(common_assets, fill_value=0)
        
        # 计算组合因子暴露
        portfolio_exposures = aligned_weights.T @ aligned_exposures
        
        # 计算因子风险
        factor_risk = np.sqrt(portfolio_exposures.T @ factor_cov @ portfolio_exposures)
        
        # 计算特异风险
        specific_risk_portfolio = np.sqrt((aligned_weights ** 2).T @ (aligned_specific_risk ** 2))
        
        # 总风险
        total_risk = np.sqrt(factor_risk ** 2 + specific_risk_portfolio ** 2)
        
        return {
            'total_risk': total_risk,
            'factor_risk': factor_risk,
            'specific_risk': specific_risk_portfolio,
            'portfolio_exposures': portfolio_exposures,
            'factor_risk_contribution': factor_risk / total_risk if total_risk > 0 else 0,
            'specific_risk_contribution': specific_risk_portfolio / total_risk if total_risk > 0 else 0
        }


class DrawdownManager:
    """回撤管理器"""
    
    def __init__(self, max_drawdown: float = 0.15):
        """初始化回撤管理器"""
        pass
    
    def calculate_drawdown(self, portfolio_value: pd.Series) -> pd.Series:
        """计算回撤"""
        if len(portfolio_value) == 0:
            return pd.Series()
        
        # 计算累积最大值
        running_max = portfolio_value.expanding().max()
        
        # 计算回撤
        drawdown = (portfolio_value - running_max) / running_max
        
        return drawdown
    
    def calculate_max_drawdown(self, portfolio_value: pd.Series) -> float:
        """计算最大回撤"""
        if len(portfolio_value) == 0:
            return 0.0
        
        drawdown = self.calculate_drawdown(portfolio_value)
        return drawdown.min()
    
    def check_drawdown_breach(self, current_drawdown: float) -> bool:
        """检查回撤违规"""
        return current_drawdown < -self.max_drawdown
    
    def calculate_drawdown_duration(self, portfolio_value: pd.Series) -> pd.Series:
        """计算回撤持续时间"""
        if len(portfolio_value) == 0:
            return pd.Series()
        
        # 计算回撤
        drawdown = self.calculate_drawdown(portfolio_value)
        
        # 计算回撤持续时间
        durations = []
        current_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
            else:
                current_duration = 0
            durations.append(current_duration)
        
        return pd.Series(durations, index=portfolio_value.index)
    
    def underwater_curve(self, portfolio_value: pd.Series) -> pd.Series:
        """水下曲线"""
        # 水下曲线就是回撤序列
        return self.calculate_drawdown(portfolio_value)


class VolatilityManager:
    """波动率管理器"""
    
    def __init__(self, target_volatility: float = 0.15):
        """初始化波动率管理器"""
        pass
    
    def calculate_realized_volatility(self, returns: pd.Series,
                                    window: int = 20) -> pd.Series:
        """计算实现波动率"""
        if len(returns) == 0:
            return pd.Series()
        
        # 计算滚动标准差
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)  # 年化
        
        return rolling_vol
    
    def calculate_garch_volatility(self, returns: pd.Series) -> pd.Series:
        """GARCH波动率预测"""
        if len(returns) == 0:
            return pd.Series()
        
        try:
            from arch import arch_model
            
            # 拟合GARCH模型
            model = arch_model(returns.dropna(), vol='Garch', p=1, q=1)
            fitted_model = model.fit(disp='off')
            
            # 预测波动率
            forecast_vol = fitted_model.conditional_volatility * np.sqrt(252)  # 年化
            
            return pd.Series(forecast_vol, index=returns.index)
            
        except ImportError:
            # 如果没有arch库，使用简单的EWMA
            return self._simple_garch_approximation(returns)
    
    def _simple_garch_approximation(self, returns: pd.Series, alpha: float = 0.94) -> pd.Series:
        """简单GARCH近似"""
        if len(returns) == 0:
            return pd.Series()
        
        # 使用EWMA近似GARCH
        ewma_vol = returns.ewm(alpha=alpha).std() * np.sqrt(252)
        
        return ewma_vol
    
    def volatility_targeting(self, weights: pd.Series,
                           forecasted_vol: float) -> pd.Series:
        """波动率目标化"""
        if forecasted_vol <= 0:
            return weights
        
        # 计算缩放因子
        scaling_factor = self.target_volatility / forecasted_vol
        
        # 应用缩放
        scaled_weights = weights * scaling_factor
        
        return scaled_weights
    
    def dynamic_volatility_scaling(self, weights: pd.Series,
                                 realized_vol: pd.Series,
                                 forecast_vol: pd.Series) -> pd.Series:
        """动态波动率调整"""
        if len(realized_vol) == 0 or len(forecast_vol) == 0:
            return weights
        
        # 计算波动率比率
        vol_ratio = realized_vol / forecast_vol
        
        # 应用动态调整
        adjusted_weights = weights / vol_ratio.iloc[-1] if len(vol_ratio) > 0 else weights
        
        return adjusted_weights


class PositionLimitManager:
    """仓位限制管理器"""
    
    def __init__(self, config: Dict):
        """初始化仓位限制管理器"""
        self.config = config
        self.max_single_position = config.get('max_single_position', 0.05)  # 5%
        self.max_sector_exposure = config.get('max_sector_exposure', 0.25)  # 25%
        self.max_concentration = config.get('max_concentration', 0.5)  # 50%
        self.max_long_position = config.get('max_long_position', 0.1)  # 10%
        self.max_short_position = config.get('max_short_position', -0.05)  # -5%
        self.min_position_size = config.get('min_position_size', 0.001)  # 0.1%
        self.max_positions = config.get('max_positions', 100)  # 最大持仓数
        self.sector_limits = config.get('sector_limits', {})
        self.violation_log = []
    
    def check_single_position_limit(self, weights: pd.Series) -> Dict[str, bool]:
        """检查单一仓位限制
        
        Args:
            weights: 权重序列
            
        Returns:
            检查结果字典
        """
        try:
            results = {}
            
            # 检查单一仓位绝对值限制
            abs_weights = weights.abs()
            max_position = abs_weights.max()
            results['max_single_position_check'] = max_position <= self.max_single_position
            
            # 检查多头仓位限制
            max_long = weights.max()
            results['max_long_position_check'] = max_long <= self.max_long_position
            
            # 检查空头仓位限制
            min_short = weights.min()
            results['max_short_position_check'] = min_short >= self.max_short_position
            
            # 检查最小仓位大小
            non_zero_positions = weights[weights.abs() > 0]
            if len(non_zero_positions) > 0:
                min_position = non_zero_positions.abs().min()
                results['min_position_size_check'] = min_position >= self.min_position_size
            else:
                results['min_position_size_check'] = True
            
            # 检查最大持仓数
            num_positions = len(weights[weights.abs() > self.min_position_size])
            results['max_positions_check'] = num_positions <= self.max_positions
            
            # 记录违规
            violations = [k for k, v in results.items() if not v]
            if violations:
                self.violation_log.append({
                    'timestamp': pd.Timestamp.now(),
                    'type': 'single_position_limit',
                    'violations': violations,
                    'max_position': max_position,
                    'num_positions': num_positions
                })
            
            return results
            
        except Exception as e:
            print(f"检查单一仓位限制失败: {e}")
            return {'error': True}
    
    def check_sector_limits(self, weights: pd.Series,
                          sector_mapping: Dict[str, str]) -> Dict[str, bool]:
        """检查行业限制
        
        Args:
            weights: 权重序列
            sector_mapping: 股票到行业的映射
            
        Returns:
            检查结果字典
        """
        try:
            results = {}
            
            if not sector_mapping:
                return {'sector_limits_check': True}
            
            # 计算行业暴露
            sector_exposures = {}
            for symbol, weight in weights.items():
                sector = sector_mapping.get(symbol, 'Unknown')
                if sector not in sector_exposures:
                    sector_exposures[sector] = 0
                sector_exposures[sector] += weight
            
            # 检查行业限制
            sector_violations = []
            for sector, exposure in sector_exposures.items():
                abs_exposure = abs(exposure)
                
                # 检查通用行业限制
                if abs_exposure > self.max_sector_exposure:
                    sector_violations.append({
                        'sector': sector,
                        'exposure': exposure,
                        'limit': self.max_sector_exposure
                    })
                
                # 检查特定行业限制
                if sector in self.sector_limits:
                    sector_limit = self.sector_limits[sector]
                    if abs_exposure > sector_limit:
                        sector_violations.append({
                            'sector': sector,
                            'exposure': exposure,
                            'limit': sector_limit
                        })
            
            results['sector_limits_check'] = len(sector_violations) == 0
            results['sector_exposures'] = sector_exposures
            results['sector_violations'] = sector_violations
            
            # 记录违规
            if sector_violations:
                self.violation_log.append({
                    'timestamp': pd.Timestamp.now(),
                    'type': 'sector_limit',
                    'violations': sector_violations
                })
            
            return results
            
        except Exception as e:
            print(f"检查行业限制失败: {e}")
            return {'error': True}
    
    def check_concentration_limits(self, weights: pd.Series,
                                 top_n: int = 10,
                                 max_concentration: float = 0.5) -> bool:
        """检查集中度限制
        
        Args:
            weights: 权重序列
            top_n: 前N大持仓
            max_concentration: 最大集中度
            
        Returns:
            是否通过检查
        """
        try:
            if len(weights) == 0:
                return True
            
            # 计算前N大持仓的集中度
            abs_weights = weights.abs()
            top_n_concentration = abs_weights.nlargest(top_n).sum()
            
            # 检查集中度限制
            concentration_check = top_n_concentration <= max_concentration
            
            # 记录违规
            if not concentration_check:
                self.violation_log.append({
                    'timestamp': pd.Timestamp.now(),
                    'type': 'concentration_limit',
                    'concentration': top_n_concentration,
                    'limit': max_concentration,
                    'top_n': top_n
                })
            
            return concentration_check
            
        except Exception as e:
            print(f"检查集中度限制失败: {e}")
            return False
    
    def apply_position_limits(self, target_weights: pd.Series) -> pd.Series:
        """应用仓位限制
        
        Args:
            target_weights: 目标权重
            
        Returns:
            调整后的权重
        """
        try:
            adjusted_weights = target_weights.copy()
            
            # 1. 应用单一仓位限制
            adjusted_weights = self._apply_single_position_limits(adjusted_weights)
            
            # 2. 应用集中度限制
            adjusted_weights = self._apply_concentration_limits(adjusted_weights)
            
            # 3. 清理小仓位
            adjusted_weights = self._clean_small_positions(adjusted_weights)
            
            # 4. 重新标准化权重
            adjusted_weights = self._normalize_weights(adjusted_weights)
            
            return adjusted_weights
            
        except Exception as e:
            print(f"应用仓位限制失败: {e}")
            return target_weights
    
    def _apply_single_position_limits(self, weights: pd.Series) -> pd.Series:
        """应用单一仓位限制"""
        try:
            adjusted_weights = weights.copy()
            
            # 限制多头仓位
            adjusted_weights = adjusted_weights.clip(upper=self.max_long_position)
            
            # 限制空头仓位
            adjusted_weights = adjusted_weights.clip(lower=self.max_short_position)
            
            # 限制绝对仓位大小
            abs_weights = adjusted_weights.abs()
            over_limit_mask = abs_weights > self.max_single_position
            
            if over_limit_mask.any():
                # 按比例缩放超限仓位
                scale_factor = self.max_single_position / abs_weights[over_limit_mask]
                adjusted_weights[over_limit_mask] *= scale_factor
            
            return adjusted_weights
            
        except Exception as e:
            print(f"应用单一仓位限制失败: {e}")
            return weights
    
    def _apply_concentration_limits(self, weights: pd.Series) -> pd.Series:
        """应用集中度限制"""
        try:
            adjusted_weights = weights.copy()
            
            # 检查前10大持仓集中度
            abs_weights = adjusted_weights.abs()
            top_10_concentration = abs_weights.nlargest(10).sum()
            
            if top_10_concentration > self.max_concentration:
                # 按比例缩放所有仓位
                scale_factor = self.max_concentration / top_10_concentration
                adjusted_weights *= scale_factor
            
            return adjusted_weights
            
        except Exception as e:
            print(f"应用集中度限制失败: {e}")
            return weights
    
    def _clean_small_positions(self, weights: pd.Series) -> pd.Series:
        """清理小仓位"""
        try:
            adjusted_weights = weights.copy()
            
            # 清理小于最小仓位的持仓
            small_positions = adjusted_weights.abs() < self.min_position_size
            adjusted_weights[small_positions] = 0
            
            # 限制持仓数量
            if len(adjusted_weights[adjusted_weights != 0]) > self.max_positions:
                # 保留绝对值最大的持仓
                abs_weights = adjusted_weights.abs()
                threshold = abs_weights.nlargest(self.max_positions).iloc[-1]
                keep_mask = abs_weights >= threshold
                adjusted_weights[~keep_mask] = 0
            
            return adjusted_weights
            
        except Exception as e:
            print(f"清理小仓位失败: {e}")
            return weights
    
    def _normalize_weights(self, weights: pd.Series) -> pd.Series:
        """重新标准化权重"""
        try:
            if weights.sum() == 0:
                return weights
            
            # 简单的标准化：保持权重和为1
            weight_sum = weights.sum()
            if abs(weight_sum) > 1e-10:
                normalized_weights = weights / weight_sum
            else:
                normalized_weights = weights
            
            return normalized_weights
            
        except Exception as e:
            print(f"标准化权重失败: {e}")
            return weights
    
    def get_violation_summary(self) -> Dict:
        """获取违规摘要"""
        try:
            if not self.violation_log:
                return {'total_violations': 0, 'violation_types': {}}
            
            # 统计违规类型
            violation_types = {}
            for violation in self.violation_log:
                v_type = violation['type']
                if v_type not in violation_types:
                    violation_types[v_type] = 0
                violation_types[v_type] += 1
            
            # 最近的违规
            recent_violations = self.violation_log[-10:] if len(self.violation_log) > 10 else self.violation_log
            
            return {
                'total_violations': len(self.violation_log),
                'violation_types': violation_types,
                'recent_violations': recent_violations
            }
            
        except Exception as e:
            print(f"获取违规摘要失败: {e}")
            return {'error': str(e)}
    
    def update_limits(self, new_limits: Dict):
        """更新限制参数"""
        try:
            for key, value in new_limits.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
            # 更新配置
            self.config.update(new_limits)
            
        except Exception as e:
            print(f"更新限制参数失败: {e}")
    
    def calculate_position_utilization(self, weights: pd.Series) -> Dict:
        """计算仓位利用率"""
        try:
            if len(weights) == 0:
                return {}
            
            abs_weights = weights.abs()
            
            # 计算各种利用率指标
            utilization = {
                'max_position_utilization': abs_weights.max() / self.max_single_position,
                'concentration_utilization': abs_weights.nlargest(10).sum() / self.max_concentration,
                'position_count_utilization': len(weights[abs_weights > self.min_position_size]) / self.max_positions,
                'gross_exposure': abs_weights.sum(),
                'net_exposure': weights.sum(),
                'long_exposure': weights[weights > 0].sum(),
                'short_exposure': weights[weights < 0].sum()
            }
            
            return utilization
            
        except Exception as e:
            print(f"计算仓位利用率失败: {e}")
            return {}


class LeverageManager:
    """杠杆管理器"""
    
    def __init__(self, max_leverage: float = 1.0, max_gross_exposure: float = None, max_net_exposure: float = None):
        """初始化杠杆管理器"""
        self.max_leverage = max_leverage
        self.max_gross_exposure = max_gross_exposure if max_gross_exposure is not None else max_leverage * 2.0  # 总暴露限制
        self.max_net_exposure = max_net_exposure if max_net_exposure is not None else max_leverage * 1.0    # 净暴露限制
        self.leverage_history = []
        self.warnings = []
    
    def calculate_leverage(self, weights: pd.Series) -> float:
        """计算杠杆率
        
        Args:
            weights: 权重序列
            
        Returns:
            杠杆率
        """
        try:
            if len(weights) == 0:
                return 0.0
            
            # 杠杆率 = 总暴露 / 净资产
            # 假设净资产为1，则杠杆率等于总暴露
            gross_exposure = self.calculate_gross_exposure(weights)
            
            return gross_exposure
            
        except Exception as e:
            print(f"计算杠杆率失败: {e}")
            return 0.0
    
    def calculate_gross_exposure(self, weights: pd.Series) -> float:
        """计算总暴露
        
        Args:
            weights: 权重序列
            
        Returns:
            总暴露（多头+空头的绝对值）
        """
        try:
            if len(weights) == 0:
                return 0.0
            
            # 总暴露 = 所有仓位绝对值之和
            gross_exposure = weights.abs().sum()
            
            return gross_exposure
            
        except Exception as e:
            print(f"计算总暴露失败: {e}")
            return 0.0
    
    def calculate_net_exposure(self, weights: pd.Series) -> float:
        """计算净暴露
        
        Args:
            weights: 权重序列
            
        Returns:
            净暴露（多头-空头）
        """
        try:
            if len(weights) == 0:
                return 0.0
            
            # 净暴露 = 多头仓位 - 空头仓位
            net_exposure = weights.sum()
            
            return net_exposure
            
        except Exception as e:
            print(f"计算净暴露失败: {e}")
            return 0.0
    
    def adjust_leverage(self, weights: pd.Series) -> pd.Series:
        """调整杠杆
        
        Args:
            weights: 权重序列
            
        Returns:
            调整后的权重
        """
        try:
            if len(weights) == 0:
                return weights
            
            adjusted_weights = weights.copy()
            
            # 1. 检查并调整总暴露
            adjusted_weights = self._adjust_gross_exposure(adjusted_weights)
            
            # 2. 检查并调整净暴露
            adjusted_weights = self._adjust_net_exposure(adjusted_weights)
            
            # 3. 记录调整后的杠杆
            self._record_leverage(adjusted_weights)
            
            return adjusted_weights
            
        except Exception as e:
            print(f"调整杠杆失败: {e}")
            return weights
    
    def _adjust_gross_exposure(self, weights: pd.Series) -> pd.Series:
        """调整总暴露"""
        try:
            gross_exposure = self.calculate_gross_exposure(weights)
            
            if gross_exposure > self.max_gross_exposure:
                # 按比例缩放所有仓位
                scale_factor = self.max_gross_exposure / gross_exposure
                adjusted_weights = weights * scale_factor
                
                # 记录警告
                self.warnings.append({
                    'timestamp': pd.Timestamp.now(),
                    'type': 'gross_exposure_adjustment',
                    'original_exposure': gross_exposure,
                    'adjusted_exposure': gross_exposure * scale_factor,
                    'scale_factor': scale_factor
                })
                
                return adjusted_weights
            
            return weights
            
        except Exception as e:
            print(f"调整总暴露失败: {e}")
            return weights
    
    def _adjust_net_exposure(self, weights: pd.Series) -> pd.Series:
        """调整净暴露"""
        try:
            net_exposure = self.calculate_net_exposure(weights)
            
            if abs(net_exposure) > self.max_net_exposure:
                # 计算需要调整的量
                excess_exposure = abs(net_exposure) - self.max_net_exposure
                
                # 选择调整策略
                if net_exposure > 0:
                    # 净多头过多，减少多头或增加空头
                    adjusted_weights = self._reduce_net_long_exposure(weights, excess_exposure)
                else:
                    # 净空头过多，减少空头或增加多头
                    adjusted_weights = self._reduce_net_short_exposure(weights, excess_exposure)
                
                # 记录警告
                self.warnings.append({
                    'timestamp': pd.Timestamp.now(),
                    'type': 'net_exposure_adjustment',
                    'original_exposure': net_exposure,
                    'adjusted_exposure': self.calculate_net_exposure(adjusted_weights),
                    'excess_exposure': excess_exposure
                })
                
                return adjusted_weights
            
            return weights
            
        except Exception as e:
            print(f"调整净暴露失败: {e}")
            return weights
    
    def _reduce_net_long_exposure(self, weights: pd.Series, excess_exposure: float) -> pd.Series:
        """减少净多头暴露"""
        try:
            adjusted_weights = weights.copy()
            
            # 获取多头仓位
            long_positions = weights[weights > 0]
            
            if len(long_positions) > 0:
                # 按比例减少多头仓位
                reduction_factor = excess_exposure / long_positions.sum()
                reduction_factor = min(reduction_factor, 1.0)  # 最多减少到0
                
                adjusted_weights[weights > 0] *= (1 - reduction_factor)
            
            return adjusted_weights
            
        except Exception as e:
            print(f"减少净多头暴露失败: {e}")
            return weights
    
    def _reduce_net_short_exposure(self, weights: pd.Series, excess_exposure: float) -> pd.Series:
        """减少净空头暴露"""
        try:
            adjusted_weights = weights.copy()
            
            # 获取空头仓位
            short_positions = weights[weights < 0]
            
            if len(short_positions) > 0:
                # 按比例减少空头仓位（增加其值，因为是负数）
                reduction_factor = excess_exposure / abs(short_positions.sum())
                reduction_factor = min(reduction_factor, 1.0)  # 最多减少到0
                
                adjusted_weights[weights < 0] *= (1 - reduction_factor)
            
            return adjusted_weights
            
        except Exception as e:
            print(f"减少净空头暴露失败: {e}")
            return weights
    
    def _record_leverage(self, weights: pd.Series):
        """记录杠杆信息"""
        try:
            leverage_info = {
                'timestamp': pd.Timestamp.now(),
                'leverage': self.calculate_leverage(weights),
                'gross_exposure': self.calculate_gross_exposure(weights),
                'net_exposure': self.calculate_net_exposure(weights),
                'long_exposure': weights[weights > 0].sum() if (weights > 0).any() else 0,
                'short_exposure': weights[weights < 0].sum() if (weights < 0).any() else 0,
                'num_positions': len(weights[weights != 0])
            }
            
            self.leverage_history.append(leverage_info)
            
            # 保持最近100条记录
            if len(self.leverage_history) > 100:
                self.leverage_history = self.leverage_history[-100:]
                
        except Exception as e:
            print(f"记录杠杆信息失败: {e}")
    
    def check_leverage_limits(self, weights: pd.Series) -> Dict[str, bool]:
        """检查杠杆限制
        
        Args:
            weights: 权重序列
            
        Returns:
            检查结果
        """
        try:
            results = {}
            
            # 检查总暴露
            gross_exposure = self.calculate_gross_exposure(weights)
            results['gross_exposure_check'] = gross_exposure <= self.max_gross_exposure
            
            # 检查净暴露
            net_exposure = self.calculate_net_exposure(weights)
            results['net_exposure_check'] = abs(net_exposure) <= self.max_net_exposure
            
            # 检查杠杆率
            leverage = self.calculate_leverage(weights)
            results['leverage_check'] = leverage <= self.max_leverage
            
            # 综合检查
            results['all_checks_passed'] = all(results.values())
            
            # 记录具体数值
            results['metrics'] = {
                'gross_exposure': gross_exposure,
                'net_exposure': net_exposure,
                'leverage': leverage,
                'max_gross_exposure': self.max_gross_exposure,
                'max_net_exposure': self.max_net_exposure,
                'max_leverage': self.max_leverage
            }
            
            return results
            
        except Exception as e:
            print(f"检查杠杆限制失败: {e}")
            return {'error': True}
    
    def get_leverage_summary(self) -> Dict:
        """获取杠杆摘要"""
        try:
            if not self.leverage_history:
                return {'no_data': True}
            
            # 最近的杠杆信息
            latest = self.leverage_history[-1]
            
            # 历史统计
            leverages = [info['leverage'] for info in self.leverage_history]
            gross_exposures = [info['gross_exposure'] for info in self.leverage_history]
            net_exposures = [info['net_exposure'] for info in self.leverage_history]
            
            summary = {
                'latest': latest,
                'statistics': {
                    'avg_leverage': np.mean(leverages),
                    'max_leverage': np.max(leverages),
                    'min_leverage': np.min(leverages),
                    'avg_gross_exposure': np.mean(gross_exposures),
                    'max_gross_exposure': np.max(gross_exposures),
                    'avg_net_exposure': np.mean(net_exposures),
                    'max_net_exposure': np.max(net_exposures),
                    'min_net_exposure': np.min(net_exposures)
                },
                'warnings_count': len(self.warnings),
                'recent_warnings': self.warnings[-5:] if len(self.warnings) > 5 else self.warnings
            }
            
            return summary
            
        except Exception as e:
            print(f"获取杠杆摘要失败: {e}")
            return {'error': str(e)}
    
    def calculate_leverage_utilization(self, weights: pd.Series) -> Dict:
        """计算杠杆利用率"""
        try:
            if len(weights) == 0:
                return {}
            
            leverage = self.calculate_leverage(weights)
            gross_exposure = self.calculate_gross_exposure(weights)
            net_exposure = self.calculate_net_exposure(weights)
            
            utilization = {
                'leverage_utilization': leverage / self.max_leverage,
                'gross_exposure_utilization': gross_exposure / self.max_gross_exposure,
                'net_exposure_utilization': abs(net_exposure) / self.max_net_exposure,
                'long_short_ratio': self._calculate_long_short_ratio(weights),
                'position_efficiency': self._calculate_position_efficiency(weights)
            }
            
            return utilization
            
        except Exception as e:
            print(f"计算杠杆利用率失败: {e}")
            return {}
    
    def _calculate_long_short_ratio(self, weights: pd.Series) -> float:
        """计算多空比率"""
        try:
            long_exposure = weights[weights > 0].sum()
            short_exposure = abs(weights[weights < 0].sum())
            
            if short_exposure == 0:
                return float('inf') if long_exposure > 0 else 0
            
            return long_exposure / short_exposure
            
        except Exception as e:
            print(f"计算多空比率失败: {e}")
            return 0.0
    
    def _calculate_position_efficiency(self, weights: pd.Series) -> float:
        """计算仓位效率"""
        try:
            if len(weights) == 0:
                return 0.0
            
            # 仓位效率 = 有效仓位数 / 总仓位数
            non_zero_positions = len(weights[weights != 0])
            total_positions = len(weights)
            
            if total_positions == 0:
                return 0.0
            
            return non_zero_positions / total_positions
            
        except Exception as e:
            print(f"计算仓位效率失败: {e}")
            return 0.0
    
    def update_leverage_limits(self, max_leverage: float = None,
                             max_gross_exposure: float = None,
                             max_net_exposure: float = None):
        """更新杠杆限制"""
        try:
            if max_leverage is not None:
                self.max_leverage = max_leverage
            
            if max_gross_exposure is not None:
                self.max_gross_exposure = max_gross_exposure
            
            if max_net_exposure is not None:
                self.max_net_exposure = max_net_exposure
                
        except Exception as e:
            print(f"更新杠杆限制失败: {e}")
    
    def get_leverage_breakdown(self, weights: pd.Series) -> Dict:
        """获取杠杆分解"""
        try:
            if len(weights) == 0:
                return {}
            
            breakdown = {
                'total_positions': len(weights),
                'active_positions': len(weights[weights != 0]),
                'long_positions': len(weights[weights > 0]),
                'short_positions': len(weights[weights < 0]),
                'long_exposure': weights[weights > 0].sum(),
                'short_exposure': weights[weights < 0].sum(),
                'gross_exposure': self.calculate_gross_exposure(weights),
                'net_exposure': self.calculate_net_exposure(weights),
                'leverage': self.calculate_leverage(weights),
                'top_5_positions': weights.abs().nlargest(5).to_dict(),
                'position_distribution': self._get_position_distribution(weights)
            }
            
            return breakdown
            
        except Exception as e:
            print(f"获取杠杆分解失败: {e}")
            return {}
    
    def _get_position_distribution(self, weights: pd.Series) -> Dict:
        """获取仓位分布"""
        try:
            abs_weights = weights.abs()
            
            # 仓位大小分布
            bins = [0, 0.01, 0.02, 0.05, 0.1, 1.0]
            labels = ['<1%', '1-2%', '2-5%', '5-10%', '>10%']
            
            distribution = {}
            for i, (start, end) in enumerate(zip(bins[:-1], bins[1:])):
                mask = (abs_weights >= start) & (abs_weights < end)
                count = mask.sum()
                total_weight = abs_weights[mask].sum()
                
                distribution[labels[i]] = {
                    'count': count,
                    'total_weight': total_weight,
                    'percentage': count / len(weights) * 100 if len(weights) > 0 else 0
                }
            
            return distribution
            
        except Exception as e:
            print(f"获取仓位分布失败: {e}")
            return {}


class CorrelationMonitor:
    """相关性监控器"""
    
    def __init__(self):
        """初始化相关性监控器"""
        self.correlation_history = []
        self.correlation_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        self.regime_change_alerts = []
    
    def calculate_rolling_correlation(self, returns: pd.DataFrame,
                                    window: int = 60) -> pd.DataFrame:
        """计算滚动相关性
        
        Args:
            returns: 收益率DataFrame
            window: 滚动窗口大小
            
        Returns:
            滚动相关性DataFrame
        """
        try:
            if len(returns) < window:
                return pd.DataFrame()
            
            # 计算滚动相关性
            rolling_corr = returns.rolling(window=window).corr()
            
            # 记录相关性历史
            self._record_correlation_history(rolling_corr)
            
            return rolling_corr
            
        except Exception as e:
            print(f"计算滚动相关性失败: {e}")
            return pd.DataFrame()
    
    def detect_correlation_regime_change(self, correlations: pd.DataFrame) -> pd.Series:
        """检测相关性状态变化
        
        Args:
            correlations: 相关性DataFrame
            
        Returns:
            状态变化序列
        """
        try:
            if len(correlations) < 2:
                return pd.Series()
            
            # 计算平均相关性时间序列
            avg_correlations = correlations.apply(
                lambda x: self._extract_correlation_values(x), axis=1
            )
            
            # 检测状态变化
            regime_changes = self._detect_regime_changes(avg_correlations)
            
            return regime_changes
            
        except Exception as e:
            print(f"检测相关性状态变化失败: {e}")
            return pd.Series()
    
    def calculate_average_correlation(self, correlation_matrix: pd.DataFrame) -> float:
        """计算平均相关性
        
        Args:
            correlation_matrix: 相关性矩阵
            
        Returns:
            平均相关性
        """
        try:
            if correlation_matrix.empty:
                return 0.0
            
            # 提取上三角矩阵的相关性值（排除对角线）
            corr_values = self._extract_correlation_values(correlation_matrix)
            
            if len(corr_values) == 0:
                return 0.0
            
            # 计算平均值
            avg_correlation = np.mean(corr_values)
            
            return avg_correlation
            
        except Exception as e:
            print(f"计算平均相关性失败: {e}")
            return 0.0
    
    def correlation_clustering(self, correlation_matrix: pd.DataFrame) -> Dict:
        """相关性聚类
        
        Args:
            correlation_matrix: 相关性矩阵
            
        Returns:
            聚类结果
        """
        try:
            if correlation_matrix.empty:
                return {}
            
            # 使用聚类算法对相关性矩阵进行聚类
            clustering_result = self._perform_correlation_clustering(correlation_matrix)
            
            return clustering_result
            
        except Exception as e:
            print(f"相关性聚类失败: {e}")
            return {}
    
    def _extract_correlation_values(self, correlation_matrix: pd.DataFrame) -> np.ndarray:
        """提取相关性值"""
        try:
            if isinstance(correlation_matrix, pd.Series):
                # 如果是Series，尝试重构为矩阵
                n = int(np.sqrt(len(correlation_matrix)))
                if n * n == len(correlation_matrix):
                    matrix = correlation_matrix.values.reshape(n, n)
                else:
                    return np.array([])
            else:
                matrix = correlation_matrix.values
            
            # 提取上三角矩阵的值（排除对角线）
            upper_triangle = np.triu(matrix, k=1)
            corr_values = upper_triangle[upper_triangle != 0]
            
            return corr_values
            
        except Exception as e:
            print(f"提取相关性值失败: {e}")
            return np.array([])
    
    def _detect_regime_changes(self, avg_correlations: pd.Series) -> pd.Series:
        """检测状态变化"""
        try:
            if len(avg_correlations) < 20:
                return pd.Series([0] * len(avg_correlations), index=avg_correlations.index)
            
            # 计算相关性变化率
            correlation_changes = avg_correlations.pct_change()
            
            # 计算滚动标准差
            rolling_std = correlation_changes.rolling(window=20).std()
            
            # 检测异常变化
            threshold = rolling_std.quantile(0.95)
            regime_changes = (correlation_changes.abs() > threshold).astype(int)
            
            # 记录状态变化
            change_points = regime_changes[regime_changes == 1].index
            for change_point in change_points:
                self.regime_change_alerts.append({
                    'timestamp': change_point,
                    'correlation_change': correlation_changes.loc[change_point],
                    'threshold': threshold,
                    'correlation_level': avg_correlations.loc[change_point]
                })
            
            return regime_changes
            
        except Exception as e:
            print(f"检测状态变化失败: {e}")
            return pd.Series([0] * len(avg_correlations), index=avg_correlations.index)
    
    def _perform_correlation_clustering(self, correlation_matrix: pd.DataFrame) -> Dict:
        """执行相关性聚类"""
        try:
            # 将相关性矩阵转换为距离矩阵
            distance_matrix = 1 - correlation_matrix.abs()
            
            # 使用层次聚类
            try:
                from scipy.cluster.hierarchy import linkage, fcluster
                from scipy.spatial.distance import squareform
                
                # 转换为距离向量
                distance_vector = squareform(distance_matrix.values)
                
                # 执行层次聚类
                linkage_matrix = linkage(distance_vector, method='ward')
                
                # 确定聚类数量（基于相关性阈值）
                n_clusters = self._determine_optimal_clusters(correlation_matrix)
                
                # 获取聚类标签
                cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                
                # 构建聚类结果
                clustering_result = {
                    'cluster_labels': dict(zip(correlation_matrix.index, cluster_labels)),
                    'n_clusters': n_clusters,
                    'cluster_summary': self._summarize_clusters(correlation_matrix, cluster_labels)
                }
                
                return clustering_result
                
            except ImportError:
                # 如果没有scipy，使用简化的聚类方法
                return self._simple_clustering(correlation_matrix)
                
        except Exception as e:
            print(f"执行相关性聚类失败: {e}")
            return {}
    
    def _determine_optimal_clusters(self, correlation_matrix: pd.DataFrame) -> int:
        """确定最优聚类数量"""
        try:
            n_assets = len(correlation_matrix)
            
            # 根据资产数量确定聚类数量
            if n_assets <= 10:
                return 2
            elif n_assets <= 50:
                return min(5, n_assets // 3)
            else:
                return min(10, n_assets // 5)
                
        except Exception as e:
            print(f"确定最优聚类数量失败: {e}")
            return 2
    
    def _summarize_clusters(self, correlation_matrix: pd.DataFrame, cluster_labels: np.ndarray) -> Dict:
        """汇总聚类结果"""
        try:
            cluster_summary = {}
            
            for cluster_id in np.unique(cluster_labels):
                # 获取该聚类的资产
                cluster_assets = correlation_matrix.index[cluster_labels == cluster_id]
                
                # 计算聚类内相关性
                cluster_corr = correlation_matrix.loc[cluster_assets, cluster_assets]
                intra_cluster_corr = self.calculate_average_correlation(cluster_corr)
                
                cluster_summary[f'cluster_{cluster_id}'] = {
                    'assets': list(cluster_assets),
                    'size': len(cluster_assets),
                    'avg_intra_correlation': intra_cluster_corr,
                    'correlation_std': np.std(self._extract_correlation_values(cluster_corr))
                }
            
            return cluster_summary
            
        except Exception as e:
            print(f"汇总聚类结果失败: {e}")
            return {}
    
    def _simple_clustering(self, correlation_matrix: pd.DataFrame) -> Dict:
        """简化的聚类方法"""
        try:
            # 基于相关性阈值的简单聚类
            clusters = {}
            processed_assets = set()
            cluster_id = 1
            
            for asset in correlation_matrix.index:
                if asset in processed_assets:
                    continue
                
                # 找到与该资产高度相关的其他资产
                high_corr_assets = [asset]
                for other_asset in correlation_matrix.index:
                    if other_asset != asset and other_asset not in processed_assets:
                        corr_value = correlation_matrix.loc[asset, other_asset]
                        if abs(corr_value) > self.correlation_thresholds['high']:
                            high_corr_assets.append(other_asset)
                
                # 创建聚类
                clusters[f'cluster_{cluster_id}'] = {
                    'assets': high_corr_assets,
                    'size': len(high_corr_assets),
                    'avg_intra_correlation': self.calculate_average_correlation(
                        correlation_matrix.loc[high_corr_assets, high_corr_assets]
                    )
                }
                
                processed_assets.update(high_corr_assets)
                cluster_id += 1
            
            return {
                'cluster_labels': {asset: i for i, cluster in enumerate(clusters.values(), 1) for asset in cluster['assets']},
                'n_clusters': len(clusters),
                'cluster_summary': clusters
            }
            
        except Exception as e:
            print(f"简化聚类失败: {e}")
            return {}
    
    def _record_correlation_history(self, correlation_data: pd.DataFrame):
        """记录相关性历史"""
        try:
            if correlation_data.empty:
                return
            
            # 计算当前时间点的平均相关性
            avg_corr = self.calculate_average_correlation(correlation_data)
            
            # 记录历史数据
            self.correlation_history.append({
                'timestamp': pd.Timestamp.now(),
                'avg_correlation': avg_corr,
                'correlation_matrix': correlation_data.copy(),
                'correlation_regime': self._classify_correlation_regime(avg_corr)
            })
            
            # 保持最近100条记录
            if len(self.correlation_history) > 100:
                self.correlation_history = self.correlation_history[-100:]
                
        except Exception as e:
            print(f"记录相关性历史失败: {e}")
    
    def _classify_correlation_regime(self, avg_correlation: float) -> str:
        """分类相关性状态"""
        try:
            if avg_correlation < self.correlation_thresholds['low']:
                return 'low'
            elif avg_correlation < self.correlation_thresholds['medium']:
                return 'medium'
            else:
                return 'high'
                
        except Exception as e:
            print(f"分类相关性状态失败: {e}")
            return 'unknown'
    
    def get_correlation_summary(self) -> Dict:
        """获取相关性摘要"""
        try:
            if not self.correlation_history:
                return {'no_data': True}
            
            # 最近的相关性信息
            latest = self.correlation_history[-1]
            
            # 历史统计
            avg_correlations = [info['avg_correlation'] for info in self.correlation_history]
            regimes = [info['correlation_regime'] for info in self.correlation_history]
            
            # 统计各状态的频率
            regime_counts = {}
            for regime in regimes:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            summary = {
                'latest': {
                    'avg_correlation': latest['avg_correlation'],
                    'correlation_regime': latest['correlation_regime'],
                    'timestamp': latest['timestamp']
                },
                'statistics': {
                    'avg_correlation': np.mean(avg_correlations),
                    'max_correlation': np.max(avg_correlations),
                    'min_correlation': np.min(avg_correlations),
                    'correlation_std': np.std(avg_correlations)
                },
                'regime_distribution': regime_counts,
                'regime_changes': len(self.regime_change_alerts),
                'recent_regime_changes': self.regime_change_alerts[-5:] if len(self.regime_change_alerts) > 5 else self.regime_change_alerts
            }
            
            return summary
            
        except Exception as e:
            print(f"获取相关性摘要失败: {e}")
            return {'error': str(e)}
    
    def monitor_correlation_breakdown(self, correlation_matrix: pd.DataFrame,
                                    threshold: float = 0.8) -> Dict:
        """监控相关性崩溃"""
        try:
            if correlation_matrix.empty:
                return {}
            
            # 计算高相关性资产对
            high_corr_pairs = []
            for i in range(len(correlation_matrix.index)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > threshold:
                        high_corr_pairs.append({
                            'asset1': correlation_matrix.index[i],
                            'asset2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            # 评估相关性崩溃风险
            breakdown_risk = self._assess_correlation_breakdown_risk(correlation_matrix)
            
            return {
                'high_correlation_pairs': high_corr_pairs,
                'breakdown_risk': breakdown_risk,
                'risk_level': self._classify_breakdown_risk(breakdown_risk)
            }
            
        except Exception as e:
            print(f"监控相关性崩溃失败: {e}")
            return {}
    
    def _assess_correlation_breakdown_risk(self, correlation_matrix: pd.DataFrame) -> float:
        """评估相关性崩溃风险"""
        try:
            # 计算相关性分布指标
            corr_values = self._extract_correlation_values(correlation_matrix)
            
            if len(corr_values) == 0:
                return 0.0
            
            # 风险指标：
            # 1. 高相关性比例
            high_corr_ratio = np.sum(corr_values > 0.8) / len(corr_values)
            
            # 2. 相关性集中度
            corr_concentration = np.std(corr_values)
            
            # 3. 极端相关性
            extreme_corr = np.sum(corr_values > 0.95) / len(corr_values)
            
            # 综合风险评分
            risk_score = (high_corr_ratio * 0.4 + 
                         corr_concentration * 0.3 + 
                         extreme_corr * 0.3)
            
            return risk_score
            
        except Exception as e:
            print(f"评估相关性崩溃风险失败: {e}")
            return 0.0
    
    def _classify_breakdown_risk(self, risk_score: float) -> str:
        """分类崩溃风险"""
        try:
            if risk_score < 0.3:
                return 'low'
            elif risk_score < 0.6:
                return 'medium'
            else:
                return 'high'
                
        except Exception as e:
            print(f"分类崩溃风险失败: {e}")
            return 'unknown'
    
    def calculate_correlation_stability(self, correlation_matrices: List[pd.DataFrame]) -> Dict:
        """计算相关性稳定性"""
        try:
            if len(correlation_matrices) < 2:
                return {'stability': 0.0}
            
            # 计算连续期间的相关性变化
            stability_metrics = []
            
            for i in range(1, len(correlation_matrices)):
                current_corr = correlation_matrices[i]
                previous_corr = correlation_matrices[i-1]
                
                # 计算相关性变化
                corr_change = current_corr - previous_corr
                change_magnitude = np.mean(np.abs(self._extract_correlation_values(corr_change)))
                
                stability_metrics.append(1 - change_magnitude)
            
            # 平均稳定性
            avg_stability = np.mean(stability_metrics)
            
            return {
                'stability': avg_stability,
                'stability_trend': stability_metrics,
                'stability_classification': self._classify_stability(avg_stability)
            }
            
        except Exception as e:
            print(f"计算相关性稳定性失败: {e}")
            return {'stability': 0.0}
    
    def _classify_stability(self, stability_score: float) -> str:
        """分类稳定性"""
        try:
            if stability_score > 0.8:
                return 'high'
            elif stability_score > 0.6:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            print(f"分类稳定性失败: {e}")
            return 'unknown'
    
    def update_correlation_thresholds(self, new_thresholds: Dict):
        """更新相关性阈值"""
        try:
            self.correlation_thresholds.update(new_thresholds)
            
        except Exception as e:
            print(f"更新相关性阈值失败: {e}")
    
    def get_correlation_alerts(self) -> List[Dict]:
        """获取相关性告警"""
        try:
            alerts = []
            
            # 状态变化告警
            alerts.extend(self.regime_change_alerts)
            
            # 最近的相关性数据
            if self.correlation_history:
                latest = self.correlation_history[-1]
                
                # 极端相关性告警
                if latest['correlation_regime'] == 'high':
                    alerts.append({
                        'type': 'high_correlation_alert',
                        'timestamp': latest['timestamp'],
                        'avg_correlation': latest['avg_correlation'],
                        'severity': 'warning'
                    })
                elif latest['correlation_regime'] == 'low':
                    alerts.append({
                        'type': 'low_correlation_alert',
                        'timestamp': latest['timestamp'],
                        'avg_correlation': latest['avg_correlation'],
                        'severity': 'info'
                    })
            
            return alerts
            
        except Exception as e:
            print(f"获取相关性告警失败: {e}")
            return []


class LiquidityRiskManager:
    """流动性风险管理器"""
    
    def __init__(self):
        """初始化流动性风险管理器"""
        self.liquidity_thresholds = {
            'very_high': 0.8,
            'high': 0.6,
            'medium': 0.4,
            'low': 0.2
        }
        self.market_impact_model_params = {
            'temporary_impact_coeff': 0.5,
            'permanent_impact_coeff': 0.3,
            'cross_impact_factor': 0.1
        }
        self.liquidity_history = []
        self.impact_estimates = []
    
    def calculate_liquidity_score(self, volume_data: pd.DataFrame,
                                market_cap: pd.DataFrame) -> pd.DataFrame:
        """计算流动性评分
        
        Args:
            volume_data: 成交量数据
            market_cap: 市值数据
            
        Returns:
            流动性评分DataFrame
        """
        try:
            if volume_data.empty and market_cap.empty:
                return pd.DataFrame()
            
            # 计算多种流动性指标
            liquidity_scores = pd.DataFrame(index=volume_data.index)
            
            # 1. 成交量指标
            if not volume_data.empty:
                # 平均日成交量 (ADV)
                adv_20 = volume_data.rolling(window=20).mean()
                adv_60 = volume_data.rolling(window=60).mean()
                
                # 成交量稳定性
                volume_stability = 1 - (volume_data.rolling(window=20).std() / adv_20)
                volume_stability = volume_stability.fillna(0).clip(0, 1)
                
                # 相对成交量（当日成交量/平均成交量）
                relative_volume = volume_data / adv_20
                volume_consistency = 1 / (1 + relative_volume.rolling(window=20).std())
                volume_consistency = volume_consistency.fillna(0).clip(0, 1)
                
                # 成交量趋势
                volume_trend = volume_data.rolling(window=20).apply(
                    lambda x: np.corrcoef(range(len(x)), x)[0, 1] if len(x) > 1 else 0
                )
                volume_trend = volume_trend.fillna(0).clip(-1, 1)
                volume_trend_score = (volume_trend + 1) / 2  # 转换到0-1
                
                # 组合成交量评分
                volume_score = (
                    0.4 * self._normalize_score(adv_20) + 
                    0.3 * volume_stability + 
                    0.2 * volume_consistency + 
                    0.1 * volume_trend_score
                )
                
                liquidity_scores = liquidity_scores.join(volume_score, rsuffix='_volume')
            
            # 2. 市值指标
            if not market_cap.empty:
                # 市值稳定性
                market_cap_stability = 1 - (market_cap.rolling(window=20).std() / market_cap.rolling(window=20).mean())
                market_cap_stability = market_cap_stability.fillna(0).clip(0, 1)
                
                # 市值评分
                market_cap_score = (
                    0.7 * self._normalize_score(market_cap) + 
                    0.3 * market_cap_stability
                )
                
                liquidity_scores = liquidity_scores.join(market_cap_score, rsuffix='_market_cap')
            
            # 3. 换手率
            if not volume_data.empty and not market_cap.empty:
                # 计算换手率 (假设价格为1，简化计算)
                turnover_rate = volume_data / market_cap
                turnover_score = self._normalize_score(turnover_rate)
                
                liquidity_scores = liquidity_scores.join(turnover_score, rsuffix='_turnover')
            
            # 4. 综合流动性评分
            score_columns = [col for col in liquidity_scores.columns if not col.endswith('_volume') 
                           and not col.endswith('_market_cap') and not col.endswith('_turnover')]
            
            if len(score_columns) == 0:
                # 如果没有其他评分列，使用所有列
                score_columns = liquidity_scores.columns
            
            if len(score_columns) > 0:
                # 权重分配
                if not volume_data.empty and not market_cap.empty:
                    weights = [0.5, 0.3, 0.2]  # 成交量、市值、换手率
                elif not volume_data.empty:
                    weights = [1.0]  # 只有成交量
                elif not market_cap.empty:
                    weights = [1.0]  # 只有市值
                else:
                    weights = [1.0 / len(score_columns)] * len(score_columns)
                
                # 确保权重长度匹配
                weights = weights[:len(score_columns)]
                if sum(weights) > 0:
                    weights = [w / sum(weights) for w in weights]  # 标准化权重
                
                # 计算综合评分
                final_scores = pd.DataFrame(index=liquidity_scores.index)
                for asset in liquidity_scores.columns:
                    if asset not in ['_volume', '_market_cap', '_turnover']:
                        asset_scores = []
                        
                        # 收集该资产的各项评分
                        if not volume_data.empty and asset in volume_data.columns:
                            vol_scores = liquidity_scores[asset + '_volume'] if asset + '_volume' in liquidity_scores.columns else liquidity_scores[asset]
                            asset_scores.append(vol_scores)
                        
                        if not market_cap.empty and asset in market_cap.columns:
                            cap_scores = liquidity_scores[asset + '_market_cap'] if asset + '_market_cap' in liquidity_scores.columns else liquidity_scores[asset]
                            asset_scores.append(cap_scores)
                        
                        if not volume_data.empty and not market_cap.empty and asset in volume_data.columns and asset in market_cap.columns:
                            turnover_scores = liquidity_scores[asset + '_turnover'] if asset + '_turnover' in liquidity_scores.columns else liquidity_scores[asset]
                            asset_scores.append(turnover_scores)
                        
                        # 加权平均
                        if asset_scores:
                            asset_weights = weights[:len(asset_scores)]
                            if sum(asset_weights) > 0:
                                weighted_score = sum(w * score for w, score in zip(asset_weights, asset_scores)) / sum(asset_weights)
                                final_scores[asset] = weighted_score
                
                # 记录历史
                self._record_liquidity_history(final_scores)
                
                return final_scores
            
            return liquidity_scores
            
        except Exception as e:
            print(f"计算流动性评分失败: {e}")
            return pd.DataFrame()
    
    def estimate_market_impact(self, trade_size: pd.Series,
                             adv: pd.Series) -> pd.Series:
        """估计市场冲击
        
        Args:
            trade_size: 交易规模
            adv: 平均日成交量
            
        Returns:
            市场冲击估计
        """
        try:
            if trade_size.empty or adv.empty:
                return pd.Series()
            
            # 对齐数据
            common_assets = trade_size.index.intersection(adv.index)
            if len(common_assets) == 0:
                return pd.Series()
            
            aligned_trade_size = trade_size.reindex(common_assets, fill_value=0)
            aligned_adv = adv.reindex(common_assets, fill_value=1)  # 避免除零
            
            # 计算参与率
            participation_rate = aligned_trade_size.abs() / aligned_adv
            participation_rate = participation_rate.fillna(0).clip(0, 1)
            
            # 临时冲击 - 基于参与率的非线性关系
            temporary_impact = self.market_impact_model_params['temporary_impact_coeff'] * (
                participation_rate ** 0.6
            )
            
            # 永久冲击 - 更小但持续
            permanent_impact = self.market_impact_model_params['permanent_impact_coeff'] * (
                participation_rate ** 0.8
            )
            
            # 总冲击
            total_impact = temporary_impact + permanent_impact
            
            # 应用方向（买入为正冲击，卖出为负冲击）
            impact_direction = np.sign(aligned_trade_size)
            total_impact = total_impact * impact_direction
            
            # 记录冲击估计
            self._record_impact_estimates(aligned_trade_size, total_impact, participation_rate)
            
            return total_impact
            
        except Exception as e:
            print(f"估计市场冲击失败: {e}")
            return pd.Series()
    
    def calculate_participation_rate(self, trade_volume: pd.Series,
                                   market_volume: pd.Series) -> pd.Series:
        """计算参与率
        
        Args:
            trade_volume: 交易量
            market_volume: 市场总交易量
            
        Returns:
            参与率序列
        """
        try:
            if trade_volume.empty or market_volume.empty:
                return pd.Series()
            
            # 对齐数据
            common_index = trade_volume.index.intersection(market_volume.index)
            if len(common_index) == 0:
                return pd.Series()
            
            aligned_trade_volume = trade_volume.reindex(common_index, fill_value=0)
            aligned_market_volume = market_volume.reindex(common_index, fill_value=1)  # 避免除零
            
            # 计算参与率
            participation_rate = aligned_trade_volume.abs() / aligned_market_volume
            participation_rate = participation_rate.fillna(0).clip(0, 1)
            
            return participation_rate
            
        except Exception as e:
            print(f"计算参与率失败: {e}")
            return pd.Series()
    
    def liquidity_adjusted_weights(self, target_weights: pd.Series,
                                 liquidity_scores: pd.Series) -> pd.Series:
        """流动性调整权重
        
        Args:
            target_weights: 目标权重
            liquidity_scores: 流动性评分
            
        Returns:
            调整后的权重
        """
        try:
            if target_weights.empty:
                return target_weights
            
            if liquidity_scores.empty:
                print("警告: 没有流动性评分，返回原始权重")
                return target_weights
            
            # 对齐数据
            common_assets = target_weights.index.intersection(liquidity_scores.index)
            if len(common_assets) == 0:
                print("警告: 权重和流动性评分没有共同资产")
                return target_weights
            
            adjusted_weights = target_weights.copy()
            
            # 应用流动性调整
            for asset in common_assets:
                original_weight = target_weights[asset]
                liquidity_score = liquidity_scores[asset]
                
                # 流动性调整因子
                adjustment_factor = self._calculate_liquidity_adjustment_factor(
                    liquidity_score, abs(original_weight)
                )
                
                # 应用调整
                adjusted_weights[asset] = original_weight * adjustment_factor
            
            # 重新标准化权重
            adjusted_weights = self._normalize_weights(adjusted_weights)
            
            return adjusted_weights
            
        except Exception as e:
            print(f"流动性调整权重失败: {e}")
            return target_weights
    
    def _normalize_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """标准化评分到0-1范围"""
        try:
            if data.empty:
                return data
            
            # 使用分位数标准化，更稳健
            normalized = data.rank(pct=True)
            return normalized.fillna(0)
            
        except Exception as e:
            print(f"标准化评分失败: {e}")
            return data
    
    def _calculate_liquidity_adjustment_factor(self, liquidity_score: float, 
                                             position_size: float) -> float:
        """计算流动性调整因子"""
        try:
            # 基于流动性评分的调整
            if liquidity_score >= self.liquidity_thresholds['very_high']:
                base_factor = 1.0  # 不调整
            elif liquidity_score >= self.liquidity_thresholds['high']:
                base_factor = 0.95
            elif liquidity_score >= self.liquidity_thresholds['medium']:
                base_factor = 0.85
            elif liquidity_score >= self.liquidity_thresholds['low']:
                base_factor = 0.7
            else:
                base_factor = 0.5  # 大幅减少低流动性资产权重
            
            # 基于仓位大小的额外调整
            if position_size > 0.05:  # 大仓位
                size_penalty = 0.9
            elif position_size > 0.02:  # 中等仓位
                size_penalty = 0.95
            else:
                size_penalty = 1.0  # 小仓位不调整
            
            return base_factor * size_penalty
            
        except Exception as e:
            print(f"计算流动性调整因子失败: {e}")
            return 1.0
    
    def _normalize_weights(self, weights: pd.Series) -> pd.Series:
        """重新标准化权重"""
        try:
            if weights.sum() == 0:
                return weights
            
            # 保持权重和为1
            total_weight = weights.sum()
            if abs(total_weight) > 1e-10:
                normalized_weights = weights / total_weight
            else:
                normalized_weights = weights
            
            return normalized_weights
            
        except Exception as e:
            print(f"标准化权重失败: {e}")
            return weights
    
    def _record_liquidity_history(self, liquidity_scores: pd.DataFrame):
        """记录流动性历史"""
        try:
            if liquidity_scores.empty:
                return
            
            # 计算平均流动性评分
            avg_liquidity = liquidity_scores.mean().mean()
            
            self.liquidity_history.append({
                'timestamp': pd.Timestamp.now(),
                'avg_liquidity_score': avg_liquidity,
                'liquidity_distribution': {
                    'very_high': (liquidity_scores >= self.liquidity_thresholds['very_high']).sum().sum(),
                    'high': ((liquidity_scores >= self.liquidity_thresholds['high']) & 
                           (liquidity_scores < self.liquidity_thresholds['very_high'])).sum().sum(),
                    'medium': ((liquidity_scores >= self.liquidity_thresholds['medium']) & 
                             (liquidity_scores < self.liquidity_thresholds['high'])).sum().sum(),
                    'low': (liquidity_scores < self.liquidity_thresholds['medium']).sum().sum()
                }
            })
            
            # 保持最近100条记录
            if len(self.liquidity_history) > 100:
                self.liquidity_history = self.liquidity_history[-100:]
                
        except Exception as e:
            print(f"记录流动性历史失败: {e}")
    
    def _record_impact_estimates(self, trade_sizes: pd.Series, 
                               impacts: pd.Series, participation_rates: pd.Series):
        """记录冲击估计"""
        try:
            self.impact_estimates.append({
                'timestamp': pd.Timestamp.now(),
                'avg_impact': impacts.abs().mean(),
                'max_impact': impacts.abs().max(),
                'avg_participation_rate': participation_rates.mean(),
                'max_participation_rate': participation_rates.max(),
                'trade_count': len(trade_sizes[trade_sizes != 0])
            })
            
            # 保持最近100条记录
            if len(self.impact_estimates) > 100:
                self.impact_estimates = self.impact_estimates[-100:]
                
        except Exception as e:
            print(f"记录冲击估计失败: {e}")
    
    def get_liquidity_summary(self) -> Dict:
        """获取流动性摘要"""
        try:
            if not self.liquidity_history:
                return {'no_data': True}
            
            # 最近的流动性信息
            latest = self.liquidity_history[-1]
            
            # 历史统计
            avg_scores = [info['avg_liquidity_score'] for info in self.liquidity_history]
            
            summary = {
                'latest': latest,
                'statistics': {
                    'avg_liquidity_score': np.mean(avg_scores),
                    'max_liquidity_score': np.max(avg_scores),
                    'min_liquidity_score': np.min(avg_scores),
                    'liquidity_trend': self._calculate_liquidity_trend(avg_scores)
                },
                'impact_summary': self.get_impact_summary()
            }
            
            return summary
            
        except Exception as e:
            print(f"获取流动性摘要失败: {e}")
            return {'error': str(e)}
    
    def get_impact_summary(self) -> Dict:
        """获取冲击摘要"""
        try:
            if not self.impact_estimates:
                return {'no_impact_data': True}
            
            # 最近的冲击信息
            latest = self.impact_estimates[-1]
            
            # 历史统计
            avg_impacts = [info['avg_impact'] for info in self.impact_estimates]
            max_impacts = [info['max_impact'] for info in self.impact_estimates]
            participation_rates = [info['avg_participation_rate'] for info in self.impact_estimates]
            
            return {
                'latest': latest,
                'statistics': {
                    'avg_impact': np.mean(avg_impacts),
                    'max_impact': np.max(max_impacts),
                    'avg_participation_rate': np.mean(participation_rates),
                    'impact_volatility': np.std(avg_impacts)
                }
            }
            
        except Exception as e:
            print(f"获取冲击摘要失败: {e}")
            return {'error': str(e)}
    
    def _calculate_liquidity_trend(self, scores: List[float]) -> str:
        """计算流动性趋势"""
        try:
            if len(scores) < 5:
                return 'insufficient_data'
            
            # 计算最近5期的平均值与之前5期的比较
            recent_avg = np.mean(scores[-5:])
            previous_avg = np.mean(scores[-10:-5]) if len(scores) >= 10 else np.mean(scores[:-5])
            
            if recent_avg > previous_avg * 1.05:
                return 'improving'
            elif recent_avg < previous_avg * 0.95:
                return 'deteriorating'
            else:
                return 'stable'
                
        except Exception as e:
            print(f"计算流动性趋势失败: {e}")
            return 'unknown'
    
    def estimate_liquidation_time(self, position_sizes: pd.Series, 
                                 adv: pd.Series, 
                                 max_participation_rate: float = 0.2) -> pd.Series:
        """估计清仓时间"""
        try:
            if position_sizes.empty or adv.empty:
                return pd.Series()
            
            # 对齐数据
            common_assets = position_sizes.index.intersection(adv.index)
            aligned_positions = position_sizes.reindex(common_assets, fill_value=0)
            aligned_adv = adv.reindex(common_assets, fill_value=1)
            
            # 计算最大日交易量
            max_daily_volume = aligned_adv * max_participation_rate
            
            # 计算清仓天数
            liquidation_days = aligned_positions.abs() / max_daily_volume
            liquidation_days = liquidation_days.fillna(0)
            
            return liquidation_days
            
        except Exception as e:
            print(f"估计清仓时间失败: {e}")
            return pd.Series()
    
    def update_liquidity_thresholds(self, new_thresholds: Dict):
        """更新流动性阈值"""
        try:
            self.liquidity_thresholds.update(new_thresholds)
            
        except Exception as e:
            print(f"更新流动性阈值失败: {e}")
    
    def calculate_liquidity_risk_metrics(self, portfolio_weights: pd.Series,
                                       liquidity_scores: pd.Series) -> Dict:
        """计算流动性风险指标"""
        try:
            if portfolio_weights.empty or liquidity_scores.empty:
                return {}
            
            # 对齐数据
            common_assets = portfolio_weights.index.intersection(liquidity_scores.index)
            if len(common_assets) == 0:
                return {}
            
            aligned_weights = portfolio_weights.reindex(common_assets, fill_value=0)
            aligned_scores = liquidity_scores.reindex(common_assets, fill_value=0)
            
            # 计算加权流动性评分
            weighted_liquidity = (aligned_weights.abs() * aligned_scores).sum()
            
            # 流动性集中度风险
            low_liquidity_mask = aligned_scores < self.liquidity_thresholds['medium']
            low_liquidity_exposure = aligned_weights[low_liquidity_mask].abs().sum()
            
            # 流动性多样化
            liquidity_diversification = 1 - (aligned_weights.abs() * (1 - aligned_scores)).sum()
            
            return {
                'weighted_liquidity_score': weighted_liquidity,
                'low_liquidity_exposure': low_liquidity_exposure,
                'liquidity_diversification': liquidity_diversification,
                'liquidity_concentration_risk': low_liquidity_exposure / aligned_weights.abs().sum() if aligned_weights.abs().sum() > 0 else 0
            }
            
        except Exception as e:
            print(f"计算流动性风险指标失败: {e}")
            return {}


class RegimeDetector:
    """市场状态检测器"""
    
    def __init__(self):
        """初始化市场状态检测器"""
        self.volatility_states = {}
        self.trend_states = {}
        self.correlation_states = {}
        self.hmm_models = {}
        self.regime_history = {}
    
    def detect_volatility_regime(self, returns: pd.Series,
                               threshold: float = 0.02) -> pd.Series:
        """检测波动率状态
        
        Args:
            returns: 收益率序列
            threshold: 高低波动率阈值
            
        Returns:
            波动率状态序列 (0: 低波动率, 1: 中波动率, 2: 高波动率)
        """
        try:
            if len(returns) == 0:
                return pd.Series(dtype=int)
            
            # 计算滚动波动率
            rolling_vol = returns.rolling(window=20).std()
            
            # 动态阈值设置
            vol_mean = rolling_vol.mean()
            vol_std = rolling_vol.std()
            
            # 定义状态阈值
            low_threshold = vol_mean - vol_std * 0.5
            high_threshold = vol_mean + vol_std * 0.5
            
            # 分类状态
            regime = pd.Series(1, index=returns.index)  # 默认中波动率
            regime[rolling_vol < low_threshold] = 0    # 低波动率
            regime[rolling_vol > high_threshold] = 2   # 高波动率
            
            # 状态平滑（避免频繁切换）
            regime = self._smooth_regime(regime)
            
            # 记录状态历史
            self.volatility_states[returns.index[-1]] = regime.iloc[-1]
            
            return regime
            
        except Exception as e:
            print(f"波动率状态检测失败: {e}")
            return pd.Series([1] * len(returns), index=returns.index)
    
    def detect_trend_regime(self, prices: pd.Series,
                          window: int = 60) -> pd.Series:
        """检测趋势状态
        
        Args:
            prices: 价格序列
            window: 趋势计算窗口
            
        Returns:
            趋势状态序列 (0: 下降, 1: 横盘, 2: 上升)
        """
        try:
            if len(prices) < window:
                return pd.Series([1] * len(prices), index=prices.index)
            
            # 计算多时间框架移动平均
            short_ma = prices.rolling(window=window//3).mean()
            medium_ma = prices.rolling(window=window//2).mean()
            long_ma = prices.rolling(window=window).mean()
            
            # 计算趋势强度
            trend_strength = (short_ma - long_ma) / long_ma
            
            # 计算价格动量
            momentum = prices.pct_change(window//4)
            
            # 组合指标
            trend_indicator = trend_strength * 0.7 + momentum * 0.3
            
            # 动态阈值
            trend_mean = trend_indicator.rolling(window=window*2).mean()
            trend_std = trend_indicator.rolling(window=window*2).std()
            
            # 定义状态阈值
            down_threshold = trend_mean - trend_std * 0.5
            up_threshold = trend_mean + trend_std * 0.5
            
            # 分类状态
            regime = pd.Series(1, index=prices.index)  # 默认横盘
            regime[trend_indicator < down_threshold] = 0  # 下降
            regime[trend_indicator > up_threshold] = 2   # 上升
            
            # 状态平滑
            regime = self._smooth_regime(regime)
            
            # 记录状态历史
            self.trend_states[prices.index[-1]] = regime.iloc[-1]
            
            return regime
            
        except Exception as e:
            print(f"趋势状态检测失败: {e}")
            return pd.Series([1] * len(prices), index=prices.index)
    
    def detect_correlation_regime(self, correlation_data: pd.DataFrame) -> pd.Series:
        """检测相关性状态
        
        Args:
            correlation_data: 相关性矩阵时间序列
            
        Returns:
            相关性状态序列 (0: 低相关, 1: 中相关, 2: 高相关)
        """
        try:
            if len(correlation_data) == 0:
                return pd.Series(dtype=int)
            
            # 计算平均相关系数
            avg_corr = correlation_data.apply(
                lambda x: x.values[np.triu_indices_from(x.values, k=1)].mean(), 
                axis=1
            )
            
            # 计算相关性变化
            corr_change = avg_corr.pct_change()
            
            # 动态阈值
            corr_mean = avg_corr.rolling(window=60).mean()
            corr_std = avg_corr.rolling(window=60).std()
            
            # 定义状态阈值
            low_threshold = corr_mean - corr_std * 0.5
            high_threshold = corr_mean + corr_std * 0.5
            
            # 分类状态
            regime = pd.Series(1, index=correlation_data.index)  # 默认中相关
            regime[avg_corr < low_threshold] = 0  # 低相关
            regime[avg_corr > high_threshold] = 2  # 高相关
            
            # 状态平滑
            regime = self._smooth_regime(regime)
            
            # 记录状态历史
            self.correlation_states[correlation_data.index[-1]] = regime.iloc[-1]
            
            return regime
            
        except Exception as e:
            print(f"相关性状态检测失败: {e}")
            return pd.Series([1] * len(correlation_data), index=correlation_data.index)
    
    def hmm_regime_detection(self, returns: pd.Series,
                           n_regimes: int = 2) -> pd.Series:
        """隐马尔可夫状态检测
        
        Args:
            returns: 收益率序列
            n_regimes: 状态数量
            
        Returns:
            HMM状态序列
        """
        try:
            if len(returns) < 50:  # 数据量不足
                return pd.Series([0] * len(returns), index=returns.index)
            
            # 准备数据
            returns_clean = returns.dropna()
            if len(returns_clean) < 50:
                return pd.Series([0] * len(returns), index=returns.index)
            
            # 尝试使用sklearn的GaussianMixture作为HMM的简化版本
            try:
                from sklearn.mixture import GaussianMixture
                
                # 准备特征矩阵
                features = self._prepare_hmm_features(returns_clean)
                
                # 拟合高斯混合模型
                gmm = GaussianMixture(n_components=n_regimes, random_state=42)
                gmm.fit(features)
                
                # 预测状态
                states = gmm.predict(features)
                
                # 创建完整的状态序列
                full_states = pd.Series(index=returns.index, dtype=int)
                full_states.loc[returns_clean.index] = states
                
                # 前向填充缺失值
                full_states = full_states.fillna(method='ffill').fillna(0)
                
                # 状态平滑
                full_states = self._smooth_regime(full_states)
                
                # 记录模型
                self.hmm_models[returns.index[-1]] = gmm
                
                return full_states
                
            except ImportError:
                # 如果没有sklearn，使用简化的基于波动率的状态检测
                return self._simple_hmm_approximation(returns, n_regimes)
                
        except Exception as e:
            print(f"HMM状态检测失败: {e}")
            return self._simple_hmm_approximation(returns, n_regimes)
    
    def _prepare_hmm_features(self, returns: pd.Series) -> np.ndarray:
        """准备HMM特征矩阵"""
        try:
            # 计算多种特征
            features = []
            
            # 收益率
            features.append(returns.values)
            
            # 绝对收益率
            features.append(np.abs(returns.values))
            
            # 滚动波动率
            rolling_vol = returns.rolling(window=5).std().fillna(0)
            features.append(rolling_vol.values)
            
            # 收益率的平方（波动率代理）
            features.append(returns.values ** 2)
            
            # 滞后收益率
            lag_returns = returns.shift(1).fillna(0)
            features.append(lag_returns.values)
            
            # 组合特征矩阵
            feature_matrix = np.column_stack(features)
            
            # 标准化
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            feature_matrix = scaler.fit_transform(feature_matrix)
            
            return feature_matrix
            
        except Exception as e:
            print(f"准备HMM特征失败: {e}")
            # 简化特征
            return returns.values.reshape(-1, 1)
    
    def _simple_hmm_approximation(self, returns: pd.Series, n_regimes: int) -> pd.Series:
        """简化的HMM近似（基于波动率聚类）"""
        try:
            # 计算滚动波动率
            rolling_vol = returns.rolling(window=20).std()
            
            # 基于波动率分位数划分状态
            if n_regimes == 2:
                # 两个状态：低波动率和高波动率
                threshold = rolling_vol.quantile(0.5)
                states = (rolling_vol > threshold).astype(int)
            elif n_regimes == 3:
                # 三个状态：低、中、高波动率
                low_threshold = rolling_vol.quantile(0.33)
                high_threshold = rolling_vol.quantile(0.67)
                states = pd.Series(1, index=returns.index)
                states[rolling_vol < low_threshold] = 0
                states[rolling_vol > high_threshold] = 2
            else:
                # 多个状态：基于分位数
                quantiles = np.linspace(0, 1, n_regimes + 1)[1:-1]
                thresholds = rolling_vol.quantile(quantiles)
                states = pd.Series(0, index=returns.index)
                for i, threshold in enumerate(thresholds):
                    states[rolling_vol > threshold] = i + 1
            
            # 状态平滑
            states = self._smooth_regime(states)
            
            return states
            
        except Exception as e:
            print(f"简化HMM近似失败: {e}")
            return pd.Series([0] * len(returns), index=returns.index)
    
    def _smooth_regime(self, regime: pd.Series, window: int = 5) -> pd.Series:
        """平滑状态序列以避免频繁切换"""
        try:
            if len(regime) < window:
                return regime
            
            smoothed = regime.copy()
            
            # 使用滑动窗口的众数进行平滑
            for i in range(window, len(regime)):
                window_data = regime.iloc[i-window:i]
                mode_value = window_data.mode()
                if len(mode_value) > 0:
                    smoothed.iloc[i] = mode_value[0]
            
            return smoothed
            
        except Exception as e:
            print(f"状态平滑失败: {e}")
            return regime
    
    def get_regime_transitions(self, regime: pd.Series) -> pd.DataFrame:
        """获取状态转换信息"""
        try:
            if len(regime) < 2:
                return pd.DataFrame()
            
            # 计算状态转换
            transitions = []
            for i in range(1, len(regime)):
                if regime.iloc[i] != regime.iloc[i-1]:
                    transitions.append({
                        'timestamp': regime.index[i],
                        'from_state': regime.iloc[i-1],
                        'to_state': regime.iloc[i],
                        'duration': i - np.where(regime.iloc[:i] != regime.iloc[i-1])[0][-1] if np.where(regime.iloc[:i] != regime.iloc[i-1])[0].size > 0 else i
                    })
            
            return pd.DataFrame(transitions)
            
        except Exception as e:
            print(f"获取状态转换信息失败: {e}")
            return pd.DataFrame()
    
    def calculate_regime_statistics(self, returns: pd.Series, regime: pd.Series) -> Dict:
        """计算各状态的统计信息"""
        try:
            if len(returns) != len(regime):
                return {}
            
            stats = {}
            unique_states = regime.unique()
            
            for state in unique_states:
                mask = regime == state
                state_returns = returns[mask]
                
                if len(state_returns) > 0:
                    stats[f'regime_{state}'] = {
                        'count': len(state_returns),
                        'mean_return': state_returns.mean(),
                        'volatility': state_returns.std(),
                        'skewness': state_returns.skew(),
                        'kurtosis': state_returns.kurtosis(),
                        'min_return': state_returns.min(),
                        'max_return': state_returns.max(),
                        'duration_avg': len(state_returns) / len(regime[regime == state].groupby((regime != state).cumsum()).size()),
                        'frequency': len(state_returns) / len(returns)
                    }
            
            return stats
            
        except Exception as e:
            print(f"计算状态统计信息失败: {e}")
            return {}
    
    def predict_regime_probability(self, returns: pd.Series, 
                                 current_features: Dict = None) -> Dict:
        """预测下一期的状态概率"""
        try:
            if len(returns) < 50:
                return {}
            
            # 使用最近的模型
            recent_model = None
            if self.hmm_models:
                recent_key = max(self.hmm_models.keys())
                recent_model = self.hmm_models[recent_key]
            
            if recent_model is None:
                return {}
            
            # 准备当前特征
            if current_features is None:
                current_features = self._prepare_hmm_features(returns.tail(1))
            
            # 预测概率
            probabilities = recent_model.predict_proba(current_features)
            
            return {
                f'regime_{i}_prob': prob[0] 
                for i, prob in enumerate(probabilities.T)
            }
            
        except Exception as e:
            print(f"预测状态概率失败: {e}")
            return {}
    
    def detect_regime_change(self, returns: pd.Series, 
                           lookback_window: int = 20,
                           significance_level: float = 0.05) -> Dict:
        """检测状态变化"""
        try:
            if len(returns) < lookback_window * 2:
                return {'regime_change': False}
            
            # 将数据分为两个窗口
            recent_data = returns.tail(lookback_window)
            previous_data = returns.iloc[-lookback_window*2:-lookback_window]
            
            # 统计检验
            from scipy import stats
            
            # 均值检验
            mean_test = stats.ttest_ind(recent_data, previous_data)
            
            # 方差检验
            var_test = stats.levene(recent_data, previous_data)
            
            # 分布检验
            ks_test = stats.ks_2samp(recent_data, previous_data)
            
            # 判断是否有显著变化
            regime_change = (
                mean_test.pvalue < significance_level or
                var_test.pvalue < significance_level or
                ks_test.pvalue < significance_level
            )
            
            return {
                'regime_change': regime_change,
                'mean_test_pvalue': mean_test.pvalue,
                'variance_test_pvalue': var_test.pvalue,
                'distribution_test_pvalue': ks_test.pvalue,
                'change_confidence': 1 - min(mean_test.pvalue, var_test.pvalue, ks_test.pvalue)
            }
            
        except Exception as e:
            print(f"检测状态变化失败: {e}")
            return {'regime_change': False, 'error': str(e)}


class StressTestManager:
    """压力测试管理器"""
    
    def __init__(self):
        """初始化压力测试管理器"""
        self.stress_scenarios = {}
        self.test_results = []
        self.sensitivity_analysis = {}
        
        # 预定义的压力测试情景
        self.default_scenarios = {
            'market_crash': {'equity_shock': -0.30, 'bond_shock': -0.10, 'commodity_shock': -0.20},
            'interest_rate_shock': {'bond_shock': -0.15, 'equity_shock': -0.05, 'currency_shock': 0.10},
            'inflation_shock': {'commodity_shock': 0.25, 'bond_shock': -0.20, 'equity_shock': -0.10},
            'liquidity_crisis': {'spread_widening': 0.50, 'equity_shock': -0.25, 'bond_shock': -0.15},
            'geopolitical_risk': {'equity_shock': -0.20, 'commodity_shock': 0.15, 'currency_shock': 0.20}
        }
    
    def scenario_stress_test(self, portfolio_weights: pd.Series,
                           stress_scenarios: Dict[str, pd.DataFrame]) -> Dict:
        """情景压力测试
        
        Args:
            portfolio_weights: 组合权重
            stress_scenarios: 压力情景字典，每个情景包含资产收益率的冲击
            
        Returns:
            压力测试结果
        """
        try:
            if portfolio_weights.empty:
                return {}
            
            # 如果没有提供压力情景，使用默认情景
            if not stress_scenarios:
                stress_scenarios = self._generate_default_scenarios(portfolio_weights)
            
            results = {}
            
            for scenario_name, scenario_data in stress_scenarios.items():
                try:
                    # 确保scenario_data是DataFrame
                    if isinstance(scenario_data, dict):
                        scenario_df = pd.DataFrame(scenario_data, index=[0])
                    else:
                        scenario_df = scenario_data
                    
                    # 计算组合在该情景下的收益
                    scenario_results = self._calculate_scenario_impact(
                        portfolio_weights, scenario_df, scenario_name
                    )
                    
                    results[scenario_name] = scenario_results
                    
                except Exception as e:
                    print(f"情景 {scenario_name} 测试失败: {e}")
                    results[scenario_name] = {'error': str(e)}
            
            # 计算综合统计
            results['summary'] = self._calculate_scenario_summary(results)
            
            # 记录测试结果
            self._record_stress_test_results(results, 'scenario')
            
            return results
            
        except Exception as e:
            print(f"情景压力测试失败: {e}")
            return {'error': str(e)}
    
    def factor_shock_test(self, portfolio_weights: pd.Series,
                         factor_exposures: pd.DataFrame,
                         factor_shocks: pd.Series) -> Dict:
        """因子冲击测试
        
        Args:
            portfolio_weights: 组合权重
            factor_exposures: 因子暴露矩阵
            factor_shocks: 因子冲击向量
            
        Returns:
            因子冲击测试结果
        """
        try:
            if portfolio_weights.empty or factor_exposures.empty or factor_shocks.empty:
                return {}
            
            # 对齐数据
            common_assets = portfolio_weights.index.intersection(factor_exposures.index)
            if len(common_assets) == 0:
                return {'error': 'No common assets between weights and exposures'}
            
            aligned_weights = portfolio_weights.reindex(common_assets, fill_value=0)
            aligned_exposures = factor_exposures.reindex(common_assets, fill_value=0)
            
            # 计算因子冲击对组合的影响
            results = {}
            
            # 对每个因子分别进行冲击测试
            for factor in factor_shocks.index:
                if factor in aligned_exposures.columns:
                    factor_impact = self._calculate_factor_impact(
                        aligned_weights, aligned_exposures, factor, factor_shocks[factor]
                    )
                    results[factor] = factor_impact
            
            # 计算所有因子同时冲击的影响
            total_impact = self._calculate_total_factor_impact(
                aligned_weights, aligned_exposures, factor_shocks
            )
            results['total_impact'] = total_impact
            
            # 计算边际影响
            marginal_impacts = self._calculate_marginal_factor_impacts(
                aligned_weights, aligned_exposures, factor_shocks
            )
            results['marginal_impacts'] = marginal_impacts
            
            # 记录测试结果
            self._record_stress_test_results(results, 'factor_shock')
            
            return results
            
        except Exception as e:
            print(f"因子冲击测试失败: {e}")
            return {'error': str(e)}
    
    def correlation_breakdown_test(self, portfolio_weights: pd.Series,
                                 normal_corr: pd.DataFrame,
                                 stress_corr: pd.DataFrame) -> Dict:
        """相关性崩溃测试
        
        Args:
            portfolio_weights: 组合权重
            normal_corr: 正常相关性矩阵
            stress_corr: 压力相关性矩阵
            
        Returns:
            相关性崩溃测试结果
        """
        try:
            if portfolio_weights.empty or normal_corr.empty or stress_corr.empty:
                return {}
            
            # 对齐数据
            common_assets = portfolio_weights.index.intersection(normal_corr.index)
            if len(common_assets) == 0:
                return {'error': 'No common assets'}
            
            aligned_weights = portfolio_weights.reindex(common_assets, fill_value=0)
            aligned_normal_corr = normal_corr.reindex(common_assets, columns=common_assets)
            aligned_stress_corr = stress_corr.reindex(common_assets, columns=common_assets)
            
            # 假设资产波动率
            asset_volatilities = pd.Series(0.20, index=common_assets)  # 假设20%年化波动率
            
            # 计算正常情况下的组合风险
            normal_risk = self._calculate_portfolio_risk(aligned_weights, aligned_normal_corr, asset_volatilities)
            
            # 计算压力情况下的组合风险
            stress_risk = self._calculate_portfolio_risk(aligned_weights, aligned_stress_corr, asset_volatilities)
            
            # 计算风险变化
            risk_change = stress_risk - normal_risk
            risk_change_pct = (stress_risk / normal_risk - 1) * 100 if normal_risk > 0 else 0
            
            # 分析相关性变化
            correlation_analysis = self._analyze_correlation_changes(aligned_normal_corr, aligned_stress_corr)
            
            # 计算分散化效应的变化
            diversification_analysis = self._analyze_diversification_breakdown(
                aligned_weights, aligned_normal_corr, aligned_stress_corr, asset_volatilities
            )
            
            results = {
                'normal_portfolio_risk': normal_risk,
                'stress_portfolio_risk': stress_risk,
                'risk_change': risk_change,
                'risk_change_percentage': risk_change_pct,
                'correlation_analysis': correlation_analysis,
                'diversification_analysis': diversification_analysis
            }
            
            # 记录测试结果
            self._record_stress_test_results(results, 'correlation_breakdown')
            
            return results
            
        except Exception as e:
            print(f"相关性崩溃测试失败: {e}")
            return {'error': str(e)}
    
    def tail_risk_scenarios(self, returns: pd.DataFrame,
                          confidence_levels: List[float]) -> Dict:
        """尾部风险情景
        
        Args:
            returns: 历史收益率数据
            confidence_levels: 置信度水平列表
            
        Returns:
            尾部风险情景结果
        """
        try:
            if returns.empty:
                return {}
            
            results = {}
            
            # 计算各种尾部风险指标
            for confidence in confidence_levels:
                tail_analysis = self._calculate_tail_risk_metrics(returns, confidence)
                results[f'confidence_{confidence}'] = tail_analysis
            
            # 极端情景分析
            extreme_scenarios = self._generate_extreme_scenarios(returns)
            results['extreme_scenarios'] = extreme_scenarios
            
            # 尾部依赖分析
            tail_dependence = self._analyze_tail_dependence(returns)
            results['tail_dependence'] = tail_dependence
            
            # 记录测试结果
            self._record_stress_test_results(results, 'tail_risk')
            
            return results
            
        except Exception as e:
            print(f"尾部风险情景分析失败: {e}")
            return {'error': str(e)}
    
    def _generate_default_scenarios(self, portfolio_weights: pd.Series) -> Dict[str, pd.DataFrame]:
        """生成默认压力情景"""
        try:
            scenarios = {}
            
            for scenario_name, scenario_params in self.default_scenarios.items():
                # 为每个资产生成压力冲击
                scenario_returns = {}
                
                for asset in portfolio_weights.index:
                    # 根据资产类型分配冲击
                    if any(keyword in asset.lower() for keyword in ['equity', 'stock', 'etf']):
                        shock = scenario_params.get('equity_shock', -0.10)
                    elif any(keyword in asset.lower() for keyword in ['bond', 'treasury', 'debt']):
                        shock = scenario_params.get('bond_shock', -0.05)
                    elif any(keyword in asset.lower() for keyword in ['commodity', 'gold', 'oil']):
                        shock = scenario_params.get('commodity_shock', 0.0)
                    else:
                        shock = scenario_params.get('equity_shock', -0.05)  # 默认
                    
                    scenario_returns[asset] = shock
                
                scenarios[scenario_name] = pd.DataFrame(scenario_returns, index=[0])
            
            return scenarios
            
        except Exception as e:
            print(f"生成默认情景失败: {e}")
            return {}
    
    def _calculate_scenario_impact(self, portfolio_weights: pd.Series, 
                                 scenario_data: pd.DataFrame, scenario_name: str) -> Dict:
        """计算情景影响"""
        try:
            # 获取共同资产
            common_assets = portfolio_weights.index.intersection(scenario_data.columns)
            if len(common_assets) == 0:
                return {'error': 'No common assets'}
            
            aligned_weights = portfolio_weights.reindex(common_assets, fill_value=0)
            aligned_scenario = scenario_data.reindex(columns=common_assets, fill_value=0)
            
            # 计算组合在该情景下的收益
            scenario_returns = (aligned_weights * aligned_scenario.iloc[0]).sum()
            
            # 计算各资产贡献
            asset_contributions = aligned_weights * aligned_scenario.iloc[0]
            
            # 计算风险指标
            worst_asset = asset_contributions.min()
            best_asset = asset_contributions.max()
            
            return {
                'scenario_name': scenario_name,
                'portfolio_return': scenario_returns,
                'worst_asset_contribution': worst_asset,
                'best_asset_contribution': best_asset,
                'total_loss': min(0, scenario_returns),
                'asset_contributions': asset_contributions.to_dict()
            }
            
        except Exception as e:
            print(f"计算情景影响失败: {e}")
            return {'error': str(e)}
    
    def _calculate_scenario_summary(self, results: Dict) -> Dict:
        """计算情景汇总统计"""
        try:
            scenario_returns = []
            scenario_losses = []
            
            for scenario_name, scenario_result in results.items():
                if isinstance(scenario_result, dict) and 'portfolio_return' in scenario_result:
                    scenario_returns.append(scenario_result['portfolio_return'])
                    scenario_losses.append(scenario_result['total_loss'])
            
            if not scenario_returns:
                return {'error': 'No valid scenario results'}
            
            return {
                'worst_case_return': min(scenario_returns),
                'best_case_return': max(scenario_returns),
                'average_return': np.mean(scenario_returns),
                'total_max_loss': min(scenario_losses),
                'scenarios_with_losses': sum(1 for loss in scenario_losses if loss < 0),
                'total_scenarios': len(scenario_returns)
            }
            
        except Exception as e:
            print(f"计算情景汇总失败: {e}")
            return {'error': str(e)}
    
    def _calculate_factor_impact(self, weights: pd.Series, exposures: pd.DataFrame, 
                               factor: str, shock: float) -> Dict:
        """计算单个因子冲击影响"""
        try:
            if factor not in exposures.columns:
                return {'error': f'Factor {factor} not found in exposures'}
            
            # 计算因子暴露
            factor_exposure = exposures[factor]
            
            # 计算组合对该因子的总暴露
            portfolio_exposure = (weights * factor_exposure).sum()
            
            # 计算因子冲击对组合的影响
            factor_impact = portfolio_exposure * shock
            
            # 计算各资产的贡献
            asset_contributions = weights * factor_exposure * shock
            
            return {
                'factor': factor,
                'shock': shock,
                'portfolio_exposure': portfolio_exposure,
                'factor_impact': factor_impact,
                'asset_contributions': asset_contributions.to_dict()
            }
            
        except Exception as e:
            print(f"计算因子影响失败: {e}")
            return {'error': str(e)}
    
    def _calculate_total_factor_impact(self, weights: pd.Series, exposures: pd.DataFrame, 
                                     factor_shocks: pd.Series) -> Dict:
        """计算所有因子同时冲击的影响"""
        try:
            total_impact = 0
            factor_impacts = {}
            
            # 对每个因子计算影响
            for factor in factor_shocks.index:
                if factor in exposures.columns:
                    factor_exposure = exposures[factor]
                    portfolio_exposure = (weights * factor_exposure).sum()
                    factor_impact = portfolio_exposure * factor_shocks[factor]
                    
                    total_impact += factor_impact
                    factor_impacts[factor] = factor_impact
            
            return {
                'total_impact': total_impact,
                'individual_factors': factor_impacts
            }
            
        except Exception as e:
            print(f"计算总因子影响失败: {e}")
            return {'error': str(e)}
    
    def _calculate_marginal_factor_impacts(self, weights: pd.Series, exposures: pd.DataFrame, 
                                         factor_shocks: pd.Series) -> Dict:
        """计算边际因子影响"""
        try:
            marginal_impacts = {}
            
            # 基准总影响
            base_impact = self._calculate_total_factor_impact(weights, exposures, factor_shocks)['total_impact']
            
            # 逐个排除因子，计算边际影响
            for factor in factor_shocks.index:
                if factor in exposures.columns:
                    # 排除该因子的冲击
                    reduced_shocks = factor_shocks.copy()
                    reduced_shocks[factor] = 0
                    
                    # 计算排除该因子后的影响
                    reduced_impact = self._calculate_total_factor_impact(weights, exposures, reduced_shocks)['total_impact']
                    
                    # 边际影响 = 总影响 - 排除该因子后的影响
                    marginal_impact = base_impact - reduced_impact
                    marginal_impacts[factor] = marginal_impact
            
            return marginal_impacts
            
        except Exception as e:
            print(f"计算边际因子影响失败: {e}")
            return {'error': str(e)}
    
    def _calculate_portfolio_risk(self, weights: pd.Series, correlation_matrix: pd.DataFrame, 
                                volatilities: pd.Series) -> float:
        """计算组合风险"""
        try:
            # 构建协方差矩阵
            covariance_matrix = correlation_matrix.multiply(volatilities, axis=0).multiply(volatilities, axis=1)
            
            # 计算组合方差
            portfolio_variance = weights.T @ covariance_matrix @ weights
            
            # 返回组合标准差
            return np.sqrt(portfolio_variance)
            
        except Exception as e:
            print(f"计算组合风险失败: {e}")
            return 0.0
    
    def _analyze_correlation_changes(self, normal_corr: pd.DataFrame, stress_corr: pd.DataFrame) -> Dict:
        """分析相关性变化"""
        try:
            # 计算相关性变化
            corr_changes = stress_corr - normal_corr
            
            # 平均相关性变化
            avg_corr_change = corr_changes.values[np.triu_indices_from(corr_changes.values, k=1)].mean()
            
            # 最大相关性增加
            max_corr_increase = corr_changes.max().max()
            
            # 最大相关性减少
            max_corr_decrease = corr_changes.min().min()
            
            return {
                'average_correlation_change': avg_corr_change,
                'max_correlation_increase': max_corr_increase,
                'max_correlation_decrease': max_corr_decrease,
                'correlation_change_matrix': corr_changes.to_dict()
            }
            
        except Exception as e:
            print(f"分析相关性变化失败: {e}")
            return {'error': str(e)}
    
    def _analyze_diversification_breakdown(self, weights: pd.Series, normal_corr: pd.DataFrame, 
                                         stress_corr: pd.DataFrame, volatilities: pd.Series) -> Dict:
        """分析分散化效应的变化"""
        try:
            # 计算正常情况下的分散化比率
            normal_portfolio_risk = self._calculate_portfolio_risk(weights, normal_corr, volatilities)
            weighted_avg_risk = (weights.abs() * volatilities).sum()
            normal_diversification_ratio = normal_portfolio_risk / weighted_avg_risk if weighted_avg_risk > 0 else 1
            
            # 计算压力情况下的分散化比率
            stress_portfolio_risk = self._calculate_portfolio_risk(weights, stress_corr, volatilities)
            stress_diversification_ratio = stress_portfolio_risk / weighted_avg_risk if weighted_avg_risk > 0 else 1
            
            # 分散化效应的变化
            diversification_change = stress_diversification_ratio - normal_diversification_ratio
            
            return {
                'normal_diversification_ratio': normal_diversification_ratio,
                'stress_diversification_ratio': stress_diversification_ratio,
                'diversification_change': diversification_change,
                'diversification_loss': max(0, diversification_change)
            }
            
        except Exception as e:
            print(f"分析分散化变化失败: {e}")
            return {'error': str(e)}
    
    def _calculate_tail_risk_metrics(self, returns: pd.DataFrame, confidence: float) -> Dict:
        """计算尾部风险指标"""
        try:
            # 计算各资产的VaR和CVaR
            asset_vars = {}
            asset_cvars = {}
            
            for asset in returns.columns:
                asset_returns = returns[asset].dropna()
                if len(asset_returns) > 0:
                    var = np.percentile(asset_returns, (1 - confidence) * 100)
                    cvar = asset_returns[asset_returns <= var].mean()
                    
                    asset_vars[asset] = var
                    asset_cvars[asset] = cvar
            
            return {
                'confidence_level': confidence,
                'asset_vars': asset_vars,
                'asset_cvars': asset_cvars
            }
            
        except Exception as e:
            print(f"计算尾部风险指标失败: {e}")
            return {'error': str(e)}
    
    def _generate_extreme_scenarios(self, returns: pd.DataFrame) -> Dict:
        """生成极端情景"""
        try:
            # 历史最差日收益
            worst_day = returns.sum(axis=1).idxmin()
            worst_day_returns = returns.loc[worst_day].to_dict()
            
            # 历史最佳日收益
            best_day = returns.sum(axis=1).idxmax()
            best_day_returns = returns.loc[best_day].to_dict()
            
            # 计算极端分位数
            extreme_low = returns.quantile(0.01).to_dict()
            extreme_high = returns.quantile(0.99).to_dict()
            
            return {
                'historical_worst_day': {
                    'date': worst_day,
                    'returns': worst_day_returns
                },
                'historical_best_day': {
                    'date': best_day,
                    'returns': best_day_returns
                },
                'extreme_low_1pct': extreme_low,
                'extreme_high_99pct': extreme_high
            }
            
        except Exception as e:
            print(f"生成极端情景失败: {e}")
            return {'error': str(e)}
    
    def _analyze_tail_dependence(self, returns: pd.DataFrame) -> Dict:
        """分析尾部依赖性"""
        try:
            # 计算资产间的尾部相关性
            tail_correlations = {}
            
            # 选择尾部事件 (5%分位数)
            threshold = returns.quantile(0.05)
            
            for i, asset1 in enumerate(returns.columns):
                for j, asset2 in enumerate(returns.columns):
                    if i < j:  # 避免重复计算
                        # 同时发生尾部事件的概率
                        joint_tail_events = ((returns[asset1] <= threshold[asset1]) & 
                                           (returns[asset2] <= threshold[asset2])).sum()
                        
                        total_observations = len(returns)
                        tail_dependence = joint_tail_events / total_observations
                        
                        tail_correlations[f'{asset1}_{asset2}'] = tail_dependence
            
            return {
                'tail_threshold': threshold.to_dict(),
                'tail_dependencies': tail_correlations
            }
            
        except Exception as e:
            print(f"分析尾部依赖失败: {e}")
            return {'error': str(e)}
    
    def _record_stress_test_results(self, results: Dict, test_type: str):
        """记录压力测试结果"""
        try:
            self.test_results.append({
                'timestamp': pd.Timestamp.now(),
                'test_type': test_type,
                'results': results
            })
            
            # 保持最近50条记录
            if len(self.test_results) > 50:
                self.test_results = self.test_results[-50:]
                
        except Exception as e:
            print(f"记录压力测试结果失败: {e}")
    
    def get_stress_test_summary(self) -> Dict:
        """获取压力测试摘要"""
        try:
            if not self.test_results:
                return {'no_data': True}
            
            # 按测试类型分组
            test_types = {}
            for result in self.test_results:
                test_type = result['test_type']
                if test_type not in test_types:
                    test_types[test_type] = []
                test_types[test_type].append(result)
            
            return {
                'total_tests': len(self.test_results),
                'test_types': {test_type: len(tests) for test_type, tests in test_types.items()},
                'latest_test': self.test_results[-1] if self.test_results else None,
                'test_history': self.test_results[-10:]  # 最近10次测试
            }
            
        except Exception as e:
            print(f"获取压力测试摘要失败: {e}")
            return {'error': str(e)}
    
    def run_comprehensive_stress_test(self, portfolio_weights: pd.Series, 
                                    market_data: Dict) -> Dict:
        """运行全面压力测试"""
        try:
            comprehensive_results = {}
            
            # 情景压力测试
            scenario_results = self.scenario_stress_test(portfolio_weights, {})
            comprehensive_results['scenario_tests'] = scenario_results
            
            # 相关性崩溃测试
            if 'correlation_matrix' in market_data:
                # 生成压力相关性矩阵（所有相关性增加到0.8）
                normal_corr = market_data['correlation_matrix']
                stress_corr = normal_corr.copy()
                stress_corr[stress_corr > 0] = 0.8
                
                corr_results = self.correlation_breakdown_test(portfolio_weights, normal_corr, stress_corr)
                comprehensive_results['correlation_breakdown'] = corr_results
            
            # 尾部风险分析
            if 'returns_data' in market_data:
                tail_results = self.tail_risk_scenarios(market_data['returns_data'], [0.95, 0.99])
                comprehensive_results['tail_risk'] = tail_results
            
            # 生成综合评估
            comprehensive_results['overall_assessment'] = self._generate_overall_assessment(comprehensive_results)
            
            return comprehensive_results
            
        except Exception as e:
            print(f"运行全面压力测试失败: {e}")
            return {'error': str(e)}
    
    def _generate_overall_assessment(self, results: Dict) -> Dict:
        """生成总体评估"""
        try:
            risk_score = 0
            risk_factors = []
            
            # 评估情景测试结果
            if 'scenario_tests' in results and 'summary' in results['scenario_tests']:
                summary = results['scenario_tests']['summary']
                if 'worst_case_return' in summary and summary['worst_case_return'] < -0.20:
                    risk_score += 30
                    risk_factors.append('severe_scenario_losses')
                elif 'worst_case_return' in summary and summary['worst_case_return'] < -0.10:
                    risk_score += 15
                    risk_factors.append('moderate_scenario_losses')
            
            # 评估相关性崩溃
            if 'correlation_breakdown' in results and 'risk_change_percentage' in results['correlation_breakdown']:
                risk_change = results['correlation_breakdown']['risk_change_percentage']
                if risk_change > 50:
                    risk_score += 25
                    risk_factors.append('high_correlation_risk')
                elif risk_change > 20:
                    risk_score += 10
                    risk_factors.append('moderate_correlation_risk')
            
            # 评估尾部风险
            if 'tail_risk' in results:
                risk_score += 10  # 基础尾部风险
                risk_factors.append('tail_risk_exposure')
            
            # 风险等级分类
            if risk_score >= 50:
                risk_level = 'high'
            elif risk_score >= 25:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'recommendations': self._generate_risk_recommendations(risk_level, risk_factors)
            }
            
        except Exception as e:
            print(f"生成总体评估失败: {e}")
            return {'error': str(e)}
    
    def _generate_risk_recommendations(self, risk_level: str, risk_factors: List[str]) -> List[str]:
        """生成风险建议"""
        recommendations = []
        
        if risk_level == 'high':
            recommendations.append('建议立即降低组合风险暴露')
            recommendations.append('考虑增加对冲工具')
            recommendations.append('重新评估风险限额')
        
        if 'severe_scenario_losses' in risk_factors:
            recommendations.append('优化资产配置以减少极端情景损失')
        
        if 'high_correlation_risk' in risk_factors:
            recommendations.append('增加资产多样化以降低相关性风险')
        
        if 'tail_risk_exposure' in risk_factors:
            recommendations.append('考虑尾部风险对冲策略')
        
        if not recommendations:
            recommendations.append('当前风险水平可接受，继续监控')
        
        return recommendations


class RiskManager:
    """风险管理主类"""
    
    def __init__(self, config: Dict = None):
        """初始化风险管理器"""
        self.config = config or {}
        self.historical_model = HistoricalRiskModel()
        self.parametric_model = ParametricRiskModel()
        self.monte_carlo_model = MonteCarloRiskModel()
        self.factor_model = FactorRiskModel()
        self.drawdown_manager = DrawdownManager()
        self.volatility_manager = VolatilityManager()
        self.position_limit_manager = PositionLimitManager(self.config)
        self.leverage_manager = LeverageManager()
        self.correlation_monitor = CorrelationMonitor()
        self.liquidity_manager = LiquidityRiskManager()
        self.regime_detector = RegimeDetector()
        self.stress_test_manager = StressTestManager()
    
    def initialize(self, config: Dict):
        """初始化配置"""
        self.config.update(config)
        
        # 更新各组件配置
        if hasattr(self.position_limit_manager, 'config'):
            self.position_limit_manager.config.update(config)
        if hasattr(self.leverage_manager, 'max_leverage'):
            self.leverage_manager.max_leverage = config.get('max_leverage', 1.0)
        if hasattr(self.volatility_manager, 'target_volatility'):
            self.volatility_manager.target_volatility = config.get('target_volatility', 0.15)
    
    def pre_trade_risk_check(self, target_weights: pd.Series,
                           current_weights: pd.Series,
                           market_data: Dict) -> Dict[str, bool]:
        """交易前风险检查"""
        checks = {}
        
        # 仓位限制检查
        checks['position_limits'] = self.position_limit_manager.check_single_position_limit(target_weights)
        
        # 杠杆检查
        target_leverage = self.leverage_manager.calculate_leverage(target_weights)
        checks['leverage_check'] = target_leverage <= self.leverage_manager.max_leverage
        
        # 集中度检查
        checks['concentration_check'] = self.position_limit_manager.check_concentration_limits(target_weights)
        
        # 换手率检查
        turnover = (target_weights - current_weights).abs().sum()
        checks['turnover_check'] = turnover <= self.config.get('max_turnover', 0.5)
        
        # 流动性检查
        if 'volume_data' in market_data:
            liquidity_scores = self.liquidity_manager.calculate_liquidity_score(
                market_data['volume_data'], market_data.get('market_cap', pd.DataFrame())
            )
            checks['liquidity_check'] = True  # 简化实现
        else:
            checks['liquidity_check'] = True
        
        return checks
    
    def post_trade_risk_monitoring(self, portfolio_weights: pd.Series,
                                 returns_data: pd.DataFrame) -> Dict:
        """交易后风险监控"""
        monitoring_results = {}
        
        # 计算组合收益
        portfolio_returns = (returns_data * portfolio_weights).sum(axis=1)
        
        # 风险指标
        monitoring_results['portfolio_volatility'] = self.historical_model.calculate_portfolio_risk(
            portfolio_weights, returns_data
        )
        monitoring_results['portfolio_var'] = self.calculate_portfolio_var(
            portfolio_weights, returns_data
        )
        
        # 回撤监控
        portfolio_value = (1 + portfolio_returns).cumprod()
        monitoring_results['current_drawdown'] = self.drawdown_manager.calculate_drawdown(portfolio_value).iloc[-1]
        monitoring_results['max_drawdown'] = self.drawdown_manager.calculate_max_drawdown(portfolio_value)
        
        # 杠杆监控
        monitoring_results['current_leverage'] = self.leverage_manager.calculate_leverage(portfolio_weights)
        
        # 相关性监控
        if len(returns_data) > 60:
            corr_matrix = self.correlation_monitor.calculate_rolling_correlation(returns_data)
            monitoring_results['avg_correlation'] = self.correlation_monitor.calculate_average_correlation(
                corr_matrix.iloc[-1].unstack().reset_index().pivot(index='level_0', columns='level_1', values=0)
            )
        
        return monitoring_results
    
    def calculate_portfolio_var(self, weights: pd.Series,
                              returns_data: pd.DataFrame,
                              confidence: float = 0.95,
                              method: str = "historical") -> float:
        """计算组合VaR"""
        # 计算组合收益
        portfolio_returns = (returns_data * weights).sum(axis=1)
        
        if method == "historical":
            return self.historical_model.calculate_var(portfolio_returns, confidence)
        elif method == "parametric":
            return self.parametric_model.calculate_var(portfolio_returns, confidence)
        elif method == "monte_carlo":
            return self.monte_carlo_model.calculate_var(portfolio_returns, confidence)
        else:
            return self.historical_model.calculate_var(portfolio_returns, confidence)
    
    def risk_attribution(self, portfolio_weights: pd.Series,
                        factor_exposures: pd.DataFrame,
                        factor_covariance: pd.DataFrame) -> Dict:
        """风险归因"""
        # 使用因子模型进行风险归因
        specific_risk = pd.Series(0.05, index=portfolio_weights.index)  # 简化的特异风险
        
        attribution = self.factor_model.calculate_portfolio_risk_decomposition(
            portfolio_weights, factor_covariance, factor_exposures, specific_risk
        )
        
        return attribution
    
    def dynamic_risk_budgeting(self, target_risk: float,
                             asset_volatilities: pd.Series,
                             correlations: pd.DataFrame) -> pd.Series:
        """动态风险预算"""
        # 使用风险平价方法
        def risk_budget_objective(weights):
            portfolio_vol = np.sqrt(weights.T @ correlations @ weights)
            marginal_contrib = correlations @ weights / portfolio_vol
            contrib = weights * marginal_contrib
            return np.sum((contrib - target_risk / len(weights)) ** 2)
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
        ]
        bounds = [(0, 1) for _ in range(len(asset_volatilities))]
        
        # 初始权重
        initial_weights = np.ones(len(asset_volatilities)) / len(asset_volatilities)
        
        try:
            from scipy.optimize import minimize
            result = minimize(risk_budget_objective, initial_weights, 
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                return pd.Series(result.x, index=asset_volatilities.index)
            else:
                return pd.Series(initial_weights, index=asset_volatilities.index)
        except:
            return pd.Series(initial_weights, index=asset_volatilities.index)
    
    def real_time_risk_monitoring(self, current_positions: pd.Series,
                                market_data: Dict) -> Dict:
        """实时风险监控"""
        alerts = {}
        
        # 杠杆监控
        current_leverage = self.leverage_manager.calculate_leverage(current_positions)
        if current_leverage > self.leverage_manager.max_leverage:
            alerts['leverage_alert'] = {
                'type': 'leverage_breach',
                'current': current_leverage,
                'limit': self.leverage_manager.max_leverage
            }
        
        # 集中度监控
        max_position = current_positions.abs().max()
        if max_position > self.config.get('max_position', 0.05):
            alerts['concentration_alert'] = {
                'type': 'concentration_breach',
                'current': max_position,
                'limit': self.config.get('max_position', 0.05)
            }
        
        # 流动性监控
        if 'volume_data' in market_data:
            low_liquidity_positions = current_positions[current_positions.abs() > 0.01]
            alerts['liquidity_alert'] = {
                'type': 'liquidity_check',
                'positions': low_liquidity_positions.to_dict()
            }
        
        return alerts
    
    def generate_risk_report(self, portfolio_data: Dict,
                           save_path: str = None) -> Dict:
        """生成风险报告"""
        report = {
            'timestamp': pd.Timestamp.now(),
            'portfolio_summary': {},
            'risk_metrics': {},
            'risk_attribution': {},
            'alerts': {}
        }
        
        # 组合概览
        if 'weights' in portfolio_data:
            weights = portfolio_data['weights']
            report['portfolio_summary']['num_positions'] = len(weights[weights.abs() > 0.001])
            report['portfolio_summary']['gross_exposure'] = weights.abs().sum()
            report['portfolio_summary']['net_exposure'] = weights.sum()
            report['portfolio_summary']['max_position'] = weights.abs().max()
        
        # 风险指标
        if 'returns' in portfolio_data:
            returns = portfolio_data['returns']
            report['risk_metrics']['volatility'] = returns.std() * np.sqrt(252)
            report['risk_metrics']['var_95'] = self.historical_model.calculate_var(returns, 0.95)
            report['risk_metrics']['max_drawdown'] = self.drawdown_manager.calculate_max_drawdown(
                (1 + returns).cumprod()
            )
        
        # 保存报告
        if save_path:
            import json
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def calculate_concentration_risk(self, weights: pd.Series) -> Dict:
        """计算集中度风险"""
        try:
            # 计算各种集中度指标
            concentration_metrics = {}
            
            # 赫芬达尔指数
            concentration_metrics['herfindahl_index'] = (weights ** 2).sum()
            
            # 前n大持仓占比
            abs_weights = weights.abs().sort_values(ascending=False)
            concentration_metrics['top_5_concentration'] = abs_weights.head(5).sum()
            concentration_metrics['top_10_concentration'] = abs_weights.head(10).sum()
            
            # 有效持仓数量
            concentration_metrics['effective_positions'] = 1 / (weights ** 2).sum()
            
            # 最大持仓占比
            concentration_metrics['max_position'] = abs_weights.max()
            
            return concentration_metrics
        except Exception as e:
            return {'error': f"计算集中度风险失败: {e}"}
    
    def analyze_correlation_risk(self, returns: pd.DataFrame) -> Dict:
        """分析相关性风险"""
        try:
            correlation_analysis = {}
            
            # 计算相关性矩阵
            corr_matrix = returns.corr()
            correlation_analysis['correlation_matrix'] = corr_matrix
            
            # 平均相关性
            correlation_analysis['average_correlation'] = self.correlation_monitor.calculate_average_correlation(corr_matrix)
            
            # 相关性聚类
            correlation_analysis['clustering'] = self.correlation_monitor.correlation_clustering(corr_matrix)
            
            # 相关性稳定性
            if len(returns) > 120:  # 需要足够的数据进行滚动计算
                rolling_corr = self.correlation_monitor.calculate_rolling_correlation(returns, window=60)
                correlation_analysis['stability'] = self.correlation_monitor.calculate_correlation_stability([rolling_corr.iloc[i:i+60] for i in range(0, len(rolling_corr)-60, 20)])
            
            return correlation_analysis
        except Exception as e:
            return {'error': f"分析相关性风险失败: {e}"}
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> Dict:
        """计算条件风险价值"""
        try:
            cvar_results = {}
            
            # 历史CVaR
            var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
            tail_losses = returns[returns <= var_threshold]
            
            if len(tail_losses) > 0:
                cvar_results['historical_cvar'] = -tail_losses.mean()
            else:
                cvar_results['historical_cvar'] = -var_threshold
            
            # 参数化CVaR（假设正态分布）
            mean_return = returns.mean()
            std_return = returns.std()
            var_param = -(mean_return + std_return * stats.norm.ppf(1 - confidence_level))
            
            # CVaR计算
            cvar_param = -(mean_return - std_return * stats.norm.pdf(stats.norm.ppf(1 - confidence_level)) / (1 - confidence_level))
            cvar_results['parametric_cvar'] = cvar_param
            
            cvar_results['confidence_level'] = confidence_level
            cvar_results['tail_observations'] = len(tail_losses)
            
            return cvar_results
        except Exception as e:
            return {'error': f"计算CVaR失败: {e}"}
    
    def adjust_portfolio_risk(self, weights: pd.Series, target_risk: float = 0.15, returns_data: pd.DataFrame = None) -> Dict:
        """调整组合风险"""
        try:
            adjusted_results = {}
            
            if returns_data is not None:
                # 计算当前风险
                current_risk = self.historical_model.calculate_portfolio_risk(weights, returns_data)
                
                # 风险调整因子
                risk_adjustment_factor = target_risk / current_risk if current_risk > 0 else 1.0
                
                # 调整权重
                adjusted_weights = weights * risk_adjustment_factor
                
                # 归一化权重
                if adjusted_weights.sum() != 0:
                    adjusted_weights = adjusted_weights / adjusted_weights.sum()
                
                adjusted_results['original_weights'] = weights
                adjusted_results['adjusted_weights'] = adjusted_weights
                adjusted_results['original_risk'] = current_risk
                adjusted_results['target_risk'] = target_risk
                adjusted_results['adjustment_factor'] = risk_adjustment_factor
            else:
                # 如果没有收益数据，使用风险平价方法
                n_assets = len(weights)
                equal_risk_weights = pd.Series(1.0 / n_assets, index=weights.index)
                
                adjusted_results['original_weights'] = weights
                adjusted_results['adjusted_weights'] = equal_risk_weights
                adjusted_results['method'] = 'equal_risk_weighting'
            
            return adjusted_results
        except Exception as e:
            return {'error': f"调整组合风险失败: {e}"}
    
    def assess_liquidity_risk(self, weights: pd.Series, liquidity_metrics: Dict) -> Dict:
        """评估流动性风险"""
        try:
            liquidity_assessment = {}
            
            # 使用流动性管理器进行评估
            if 'volume_data' in liquidity_metrics:
                volume_data = liquidity_metrics['volume_data']
                market_cap = liquidity_metrics.get('market_cap', pd.DataFrame())
                
                # 计算流动性得分
                liquidity_scores = self.liquidity_manager.calculate_liquidity_score(volume_data, market_cap)
                
                # 流动性风险指标
                liquidity_assessment['liquidity_scores'] = liquidity_scores
                liquidity_assessment['liquidity_risk_metrics'] = self.liquidity_manager.calculate_liquidity_risk_metrics(weights, liquidity_scores.iloc[-1] if not liquidity_scores.empty else pd.Series())
                
                # 估计市场冲击
                if 'trade_sizes' in liquidity_metrics:
                    trade_sizes = liquidity_metrics['trade_sizes']
                    adv = volume_data.mean() if not volume_data.empty else pd.Series()
                    market_impact = self.liquidity_manager.estimate_market_impact(trade_sizes, adv)
                    liquidity_assessment['market_impact'] = market_impact
            else:
                # 简化的流动性风险评估
                liquidity_assessment['liquidity_risk'] = 'low'  # 默认低风险
                liquidity_assessment['method'] = 'simplified_assessment'
            
            return liquidity_assessment
        except Exception as e:
            return {'error': f"评估流动性风险失败: {e}"}
    
    def calculate_maximum_drawdown(self, returns: pd.Series) -> Dict:
        """计算最大回撤"""
        try:
            # 计算累积收益
            cumulative_returns = (1 + returns).cumprod()
            
            # 使用回撤管理器计算最大回撤
            max_drawdown = self.drawdown_manager.calculate_max_drawdown(cumulative_returns)
            drawdown_series = self.drawdown_manager.calculate_drawdown(cumulative_returns)
            
            # 回撤统计
            drawdown_stats = {
                'max_drawdown': max_drawdown,
                'current_drawdown': drawdown_series.iloc[-1] if len(drawdown_series) > 0 else 0,
                'avg_drawdown': drawdown_series.mean(),
                'drawdown_volatility': drawdown_series.std(),
                'underwater_periods': len(drawdown_series[drawdown_series < -0.01])  # 回撤超过1%的期间
            }
            
            return drawdown_stats
        except Exception as e:
            return {'error': f"计算最大回撤失败: {e}"}
    
    def monte_carlo_simulation(self, weights: pd.Series, returns_data: pd.DataFrame, n_simulations: int = 1000) -> Dict:
        """蒙特卡洛模拟"""
        try:
            # 使用蒙特卡洛风险模型
            simulated_returns = self.monte_carlo_model.simulate_portfolio_returns(weights, returns_data, time_horizon=252)
            
            # 统计分析
            simulation_results = {
                'simulated_returns': simulated_returns,
                'mean_return': np.mean(simulated_returns),
                'std_return': np.std(simulated_returns),
                'var_95': np.percentile(simulated_returns, 5),
                'var_99': np.percentile(simulated_returns, 1),
                'best_case': np.percentile(simulated_returns, 95),
                'worst_case': np.percentile(simulated_returns, 5),
                'probability_of_loss': np.sum(simulated_returns < 0) / len(simulated_returns)
            }
            
            return simulation_results
        except Exception as e:
            return {'error': f"蒙特卡洛模拟失败: {e}"}
    
    def calculate_portfolio_risk(self, weights: pd.Series, cov_matrix: pd.DataFrame) -> Dict:
        """计算组合风险"""
        try:
            # 对齐权重和协方差矩阵
            common_assets = weights.index.intersection(cov_matrix.index)
            aligned_weights = weights.reindex(common_assets, fill_value=0)
            aligned_cov = cov_matrix.reindex(common_assets).reindex(common_assets, axis=1)
            
            # 计算组合方差和风险
            portfolio_variance = aligned_weights.T @ aligned_cov @ aligned_weights
            portfolio_risk = np.sqrt(portfolio_variance)
            
            # 计算各资产对组合风险的贡献
            marginal_contrib = aligned_cov @ aligned_weights / portfolio_risk if portfolio_risk > 0 else aligned_weights * 0
            risk_contrib = aligned_weights * marginal_contrib
            
            risk_results = {
                'portfolio_variance': portfolio_variance,
                'portfolio_risk': portfolio_risk,
                'annualized_risk': portfolio_risk * np.sqrt(252),
                'risk_contributions': risk_contrib,
                'marginal_contributions': marginal_contrib,
                'percentage_contributions': risk_contrib / risk_contrib.sum() if risk_contrib.sum() != 0 else risk_contrib
            }
            
            return risk_results
        except Exception as e:
            return {'error': f"计算组合风险失败: {e}"}
    
    def detect_market_regime(self, returns: pd.Series) -> Dict:
        """检测市场状态"""
        try:
            regime_detection = {}
            
            # 波动率状态检测
            vol_regime = self.regime_detector.detect_volatility_regime(returns)
            regime_detection['volatility_regime'] = vol_regime
            
            # 趋势状态检测（需要价格数据）
            cumulative_prices = (1 + returns).cumprod()
            trend_regime = self.regime_detector.detect_trend_regime(cumulative_prices)
            regime_detection['trend_regime'] = trend_regime
            
            # HMM状态检测
            hmm_regime = self.regime_detector.hmm_regime_detection(returns)
            regime_detection['hmm_regime'] = hmm_regime
            
            # 状态统计
            regime_stats = self.regime_detector.calculate_regime_statistics(returns, hmm_regime)
            regime_detection['regime_statistics'] = regime_stats
            
            # 状态变化检测
            regime_change = self.regime_detector.detect_regime_change(returns)
            regime_detection['regime_change_detection'] = regime_change
            
            return regime_detection
        except Exception as e:
            return {'error': f"检测市场状态失败: {e}"}
    
    def calculate_risk_attribution(self, weights: pd.Series, factor_exposures: pd.DataFrame, factor_covariance: pd.DataFrame) -> Dict:
        """计算风险归因"""
        try:
            # 使用因子模型进行风险归因
            specific_risk = pd.Series(0.05, index=weights.index)  # 简化的特异风险
            
            attribution_results = self.factor_model.calculate_portfolio_risk_decomposition(
                weights, factor_covariance, factor_exposures, specific_risk
            )
            
            return attribution_results
        except Exception as e:
            return {'error': f"计算风险归因失败: {e}"}
    
    def calculate_risk_budget(self, weights: pd.Series, target_risk_budget: Dict) -> Dict:
        """计算风险预算"""
        try:
            risk_budget_results = {}
            
            # 计算当前风险预算分配
            total_risk = weights.abs().sum()
            current_risk_budget = (weights.abs() / total_risk).to_dict() if total_risk > 0 else {}
            
            # 与目标风险预算比较
            budget_deviation = {}
            for asset in weights.index:
                current_allocation = current_risk_budget.get(asset, 0)
                target_allocation = target_risk_budget.get(asset, 0)
                budget_deviation[asset] = current_allocation - target_allocation
            
            risk_budget_results['current_budget'] = current_risk_budget
            risk_budget_results['target_budget'] = target_risk_budget
            risk_budget_results['budget_deviation'] = budget_deviation
            risk_budget_results['total_deviation'] = sum(abs(v) for v in budget_deviation.values())
            
            return risk_budget_results
        except Exception as e:
            return {'error': f"计算风险预算失败: {e}"}
    
    def run_stress_test(self, weights: pd.Series, stress_scenarios: Dict) -> Dict:
        """运行压力测试"""
        try:
            # 使用压力测试管理器
            stress_results = self.stress_test_manager.scenario_stress_test(weights, stress_scenarios)
            
            # 运行综合压力测试
            if 'market_data' in stress_scenarios:
                comprehensive_results = self.stress_test_manager.run_comprehensive_stress_test(
                    weights, stress_scenarios['market_data']
                )
                stress_results['comprehensive_test'] = comprehensive_results
            
            return stress_results
        except Exception as e:
            return {'error': f"运行压力测试失败: {e}"}
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> Dict:
        """计算风险价值"""
        try:
            var_results = {}
            
            # 历史VaR
            var_results['historical_var'] = self.historical_model.calculate_var(returns, confidence_level)
            
            # 参数化VaR
            var_results['parametric_var'] = self.parametric_model.calculate_var(returns, confidence_level)
            
            # 蒙特卡洛VaR
            var_results['monte_carlo_var'] = self.monte_carlo_model.calculate_var(returns, confidence_level)
            
            # VaR统计
            var_results['confidence_level'] = confidence_level
            var_results['time_horizon'] = '1 day'
            var_results['sample_size'] = len(returns)
            
            return var_results
        except Exception as e:
            return {'error': f"计算VaR失败: {e}"} 