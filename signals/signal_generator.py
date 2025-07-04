#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号生成模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from abc import ABC, abstractmethod
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from utils.logger import Logger


class BaseSignalGenerator(ABC):
    """信号生成器基类"""
    
    def __init__(self, name: str):
        """初始化信号生成器"""
        self.name = name
        self.logger = Logger().get_logger(f"signal_{name}")
    
    @abstractmethod
    def generate_signals(self, factors: pd.DataFrame, 
                        additional_data: Dict = None) -> pd.DataFrame:
        """生成信号"""
        pass
    
    def preprocess_factors(self, factors: pd.DataFrame) -> pd.DataFrame:
        """因子预处理"""
        try:
            processed_factors = factors.copy()
            
            # 处理缺失值
            processed_factors = processed_factors.ffill().fillna(0)
            
            # 去除极值
            for col in processed_factors.columns:
                if processed_factors[col].dtype in ['float64', 'int64']:
                    # 使用3倍标准差去除极值
                    mean_val = processed_factors[col].mean()
                    std_val = processed_factors[col].std()
                    if std_val > 0:
                        processed_factors[col] = processed_factors[col].clip(
                            lower=mean_val - 3 * std_val,
                            upper=mean_val + 3 * std_val
                        )
            
            # 标准化（只对数值列）
            numeric_columns = processed_factors.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                scaler = StandardScaler()
                processed_factors[numeric_columns] = scaler.fit_transform(processed_factors[numeric_columns])
            
            return processed_factors
            
        except Exception as e:
            self.logger.error(f"因子预处理失败: {e}")
            return factors
    
    def validate_signals(self, signals: pd.DataFrame) -> bool:
        """验证信号有效性"""
        try:
            # 检查是否有空值
            if signals.isnull().any().any():
                self.logger.warning("信号中包含空值")
                return False
            
            # 检查信号范围
            if (signals < -1).any().any() or (signals > 1).any().any():
                self.logger.warning("信号超出[-1, 1]范围")
                return False
            
            # 检查是否有非数值
            if not signals.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all():
                self.logger.warning("信号包含非数值类型")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"信号验证失败: {e}")
            return False


class ThresholdSignalGenerator(BaseSignalGenerator):
    """阈值信号生成器"""
    
    def __init__(self, thresholds: Dict[str, float] = None):
        """初始化阈值信号生成器"""
        super().__init__("threshold")
        self.thresholds = thresholds or {}
        self.dynamic_thresholds = {}
    
    def generate_signals(self, factors: pd.DataFrame,
                        additional_data: Dict = None) -> pd.DataFrame:
        """基于阈值生成信号"""
        try:
            # 预处理因子
            processed_factors = self.preprocess_factors(factors)
            
            # 初始化信号DataFrame
            signals = pd.DataFrame(0.0, index=factors.index, columns=factors.columns)
            
            for factor_name in processed_factors.columns:
                factor_data = processed_factors[factor_name]
                
                if factor_name in self.thresholds:
                    # 使用预设阈值
                    threshold = self.thresholds[factor_name]
                    signals[factor_name] = np.where(factor_data > threshold, 1.0,
                                                  np.where(factor_data < -threshold, -1.0, 0.0))
                elif factor_name in self.dynamic_thresholds:
                    # 使用动态阈值
                    upper_threshold = self.dynamic_thresholds[factor_name]['upper']
                    lower_threshold = self.dynamic_thresholds[factor_name]['lower']
                    signals[factor_name] = np.where(factor_data > upper_threshold, 1.0,
                                                  np.where(factor_data < lower_threshold, -1.0, 0.0))
                else:
                    # 使用默认阈值（1倍标准差）
                    threshold = 1.0
                    signals[factor_name] = np.where(factor_data > threshold, 1.0,
                                                  np.where(factor_data < -threshold, -1.0, 0.0))
            
            # 验证信号
            if not self.validate_signals(signals):
                self.logger.warning("生成的信号未通过验证")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"生成阈值信号失败: {e}")
            return pd.DataFrame()
    
    def set_dynamic_thresholds(self, factors: pd.DataFrame,
                              method: str = "quantile",
                              quantiles: Tuple[float, float] = (0.3, 0.7)):
        """设置动态阈值"""
        try:
            self.dynamic_thresholds = {}
            
            for factor_name in factors.columns:
                factor_data = factors[factor_name].dropna()
                
                if method == "quantile":
                    lower_threshold = factor_data.quantile(quantiles[0])
                    upper_threshold = factor_data.quantile(quantiles[1])
                elif method == "std":
                    mean_val = factor_data.mean()
                    std_val = factor_data.std()
                    lower_threshold = mean_val - std_val
                    upper_threshold = mean_val + std_val
                else:
                    raise ValueError(f"不支持的阈值方法: {method}")
                
                self.dynamic_thresholds[factor_name] = {
                    'lower': lower_threshold,
                    'upper': upper_threshold
                }
            
            self.logger.info(f"动态阈值设置完成，使用方法: {method}")
            
        except Exception as e:
            self.logger.error(f"设置动态阈值失败: {e}")


class RankingSignalGenerator(BaseSignalGenerator):
    """排序信号生成器"""
    
    def __init__(self, top_pct: float = 0.2, bottom_pct: float = 0.2):
        """初始化排序信号生成器"""
        super().__init__("ranking")
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
    
    def generate_signals(self, factors: pd.DataFrame,
                        additional_data: Dict = None) -> pd.DataFrame:
        """基于排序生成信号"""
        try:
            # 预处理因子
            processed_factors = self.preprocess_factors(factors)
            
            # 使用横截面排序
            signals = self.cross_sectional_ranking(processed_factors)
            
            # 验证信号
            if not self.validate_signals(signals):
                self.logger.warning("生成的信号未通过验证")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"生成排序信号失败: {e}")
            return pd.DataFrame()
    
    def cross_sectional_ranking(self, factors: pd.DataFrame) -> pd.DataFrame:
        """横截面排序"""
        try:
            signals = pd.DataFrame(0.0, index=factors.index, columns=factors.columns)
            
            for timestamp in factors.index:
                for factor_name in factors.columns:
                    factor_values = factors.loc[timestamp, :].dropna()
                    
                    if len(factor_values) > 0:
                        # 计算排序
                        ranks = factor_values.rank(method='average', ascending=False)
                        percentiles = ranks / len(ranks)
                        
                        # 生成信号
                        for symbol in factor_values.index:
                            if symbol in factors.columns:
                                pct = percentiles[symbol]
                                if pct <= self.top_pct:
                                    signals.loc[timestamp, symbol] = 1.0
                                elif pct >= (1 - self.bottom_pct):
                                    signals.loc[timestamp, symbol] = -1.0
                                else:
                                    signals.loc[timestamp, symbol] = 0.0
            
            return signals
            
        except Exception as e:
            self.logger.error(f"横截面排序失败: {e}")
            return pd.DataFrame()
    
    def time_series_ranking(self, factors: pd.DataFrame,
                          lookback: int = 20) -> pd.DataFrame:
        """时间序列排序"""
        try:
            signals = pd.DataFrame(0.0, index=factors.index, columns=factors.columns)
            
            for factor_name in factors.columns:
                factor_data = factors[factor_name]
                
                # 计算滚动排序
                rolling_ranks = factor_data.rolling(window=lookback).rank(method='average')
                rolling_percentiles = rolling_ranks / lookback
                
                # 生成信号
                signals[factor_name] = np.where(rolling_percentiles <= self.top_pct, 1.0,
                                              np.where(rolling_percentiles >= (1 - self.bottom_pct), -1.0, 0.0))
            
            return signals
            
        except Exception as e:
            self.logger.error(f"时间序列排序失败: {e}")
            return pd.DataFrame()


class MLSignalGenerator(BaseSignalGenerator):
    """机器学习信号生成器"""
    
    def __init__(self, model_type: str = "classification",
                 prediction_horizon: int = 1):
        """初始化ML信号生成器"""
        super().__init__("ml")
        self.model_type = model_type
        self.prediction_horizon = prediction_horizon
        self.models = {}
        self.feature_scaler = StandardScaler()
        self.is_fitted = False
    
    def generate_signals(self, factors: pd.DataFrame,
                        additional_data: Dict = None) -> pd.DataFrame:
        """基于ML模型生成信号"""
        try:
            if not self.is_fitted:
                self.logger.warning("ML模型未训练，无法生成信号")
                return pd.DataFrame()
            
            # 预处理因子
            processed_factors = self.preprocess_factors(factors)
            
            # 创建特征矩阵
            X = self._create_feature_matrix(processed_factors)
            
            # 生成预测
            predictions = {}
            for symbol in factors.columns:
                if symbol in self.models:
                    model = self.models[symbol]
                    symbol_features = X.loc[:, X.columns.str.startswith(symbol)]
                    
                    if not symbol_features.empty:
                        pred = model.predict(symbol_features)
                        predictions[symbol] = pred
            
            # 转换为信号
            if predictions:
                pred_df = pd.DataFrame(predictions, index=factors.index)
                
                if self.model_type == "regression":
                    signals = self.convert_regression_to_signals(pred_df)
                else:
                    signals = self._convert_classification_to_signals(pred_df)
                
                return signals
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"ML信号生成失败: {e}")
            return pd.DataFrame()
    
    def prepare_target_labels(self, returns: pd.DataFrame,
                             method: str = "quantile",
                             n_classes: int = 3) -> pd.DataFrame:
        """准备目标标签"""
        try:
            # 计算前向收益率
            forward_returns = returns.shift(-self.prediction_horizon)
            
            if method == "quantile":
                # 使用分位数方法
                labels = pd.DataFrame(index=returns.index, columns=returns.columns)
                
                for col in returns.columns:
                    col_returns = forward_returns[col].dropna()
                    if len(col_returns) > 0:
                        if n_classes == 3:
                            # 三分类：买入(2)、持有(1)、卖出(0)
                            q33 = col_returns.quantile(0.33)
                            q67 = col_returns.quantile(0.67)
                            
                            labels[col] = np.where(forward_returns[col] >= q67, 2,
                                                 np.where(forward_returns[col] <= q33, 0, 1))
                        elif n_classes == 2:
                            # 二分类：买入(1)、卖出(0)
                            median = col_returns.median()
                            labels[col] = np.where(forward_returns[col] >= median, 1, 0)
                        else:
                            # 多分类
                            quantiles = np.linspace(0, 1, n_classes + 1)
                            labels[col] = pd.cut(forward_returns[col], 
                                               bins=col_returns.quantile(quantiles),
                                               labels=range(n_classes),
                                               include_lowest=True)
                
            elif method == "threshold":
                # 使用阈值方法
                labels = pd.DataFrame(index=returns.index, columns=returns.columns)
                
                for col in returns.columns:
                    col_returns = forward_returns[col]
                    threshold = col_returns.std() * 0.5  # 使用0.5倍标准差作为阈值
                    
                    if n_classes == 3:
                        labels[col] = np.where(col_returns >= threshold, 2,
                                             np.where(col_returns <= -threshold, 0, 1))
                    else:
                        labels[col] = np.where(col_returns >= 0, 1, 0)
                        
            elif method == "zscore":
                # 使用Z分数方法
                labels = pd.DataFrame(index=returns.index, columns=returns.columns)
                
                for col in returns.columns:
                    z_scores = (forward_returns[col] - forward_returns[col].mean()) / forward_returns[col].std()
                    
                    if n_classes == 3:
                        labels[col] = np.where(z_scores >= 0.5, 2,
                                             np.where(z_scores <= -0.5, 0, 1))
                    else:
                        labels[col] = np.where(z_scores >= 0, 1, 0)
            
            return labels
            
        except Exception as e:
            self.logger.error(f"准备目标标签失败: {e}")
            return pd.DataFrame()
    
    def convert_regression_to_signals(self, predictions: pd.DataFrame,
                                    method: str = "quantile") -> pd.DataFrame:
        """将回归预测转换为信号"""
        try:
            signals = pd.DataFrame(index=predictions.index, columns=predictions.columns)
            
            for col in predictions.columns:
                pred_values = predictions[col].dropna()
                
                if method == "quantile":
                    # 使用分位数转换
                    q33 = pred_values.quantile(0.33)
                    q67 = pred_values.quantile(0.67)
                    
                    signals[col] = np.where(predictions[col] >= q67, 1.0,
                                          np.where(predictions[col] <= q33, -1.0, 0.0))
                    
                elif method == "zscore":
                    # 使用Z分数转换
                    z_scores = (predictions[col] - predictions[col].mean()) / predictions[col].std()
                    signals[col] = np.where(z_scores >= 0.5, 1.0,
                                          np.where(z_scores <= -0.5, -1.0, 0.0))
                    
                elif method == "threshold":
                    # 使用阈值转换
                    threshold = pred_values.std() * 0.5
                    signals[col] = np.where(predictions[col] >= threshold, 1.0,
                                          np.where(predictions[col] <= -threshold, -1.0, 0.0))
                else:
                    # 直接标准化
                    signals[col] = (predictions[col] - predictions[col].mean()) / predictions[col].std()
                    signals[col] = signals[col].clip(-1.0, 1.0)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"回归预测转换信号失败: {e}")
            return pd.DataFrame()
    
    def _create_feature_matrix(self, factors: pd.DataFrame) -> pd.DataFrame:
        """创建特征矩阵"""
        try:
            # 创建滞后特征
            feature_matrix = pd.DataFrame(index=factors.index)
            
            for col in factors.columns:
                # 当前值
                feature_matrix[f"{col}_current"] = factors[col]
                
                # 滞后值
                for lag in [1, 2, 3, 5]:
                    feature_matrix[f"{col}_lag_{lag}"] = factors[col].shift(lag)
                
                # 滚动统计
                feature_matrix[f"{col}_ma_5"] = factors[col].rolling(5).mean()
                feature_matrix[f"{col}_ma_10"] = factors[col].rolling(10).mean()
                feature_matrix[f"{col}_std_5"] = factors[col].rolling(5).std()
                feature_matrix[f"{col}_std_10"] = factors[col].rolling(10).std()
                
                # 技术指标
                feature_matrix[f"{col}_rsi"] = self._calculate_rsi(factors[col])
                feature_matrix[f"{col}_momentum"] = factors[col] / factors[col].shift(5) - 1
            
            # 填充缺失值
            feature_matrix = feature_matrix.fillna(0)
            
            return feature_matrix
            
        except Exception as e:
            self.logger.error(f"创建特征矩阵失败: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """计算RSI指标"""
        try:
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
            
        except Exception as e:
            self.logger.error(f"计算RSI失败: {e}")
            return pd.Series(index=series.index)
    
    def _convert_classification_to_signals(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """将分类预测转换为信号"""
        try:
            signals = pd.DataFrame(index=predictions.index, columns=predictions.columns)
            
            for col in predictions.columns:
                # 假设分类结果：0=卖出(-1), 1=持有(0), 2=买入(1)
                signals[col] = np.where(predictions[col] == 2, 1.0,
                                      np.where(predictions[col] == 0, -1.0, 0.0))
            
            return signals
            
        except Exception as e:
            self.logger.error(f"分类预测转换信号失败: {e}")
            return pd.DataFrame()
    
    def fit(self, factors: pd.DataFrame, returns: pd.DataFrame, 
            model_config: Dict = None):
        """训练ML模型"""
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.model_selection import TimeSeriesSplit
            
            model_config = model_config or {}
            
            # 准备目标标签
            if self.model_type == "classification":
                labels = self.prepare_target_labels(returns, 
                                                  method=model_config.get("label_method", "quantile"),
                                                  n_classes=model_config.get("n_classes", 3))
            else:
                labels = returns.shift(-self.prediction_horizon)
            
            # 创建特征矩阵
            X = self._create_feature_matrix(factors)
            
            # 对每个股票训练模型
            for symbol in factors.columns:
                if symbol in labels.columns:
                    y = labels[symbol].dropna()
                    symbol_features = X.loc[y.index, X.columns.str.startswith(symbol)]
                    
                    if len(y) > 50 and not symbol_features.empty:  # 至少需要50个样本
                        # 创建模型
                        if self.model_type == "classification":
                            model = RandomForestClassifier(
                                n_estimators=model_config.get("n_estimators", 100),
                                max_depth=model_config.get("max_depth", 5),
                                random_state=42
                            )
                        else:
                            model = RandomForestRegressor(
                                n_estimators=model_config.get("n_estimators", 100),
                                max_depth=model_config.get("max_depth", 5),
                                random_state=42
                            )
                        
                        # 训练模型
                        model.fit(symbol_features, y)
                        self.models[symbol] = model
                        
                        self.logger.info(f"模型训练完成: {symbol}")
            
            self.is_fitted = True
            self.logger.info("ML信号生成器训练完成")
            
        except Exception as e:
            self.logger.error(f"训练ML模型失败: {e}")
            raise


class CompositeSignalGenerator(BaseSignalGenerator):
    """复合信号生成器"""
    
    def __init__(self, sub_generators: List[BaseSignalGenerator],
                 combination_method: str = "weighted_average"):
        """初始化复合信号生成器"""
        super().__init__("composite")
        self.sub_generators = sub_generators
        self.combination_method = combination_method
        self.weights = {}
        self.performance_history = {}
    
    def generate_signals(self, factors: pd.DataFrame,
                        additional_data: Dict = None) -> pd.DataFrame:
        """生成复合信号"""
        try:
            # 生成各个子生成器的信号
            signals_dict = {}
            
            for generator in self.sub_generators:
                try:
                    signals = generator.generate_signals(factors, additional_data)
                    if not signals.empty:
                        signals_dict[generator.name] = signals
                        self.logger.info(f"子生成器 {generator.name} 信号生成完成")
                except Exception as e:
                    self.logger.error(f"子生成器 {generator.name} 信号生成失败: {e}")
                    continue
            
            if not signals_dict:
                self.logger.warning("没有子生成器成功生成信号")
                return pd.DataFrame()
            
            # 组合信号
            if self.combination_method == "adaptive":
                performance_data = additional_data.get("performance_data") if additional_data else None
                if performance_data is not None:
                    combined_signals = self.adaptive_combination(signals_dict, performance_data)
                else:
                    self.logger.warning("自适应组合需要性能数据，使用加权平均方法")
                    combined_signals = self.combine_signals(signals_dict)
            else:
                combined_signals = self.combine_signals(signals_dict, self.weights)
            
            # 验证组合信号
            if not self.validate_signals(combined_signals):
                self.logger.warning("组合信号验证失败")
            
            return combined_signals
            
        except Exception as e:
            self.logger.error(f"生成复合信号失败: {e}")
            return pd.DataFrame()
    
    def combine_signals(self, signals_dict: Dict[str, pd.DataFrame],
                       weights: Dict[str, float] = None) -> pd.DataFrame:
        """组合多个信号"""
        try:
            if not signals_dict:
                return pd.DataFrame()
            
            # 确保所有信号具有相同的index和columns
            first_signal = list(signals_dict.values())[0]
            combined_signals = pd.DataFrame(0.0, index=first_signal.index, columns=first_signal.columns)
            
            if weights is None:
                # 等权重
                weights = {name: 1.0 / len(signals_dict) for name in signals_dict.keys()}
            else:
                # 归一化权重
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {name: weight / total_weight for name, weight in weights.items()}
                else:
                    weights = {name: 1.0 / len(signals_dict) for name in signals_dict.keys()}
            
            if self.combination_method == "weighted_average":
                # 加权平均
                for name, signals in signals_dict.items():
                    weight = weights.get(name, 1.0 / len(signals_dict))
                    # 确保信号对齐
                    aligned_signals = signals.reindex(index=combined_signals.index, 
                                                    columns=combined_signals.columns, 
                                                    fill_value=0.0)
                    combined_signals += weight * aligned_signals
                    
            elif self.combination_method == "max_signal":
                # 使用最大信号
                for i, (name, signals) in enumerate(signals_dict.items()):
                    aligned_signals = signals.reindex(index=combined_signals.index,
                                                    columns=combined_signals.columns,
                                                    fill_value=0.0)
                    if i == 0:
                        combined_signals = aligned_signals.copy()
                    else:
                        combined_signals = np.where(np.abs(aligned_signals) > np.abs(combined_signals),
                                                  aligned_signals, combined_signals)
                        
            elif self.combination_method == "majority_vote":
                # 多数投票
                for name, signals in signals_dict.items():
                    aligned_signals = signals.reindex(index=combined_signals.index,
                                                    columns=combined_signals.columns,
                                                    fill_value=0.0)
                    # 转换为-1, 0, 1信号
                    vote_signals = np.where(aligned_signals > 0.1, 1,
                                          np.where(aligned_signals < -0.1, -1, 0))
                    combined_signals += vote_signals
                
                # 归一化投票结果
                combined_signals = np.where(combined_signals > 0, 1.0,
                                          np.where(combined_signals < 0, -1.0, 0.0))
                
            elif self.combination_method == "rank_combination":
                # 排名组合
                rank_signals = pd.DataFrame(0.0, index=combined_signals.index, columns=combined_signals.columns)
                
                for name, signals in signals_dict.items():
                    aligned_signals = signals.reindex(index=combined_signals.index,
                                                    columns=combined_signals.columns,
                                                    fill_value=0.0)
                    # 对每个时间点进行排名
                    for timestamp in aligned_signals.index:
                        timestamp_signals = aligned_signals.loc[timestamp]
                        ranks = timestamp_signals.rank(method='average', ascending=False)
                        normalized_ranks = (ranks - ranks.mean()) / ranks.std()
                        rank_signals.loc[timestamp] += normalized_ranks.fillna(0)
                
                combined_signals = rank_signals / len(signals_dict)
            
            # 限制信号范围
            combined_signals = combined_signals.clip(-1.0, 1.0)
            
            return combined_signals
            
        except Exception as e:
            self.logger.error(f"组合信号失败: {e}")
            return pd.DataFrame()
    
    def adaptive_combination(self, signals_dict: Dict[str, pd.DataFrame],
                           performance_data: pd.DataFrame) -> pd.DataFrame:
        """自适应信号组合"""
        try:
            # 计算每个子生成器的历史性能
            performance_weights = {}
            
            for name in signals_dict.keys():
                if name in performance_data.columns:
                    # 使用最近的性能数据计算权重
                    recent_performance = performance_data[name].tail(20)  # 最近20个周期
                    
                    # 计算性能指标
                    avg_performance = recent_performance.mean()
                    volatility = recent_performance.std()
                    
                    # 计算夏普比率作为权重
                    if volatility > 0:
                        sharpe_ratio = avg_performance / volatility
                        performance_weights[name] = max(0, sharpe_ratio)  # 确保权重非负
                    else:
                        performance_weights[name] = 0.0
                else:
                    performance_weights[name] = 1.0  # 默认权重
            
            # 检查是否有有效权重
            total_weight = sum(performance_weights.values())
            if total_weight <= 0:
                self.logger.warning("所有子生成器性能权重为0，使用等权重")
                performance_weights = {name: 1.0 for name in signals_dict.keys()}
            
            # 使用性能权重组合信号
            combined_signals = self.combine_signals(signals_dict, performance_weights)
            
            # 更新权重记录
            self.weights = performance_weights
            
            self.logger.info(f"自适应组合完成，权重: {performance_weights}")
            return combined_signals
            
        except Exception as e:
            self.logger.error(f"自适应信号组合失败: {e}")
            # 回退到等权重组合
            return self.combine_signals(signals_dict)
    
    def update_performance_history(self, performance_data: Dict[str, float]):
        """更新性能历史"""
        try:
            timestamp = pd.Timestamp.now()
            
            for name, performance in performance_data.items():
                if name not in self.performance_history:
                    self.performance_history[name] = []
                
                self.performance_history[name].append({
                    'timestamp': timestamp,
                    'performance': performance
                })
                
                # 保留最近100个记录
                if len(self.performance_history[name]) > 100:
                    self.performance_history[name] = self.performance_history[name][-100:]
            
            self.logger.info("性能历史更新完成")
            
        except Exception as e:
            self.logger.error(f"更新性能历史失败: {e}")
    
    def set_weights(self, weights: Dict[str, float]):
        """设置组合权重"""
        try:
            # 归一化权重
            total_weight = sum(weights.values())
            if total_weight > 0:
                self.weights = {name: weight / total_weight for name, weight in weights.items()}
            else:
                self.weights = {name: 1.0 / len(weights) for name in weights.keys()}
            
            self.logger.info(f"设置组合权重: {self.weights}")
            
        except Exception as e:
            self.logger.error(f"设置权重失败: {e}")
    
    def get_sub_generator_signals(self, factors: pd.DataFrame,
                                 additional_data: Dict = None) -> Dict[str, pd.DataFrame]:
        """获取各子生成器的信号"""
        try:
            signals_dict = {}
            
            for generator in self.sub_generators:
                try:
                    signals = generator.generate_signals(factors, additional_data)
                    if not signals.empty:
                        signals_dict[generator.name] = signals
                except Exception as e:
                    self.logger.error(f"获取子生成器 {generator.name} 信号失败: {e}")
                    continue
            
            return signals_dict
            
        except Exception as e:
            self.logger.error(f"获取子生成器信号失败: {e}")
            return {}


class MultiTimeframeSignalGenerator(BaseSignalGenerator):
    """多时间框架信号生成器"""
    
    def __init__(self, timeframes: List[str] = ["1min", "5min", "15min", "1h"]):
        """初始化多时间框架信号生成器"""
        super().__init__("multi_timeframe")
        self.timeframes = timeframes
        self.timeframe_generators = {}
        self.timeframe_weights = {}
        
        # 设置默认权重（长周期权重更高）
        for i, tf in enumerate(timeframes):
            self.timeframe_weights[tf] = (i + 1) / len(timeframes)
    
    def generate_signals(self, factors: pd.DataFrame,
                        additional_data: Dict = None) -> pd.DataFrame:
        """生成多时间框架信号"""
        try:
            timeframe_signals = {}
            
            # 为每个时间框架生成信号
            for timeframe in self.timeframes:
                try:
                    # 调整因子数据到对应时间框架
                    tf_factors = self._resample_factors(factors, timeframe)
                    
                    if not tf_factors.empty:
                        # 为该时间框架生成信号
                        if timeframe in self.timeframe_generators:
                            generator = self.timeframe_generators[timeframe]
                            tf_signals = generator.generate_signals(tf_factors, additional_data)
                        else:
                            # 使用默认的排序信号生成器
                            generator = RankingSignalGenerator()
                            tf_signals = generator.generate_signals(tf_factors, additional_data)
                        
                        if not tf_signals.empty:
                            timeframe_signals[timeframe] = tf_signals
                            self.logger.info(f"时间框架 {timeframe} 信号生成完成")
                        
                except Exception as e:
                    self.logger.error(f"时间框架 {timeframe} 信号生成失败: {e}")
                    continue
            
            if not timeframe_signals:
                self.logger.warning("没有时间框架成功生成信号")
                return pd.DataFrame()
            
            # 对齐和组合不同时间框架的信号
            aligned_signals = self.align_timeframes(timeframe_signals)
            
            # 按时间框架加权
            weighted_signals = self.weight_by_timeframe(aligned_signals, self.timeframe_weights)
            
            return weighted_signals
            
        except Exception as e:
            self.logger.error(f"生成多时间框架信号失败: {e}")
            return pd.DataFrame()
    
    def align_timeframes(self, signals_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """对齐不同时间框架的信号"""
        try:
            if not signals_dict:
                return pd.DataFrame()
            
            # 找到最高频率的时间框架（通常是第一个）
            base_timeframe = self.timeframes[0]
            if base_timeframe not in signals_dict:
                base_timeframe = list(signals_dict.keys())[0]
            
            base_signals = signals_dict[base_timeframe]
            aligned_signals = pd.DataFrame(index=base_signals.index, columns=base_signals.columns)
            
            # 初始化结果
            combined_signals = pd.DataFrame(0.0, index=base_signals.index, columns=base_signals.columns)
            
            for timeframe, signals in signals_dict.items():
                # 将低频信号向前填充到高频
                if timeframe != base_timeframe:
                    resampled_signals = self._forward_fill_signals(signals, base_signals.index)
                else:
                    resampled_signals = signals
                
                # 确保索引和列对齐
                aligned = resampled_signals.reindex(index=base_signals.index, 
                                                  columns=base_signals.columns, 
                                                  fill_value=0.0)
                
                # 添加时间框架权重
                weight = self.timeframe_weights.get(timeframe, 1.0)
                combined_signals += weight * aligned
            
            # 归一化
            total_weight = sum(self.timeframe_weights.get(tf, 1.0) for tf in signals_dict.keys())
            if total_weight > 0:
                combined_signals /= total_weight
            
            # 限制信号范围
            combined_signals = combined_signals.clip(-1.0, 1.0)
            
            return combined_signals
            
        except Exception as e:
            self.logger.error(f"对齐时间框架失败: {e}")
            return pd.DataFrame()
    
    def weight_by_timeframe(self, signals: pd.DataFrame,
                           timeframe_weights: Dict[str, float]) -> pd.DataFrame:
        """按时间框架加权"""
        try:
            # 这个方法在align_timeframes中已经实现了权重逻辑
            # 这里可以进行额外的加权处理
            
            weighted_signals = signals.copy()
            
            # 可以根据市场状态动态调整权重
            # 例如：在高波动时期增加短周期权重，在趋势市场增加长周期权重
            
            return weighted_signals
            
        except Exception as e:
            self.logger.error(f"时间框架加权失败: {e}")
            return signals
    
    def _resample_factors(self, factors: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """将因子数据重采样到指定时间框架"""
        try:
            if timeframe == "1min":
                # 原始数据，不需要重采样
                return factors
            
            # 定义时间频率映射
            freq_mapping = {
                "5min": "5T",
                "15min": "15T", 
                "30min": "30T",
                "1h": "1H",
                "4h": "4H",
                "1d": "1D"
            }
            
            freq = freq_mapping.get(timeframe)
            if freq is None:
                self.logger.warning(f"不支持的时间框架: {timeframe}")
                return factors
            
            # 重采样：使用最后一个值
            resampled_factors = factors.resample(freq).last()
            
            # 删除空值行
            resampled_factors = resampled_factors.dropna(how='all')
            
            return resampled_factors
            
        except Exception as e:
            self.logger.error(f"重采样因子失败: {e}")
            return pd.DataFrame()
    
    def _forward_fill_signals(self, signals: pd.DataFrame, target_index: pd.Index) -> pd.DataFrame:
        """前向填充信号到目标索引"""
        try:
            # 重建索引并前向填充
            reindexed_signals = signals.reindex(target_index, method='ffill')
            
            # 后向填充开始的空值
            reindexed_signals = reindexed_signals.fillna(method='bfill')
            
            # 最后填充剩余空值为0
            reindexed_signals = reindexed_signals.fillna(0.0)
            
            return reindexed_signals
            
        except Exception as e:
            self.logger.error(f"前向填充信号失败: {e}")
            return pd.DataFrame()
    
    def set_timeframe_generator(self, timeframe: str, generator: BaseSignalGenerator):
        """为特定时间框架设置信号生成器"""
        try:
            self.timeframe_generators[timeframe] = generator
            self.logger.info(f"为时间框架 {timeframe} 设置信号生成器: {generator.name}")
            
        except Exception as e:
            self.logger.error(f"设置时间框架生成器失败: {e}")
    
    def set_timeframe_weights(self, weights: Dict[str, float]):
        """设置时间框架权重"""
        try:
            # 归一化权重
            total_weight = sum(weights.values())
            if total_weight > 0:
                self.timeframe_weights = {tf: weight / total_weight 
                                        for tf, weight in weights.items()}
            else:
                self.timeframe_weights = {tf: 1.0 / len(weights) 
                                        for tf in weights.keys()}
            
            self.logger.info(f"设置时间框架权重: {self.timeframe_weights}")
            
        except Exception as e:
            self.logger.error(f"设置时间框架权重失败: {e}")
    
    def get_timeframe_signals(self, factors: pd.DataFrame,
                             additional_data: Dict = None) -> Dict[str, pd.DataFrame]:
        """获取各时间框架的信号"""
        try:
            timeframe_signals = {}
            
            for timeframe in self.timeframes:
                try:
                    tf_factors = self._resample_factors(factors, timeframe)
                    
                    if not tf_factors.empty:
                        if timeframe in self.timeframe_generators:
                            generator = self.timeframe_generators[timeframe]
                            tf_signals = generator.generate_signals(tf_factors, additional_data)
                        else:
                            generator = RankingSignalGenerator()
                            tf_signals = generator.generate_signals(tf_factors, additional_data)
                        
                        if not tf_signals.empty:
                            timeframe_signals[timeframe] = tf_signals
                            
                except Exception as e:
                    self.logger.error(f"获取时间框架 {timeframe} 信号失败: {e}")
                    continue
            
            return timeframe_signals
            
        except Exception as e:
            self.logger.error(f"获取时间框架信号失败: {e}")
            return {}
    
    def analyze_timeframe_contribution(self, signals_dict: Dict[str, pd.DataFrame],
                                     returns: pd.DataFrame) -> Dict[str, float]:
        """分析各时间框架的贡献度"""
        try:
            contribution_scores = {}
            
            for timeframe, signals in signals_dict.items():
                # 计算该时间框架信号的IC
                forward_returns = returns.shift(-1)
                
                # 对齐信号和收益率
                aligned_signals = signals.reindex(index=forward_returns.index, 
                                                columns=forward_returns.columns,
                                                fill_value=0.0)
                
                # 计算IC
                ic_values = []
                for col in aligned_signals.columns:
                    if col in forward_returns.columns:
                        signal_col = aligned_signals[col].dropna()
                        return_col = forward_returns[col].reindex(signal_col.index).dropna()
                        
                        if len(signal_col) > 10 and len(return_col) > 10:
                            correlation = signal_col.corr(return_col)
                            if not np.isnan(correlation):
                                ic_values.append(correlation)
                
                if ic_values:
                    contribution_scores[timeframe] = np.mean(ic_values)
                else:
                    contribution_scores[timeframe] = 0.0
            
            self.logger.info(f"时间框架贡献度分析完成: {contribution_scores}")
            return contribution_scores
            
        except Exception as e:
            self.logger.error(f"时间框架贡献度分析失败: {e}")
            return {}


class SignalFilter:
    """信号过滤器"""
    
    def __init__(self):
        """初始化信号过滤器"""
        self.logger = Logger().get_logger("signal_filter")
    
    def filter_by_volatility(self, signals: pd.DataFrame,
                           volatility: pd.DataFrame,
                           vol_threshold: float = 0.3) -> pd.DataFrame:
        """基于波动率过滤信号"""
        try:
            filtered_signals = signals.copy()
            
            # 过滤高波动率股票的信号
            high_vol_mask = volatility > vol_threshold
            filtered_signals[high_vol_mask] = 0.0
            
            return filtered_signals
        except Exception as e:
            self.logger.error(f"波动率过滤失败: {e}")
            return signals
    
    def filter_by_liquidity(self, signals: pd.DataFrame,
                          volume: pd.DataFrame,
                          min_volume: float = 1000000) -> pd.DataFrame:
        """基于流动性过滤信号"""
        try:
            filtered_signals = signals.copy()
            
            # 过滤低流动性股票的信号
            low_liquidity_mask = volume < min_volume
            filtered_signals[low_liquidity_mask] = 0.0
            
            return filtered_signals
        except Exception as e:
            self.logger.error(f"流动性过滤失败: {e}")
            return signals
    
    def filter_by_market_regime(self, signals: pd.DataFrame,
                               market_regime: pd.Series) -> pd.DataFrame:
        """基于市场状态过滤信号"""
        pass
    
    def filter_by_sector(self, signals: pd.DataFrame,
                        sector_exposure: pd.DataFrame,
                        max_sector_weight: float = 0.3) -> pd.DataFrame:
        """基于行业暴露过滤信号"""
        pass
    
    def apply_stop_loss(self, signals: pd.DataFrame,
                       returns: pd.DataFrame,
                       stop_loss_pct: float = 0.05) -> pd.DataFrame:
        """应用止损"""
        pass


class SignalValidator:
    """信号验证器"""
    
    def __init__(self):
        """初始化信号验证器"""
        self.logger = Logger().get_logger("signal_validator")
    
    def validate_signal_distribution(self, signals: pd.DataFrame) -> Dict[str, bool]:
        """验证信号分布"""
        try:
            validation_results = {}
            
            # 检查是否为空
            validation_results['not_empty'] = not signals.empty
            
            # 检查是否有数值
            validation_results['has_numeric'] = signals.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
            
            # 检查信号范围
            validation_results['in_range'] = (signals >= -1).all().all() and (signals <= 1).all().all()
            
            # 检查是否有过多的零值
            zero_ratio = (signals == 0).sum().sum() / signals.size
            validation_results['not_too_many_zeros'] = zero_ratio < 0.9
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"信号分布验证失败: {e}")
            return {'error': True}
    
    def check_signal_stability(self, signals: pd.DataFrame,
                              window: int = 20) -> pd.DataFrame:
        """检查信号稳定性"""
        try:
            # 计算滚动标准差作为稳定性指标
            stability = signals.rolling(window=window).std()
            return stability
            
        except Exception as e:
            self.logger.error(f"信号稳定性检查失败: {e}")
            return pd.DataFrame()
    
    def detect_signal_anomalies(self, signals: pd.DataFrame) -> pd.DataFrame:
        """检测信号异常"""
        try:
            # 使用3倍标准差检测异常
            anomalies = pd.DataFrame(False, index=signals.index, columns=signals.columns)
            
            for col in signals.columns:
                col_data = signals[col]
                mean_val = col_data.mean()
                std_val = col_data.std()
                
                anomalies[col] = (col_data > mean_val + 3 * std_val) | (col_data < mean_val - 3 * std_val)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"信号异常检测失败: {e}")
            return pd.DataFrame()
    
    def calculate_signal_turnover(self, signals: pd.DataFrame) -> pd.Series:
        """计算信号换手率"""
        pass


class SignalAnalyzer:
    """信号分析器"""
    
    def __init__(self):
        """初始化信号分析器"""
        self.logger = Logger().get_logger("signal_analyzer")
    
    def analyze_signal_performance(self, signals: pd.DataFrame,
                                 returns: pd.DataFrame,
                                 periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """分析信号表现"""
        try:
            performance_results = {}
            
            for period in periods:
                # 计算前向收益率
                forward_returns = returns.shift(-period)
                
                # 计算信号与收益率的相关性
                corr = signals.corrwith(forward_returns, axis=0)
                performance_results[f'ic_{period}d'] = corr
                
                # 计算信号收益率
                signal_returns = signals.shift(1) * forward_returns
                performance_results[f'signal_return_{period}d'] = signal_returns.mean()
            
            return pd.DataFrame(performance_results)
            
        except Exception as e:
            self.logger.error(f"信号表现分析失败: {e}")
            return pd.DataFrame()
    
    def calculate_signal_ic(self, signals: pd.DataFrame,
                          returns: pd.DataFrame) -> pd.DataFrame:
        """计算信号IC"""
        try:
            # 计算信号与未来收益率的相关性
            forward_returns = returns.shift(-1)
            
            # 计算IC
            ic = pd.DataFrame(index=signals.index, columns=signals.columns)
            
            for timestamp in signals.index:
                for factor in signals.columns:
                    if timestamp in forward_returns.index:
                        signal_values = signals.loc[timestamp, :]
                        return_values = forward_returns.loc[timestamp, :]
                        
                        # 去除空值
                        valid_mask = ~(signal_values.isnull() | return_values.isnull())
                        
                        if valid_mask.sum() > 10:  # 至少需要10个有效值
                            correlation = signal_values[valid_mask].corr(return_values[valid_mask])
                            ic.loc[timestamp, factor] = correlation
            
            return ic
            
        except Exception as e:
            self.logger.error(f"信号IC计算失败: {e}")
            return pd.DataFrame()
    
    def analyze_signal_decay(self, signals: pd.DataFrame,
                           returns: pd.DataFrame,
                           max_periods: int = 20) -> pd.DataFrame:
        """分析信号衰减"""
        pass
    
    def sector_signal_analysis(self, signals: pd.DataFrame,
                             returns: pd.DataFrame,
                             sector_data: pd.DataFrame) -> pd.DataFrame:
        """行业信号分析"""
        pass


class SignalGenerator:
    """信号生成主类"""
    
    def __init__(self, config: Dict = None):
        """初始化信号生成器"""
        self.config = config or {}
        self.logger = Logger().get_logger("signal_generator")
        
        # 初始化各种信号生成器
        self.threshold_generator = ThresholdSignalGenerator()
        self.ranking_generator = RankingSignalGenerator()
        self.signal_filter = SignalFilter()
        self.signal_validator = SignalValidator()
        self.signal_analyzer = SignalAnalyzer()
        
        # 信号缓存
        self.signal_cache = {}
        
    def initialize(self, config: Dict):
        """初始化配置"""
        try:
            self.config = config
            
            # 配置各种生成器
            if 'threshold_params' in config:
                self.threshold_generator = ThresholdSignalGenerator(
                    thresholds=config['threshold_params']
                )
            
            if 'ranking_params' in config:
                self.ranking_generator = RankingSignalGenerator(
                    top_pct=config['ranking_params'].get('top_pct', 0.2),
                    bottom_pct=config['ranking_params'].get('bottom_pct', 0.2)
                )
            
            self.logger.info("信号生成器初始化完成")
            
        except Exception as e:
            self.logger.error(f"信号生成器初始化失败: {e}")
            raise
    
    def generate_raw_signals(self, factors: pd.DataFrame,
                           method: str = "ranking",
                           **kwargs) -> pd.DataFrame:
        """生成原始信号"""
        try:
            if method == "ranking":
                signals = self.ranking_generator.generate_signals(factors, kwargs)
            elif method == "threshold":
                signals = self.threshold_generator.generate_signals(factors, kwargs)
            else:
                raise ValueError(f"不支持的信号生成方法: {method}")
            
            self.logger.info(f"使用{method}方法生成信号完成")
            return signals
            
        except Exception as e:
            self.logger.error(f"生成原始信号失败: {e}")
            return pd.DataFrame()
    
    def generate_multi_horizon_signals(self, factors: pd.DataFrame,
                                     horizons: List[int] = [1, 5, 10, 20]) -> Dict[int, pd.DataFrame]:
        """生成多周期信号"""
        try:
            horizon_signals = {}
            
            for horizon in horizons:
                # 为每个周期生成信号
                horizon_factors = factors.copy()
                
                # 根据周期调整因子
                if horizon > 1:
                    # 对于长周期，使用滚动平均
                    horizon_factors = horizon_factors.rolling(window=horizon).mean()
                
                signals = self.generate_raw_signals(
                    horizon_factors, 
                    method=self.config.get('signal_method', 'ranking')
                )
                
                horizon_signals[horizon] = signals
            
            self.logger.info(f"生成多周期信号完成，周期数: {len(horizons)}")
            return horizon_signals
            
        except Exception as e:
            self.logger.error(f"生成多周期信号失败: {e}")
            return {}
    
    def generate_ensemble_signals(self, factors: pd.DataFrame,
                                methods: List[str] = ["ranking", "threshold"]) -> pd.DataFrame:
        """生成集成信号"""
        try:
            all_signals = {}
            
            for method in methods:
                signals = self.generate_raw_signals(factors, method)
                if not signals.empty:
                    all_signals[method] = signals
            
            if not all_signals:
                return pd.DataFrame()
            
            # 平均集成
            ensemble_signals = pd.DataFrame(0.0, index=factors.index, columns=factors.columns)
            
            for method_signals in all_signals.values():
                ensemble_signals += method_signals
            
            ensemble_signals /= len(all_signals)
            
            self.logger.info(f"生成集成信号完成，使用方法: {methods}")
            return ensemble_signals
            
        except Exception as e:
            self.logger.error(f"生成集成信号失败: {e}")
            return pd.DataFrame()
    
    def post_process_signals(self, signals: pd.DataFrame,
                           processing_config: Dict) -> pd.DataFrame:
        """信号后处理"""
        try:
            processed_signals = signals.copy()
            
            # 应用信号过滤
            if 'volatility_filter' in processing_config:
                processed_signals = self.signal_filter.filter_by_volatility(
                    processed_signals, 
                    volatility=processing_config['volatility_data'],
                    vol_threshold=processing_config['volatility_filter']['threshold']
                )
            
            if 'liquidity_filter' in processing_config:
                processed_signals = self.signal_filter.filter_by_liquidity(
                    processed_signals,
                    volume=processing_config['volume_data'],
                    min_volume=processing_config['liquidity_filter']['min_volume']
                )
            
            # 信号平滑
            if processing_config.get('smooth_signals', False):
                smooth_window = processing_config.get('smooth_window', 3)
                processed_signals = processed_signals.rolling(window=smooth_window).mean()
            
            # 信号截断
            if processing_config.get('clip_signals', True):
                processed_signals = processed_signals.clip(-1.0, 1.0)
            
            self.logger.info("信号后处理完成")
            return processed_signals
            
        except Exception as e:
            self.logger.error(f"信号后处理失败: {e}")
            return signals
    
    def validate_signals(self, signals: pd.DataFrame,
                        validation_config: Dict) -> Dict[str, bool]:
        """验证信号"""
        try:
            validation_results = {}
            
            # 基本验证
            validation_results['basic_validation'] = self.signal_validator.validate_signal_distribution(signals)
            
            # 稳定性验证
            if validation_config.get('check_stability', True):
                stability_window = validation_config.get('stability_window', 20)
                stability_results = self.signal_validator.check_signal_stability(signals, stability_window)
                validation_results['stability_check'] = not stability_results.isnull().all().all()
            
            # 异常检测
            if validation_config.get('detect_anomalies', True):
                anomaly_results = self.signal_validator.detect_signal_anomalies(signals)
                validation_results['anomaly_detection'] = anomaly_results.sum().sum() == 0
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"信号验证失败: {e}")
            return {'error': True}
    
    def optimize_signal_parameters(self, factors: pd.DataFrame,
                                 returns: pd.DataFrame,
                                 param_ranges: Dict) -> Dict:
        """优化信号参数"""
        try:
            from scipy.optimize import minimize
            
            def objective_function(params):
                # 根据参数生成信号
                if 'ranking' in param_ranges:
                    top_pct, bottom_pct = params[:2]
                    generator = RankingSignalGenerator(top_pct=top_pct, bottom_pct=bottom_pct)
                else:
                    generator = self.ranking_generator
                
                signals = generator.generate_signals(factors)
                
                # 计算IC作为目标函数
                ic = self.signal_analyzer.calculate_signal_ic(signals, returns)
                return -ic.mean().mean()  # 负号因为要最大化IC
            
            # 设置优化边界
            bounds = []
            initial_guess = []
            
            if 'ranking' in param_ranges:
                bounds.extend([(0.1, 0.4), (0.1, 0.4)])  # top_pct, bottom_pct
                initial_guess.extend([0.2, 0.2])
            
            # 执行优化
            result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')
            
            optimal_params = {
                'top_pct': result.x[0],
                'bottom_pct': result.x[1],
                'optimal_ic': -result.fun
            }
            
            self.logger.info(f"参数优化完成: {optimal_params}")
            return optimal_params
            
        except Exception as e:
            self.logger.error(f"参数优化失败: {e}")
            return {}
    
    def backtest_signals(self, signals: pd.DataFrame,
                       returns: pd.DataFrame,
                       transaction_costs: float = 0.001) -> Dict:
        """信号回测"""
        try:
            # 计算信号收益
            signal_returns = signals.shift(1) * returns
            
            # 计算交易成本
            turnover = signals.diff().abs().sum(axis=1)
            transaction_cost_series = turnover * transaction_costs
            
            # 计算净收益
            net_returns = signal_returns.sum(axis=1) - transaction_cost_series
            
            # 计算回测指标
            backtest_results = {
                'total_return': net_returns.cumsum().iloc[-1],
                'annual_return': net_returns.mean() * 252,
                'volatility': net_returns.std() * np.sqrt(252),
                'sharpe_ratio': net_returns.mean() / net_returns.std() * np.sqrt(252),
                'max_drawdown': (net_returns.cumsum() - net_returns.cumsum().cummax()).min(),
                'win_rate': (net_returns > 0).mean(),
                'average_turnover': turnover.mean(),
                'ic_mean': self.signal_analyzer.calculate_signal_ic(signals, returns).mean().mean(),
                'ic_std': self.signal_analyzer.calculate_signal_ic(signals, returns).std().mean()
            }
            
            self.logger.info("信号回测完成")
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"信号回测失败: {e}")
            return {}
    
    def real_time_signal_generation(self, latest_factors: pd.DataFrame) -> pd.DataFrame:
        """实时信号生成"""
        try:
            # 使用配置的默认方法生成信号
            method = self.config.get('signal_method', 'ranking')
            signals = self.generate_raw_signals(latest_factors, method)
            
            # 应用实时过滤
            if self.config.get('real_time_filter', True):
                processing_config = self.config.get('processing_config', {})
                signals = self.post_process_signals(signals, processing_config)
            
            # 缓存信号
            timestamp = latest_factors.index[-1]
            self.signal_cache[timestamp] = signals
            
            # 清理旧缓存
            if len(self.signal_cache) > 100:
                oldest_key = min(self.signal_cache.keys())
                del self.signal_cache[oldest_key]
            
            self.logger.info("实时信号生成完成")
            return signals
            
        except Exception as e:
            self.logger.error(f"实时信号生成失败: {e}")
            return pd.DataFrame()
    
    def generate_signals(self, factors: pd.DataFrame, method: str = "ranking") -> pd.DataFrame:
        """生成信号的主要接口"""
        try:
            # 生成原始信号
            raw_signals = self.generate_raw_signals(factors, method)
            
            # 后处理
            processing_config = self.config.get('processing_config', {})
            processed_signals = self.post_process_signals(raw_signals, processing_config)
            
            # 验证
            validation_config = self.config.get('validation_config', {})
            validation_results = self.validate_signals(processed_signals, validation_config)
            
            if not validation_results.get('basic_validation', {}).get('not_empty', False):
                self.logger.warning("信号验证未通过")
            
            return processed_signals
            
        except Exception as e:
            self.logger.error(f"信号生成失败: {e}")
            return pd.DataFrame() 