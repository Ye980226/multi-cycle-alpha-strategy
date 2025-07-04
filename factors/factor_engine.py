#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子引擎主模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from .technical_factors import TechnicalFactors
from .fundamental_factors import FundamentalFactors  
from .sentiment_factors import SentimentFactors
from .factor_utils import FactorProcessor, FactorValidator, FactorSelector
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import Logger
from config.strategy_config import StrategyConfig


class FactorEngine:
    """因子引擎主类"""
    
    def __init__(self, config: Dict = None):
        """初始化因子引擎"""
        self.config = config or {}
        self.logger = Logger()
        
        # 初始化因子计算器
        self.technical_factors = TechnicalFactors(self.logger)
        self.fundamental_factors = FundamentalFactors(self.logger)
        self.sentiment_factors = SentimentFactors(self.logger)
        
        # 初始化因子处理器
        self.factor_processor = FactorProcessor(self.logger)
        self.factor_validator = FactorValidator(self.logger)
        self.factor_selector = FactorSelector(self.logger)
        
        # 设置默认参数
        self.technical_periods = self.config.get('technical_periods', [5, 10, 20, 60])
        self.factor_groups = self.config.get('factor_groups', ['technical', 'fundamental', 'sentiment'])
        self.max_workers = self.config.get('max_workers', 4)
        self.use_multiprocessing = self.config.get('use_multiprocessing', False)
        self.winsorize_quantiles = self.config.get('winsorize_quantiles', (0.01, 0.99))
        self.standardize_method = self.config.get('standardize_method', 'zscore')
        self.correlation_threshold = self.config.get('correlation_threshold', 0.8)
        
        # 缓存
        self.factor_cache = {}
        self.correlation_cache = {}
        
        self.logger.info("FactorEngine initialized successfully")
    
    def initialize(self, config: Dict):
        """初始化配置"""
        try:
            self.config.update(config)
            
            # 设置并发参数
            self.max_workers = self.config.get('max_workers', 4)
            self.use_multiprocessing = self.config.get('use_multiprocessing', False)
            
            # 设置因子计算参数
            self.technical_periods = self.config.get('technical_periods', [5, 10, 20, 60])
            self.factor_groups = self.config.get('factor_groups', ['technical', 'fundamental', 'sentiment'])
            
            # 设置因子处理参数
            self.winsorize_quantiles = self.config.get('winsorize_quantiles', (0.01, 0.99))
            self.standardize_method = self.config.get('standardize_method', 'zscore')
            self.correlation_threshold = self.config.get('correlation_threshold', 0.8)
            
            self.logger.info(f"FactorEngine configuration updated: {len(self.config)} parameters")
            
        except Exception as e:
            self.logger.error(f"Error initializing FactorEngine: {str(e)}")
            raise
    
    def calculate_all_factors(self, data: pd.DataFrame, 
                             factor_groups: List[str] = None) -> pd.DataFrame:
        """计算所有因子"""
        try:
            if factor_groups is None:
                factor_groups = self.factor_groups
            
            all_factors = pd.DataFrame(index=data.index)
            
            # 并行计算不同类型的因子
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                
                if 'technical' in factor_groups:
                    futures['technical'] = executor.submit(
                        self.calculate_technical_factors, data, self.technical_periods
                    )
                
                if 'fundamental' in factor_groups:
                    # 需要基本面数据，这里使用模拟数据
                    fundamental_data = self._generate_mock_fundamental_data(data)
                    futures['fundamental'] = executor.submit(
                        self.calculate_fundamental_factors, data, fundamental_data
                    )
                
                if 'sentiment' in factor_groups:
                    futures['sentiment'] = executor.submit(
                        self.calculate_sentiment_factors, data
                    )
                
                # 收集结果
                for factor_type, future in futures.items():
                    try:
                        factors = future.result()
                        if not factors.empty:
                            # 过滤掉原始数据列，只保留因子列
                            original_columns = ['symbol', 'open', 'high', 'low', 'close', 'volume']
                            factor_columns = [col for col in factors.columns if col not in original_columns]
                            
                            if factor_columns:
                                factor_only = factors[factor_columns]
                                if all_factors.empty:
                                    all_factors = factor_only
                                else:
                                    all_factors = all_factors.join(factor_only, how='outer')
                                self.logger.info(f"Added {len(factor_columns)} {factor_type} factors")
                            else:
                                self.logger.warning(f"No factor columns found for {factor_type}")
                    except Exception as e:
                        self.logger.error(f"Error calculating {factor_type} factors: {str(e)}")
            
            # 缓存结果
            self.factor_cache['all_factors'] = all_factors
            
            self.logger.info(f"Calculated total {len(all_factors.columns)} factors")
            return all_factors
            
        except Exception as e:
            self.logger.error(f"Error calculating all factors: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def calculate_technical_factors(self, data: pd.DataFrame, 
                                   periods: List[int] = None) -> pd.DataFrame:
        """计算技术因子"""
        try:
            if periods is None:
                periods = self.technical_periods
            
            # 计算所有技术因子
            factors = self.technical_factors.calculate_all_factors(data, periods)
            
            self.logger.info(f"Calculated {len(factors.columns)} technical factors")
            return factors
            
        except Exception as e:
            self.logger.error(f"Error calculating technical factors: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def calculate_fundamental_factors(self, price_data: pd.DataFrame,
                                     fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """计算基本面因子"""
        try:
            # 计算所有基本面因子
            factors = self.fundamental_factors.calculate_all_factors(price_data, fundamental_data)
            
            self.logger.info(f"Calculated {len(factors.columns)} fundamental factors")
            return factors
            
        except Exception as e:
            self.logger.error(f"Error calculating fundamental factors: {str(e)}")
            return pd.DataFrame(index=price_data.index)
    
    def calculate_sentiment_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算情绪因子"""
        try:
            # 计算所有情绪因子
            factors = self.sentiment_factors.calculate_all_factors(data)
            
            self.logger.info(f"Calculated {len(factors.columns)} sentiment factors")
            return factors
            
        except Exception as e:
            self.logger.error(f"Error calculating sentiment factors: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def calculate_multi_timeframe_factors(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """计算多时间框架因子"""
        try:
            multi_factors = pd.DataFrame()
            
            for timeframe, df in data.items():
                # 为每个时间框架计算因子
                factors = self.calculate_all_factors(df)
                
                # 添加时间框架前缀
                factors.columns = [f"{timeframe}_{col}" for col in factors.columns]
                
                if multi_factors.empty:
                    multi_factors = factors
                else:
                    multi_factors = multi_factors.join(factors, how='outer')
            
            # 计算跨时间框架的因子
            cross_timeframe_factors = self._calculate_cross_timeframe_factors(data)
            if not cross_timeframe_factors.empty:
                multi_factors = multi_factors.join(cross_timeframe_factors, how='outer')
            
            self.logger.info(f"Calculated {len(multi_factors.columns)} multi-timeframe factors")
            return multi_factors
            
        except Exception as e:
            self.logger.error(f"Error calculating multi-timeframe factors: {str(e)}")
            return pd.DataFrame()
    
    def process_factors(self, factors: pd.DataFrame, 
                       processing_config: Dict) -> pd.DataFrame:
        """因子预处理"""
        try:
            processed_factors = factors.copy()
            
            # 去极值
            if processing_config.get('winsorize', True):
                quantiles = processing_config.get('winsorize_quantiles', self.winsorize_quantiles)
                processed_factors = self.factor_processor.winsorize(processed_factors, quantiles)
            
            # 填充缺失值
            if processing_config.get('fill_missing', True):
                fill_method = processing_config.get('fill_method', 'forward')
                processed_factors = self.factor_processor.fill_missing_values(processed_factors, fill_method)
            
            # 标准化
            if processing_config.get('standardize', True):
                method = processing_config.get('standardize_method', self.standardize_method)
                processed_factors = self.factor_processor.standardize(processed_factors, method)
            
            # 中性化
            if processing_config.get('neutralize', False):
                industry_data = processing_config.get('industry_data')
                market_cap = processing_config.get('market_cap')
                processed_factors = self.factor_processor.neutralize(
                    processed_factors, industry_data, market_cap
                )
            
            self.logger.info(f"Processed {len(processed_factors.columns)} factors")
            return processed_factors
            
        except Exception as e:
            self.logger.error(f"Error processing factors: {str(e)}")
            return factors
    
    def validate_factors(self, factors: pd.DataFrame) -> Dict[str, bool]:
        """验证因子质量"""
        try:
            validation_results = self.factor_validator.validate_factor_data(factors)
            
            # 提取整体质量判断
            quality_summary = {}
            for factor, results in validation_results.items():
                quality_summary[factor] = results.get('overall_quality', False)
            
            # 统计
            total_factors = len(quality_summary)
            good_factors = sum(quality_summary.values())
            
            self.logger.info(f"Validated {total_factors} factors, {good_factors} passed quality checks")
            return quality_summary
            
        except Exception as e:
            self.logger.error(f"Error validating factors: {str(e)}")
            return {}
    
    def get_factor_correlation_matrix(self, factors: pd.DataFrame) -> pd.DataFrame:
        """获取因子相关性矩阵"""
        try:
            # 检查缓存
            cache_key = f"corr_{len(factors.columns)}_{len(factors)}"
            if cache_key in self.correlation_cache:
                return self.correlation_cache[cache_key]
            
            correlation_matrix = factors.corr()
            
            # 缓存结果
            self.correlation_cache[cache_key] = correlation_matrix
            
            self.logger.info(f"Calculated correlation matrix for {len(factors.columns)} factors")
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {str(e)}")
            return pd.DataFrame()
    
    def select_factors(self, factors: pd.DataFrame, returns: pd.DataFrame,
                      method: str = "ic", top_k: int = 20) -> List[str]:
        """因子选择"""
        try:
            if method == "ic":
                selected_factors = self.factor_selector.select_by_ic(factors, returns, top_k)
            elif method == "correlation":
                selected_factors = self.factor_selector.select_by_correlation(
                    factors, self.correlation_threshold
                )
            elif method == "pca":
                selected_factors = self.factor_selector.select_by_pca(factors, top_k)
            elif method == "mutual_info":
                selected_factors = self.factor_selector.select_by_mutual_information(
                    factors, returns, top_k
                )
            else:
                self.logger.warning(f"Unknown factor selection method: {method}")
                return list(factors.columns)
            
            self.logger.info(f"Selected {len(selected_factors)} factors using {method} method")
            return selected_factors
            
        except Exception as e:
            self.logger.error(f"Error selecting factors: {str(e)}")
            return list(factors.columns)
    
    def calculate_factor_ic(self, factors: pd.DataFrame, 
                           returns: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """计算因子IC"""
        try:
            ic_results = self.factor_validator.calculate_factor_ic(factors, returns, periods)
            
            self.logger.info(f"Calculated IC for {len(factors.columns)} factors")
            return ic_results
            
        except Exception as e:
            self.logger.error(f"Error calculating factor IC: {str(e)}")
            return pd.DataFrame()
    
    def calculate_factor_turnover(self, factors: pd.DataFrame) -> pd.DataFrame:
        """计算因子换手率"""
        try:
            turnover_results = self.factor_validator.calculate_factor_turnover(factors)
            
            self.logger.info(f"Calculated turnover for {len(factors.columns)} factors")
            return turnover_results
            
        except Exception as e:
            self.logger.error(f"Error calculating factor turnover: {str(e)}")
            return pd.DataFrame()
    
    def neutralize_factors(self, factors: pd.DataFrame, 
                          industry_data: pd.DataFrame = None,
                          market_cap: pd.DataFrame = None) -> pd.DataFrame:
        """因子中性化"""
        try:
            neutralized_factors = self.factor_processor.neutralize(factors, industry_data, market_cap)
            
            self.logger.info(f"Neutralized {len(factors.columns)} factors")
            return neutralized_factors
            
        except Exception as e:
            self.logger.error(f"Error neutralizing factors: {str(e)}")
            return factors
    
    def standardize_factors(self, factors: pd.DataFrame, 
                           method: str = "zscore") -> pd.DataFrame:
        """因子标准化"""
        try:
            standardized_factors = self.factor_processor.standardize(factors, method)
            
            self.logger.info(f"Standardized {len(factors.columns)} factors using {method}")
            return standardized_factors
            
        except Exception as e:
            self.logger.error(f"Error standardizing factors: {str(e)}")
            return factors
    
    def winsorize_factors(self, factors: pd.DataFrame, 
                         quantiles: tuple = (0.01, 0.99)) -> pd.DataFrame:
        """因子去极值"""
        try:
            winsorized_factors = self.factor_processor.winsorize(factors, quantiles)
            
            self.logger.info(f"Winsorized {len(factors.columns)} factors")
            return winsorized_factors
            
        except Exception as e:
            self.logger.error(f"Error winsorizing factors: {str(e)}")
            return factors
    
    def calculate_composite_factor(self, factors: pd.DataFrame,
                                  weights: Dict[str, float] = None,
                                  method: str = "equal_weight") -> pd.Series:
        """计算复合因子"""
        try:
            if method == "equal_weight":
                # 等权重
                composite = factors.mean(axis=1)
            elif method == "ic_weight":
                # 基于IC加权
                if weights is None:
                    # 如果没有提供权重，使用等权重
                    composite = factors.mean(axis=1)
                else:
                    # 使用提供的权重
                    weight_series = pd.Series(weights)
                    # 只保留存在的因子
                    common_factors = factors.columns.intersection(weight_series.index)
                    if len(common_factors) > 0:
                        composite = (factors[common_factors] * weight_series[common_factors]).sum(axis=1)
                    else:
                        composite = factors.mean(axis=1)
            elif method == "pca_weight":
                # 基于PCA加权
                from sklearn.decomposition import PCA
                pca = PCA(n_components=1)
                factors_filled = factors.fillna(0)
                pca.fit(factors_filled)
                composite = pd.Series(pca.transform(factors_filled).flatten(), index=factors.index)
            else:
                # 默认等权重
                composite = factors.mean(axis=1)
            
            composite.name = f"composite_{method}"
            
            self.logger.info(f"Calculated composite factor using {method} method")
            return composite
            
        except Exception as e:
            self.logger.error(f"Error calculating composite factor: {str(e)}")
            return pd.Series(index=factors.index)
    
    def backtest_factor(self, factor: pd.Series, returns: Union[pd.Series, pd.DataFrame],
                       periods: List[int] = [1, 5, 10, 20]) -> Dict:
        """因子回测"""
        try:
            backtest_results = {}
            
            # 如果returns是DataFrame，使用第一列
            if isinstance(returns, pd.DataFrame):
                returns_series = returns.iloc[:, 0]
            else:
                returns_series = returns
            
            for period in periods:
                # 计算前瞻收益
                future_returns = returns_series.shift(-period)
                
                # 对齐数据
                aligned_factor = factor.dropna()
                aligned_returns = future_returns.reindex(aligned_factor.index).dropna()
                
                # 找到共同的有效数据
                common_index = aligned_factor.index.intersection(aligned_returns.index)
                
                if len(common_index) > 20:
                    factor_values = aligned_factor.loc[common_index]
                    return_values = aligned_returns.loc[common_index]
                    
                    # 计算分层回测
                    quantiles = 5
                    factor_ranks = factor_values.rank(pct=True)
                    factor_quantiles = pd.cut(factor_ranks, quantiles, labels=False)
                    
                    # 计算每个分位数的平均收益
                    quantile_returns = []
                    for q in range(quantiles):
                        mask = factor_quantiles == q
                        if mask.sum() > 0:
                            avg_return = return_values[mask].mean()
                            quantile_returns.append(avg_return)
                        else:
                            quantile_returns.append(0)
                    
                    # 计算多空收益
                    long_short_return = quantile_returns[-1] - quantile_returns[0]
                    
                    # 计算IC
                    ic = factor_values.corr(return_values)
                    
                    backtest_results[f"period_{period}"] = {
                        'quantile_returns': quantile_returns,
                        'long_short_return': long_short_return,
                        'ic': ic,
                        'sample_size': len(common_index)
                    }
            
            self.logger.info(f"Backtested factor across {len(periods)} periods")
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"Error backtesting factor: {str(e)}")
            return {}
    
    def get_factor_statistics(self, factors: pd.DataFrame) -> pd.DataFrame:
        """获取因子统计信息"""
        try:
            stats_data = []
            
            for col in factors.columns:
                factor_data = factors[col]
                
                stats = {
                    'factor': col,
                    'count': factor_data.count(),
                    'mean': factor_data.mean(),
                    'std': factor_data.std(),
                    'min': factor_data.min(),
                    'max': factor_data.max(),
                    'skew': factor_data.skew(),
                    'kurtosis': factor_data.kurtosis(),
                    'missing_ratio': factor_data.isna().sum() / len(factor_data)
                }
                
                stats_data.append(stats)
            
            stats_df = pd.DataFrame(stats_data)
            
            self.logger.info(f"Calculated statistics for {len(factors.columns)} factors")
            return stats_df
            
        except Exception as e:
            self.logger.error(f"Error calculating factor statistics: {str(e)}")
            return pd.DataFrame()
    
    def save_factors(self, factors: pd.DataFrame, file_path: str):
        """保存因子数据"""
        try:
            # 根据文件扩展名选择保存格式
            if file_path.endswith('.csv'):
                factors.to_csv(file_path)
            elif file_path.endswith('.parquet'):
                factors.to_parquet(file_path)
            elif file_path.endswith('.pkl'):
                factors.to_pickle(file_path)
            else:
                # 默认使用parquet格式
                factors.to_parquet(file_path + '.parquet')
            
            self.logger.info(f"Saved {len(factors.columns)} factors to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving factors: {str(e)}")
    
    def load_factors(self, file_path: str) -> pd.DataFrame:
        """加载因子数据"""
        try:
            # 根据文件扩展名选择加载格式
            if file_path.endswith('.csv'):
                factors = pd.read_csv(file_path, index_col=0)
            elif file_path.endswith('.parquet'):
                factors = pd.read_parquet(file_path)
            elif file_path.endswith('.pkl'):
                factors = pd.read_pickle(file_path)
            else:
                # 尝试不同格式
                for ext in ['.parquet', '.csv', '.pkl']:
                    try:
                        if ext == '.parquet':
                            factors = pd.read_parquet(file_path + ext)
                        elif ext == '.csv':
                            factors = pd.read_csv(file_path + ext, index_col=0)
                        else:
                            factors = pd.read_pickle(file_path + ext)
                        break
                    except:
                        continue
                else:
                    raise FileNotFoundError(f"Cannot load factors from {file_path}")
            
            self.logger.info(f"Loaded {len(factors.columns)} factors from {file_path}")
            return factors
            
        except Exception as e:
            self.logger.error(f"Error loading factors: {str(e)}")
            return pd.DataFrame()
    
    def update_factors(self, new_data: pd.DataFrame, 
                      existing_factors: pd.DataFrame) -> pd.DataFrame:
        """增量更新因子"""
        try:
            # 计算新数据的因子
            new_factors = self.calculate_all_factors(new_data)
            
            # 合并新旧因子
            if existing_factors.empty:
                updated_factors = new_factors
            else:
                # 确保列对齐
                common_columns = existing_factors.columns.intersection(new_factors.columns)
                if len(common_columns) > 0:
                    updated_factors = pd.concat([
                        existing_factors[common_columns],
                        new_factors[common_columns]
                    ])
                else:
                    updated_factors = new_factors
            
            # 去重
            updated_factors = updated_factors[~updated_factors.index.duplicated(keep='last')]
            
            self.logger.info(f"Updated factors with {len(new_factors)} new observations")
            return updated_factors
            
        except Exception as e:
            self.logger.error(f"Error updating factors: {str(e)}")
            return existing_factors
    
    def _generate_mock_fundamental_data(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """生成模拟基本面数据"""
        try:
            # 生成基本的财务指标
            fundamental_data = pd.DataFrame(index=price_data.index)
            
            # 模拟PE、PB等估值指标
            fundamental_data['pe_ratio'] = np.random.uniform(5, 50, len(price_data))
            fundamental_data['pb_ratio'] = np.random.uniform(0.5, 5, len(price_data))
            fundamental_data['ps_ratio'] = np.random.uniform(0.5, 10, len(price_data))
            
            # 模拟盈利能力指标
            fundamental_data['roe'] = np.random.uniform(0.05, 0.3, len(price_data))
            fundamental_data['roa'] = np.random.uniform(0.02, 0.15, len(price_data))
            
            # 模拟成长性指标
            fundamental_data['revenue_growth'] = np.random.uniform(-0.2, 0.5, len(price_data))
            fundamental_data['profit_growth'] = np.random.uniform(-0.3, 0.8, len(price_data))
            
            return fundamental_data
            
        except Exception as e:
            self.logger.error(f"Error generating mock fundamental data: {str(e)}")
            return pd.DataFrame(index=price_data.index)
    
    def _calculate_cross_timeframe_factors(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """计算跨时间框架因子"""
        try:
            cross_factors = pd.DataFrame()
            
            # 获取所有时间框架的收益率
            timeframes = list(data.keys())
            
            if len(timeframes) >= 2:
                # 计算不同时间框架间的相关性
                for i in range(len(timeframes)):
                    for j in range(i+1, len(timeframes)):
                        tf1, tf2 = timeframes[i], timeframes[j]
                        
                        # 计算收益率
                        returns1 = data[tf1]['close'].pct_change()
                        returns2 = data[tf2]['close'].pct_change()
                        
                        # 计算滚动相关性
                        rolling_corr = returns1.rolling(20).corr(returns2)
                        cross_factors[f'corr_{tf1}_{tf2}'] = rolling_corr
                        
                        # 计算相对强度
                        relative_strength = returns1.rolling(20).mean() / returns2.rolling(20).mean()
                        cross_factors[f'rel_strength_{tf1}_{tf2}'] = relative_strength
            
            return cross_factors
            
        except Exception as e:
            self.logger.error(f"Error calculating cross-timeframe factors: {str(e)}")
            return pd.DataFrame() 