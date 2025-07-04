#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子处理和验证工具类
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import Logger


class FactorProcessor:
    """因子处理器"""
    
    def __init__(self, logger: Logger = None):
        self.logger = logger or Logger()
    
    def winsorize(self, data: pd.DataFrame, quantiles: Tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame:
        """因子去极值处理"""
        try:
            result = data.copy()
            
            # 只处理数值列
            numeric_columns = result.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                lower_bound = result[col].quantile(quantiles[0])
                upper_bound = result[col].quantile(quantiles[1])
                result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
            
            self.logger.info(f"Winsorized {len(result.columns)} factors with quantiles {quantiles}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in winsorizing factors: {str(e)}")
            return data
    
    def standardize(self, data: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
        """因子标准化"""
        try:
            result = data.copy()
            
            # 只处理数值列
            numeric_columns = result.select_dtypes(include=[np.number]).columns
            numeric_data = result[numeric_columns]
            
            if len(numeric_columns) == 0:
                self.logger.warning("No numeric columns found for standardization")
                return result
            
            if method == "zscore":
                # Z-Score标准化
                standardized = (numeric_data - numeric_data.mean()) / numeric_data.std()
            elif method == "robust":
                # 稳健标准化
                scaler = RobustScaler()
                standardized = pd.DataFrame(
                    scaler.fit_transform(numeric_data),
                    index=numeric_data.index,
                    columns=numeric_data.columns
                )
            elif method == "minmax":
                # 最小-最大标准化
                standardized = (numeric_data - numeric_data.min()) / (numeric_data.max() - numeric_data.min())
            elif method == "rank":
                # 秩标准化
                standardized = numeric_data.rank(pct=True)
            else:
                standardized = numeric_data
            
            # 更新结果
            result[numeric_columns] = standardized
            
            self.logger.info(f"Standardized {len(result.columns)} factors using {method} method")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in standardizing factors: {str(e)}")
            return data
    
    def neutralize(self, factors: pd.DataFrame, industry_data: pd.DataFrame = None,
                   market_cap: pd.DataFrame = None) -> pd.DataFrame:
        """因子中性化处理"""
        try:
            result = factors.copy()
            
            # 行业中性化
            if industry_data is not None:
                for col in result.columns:
                    if col in result.columns:
                        # 按行业分组，计算行业内标准化
                        for industry in industry_data.unique():
                            industry_mask = industry_data == industry
                            if industry_mask.sum() > 1:
                                industry_mean = result.loc[industry_mask, col].mean()
                                industry_std = result.loc[industry_mask, col].std()
                                if industry_std > 0:
                                    result.loc[industry_mask, col] = (
                                        result.loc[industry_mask, col] - industry_mean
                                    ) / industry_std
            
            # 市值中性化
            if market_cap is not None:
                for col in result.columns:
                    if col in result.columns:
                        # 对市值进行回归，取残差
                        valid_mask = ~(result[col].isna() | market_cap.isna())
                        if valid_mask.sum() > 10:
                            X = market_cap[valid_mask].values.reshape(-1, 1)
                            y = result.loc[valid_mask, col].values
                            
                            reg = LinearRegression()
                            reg.fit(X, y)
                            
                            # 用残差替换原始值
                            result.loc[valid_mask, col] = y - reg.predict(X)
            
            self.logger.info(f"Neutralized {len(result.columns)} factors")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in neutralizing factors: {str(e)}")
            return factors
    
    def fill_missing_values(self, data: pd.DataFrame, method: str = "forward") -> pd.DataFrame:
        """填充缺失值"""
        try:
            result = data.copy()
            
            if method == "forward":
                result = result.ffill()
            elif method == "backward":
                result = result.bfill()
            elif method == "mean":
                result = result.fillna(result.mean())
            elif method == "median":
                result = result.fillna(result.median())
            elif method == "zero":
                result = result.fillna(0)
            elif method == "interpolate":
                result = result.interpolate()
            
            self.logger.info(f"Filled missing values using {method} method")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in filling missing values: {str(e)}")
            return data
    
    def remove_outliers(self, data: pd.DataFrame, method: str = "iqr", threshold: float = 1.5) -> pd.DataFrame:
        """移除异常值"""
        try:
            result = data.copy()
            
            if method == "iqr":
                # 使用IQR方法
                Q1 = result.quantile(0.25)
                Q3 = result.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                result = result[(result >= lower_bound) & (result <= upper_bound)]
                
            elif method == "zscore":
                # 使用Z-Score方法
                z_scores = np.abs(stats.zscore(result))
                result = result[(z_scores < threshold).all(axis=1)]
            
            self.logger.info(f"Removed outliers using {method} method")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in removing outliers: {str(e)}")
            return data
    
    def calculate_rolling_features(self, data: pd.DataFrame, window: int = 20,
                                  features: List[str] = None) -> pd.DataFrame:
        """计算滚动特征"""
        try:
            if features is None:
                features = ['mean', 'std', 'min', 'max', 'skew', 'kurt']
            
            result = pd.DataFrame(index=data.index)
            
            for col in data.columns:
                for feature in features:
                    if feature == 'mean':
                        result[f"{col}_roll_{feature}_{window}"] = data[col].rolling(window).mean()
                    elif feature == 'std':
                        result[f"{col}_roll_{feature}_{window}"] = data[col].rolling(window).std()
                    elif feature == 'min':
                        result[f"{col}_roll_{feature}_{window}"] = data[col].rolling(window).min()
                    elif feature == 'max':
                        result[f"{col}_roll_{feature}_{window}"] = data[col].rolling(window).max()
                    elif feature == 'skew':
                        result[f"{col}_roll_{feature}_{window}"] = data[col].rolling(window).skew()
                    elif feature == 'kurt':
                        result[f"{col}_roll_{feature}_{window}"] = data[col].rolling(window).kurt()
            
            self.logger.info(f"Calculated rolling features for {len(data.columns)} factors")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in calculating rolling features: {str(e)}")
            return pd.DataFrame(index=data.index)


class FactorValidator:
    """因子验证器"""
    
    def __init__(self, logger: Logger = None):
        self.logger = logger or Logger()
    
    def validate_factor_data(self, factors: pd.DataFrame) -> Dict[str, Dict]:
        """验证因子数据质量"""
        try:
            validation_results = {}
            
            for col in factors.columns:
                factor_data = factors[col]
                
                # 检查是否为数值列
                if not pd.api.types.is_numeric_dtype(factor_data):
                    validation_results[col] = {
                        'overall_quality': False,
                        'reason': 'Non-numeric data type'
                    }
                    continue
                
                # 基本统计
                basic_stats = {
                    'count': factor_data.count(),
                    'mean': factor_data.mean(),
                    'std': factor_data.std(),
                    'min': factor_data.min(),
                    'max': factor_data.max(),
                    'missing_ratio': factor_data.isna().sum() / len(factor_data)
                }
                
                # 质量检查
                quality_checks = {
                    'is_numeric': pd.api.types.is_numeric_dtype(factor_data),
                    'has_variance': factor_data.std() > 1e-8,
                    'missing_acceptable': basic_stats['missing_ratio'] < 0.5,
                    'no_inf_values': not np.isinf(factor_data).any(),
                    'sufficient_data': factor_data.count() >= 100
                }
                
                # 分布检查
                distribution_stats = {
                    'skewness': factor_data.skew(),
                    'kurtosis': factor_data.kurtosis(),
                    'normality_test': stats.jarque_bera(factor_data.dropna())[1] > 0.05 if factor_data.count() > 10 else False
                }
                
                validation_results[col] = {
                    'basic_stats': basic_stats,
                    'quality_checks': quality_checks,
                    'distribution_stats': distribution_stats,
                    'overall_quality': all(quality_checks.values())
                }
            
            self.logger.info(f"Validated {len(factors.columns)} factors")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error in validating factors: {str(e)}")
            return {}
    
    def calculate_factor_ic(self, factors: pd.DataFrame, returns: pd.DataFrame,
                           periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """计算因子IC"""
        try:
            ic_results = []
            
            for period in periods:
                # 计算前瞻收益
                future_returns = returns.shift(-period)
                
                for col in factors.columns:
                    if col in factors.columns:
                        # 对齐数据
                        aligned_factor = factors[col].dropna()
                        aligned_returns = future_returns.reindex(aligned_factor.index).dropna()
                        
                        # 找到共同的有效数据
                        common_index = aligned_factor.index.intersection(aligned_returns.index)
                        
                        if len(common_index) > 10:
                            factor_values = aligned_factor.loc[common_index]
                            return_values = aligned_returns.loc[common_index]
                            
                            # 计算Pearson相关系数
                            ic_pearson, p_value_pearson = pearsonr(factor_values, return_values)
                            
                            # 计算Spearman相关系数
                            ic_spearman, p_value_spearman = spearmanr(factor_values, return_values)
                            
                            ic_results.append({
                                'factor': col,
                                'period': period,
                                'ic_pearson': ic_pearson,
                                'ic_spearman': ic_spearman,
                                'p_value_pearson': p_value_pearson,
                                'p_value_spearman': p_value_spearman,
                                'sample_size': len(common_index)
                            })
            
            ic_df = pd.DataFrame(ic_results)
            self.logger.info(f"Calculated IC for {len(factors.columns)} factors across {len(periods)} periods")
            return ic_df
            
        except Exception as e:
            self.logger.error(f"Error in calculating factor IC: {str(e)}")
            return pd.DataFrame()
    
    def calculate_factor_turnover(self, factors: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:
        """计算因子换手率"""
        try:
            turnover_results = []
            
            for col in factors.columns:
                factor_data = factors[col].dropna()
                
                if len(factor_data) < 20:
                    continue
                
                # 计算分位数
                factor_ranks = factor_data.rank(pct=True)
                factor_quantiles = pd.cut(factor_ranks, quantiles, labels=False)
                
                # 计算换手率
                turnover_rates = []
                for i in range(1, len(factor_quantiles)):
                    if not pd.isna(factor_quantiles.iloc[i]) and not pd.isna(factor_quantiles.iloc[i-1]):
                        changed = factor_quantiles.iloc[i] != factor_quantiles.iloc[i-1]
                        turnover_rates.append(changed)
                
                avg_turnover = np.mean(turnover_rates) if turnover_rates else 0
                
                turnover_results.append({
                    'factor': col,
                    'avg_turnover': avg_turnover,
                    'sample_size': len(turnover_rates)
                })
            
            turnover_df = pd.DataFrame(turnover_results)
            self.logger.info(f"Calculated turnover for {len(factors.columns)} factors")
            return turnover_df
            
        except Exception as e:
            self.logger.error(f"Error in calculating factor turnover: {str(e)}")
            return pd.DataFrame()
    
    def check_factor_correlation(self, factors: pd.DataFrame, threshold: float = 0.8) -> Dict:
        """检查因子相关性"""
        try:
            correlation_matrix = factors.corr()
            
            # 找到高相关性的因子对
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > threshold:
                        high_corr_pairs.append({
                            'factor1': correlation_matrix.columns[i],
                            'factor2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            # 计算因子多重共线性
            multicollinearity_info = {}
            for col in factors.columns:
                other_factors = factors.drop(columns=[col])
                if len(other_factors.columns) > 0:
                    # 计算VIF (方差膨胀因子)
                    try:
                        X = other_factors.dropna()
                        y = factors[col].dropna()
                        
                        # 对齐数据
                        common_index = X.index.intersection(y.index)
                        if len(common_index) > 10:
                            X_aligned = X.loc[common_index]
                            y_aligned = y.loc[common_index]
                            
                            reg = LinearRegression()
                            reg.fit(X_aligned, y_aligned)
                            r_squared = reg.score(X_aligned, y_aligned)
                            
                            vif = 1 / (1 - r_squared) if r_squared < 0.999 else float('inf')
                            multicollinearity_info[col] = {
                                'vif': vif,
                                'r_squared': r_squared
                            }
                    except:
                        multicollinearity_info[col] = {
                            'vif': np.nan,
                            'r_squared': np.nan
                        }
            
            result = {
                'correlation_matrix': correlation_matrix,
                'high_corr_pairs': high_corr_pairs,
                'multicollinearity_info': multicollinearity_info,
                'summary': {
                    'total_factors': len(factors.columns),
                    'high_corr_pairs_count': len(high_corr_pairs),
                    'avg_correlation': correlation_matrix.abs().mean().mean()
                }
            }
            
            self.logger.info(f"Checked correlation for {len(factors.columns)} factors")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in checking factor correlation: {str(e)}")
            return {}
    
    def validate_time_series_properties(self, factors: pd.DataFrame) -> Dict:
        """验证时间序列特性"""
        try:
            results = {}
            
            for col in factors.columns:
                factor_data = factors[col].dropna()
                
                if len(factor_data) < 20:
                    continue
                
                # 平稳性检验
                try:
                    from statsmodels.tsa.stattools import adfuller
                    adf_result = adfuller(factor_data)
                    is_stationary = adf_result[1] < 0.05
                except:
                    is_stationary = None
                
                # 自相关检验
                autocorr_1 = factor_data.autocorr(lag=1)
                autocorr_5 = factor_data.autocorr(lag=5)
                
                # 趋势检验
                time_index = np.arange(len(factor_data))
                trend_corr, _ = pearsonr(time_index, factor_data.values)
                
                results[col] = {
                    'is_stationary': is_stationary,
                    'autocorr_lag1': autocorr_1,
                    'autocorr_lag5': autocorr_5,
                    'trend_correlation': trend_corr,
                    'has_trend': abs(trend_corr) > 0.1
                }
            
            self.logger.info(f"Validated time series properties for {len(factors.columns)} factors")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in validating time series properties: {str(e)}")
            return {}


class FactorSelector:
    """因子选择器"""
    
    def __init__(self, logger: Logger = None):
        self.logger = logger or Logger()
    
    def select_by_ic(self, factors: pd.DataFrame, returns: pd.DataFrame,
                     top_k: int = 20, ic_method: str = "pearson") -> List[str]:
        """基于IC选择因子"""
        try:
            validator = FactorValidator(self.logger)
            ic_results = validator.calculate_factor_ic(factors, returns)
            
            if ic_results.empty:
                return []
            
            # 选择IC方法
            ic_col = f"ic_{ic_method}"
            if ic_col not in ic_results.columns:
                ic_col = "ic_pearson"
            
            # 按IC绝对值排序
            ic_summary = ic_results.groupby('factor')[ic_col].agg(['mean', 'std', 'count']).reset_index()
            ic_summary['ic_abs_mean'] = ic_summary['mean'].abs()
            ic_summary['ic_stability'] = ic_summary['mean'] / ic_summary['std']
            
            # 综合评分
            ic_summary['score'] = ic_summary['ic_abs_mean'] * 0.7 + ic_summary['ic_stability'].abs() * 0.3
            
            # 选择top_k个因子
            selected_factors = ic_summary.nlargest(top_k, 'score')['factor'].tolist()
            
            self.logger.info(f"Selected {len(selected_factors)} factors using IC method")
            return selected_factors
            
        except Exception as e:
            self.logger.error(f"Error in selecting factors by IC: {str(e)}")
            return []
    
    def select_by_correlation(self, factors: pd.DataFrame, threshold: float = 0.8,
                             method: str = "keep_first") -> List[str]:
        """基于相关性选择因子"""
        try:
            correlation_matrix = factors.corr().abs()
            
            # 找到高相关性的因子对
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > threshold:
                        high_corr_pairs.append((i, j, correlation_matrix.iloc[i, j]))
            
            # 选择保留的因子
            factors_to_remove = set()
            
            if method == "keep_first":
                # 保留排序靠前的因子
                for i, j, corr in high_corr_pairs:
                    if j not in factors_to_remove:
                        factors_to_remove.add(j)
            elif method == "keep_variance":
                # 保留方差较大的因子
                for i, j, corr in high_corr_pairs:
                    var_i = factors.iloc[:, i].var()
                    var_j = factors.iloc[:, j].var()
                    if var_i > var_j:
                        factors_to_remove.add(j)
                    else:
                        factors_to_remove.add(i)
            
            selected_factors = [col for idx, col in enumerate(factors.columns) 
                              if idx not in factors_to_remove]
            
            self.logger.info(f"Selected {len(selected_factors)} factors after correlation filtering")
            return selected_factors
            
        except Exception as e:
            self.logger.error(f"Error in selecting factors by correlation: {str(e)}")
            return list(factors.columns)
    
    def select_by_pca(self, factors: pd.DataFrame, n_components: int = 20,
                      variance_threshold: float = 0.95) -> List[str]:
        """基于PCA选择因子"""
        try:
            # 标准化数据
            scaler = StandardScaler()
            factors_scaled = scaler.fit_transform(factors.fillna(0))
            
            # PCA分析
            pca = PCA(n_components=min(n_components, len(factors.columns)))
            pca.fit(factors_scaled)
            
            # 累计解释方差
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            
            # 选择达到方差阈值的主成分数量
            n_components_selected = np.argmax(cumulative_variance >= variance_threshold) + 1
            
            # 获取每个原始因子在主成分中的权重
            components = pca.components_[:n_components_selected]
            factor_importance = np.abs(components).mean(axis=0)
            
            # 选择重要性最高的因子
            factor_indices = np.argsort(factor_importance)[::-1][:n_components_selected]
            selected_factors = [factors.columns[idx] for idx in factor_indices]
            
            self.logger.info(f"Selected {len(selected_factors)} factors using PCA")
            return selected_factors
            
        except Exception as e:
            self.logger.error(f"Error in selecting factors by PCA: {str(e)}")
            return list(factors.columns)
    
    def select_by_mutual_information(self, factors: pd.DataFrame, returns: pd.DataFrame,
                                   top_k: int = 20) -> List[str]:
        """基于互信息选择因子"""
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            # 对齐数据
            aligned_data = factors.join(returns, how='inner').dropna()
            
            if len(aligned_data) < 50:
                return []
            
            X = aligned_data[factors.columns]
            y = aligned_data[returns.name] if hasattr(returns, 'name') else aligned_data.iloc[:, -1]
            
            # 计算互信息
            mi_scores = mutual_info_regression(X, y)
            
            # 选择top_k个因子
            factor_scores = list(zip(factors.columns, mi_scores))
            factor_scores.sort(key=lambda x: x[1], reverse=True)
            
            selected_factors = [factor for factor, score in factor_scores[:top_k]]
            
            self.logger.info(f"Selected {len(selected_factors)} factors using mutual information")
            return selected_factors
            
        except Exception as e:
            self.logger.error(f"Error in selecting factors by mutual information: {str(e)}")
            return []