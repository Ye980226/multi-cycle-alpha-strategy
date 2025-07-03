#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略配置管理模块
"""

import yaml
import json
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class DataConfig:
    """数据配置"""
    data_source: str = "akshare"  # 数据源: akshare, tushare, wind等
    frequency: str = "1min"  # 数据频率: 1min, 5min, 15min, 30min, 1h, 1d
    universe: List[str] = None  # 股票池
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"
    benchmark: str = "000300.SH"  # 基准指数
    
    def __post_init__(self):
        if self.universe is None:
            self.universe = ["HS300", "ZZ500"]  # 默认股票池


@dataclass 
class FactorConfig:
    """因子配置"""
    factor_groups: List[str] = None  # 因子组: technical, fundamental, sentiment
    lookback_periods: List[int] = None  # 回望期: [5, 10, 20, 60]
    factor_update_frequency: str = "1d"  # 因子更新频率
    factor_neutralization: bool = True  # 因子中性化
    factor_standardization: str = "zscore"  # 标准化方法: zscore, minmax, rank
    
    def __post_init__(self):
        if self.factor_groups is None:
            self.factor_groups = ["technical", "fundamental", "sentiment"]
        if self.lookback_periods is None:
            self.lookback_periods = [5, 10, 20, 60]


@dataclass
class ModelConfig:
    """模型配置"""
    model_types: List[str] = None  # 模型类型
    ensemble_method: str = "weighted_average"  # 集成方法
    retrain_frequency: str = "1w"  # 重训练频率
    validation_method: str = "time_series_split"  # 验证方法
    max_features: int = 50  # 最大特征数
    
    def __post_init__(self):
        if self.model_types is None:
            self.model_types = ["lightgbm", "xgboost", "linear", "neural_network"]


@dataclass
class PortfolioConfig:
    """组合配置"""
    optimization_method: str = "mean_variance"  # 优化方法
    rebalance_frequency: str = "1d"  # 再平衡频率
    max_position: float = 0.05  # 最大持仓比例
    max_turnover: float = 0.5  # 最大换手率
    transaction_cost: float = 0.001  # 交易成本
    risk_budget: float = 0.15  # 风险预算
    cycles: List[str] = None  # 多周期设置
    
    def __post_init__(self):
        if self.cycles is None:
            self.cycles = ["intraday", "daily", "weekly"]


@dataclass
class RiskConfig:
    """风险配置"""
    max_drawdown: float = 0.1  # 最大回撤
    var_confidence: float = 0.95  # VaR置信度
    sector_exposure_limit: float = 0.3  # 行业暴露限制
    single_stock_limit: float = 0.05  # 单股持仓限制
    beta_range: tuple = (0.8, 1.2)  # Beta范围
    
    
class StrategyConfig:
    """策略总配置"""
    
    def __init__(self, config_path: str = None):
        """初始化配置"""
        pass
    
    def load_config(self, config_path: str):
        """加载配置文件"""
        pass
    
    def save_config(self, config_path: str):
        """保存配置文件"""
        pass
    
    def get_data_config(self) -> DataConfig:
        """获取数据配置"""
        pass
    
    def get_factor_config(self) -> FactorConfig:
        """获取因子配置"""
        pass
    
    def get_model_config(self) -> ModelConfig:
        """获取模型配置"""
        pass
    
    def get_portfolio_config(self) -> PortfolioConfig:
        """获取组合配置"""
        pass
    
    def get_risk_config(self) -> RiskConfig:
        """获取风险配置"""
        pass
    
    def validate_config(self) -> bool:
        """验证配置有效性"""
        pass
    
    def update_config(self, section: str, **kwargs):
        """更新配置"""
        pass 