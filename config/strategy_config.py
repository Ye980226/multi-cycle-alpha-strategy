#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略配置管理模块
"""

import yaml
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime

@dataclass
class DataConfig:
    """数据配置"""
    data_source: str = "akshare"  # 数据源: akshare, tushare, wind等
    frequency: str = "1min"  # 数据频率: 1min, 5min, 15min, 30min, 1h, 1d
    universe: List[str] = None  # 股票池
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"
    benchmark: str = "000300.SH"  # 基准指数
    
    # 真实数据源配置
    akshare: Dict[str, Any] = None  # akshare配置
    tushare: Dict[str, Any] = None  # tushare配置
    minute_data: Dict[str, Any] = None  # 分钟级数据配置
    test: Dict[str, Any] = None  # 测试配置
    
    def __post_init__(self):
        if self.universe is None:
            self.universe = ["HS300", "ZZ500"]  # 默认股票池
        
        # 设置默认的数据源配置
        if self.akshare is None:
            self.akshare = {
                "timeout": 30,
                "retry_times": 3,
                "retry_delay": 1,
                "cache_enabled": True,
                "cache_expire_hours": 24
            }
        
        if self.tushare is None:
            self.tushare = {
                "token": "",
                "timeout": 30,
                "retry_times": 3,
                "cache_enabled": True
            }
        
        if self.minute_data is None:
            self.minute_data = {
                "trading_sessions": {
                    "morning": ["09:30", "11:30"],
                    "afternoon": ["13:00", "15:00"]
                },
                "exclude_weekends": True,
                "exclude_holidays": True,
                "fill_missing_method": "forward_fill"
            }
        
        if self.test is None:
            self.test = {
                "symbols": ["000001", "000002", "600000", "600036"],
                "test_days": 30,
                "enable_simulation": True
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataConfig':
        """从字典创建配置对象"""
        return cls(**data)

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
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FactorConfig':
        """从字典创建配置对象"""
        return cls(**data)

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
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """从字典创建配置对象"""
        return cls(**data)

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
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortfolioConfig':
        """从字典创建配置对象"""
        return cls(**data)

@dataclass
class RiskConfig:
    """风险配置"""
    max_drawdown: float = 0.1  # 最大回撤
    var_confidence: float = 0.95  # VaR置信度
    sector_exposure_limit: float = 0.3  # 行业暴露限制
    single_stock_limit: float = 0.05  # 单股持仓限制
    beta_range: tuple = (0.8, 1.2)  # Beta范围
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 将tuple转换为list，便于JSON序列化
        data['beta_range'] = list(data['beta_range'])
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskConfig':
        """从字典创建配置对象"""
        if 'beta_range' in data and isinstance(data['beta_range'], list):
            data['beta_range'] = tuple(data['beta_range'])
        return cls(**data)

@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 1000000  # 初始资金
    benchmark: str = "000300.SH"  # 基准指数
    transaction_cost: float = 0.001  # 交易成本
    slippage: float = 0.0005  # 滑点
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BacktestConfig':
        """从字典创建配置对象"""
        return cls(**data)

@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"  # 日志级别
    file_path: str = "./logs/strategy.log"  # 日志文件路径
    max_file_size: str = "10MB"  # 最大文件大小
    backup_count: int = 5  # 备份文件数量
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LoggingConfig':
        """从字典创建配置对象"""
        return cls(**data)

@dataclass
class MonitoringConfig:
    """监控配置"""
    alert_thresholds: Dict[str, float] = None  # 告警阈值
    notification: Dict[str, bool] = None  # 通知配置
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "max_drawdown": 0.08,
                "min_sharpe": 0.5,
                "max_volatility": 0.25
            }
        if self.notification is None:
            self.notification = {
                "email": True,
                "sms": False,
                "webhook": False
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MonitoringConfig':
        """从字典创建配置对象"""
        return cls(**data)

class StrategyConfig:
    """策略总配置"""
    
    def __init__(self, config_path: str = None):
        """初始化配置"""
        # 初始化所有配置对象
        self.data_config = DataConfig()
        self.factor_config = FactorConfig()
        self.model_config = ModelConfig()
        self.portfolio_config = PortfolioConfig()
        self.risk_config = RiskConfig()
        self.backtest_config = BacktestConfig()
        self.logging_config = LoggingConfig()
        self.monitoring_config = MonitoringConfig()
        
        # 配置文件路径
        self.config_path = config_path or "./config/default_config.yaml"
        
        # 加载配置
        if os.path.exists(self.config_path):
            self.load_config(self.config_path)
        else:
            print(f"配置文件 {self.config_path} 不存在，使用默认配置")
    
    def load_config(self, config_path: str):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                print("配置文件为空，使用默认配置")
                return
            
            # 更新各个配置
            if 'data' in config_data:
                self.data_config = DataConfig.from_dict(config_data['data'])
            
            if 'factors' in config_data:
                self.factor_config = FactorConfig.from_dict(config_data['factors'])
            
            if 'models' in config_data:
                self.model_config = ModelConfig.from_dict(config_data['models'])
            
            if 'portfolio' in config_data:
                self.portfolio_config = PortfolioConfig.from_dict(config_data['portfolio'])
            
            if 'risk' in config_data:
                self.risk_config = RiskConfig.from_dict(config_data['risk'])
            
            if 'backtest' in config_data:
                self.backtest_config = BacktestConfig.from_dict(config_data['backtest'])
            
            if 'logging' in config_data:
                self.logging_config = LoggingConfig.from_dict(config_data['logging'])
            
            if 'monitoring' in config_data:
                self.monitoring_config = MonitoringConfig.from_dict(config_data['monitoring'])
            
            print(f"配置文件 {config_path} 加载成功")
            
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            print("使用默认配置")
    
    def save_config(self, config_path: str):
        """保存配置文件"""
        try:
            config_data = {
                'data': self.data_config.to_dict(),
                'factors': self.factor_config.to_dict(),
                'models': self.model_config.to_dict(),
                'portfolio': self.portfolio_config.to_dict(),
                'risk': self.risk_config.to_dict(),
                'backtest': self.backtest_config.to_dict(),
                'logging': self.logging_config.to_dict(),
                'monitoring': self.monitoring_config.to_dict()
            }
            
            # 创建目录
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            print(f"配置文件已保存到 {config_path}")
            
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def get_data_config(self) -> DataConfig:
        """获取数据配置"""
        return self.data_config
    
    def get_factor_config(self) -> FactorConfig:
        """获取因子配置"""
        return self.factor_config
    
    def get_model_config(self) -> ModelConfig:
        """获取模型配置"""
        return self.model_config
    
    def get_portfolio_config(self) -> PortfolioConfig:
        """获取组合配置"""
        return self.portfolio_config
    
    def get_risk_config(self) -> RiskConfig:
        """获取风险配置"""
        return self.risk_config
    
    def get_backtest_config(self) -> BacktestConfig:
        """获取回测配置"""
        return self.backtest_config
    
    def get_logging_config(self) -> LoggingConfig:
        """获取日志配置"""
        return self.logging_config
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """获取监控配置"""
        return self.monitoring_config
    
    def validate_config(self) -> bool:
        """验证配置有效性"""
        try:
            # 验证数据配置
            if not self.data_config.data_source:
                print("错误：数据源不能为空")
                return False
            
            if not self.data_config.frequency:
                print("错误：数据频率不能为空")
                return False
            
            if not self.data_config.universe:
                print("错误：股票池不能为空")
                return False
            
            # 验证因子配置
            if not self.factor_config.factor_groups:
                print("错误：因子组不能为空")
                return False
            
            if not self.factor_config.lookback_periods:
                print("错误：回望期不能为空")
                return False
            
            # 验证模型配置
            if not self.model_config.model_types:
                print("错误：模型类型不能为空")
                return False
            
            # 验证组合配置
            if self.portfolio_config.max_position <= 0 or self.portfolio_config.max_position > 1:
                print("错误：最大持仓比例必须在(0,1]范围内")
                return False
            
            if self.portfolio_config.max_turnover <= 0 or self.portfolio_config.max_turnover > 1:
                print("错误：最大换手率必须在(0,1]范围内")
                return False
            
            # 验证风险配置
            if self.risk_config.max_drawdown <= 0 or self.risk_config.max_drawdown > 1:
                print("错误：最大回撤必须在(0,1]范围内")
                return False
            
            if self.risk_config.var_confidence <= 0 or self.risk_config.var_confidence >= 1:
                print("错误：VaR置信度必须在(0,1)范围内")
                return False
            
            # 验证回测配置
            if self.backtest_config.initial_capital <= 0:
                print("错误：初始资金必须大于0")
                return False
            
            print("配置验证通过")
            return True
            
        except Exception as e:
            print(f"配置验证失败: {e}")
            return False
    
    def update_config(self, section: str, **kwargs):
        """更新配置"""
        try:
            if section == "data":
                for key, value in kwargs.items():
                    if hasattr(self.data_config, key):
                        setattr(self.data_config, key, value)
            
            elif section == "factors":
                for key, value in kwargs.items():
                    if hasattr(self.factor_config, key):
                        setattr(self.factor_config, key, value)
            
            elif section == "models":
                for key, value in kwargs.items():
                    if hasattr(self.model_config, key):
                        setattr(self.model_config, key, value)
            
            elif section == "portfolio":
                for key, value in kwargs.items():
                    if hasattr(self.portfolio_config, key):
                        setattr(self.portfolio_config, key, value)
            
            elif section == "risk":
                for key, value in kwargs.items():
                    if hasattr(self.risk_config, key):
                        setattr(self.risk_config, key, value)
            
            elif section == "backtest":
                for key, value in kwargs.items():
                    if hasattr(self.backtest_config, key):
                        setattr(self.backtest_config, key, value)
            
            elif section == "logging":
                for key, value in kwargs.items():
                    if hasattr(self.logging_config, key):
                        setattr(self.logging_config, key, value)
            
            elif section == "monitoring":
                for key, value in kwargs.items():
                    if hasattr(self.monitoring_config, key):
                        setattr(self.monitoring_config, key, value)
            
            else:
                print(f"未知的配置段: {section}")
                return
            
            print(f"配置段 {section} 更新成功")
            
        except Exception as e:
            print(f"更新配置失败: {e}")
    
    def get_all_config(self) -> Dict[str, Any]:
        """获取所有配置"""
        return {
            'data': self.data_config.to_dict(),
            'factors': self.factor_config.to_dict(),
            'models': self.model_config.to_dict(),
            'portfolio': self.portfolio_config.to_dict(),
            'risk': self.risk_config.to_dict(),
            'backtest': self.backtest_config.to_dict(),
            'logging': self.logging_config.to_dict(),
            'monitoring': self.monitoring_config.to_dict()
        }
    
    def print_config(self):
        """打印配置信息"""
        print("\n========== 策略配置信息 ==========")
        print(f"数据配置: {self.data_config}")
        print(f"因子配置: {self.factor_config}")
        print(f"模型配置: {self.model_config}")
        print(f"组合配置: {self.portfolio_config}")
        print(f"风险配置: {self.risk_config}")
        print(f"回测配置: {self.backtest_config}")
        print(f"日志配置: {self.logging_config}")
        print(f"监控配置: {self.monitoring_config}")
        print("=====================================")
    
    def create_backup(self, backup_path: str = None):
        """创建配置备份"""
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"./config/backup/config_backup_{timestamp}.yaml"
        
        self.save_config(backup_path)
        print(f"配置备份已创建: {backup_path}")
        return backup_path 