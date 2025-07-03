#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多周期Alpha策略框架

这是一个完整的量化投资策略框架，专为多周期alpha策略设计。
框架包含数据管理、因子生成、模型训练、信号生成、组合优化、
回测分析、风险管理、交易执行和监控等完整功能模块。

主要特性：
- 支持分钟级高频数据处理
- 多时间周期因子计算和信号生成
- 先进的机器学习模型集成
- 灵活的组合优化算法
- 全面的风险管理系统
- 专业的回测和分析工具
- 实时监控和告警系统

作者: AI Assistant
版本: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

# 导入主要类
from .main import MultiCycleAlphaStrategy
from .config.strategy_config import StrategyConfig
from .data.data_manager import DataManager
from .factors.factor_engine import FactorEngine
from .models.model_manager import ModelManager
from .signals.signal_generator import SignalGenerator
from .portfolio.portfolio_optimizer import PortfolioOptimizer
from .backtest.backtest_engine import BacktestEngine
from .risk.risk_manager import RiskManager
from .execution.trade_executor import TradeExecutor
from .monitoring.performance_monitor import PerformanceMonitor
from .utils.logger import Logger

__all__ = [
    "MultiCycleAlphaStrategy",
    "StrategyConfig", 
    "DataManager",
    "FactorEngine",
    "ModelManager", 
    "SignalGenerator",
    "PortfolioOptimizer",
    "BacktestEngine",
    "RiskManager",
    "TradeExecutor",
    "PerformanceMonitor",
    "Logger"
] 