#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测模块
提供完整的回测引擎，包括交易模拟、持仓跟踪、再平衡调度等功能
"""

from .backtest_engine import (
    BacktestConfig,
    TradeSimulator,
    PositionTracker,
    RebalanceScheduler,
    MultiFrequencyBacktester,
    BacktestEngine
)

__all__ = [
    # 配置
    'BacktestConfig',
    
    # 核心组件
    'TradeSimulator',
    'PositionTracker',
    'RebalanceScheduler',
    'MultiFrequencyBacktester',
    
    # 主要接口
    'BacktestEngine'
] 