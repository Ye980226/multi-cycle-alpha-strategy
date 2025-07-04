#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号生成模块
提供多种信号生成方法，包括阈值、排序、机器学习、复合信号等
"""

from .signal_generator import (
    BaseSignalGenerator,
    ThresholdSignalGenerator,
    RankingSignalGenerator,
    MLSignalGenerator,
    CompositeSignalGenerator,
    MultiTimeframeSignalGenerator,
    SignalFilter,
    SignalValidator,
    SignalAnalyzer,
    SignalGenerator
)

__all__ = [
    'BaseSignalGenerator',
    'ThresholdSignalGenerator',
    'RankingSignalGenerator', 
    'MLSignalGenerator',
    'CompositeSignalGenerator',
    'MultiTimeframeSignalGenerator',
    'SignalFilter',
    'SignalValidator',
    'SignalAnalyzer',
    'SignalGenerator'
] 