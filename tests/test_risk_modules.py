#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风险管理模块新功能测试文件
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 导入被测试的模块
from risk.risk_manager import (
    PositionLimitManager, 
    LeverageManager, 
    CorrelationMonitor, 
    LiquidityRiskManager, 
    StressTestManager, 
    RegimeDetector
)


class TestPositionLimitManager(unittest.TestCase):
    """测试仓位限制管理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = {
            'max_single_position': 0.1,
            'max_long_position': 0.08,
            'max_short_position': 0.05,
            'min_position_size': 0.01,
            'max_positions': 50,
            'sector_limits': {
                'technology': 0.3,
                'finance': 0.2,
                'healthcare': 0.15
            }
        }
        self.manager = PositionLimitManager(self.config)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.manager)
        self.assertEqual(self.manager.max_single_position, 0.1)
        self.assertEqual(self.manager.max_long_position, 0.08)
        self.assertEqual(self.manager.max_short_position, 0.05)
    
    def test_check_single_position_limit(self):
        """测试单一仓位限制检查"""
        # 创建测试权重
        weights = pd.Series([0.05, 0.03, -0.02, 0.15], index=['A', 'B', 'C', 'D'])
        
        # 检查仓位限制
        results = self.manager.check_single_position_limit(weights)
        
        # 验证结果
        self.assertIsInstance(results, dict)
        self.assertIn('single_position_check', results)
        self.assertIn('long_position_check', results)
        self.assertIn('short_position_check', results)
        self.assertIn('violations', results)
        
        # 检查违规情况
        self.assertFalse(results['single_position_check'])  # D违反了10%限制
        self.assertFalse(results['long_position_check'])  # D违反了8%多头限制
    
    def test_check_sector_limits(self):
        """测试行业限制检查"""
        # 创建测试权重
        weights = pd.Series([0.15, 0.10, 0.05, 0.08], index=['TECH1', 'TECH2', 'FIN1', 'HEALTH1'])
        
        # 创建行业映射
        sector_mapping = {
            'TECH1': 'technology',
            'TECH2': 'technology',
            'FIN1': 'finance',
            'HEALTH1': 'healthcare'
        }
        
        # 检查行业限制
        results = self.manager.check_sector_limits(weights, sector_mapping)
        
        # 验证结果
        self.assertIsInstance(results, dict)
        self.assertIn('sector_check', results)
        self.assertIn('violations', results)
        
        # 检查违规情况 (technology = 0.15 + 0.10 = 0.25 < 0.3, 应该通过)
        self.assertTrue(results['sector_check'])
    
    def test_check_concentration_limits(self):
        """测试集中度限制检查"""
        # 创建测试权重
        weights = pd.Series([0.3, 0.2, 0.15, 0.1, 0.25], index=['A', 'B', 'C', 'D', 'E'])
        
        # 检查集中度限制
        result = self.manager.check_concentration_limits(weights, top_n=3, max_concentration=0.5)
        
        # 验证结果
        self.assertIsInstance(result, bool)
        # 前3大仓位: 0.3 + 0.25 + 0.2 = 0.75 > 0.5，应该违反限制
        self.assertFalse(result)
    
    def test_apply_position_limits(self):
        """测试应用仓位限制"""
        # 创建违反限制的权重
        weights = pd.Series([0.15, 0.12, -0.08, 0.05], index=['A', 'B', 'C', 'D'])
        
        # 应用限制
        adjusted_weights = self.manager.apply_position_limits(weights)
        
        # 验证结果
        self.assertIsInstance(adjusted_weights, pd.Series)
        self.assertTrue(all(adjusted_weights.abs() <= self.config['max_single_position']))
        self.assertTrue(all(adjusted_weights[adjusted_weights > 0] <= self.config['max_long_position']))
        self.assertTrue(all(adjusted_weights[adjusted_weights < 0] >= -self.config['max_short_position']))
    
    def test_update_limits(self):
        """测试更新限制"""
        new_limits = {'max_single_position': 0.15}
        self.manager.update_limits(new_limits)
        
        self.assertEqual(self.manager.max_single_position, 0.15)
    
    def test_calculate_position_utilization(self):
        """测试仓位利用率计算"""
        weights = pd.Series([0.05, 0.03, -0.02, 0.04], index=['A', 'B', 'C', 'D'])
        
        utilization = self.manager.calculate_position_utilization(weights)
        
        self.assertIsInstance(utilization, dict)
        self.assertIn('position_utilization', utilization)
        self.assertIn('long_utilization', utilization)
        self.assertIn('short_utilization', utilization)


class TestLeverageManager(unittest.TestCase):
    """测试杠杆管理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.manager = LeverageManager(max_leverage=2.0, max_gross_exposure=1.5, max_net_exposure=0.8)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.manager)
        self.assertEqual(self.manager.max_leverage, 2.0)
        self.assertEqual(self.manager.max_gross_exposure, 1.5)
        self.assertEqual(self.manager.max_net_exposure, 0.8)
    
    def test_calculate_leverage(self):
        """测试杠杆计算"""
        weights = pd.Series([0.6, 0.3, -0.2, 0.1], index=['A', 'B', 'C', 'D'])
        
        leverage = self.manager.calculate_leverage(weights)
        
        self.assertIsInstance(leverage, float)
        self.assertGreater(leverage, 0)
        # 总暴露 = 0.6 + 0.3 + 0.2 + 0.1 = 1.2
        self.assertAlmostEqual(leverage, 1.2, places=2)
    
    def test_calculate_gross_exposure(self):
        """测试总暴露计算"""
        weights = pd.Series([0.4, 0.3, -0.2, 0.1], index=['A', 'B', 'C', 'D'])
        
        gross_exposure = self.manager.calculate_gross_exposure(weights)
        
        self.assertIsInstance(gross_exposure, float)
        # 总暴露 = |0.4| + |0.3| + |-0.2| + |0.1| = 1.0
        self.assertAlmostEqual(gross_exposure, 1.0, places=2)
    
    def test_calculate_net_exposure(self):
        """测试净暴露计算"""
        weights = pd.Series([0.4, 0.3, -0.2, 0.1], index=['A', 'B', 'C', 'D'])
        
        net_exposure = self.manager.calculate_net_exposure(weights)
        
        self.assertIsInstance(net_exposure, float)
        # 净暴露 = 0.4 + 0.3 - 0.2 + 0.1 = 0.6
        self.assertAlmostEqual(net_exposure, 0.6, places=2)
    
    def test_adjust_leverage(self):
        """测试杠杆调整"""
        # 创建超过限制的权重
        weights = pd.Series([0.8, 0.6, -0.4, 0.3], index=['A', 'B', 'C', 'D'])
        
        adjusted_weights = self.manager.adjust_leverage(weights)
        
        self.assertIsInstance(adjusted_weights, pd.Series)
        # 调整后的权重应该满足限制
        adjusted_gross = self.manager.calculate_gross_exposure(adjusted_weights)
        self.assertLessEqual(adjusted_gross, self.manager.max_gross_exposure)
    
    def test_check_leverage_limits(self):
        """测试杠杆限制检查"""
        weights = pd.Series([0.5, 0.4, -0.3, 0.2], index=['A', 'B', 'C', 'D'])
        
        results = self.manager.check_leverage_limits(weights)
        
        self.assertIsInstance(results, dict)
        self.assertIn('gross_exposure_check', results)
        self.assertIn('net_exposure_check', results)
        self.assertIn('leverage_check', results)
        self.assertIn('all_checks_passed', results)
        self.assertIn('metrics', results)
    
    def test_update_leverage_limits(self):
        """测试更新杠杆限制"""
        self.manager.update_leverage_limits(max_leverage=3.0)
        
        self.assertEqual(self.manager.max_leverage, 3.0)
    
    def test_get_leverage_breakdown(self):
        """测试杠杆分解"""
        weights = pd.Series([0.4, 0.3, -0.2, 0.1], index=['A', 'B', 'C', 'D'])
        
        breakdown = self.manager.get_leverage_breakdown(weights)
        
        self.assertIsInstance(breakdown, dict)
        self.assertIn('long_exposure', breakdown)
        self.assertIn('short_exposure', breakdown)
        self.assertIn('long_short_ratio', breakdown)
        self.assertIn('position_distribution', breakdown)


class TestCorrelationMonitor(unittest.TestCase):
    """测试相关性监控器"""
    
    def setUp(self):
        """设置测试环境"""
        self.monitor = CorrelationMonitor()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.monitor)
        self.assertIsInstance(self.monitor.correlation_history, list)
        self.assertIsInstance(self.monitor.alerts, list)
    
    def test_calculate_rolling_correlation(self):
        """测试滚动相关性计算"""
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        returns = pd.DataFrame({
            'A': np.random.normal(0, 0.02, 100),
            'B': np.random.normal(0, 0.025, 100),
            'C': np.random.normal(0, 0.015, 100)
        }, index=dates)
        
        # 计算滚动相关性
        rolling_corr = self.monitor.calculate_rolling_correlation(returns, window=30)
        
        self.assertIsInstance(rolling_corr, pd.DataFrame)
        self.assertGreater(len(rolling_corr), 0)
        # 检查相关性值在-1到1之间
        self.assertTrue(rolling_corr.abs().max().max() <= 1.0)
    
    def test_detect_correlation_regime_change(self):
        """测试相关性状态变化检测"""
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        correlations = pd.DataFrame({
            'A_B': np.random.uniform(0.2, 0.8, 100),
            'A_C': np.random.uniform(0.1, 0.6, 100),
            'B_C': np.random.uniform(0.3, 0.7, 100)
        }, index=dates)
        
        # 检测状态变化
        regime_changes = self.monitor.detect_correlation_regime_change(correlations)
        
        self.assertIsInstance(regime_changes, pd.Series)
        self.assertEqual(len(regime_changes), len(correlations))
    
    def test_calculate_average_correlation(self):
        """测试平均相关性计算"""
        # 创建测试相关性矩阵
        corr_matrix = pd.DataFrame({
            'A': [1.0, 0.3, 0.5],
            'B': [0.3, 1.0, 0.4],
            'C': [0.5, 0.4, 1.0]
        }, index=['A', 'B', 'C'])
        
        avg_corr = self.monitor.calculate_average_correlation(corr_matrix)
        
        self.assertIsInstance(avg_corr, float)
        self.assertGreater(avg_corr, 0)
        self.assertLess(avg_corr, 1)
    
    def test_correlation_clustering(self):
        """测试相关性聚类"""
        # 创建测试相关性矩阵
        corr_matrix = pd.DataFrame({
            'A': [1.0, 0.8, 0.2, 0.1],
            'B': [0.8, 1.0, 0.1, 0.3],
            'C': [0.2, 0.1, 1.0, 0.7],
            'D': [0.1, 0.3, 0.7, 1.0]
        }, index=['A', 'B', 'C', 'D'])
        
        clustering_result = self.monitor.correlation_clustering(corr_matrix)
        
        self.assertIsInstance(clustering_result, dict)
        self.assertIn('clusters', clustering_result)
        self.assertIn('n_clusters', clustering_result)
    
    def test_monitor_correlation_breakdown(self):
        """测试相关性崩溃监控"""
        # 创建测试相关性矩阵
        corr_matrix = pd.DataFrame({
            'A': [1.0, 0.9, 0.8, 0.85],
            'B': [0.9, 1.0, 0.85, 0.9],
            'C': [0.8, 0.85, 1.0, 0.87],
            'D': [0.85, 0.9, 0.87, 1.0]
        }, index=['A', 'B', 'C', 'D'])
        
        breakdown_result = self.monitor.monitor_correlation_breakdown(corr_matrix, threshold=0.8)
        
        self.assertIsInstance(breakdown_result, dict)
        self.assertIn('high_correlation_pairs', breakdown_result)
        self.assertIn('breakdown_risk', breakdown_result)
        self.assertIn('risk_level', breakdown_result)
    
    def test_calculate_correlation_stability(self):
        """测试相关性稳定性计算"""
        # 创建多个相关性矩阵
        corr_matrices = []
        for i in range(5):
            corr_matrix = pd.DataFrame({
                'A': [1.0, 0.3 + i*0.1, 0.5],
                'B': [0.3 + i*0.1, 1.0, 0.4],
                'C': [0.5, 0.4, 1.0]
            }, index=['A', 'B', 'C'])
            corr_matrices.append(corr_matrix)
        
        stability_result = self.monitor.calculate_correlation_stability(corr_matrices)
        
        self.assertIsInstance(stability_result, dict)
        self.assertIn('stability_score', stability_result)
        self.assertIn('stability_level', stability_result)


class TestLiquidityRiskManager(unittest.TestCase):
    """测试流动性风险管理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.manager = LiquidityRiskManager()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.manager)
        self.assertIsInstance(self.manager.liquidity_thresholds, dict)
        self.assertIsInstance(self.manager.liquidity_history, list)
    
    def test_calculate_liquidity_score(self):
        """测试流动性评分计算"""
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        volume_data = pd.DataFrame({
            'A': np.random.uniform(1000000, 5000000, 100),
            'B': np.random.uniform(500000, 2000000, 100),
            'C': np.random.uniform(800000, 3000000, 100)
        }, index=dates)
        
        market_cap = pd.DataFrame({
            'A': np.random.uniform(1e9, 5e9, 100),
            'B': np.random.uniform(500e6, 2e9, 100),
            'C': np.random.uniform(800e6, 3e9, 100)
        }, index=dates)
        
        # 计算流动性评分
        liquidity_scores = self.manager.calculate_liquidity_score(volume_data, market_cap)
        
        self.assertIsInstance(liquidity_scores, pd.DataFrame)
        if not liquidity_scores.empty:
            self.assertTrue(all(liquidity_scores.min() >= 0))
            self.assertTrue(all(liquidity_scores.max() <= 1))
    
    def test_estimate_market_impact(self):
        """测试市场冲击估计"""
        # 创建测试数据
        trade_size = pd.Series([100000, 50000, 80000], index=['A', 'B', 'C'])
        adv = pd.Series([1000000, 500000, 800000], index=['A', 'B', 'C'])
        
        # 估计市场冲击
        market_impact = self.manager.estimate_market_impact(trade_size, adv)
        
        self.assertIsInstance(market_impact, pd.Series)
        self.assertEqual(len(market_impact), len(trade_size))
    
    def test_calculate_participation_rate(self):
        """测试参与率计算"""
        # 创建测试数据
        trade_volume = pd.Series([10000, 5000, 8000], index=['A', 'B', 'C'])
        market_volume = pd.Series([100000, 50000, 80000], index=['A', 'B', 'C'])
        
        # 计算参与率
        participation_rate = self.manager.calculate_participation_rate(trade_volume, market_volume)
        
        self.assertIsInstance(participation_rate, pd.Series)
        self.assertTrue(all(participation_rate >= 0))
        self.assertTrue(all(participation_rate <= 1))
    
    def test_liquidity_adjusted_weights(self):
        """测试流动性调整权重"""
        # 创建测试数据
        target_weights = pd.Series([0.4, 0.3, 0.3], index=['A', 'B', 'C'])
        liquidity_scores = pd.Series([0.8, 0.5, 0.7], index=['A', 'B', 'C'])
        
        # 调整权重
        adjusted_weights = self.manager.liquidity_adjusted_weights(target_weights, liquidity_scores)
        
        self.assertIsInstance(adjusted_weights, pd.Series)
        self.assertAlmostEqual(adjusted_weights.sum(), 1.0, places=5)
    
    def test_estimate_liquidation_time(self):
        """测试清仓时间估计"""
        # 创建测试数据
        position_sizes = pd.Series([100000, 50000, 80000], index=['A', 'B', 'C'])
        adv = pd.Series([1000000, 500000, 800000], index=['A', 'B', 'C'])
        
        # 估计清仓时间
        liquidation_time = self.manager.estimate_liquidation_time(position_sizes, adv)
        
        self.assertIsInstance(liquidation_time, pd.Series)
        self.assertTrue(all(liquidation_time >= 0))
    
    def test_calculate_liquidity_risk_metrics(self):
        """测试流动性风险指标计算"""
        # 创建测试数据
        portfolio_weights = pd.Series([0.4, 0.3, 0.3], index=['A', 'B', 'C'])
        liquidity_scores = pd.Series([0.8, 0.5, 0.7], index=['A', 'B', 'C'])
        
        # 计算风险指标
        risk_metrics = self.manager.calculate_liquidity_risk_metrics(portfolio_weights, liquidity_scores)
        
        self.assertIsInstance(risk_metrics, dict)
        if risk_metrics:
            self.assertIn('weighted_liquidity_score', risk_metrics)
            self.assertIn('liquidity_diversification', risk_metrics)


class TestStressTestManager(unittest.TestCase):
    """测试压力测试管理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.manager = StressTestManager()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.manager)
        self.assertIsInstance(self.manager.default_scenarios, dict)
        self.assertIsInstance(self.manager.test_results, list)
    
    def test_scenario_stress_test(self):
        """测试情景压力测试"""
        # 创建测试数据
        portfolio_weights = pd.Series([0.4, 0.3, 0.3], index=['EQUITY1', 'BOND1', 'COMMODITY1'])
        
        # 使用默认情景
        stress_results = self.manager.scenario_stress_test(portfolio_weights, {})
        
        self.assertIsInstance(stress_results, dict)
        self.assertIn('summary', stress_results)
        if 'market_crash' in stress_results:
            self.assertIn('portfolio_return', stress_results['market_crash'])
    
    def test_factor_shock_test(self):
        """测试因子冲击测试"""
        # 创建测试数据
        portfolio_weights = pd.Series([0.4, 0.3, 0.3], index=['A', 'B', 'C'])
        factor_exposures = pd.DataFrame({
            'market': [1.0, 1.2, 0.8],
            'size': [0.5, -0.2, 0.3],
            'value': [-0.1, 0.4, 0.2]
        }, index=['A', 'B', 'C'])
        factor_shocks = pd.Series([-0.2, 0.1, -0.1], index=['market', 'size', 'value'])
        
        # 执行因子冲击测试
        shock_results = self.manager.factor_shock_test(portfolio_weights, factor_exposures, factor_shocks)
        
        self.assertIsInstance(shock_results, dict)
        if 'total_impact' in shock_results:
            self.assertIn('total_impact', shock_results['total_impact'])
    
    def test_correlation_breakdown_test(self):
        """测试相关性崩溃测试"""
        # 创建测试数据
        portfolio_weights = pd.Series([0.4, 0.3, 0.3], index=['A', 'B', 'C'])
        normal_corr = pd.DataFrame({
            'A': [1.0, 0.3, 0.2],
            'B': [0.3, 1.0, 0.4],
            'C': [0.2, 0.4, 1.0]
        }, index=['A', 'B', 'C'])
        stress_corr = pd.DataFrame({
            'A': [1.0, 0.8, 0.8],
            'B': [0.8, 1.0, 0.8],
            'C': [0.8, 0.8, 1.0]
        }, index=['A', 'B', 'C'])
        
        # 执行相关性崩溃测试
        breakdown_results = self.manager.correlation_breakdown_test(portfolio_weights, normal_corr, stress_corr)
        
        self.assertIsInstance(breakdown_results, dict)
        if 'normal_portfolio_risk' in breakdown_results:
            self.assertIn('stress_portfolio_risk', breakdown_results)
            self.assertIn('risk_change', breakdown_results)
    
    def test_tail_risk_scenarios(self):
        """测试尾部风险情景"""
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        returns = pd.DataFrame({
            'A': np.random.normal(0.001, 0.02, 252),
            'B': np.random.normal(0.0005, 0.025, 252),
            'C': np.random.normal(0.0015, 0.015, 252)
        }, index=dates)
        
        # 执行尾部风险分析
        tail_results = self.manager.tail_risk_scenarios(returns, [0.95, 0.99])
        
        self.assertIsInstance(tail_results, dict)
        if 'confidence_0.95' in tail_results:
            self.assertIn('confidence_level', tail_results['confidence_0.95'])
        if 'extreme_scenarios' in tail_results:
            self.assertIn('historical_worst_day', tail_results['extreme_scenarios'])
    
    def test_run_comprehensive_stress_test(self):
        """测试综合压力测试"""
        # 创建测试数据
        portfolio_weights = pd.Series([0.4, 0.3, 0.3], index=['A', 'B', 'C'])
        market_data = {
            'correlation_matrix': pd.DataFrame({
                'A': [1.0, 0.3, 0.2],
                'B': [0.3, 1.0, 0.4],
                'C': [0.2, 0.4, 1.0]
            }, index=['A', 'B', 'C']),
            'returns_data': pd.DataFrame({
                'A': np.random.normal(0.001, 0.02, 100),
                'B': np.random.normal(0.0005, 0.025, 100),
                'C': np.random.normal(0.0015, 0.015, 100)
            })
        }
        
        # 执行综合压力测试
        comprehensive_results = self.manager.run_comprehensive_stress_test(portfolio_weights, market_data)
        
        self.assertIsInstance(comprehensive_results, dict)
        if 'overall_assessment' in comprehensive_results:
            self.assertIn('risk_level', comprehensive_results['overall_assessment'])
            self.assertIn('recommendations', comprehensive_results['overall_assessment'])


class TestRegimeDetector(unittest.TestCase):
    """测试市场状态检测器"""
    
    def setUp(self):
        """设置测试环境"""
        self.detector = RegimeDetector()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.detector)
        self.assertIsInstance(self.detector.volatility_states, dict)
        self.assertIsInstance(self.detector.trend_states, dict)
    
    def test_detect_volatility_regime(self):
        """测试波动率状态检测"""
        # 创建测试数据
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        
        # 检测波动率状态
        volatility_regime = self.detector.detect_volatility_regime(returns)
        
        self.assertIsInstance(volatility_regime, pd.Series)
        self.assertEqual(len(volatility_regime), len(returns))
        self.assertTrue(all(volatility_regime.isin([0, 1, 2])))
    
    def test_detect_trend_regime(self):
        """测试趋势状态检测"""
        # 创建测试数据
        prices = pd.Series(np.cumsum(np.random.normal(0.001, 0.02, 100)) + 100)
        
        # 检测趋势状态
        trend_regime = self.detector.detect_trend_regime(prices)
        
        self.assertIsInstance(trend_regime, pd.Series)
        self.assertEqual(len(trend_regime), len(prices))
        self.assertTrue(all(trend_regime.isin([0, 1, 2])))
    
    def test_detect_correlation_regime(self):
        """测试相关性状态检测"""
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        correlation_data = pd.DataFrame({
            'A_B': np.random.uniform(0.2, 0.8, 100),
            'A_C': np.random.uniform(0.1, 0.6, 100),
            'B_C': np.random.uniform(0.3, 0.7, 100)
        }, index=dates)
        
        # 检测相关性状态
        correlation_regime = self.detector.detect_correlation_regime(correlation_data)
        
        self.assertIsInstance(correlation_regime, pd.Series)
        self.assertEqual(len(correlation_regime), len(correlation_data))
        self.assertTrue(all(correlation_regime.isin([0, 1, 2])))
    
    def test_hmm_regime_detection(self):
        """测试HMM状态检测"""
        # 创建测试数据
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        
        # 检测HMM状态
        hmm_regime = self.detector.hmm_regime_detection(returns, n_regimes=2)
        
        self.assertIsInstance(hmm_regime, pd.Series)
        self.assertEqual(len(hmm_regime), len(returns))
        self.assertTrue(all(hmm_regime.isin([0, 1])))
    
    def test_get_regime_transitions(self):
        """测试状态转换分析"""
        # 创建测试状态序列
        regime = pd.Series([0, 0, 1, 1, 1, 0, 2, 2, 0, 1])
        
        # 分析状态转换
        transitions = self.detector.get_regime_transitions(regime)
        
        self.assertIsInstance(transitions, pd.DataFrame)
        self.assertGreater(len(transitions), 0)
    
    def test_calculate_regime_statistics(self):
        """测试状态统计"""
        # 创建测试数据
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        regime = pd.Series(np.random.choice([0, 1, 2], 100))
        
        # 计算状态统计
        statistics = self.detector.calculate_regime_statistics(returns, regime)
        
        self.assertIsInstance(statistics, dict)
        if statistics:
            self.assertIn('regime_returns', statistics)
            self.assertIn('regime_volatilities', statistics)
    
    def test_predict_regime_probability(self):
        """测试状态概率预测"""
        # 创建测试数据
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        
        # 预测状态概率
        probabilities = self.detector.predict_regime_probability(returns)
        
        self.assertIsInstance(probabilities, dict)
        if probabilities:
            self.assertIn('regime_probabilities', probabilities)
    
    def test_detect_regime_change(self):
        """测试状态变化检测"""
        # 创建测试数据
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        
        # 检测状态变化
        change_detection = self.detector.detect_regime_change(returns)
        
        self.assertIsInstance(change_detection, dict)
        if change_detection:
            self.assertIn('regime_change_detected', change_detection)
            self.assertIn('change_probability', change_detection)


if __name__ == '__main__':
    # 运行所有测试
    unittest.main(verbosity=2)