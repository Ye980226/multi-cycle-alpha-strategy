#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能监控模块测试文件
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# 导入被测试的模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.performance_monitor import PerformanceMonitor


class TestPerformanceMonitor(unittest.TestCase):
    """测试性能监控器"""
    
    def setUp(self):
        """设置测试环境"""
        self.monitor = PerformanceMonitor()
    
    def test_monitor_initialization(self):
        """测试监控器初始化"""
        self.assertIsNotNone(self.monitor)
        self.assertIsNotNone(self.monitor.metrics_calculator)
        self.assertIsNotNone(self.monitor.real_time_monitor)
        self.assertIsNotNone(self.monitor.alert_manager)
    
    def test_portfolio_performance_monitoring(self):
        """测试组合性能监控"""
        # 准备测试数据
        portfolio_returns = pd.Series(
            np.random.normal(0.001, 0.02, 252),
            index=pd.date_range('2023-01-01', periods=252, freq='D')
        )
        
        benchmark_returns = pd.Series(
            np.random.normal(0.0008, 0.015, 252),
            index=pd.date_range('2023-01-01', periods=252, freq='D')
        )
        
        # 监控组合性能
        result = self.monitor.monitor_portfolio_performance(
            portfolio_returns, benchmark_returns
        )
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn('performance_metrics', result)
        self.assertIn('risk_metrics', result)
        self.assertIn('benchmark_comparison', result)
    
    def test_real_time_monitoring(self):
        """测试实时监控"""
        # 准备测试数据
        current_positions = pd.Series([0.4, 0.3, 0.3], index=['AAPL', 'GOOGL', 'MSFT'])
        market_data = {
            'prices': pd.Series([150.0, 2800.0, 300.0], index=['AAPL', 'GOOGL', 'MSFT']),
            'volumes': pd.Series([1000000, 500000, 800000], index=['AAPL', 'GOOGL', 'MSFT']),
            'returns': pd.Series([0.01, -0.02, 0.005], index=['AAPL', 'GOOGL', 'MSFT'])
        }
        
        # 执行实时监控
        monitoring_result = self.monitor.real_time_monitor.check_real_time_metrics(
            current_positions, market_data
        )
        
        # 验证结果
        self.assertIsInstance(monitoring_result, dict)
        self.assertIn('portfolio_value', monitoring_result)
        self.assertIn('daily_pnl', monitoring_result)
        self.assertIn('alerts', monitoring_result)
    
    def test_performance_metrics_calculation(self):
        """测试性能指标计算"""
        # 准备测试数据
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        
        # 计算性能指标
        metrics = self.monitor.metrics_calculator.calculate_performance_metrics(returns)
        
        # 验证结果
        self.assertIsInstance(metrics, dict)
        self.assertIn('total_return', metrics)
        self.assertIn('annualized_return', metrics)
        self.assertIn('volatility', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('calmar_ratio', metrics)
    
    def test_risk_metrics_calculation(self):
        """测试风险指标计算"""
        # 准备测试数据
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        
        # 计算风险指标
        risk_metrics = self.monitor.metrics_calculator.calculate_risk_metrics(returns)
        
        # 验证结果
        self.assertIsInstance(risk_metrics, dict)
        self.assertIn('var_95', risk_metrics)
        self.assertIn('cvar_95', risk_metrics)
        self.assertIn('skewness', risk_metrics)
        self.assertIn('kurtosis', risk_metrics)
    
    def test_benchmark_comparison(self):
        """测试基准比较"""
        # 准备测试数据
        portfolio_returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        benchmark_returns = pd.Series(np.random.normal(0.0008, 0.015, 252))
        
        # 执行基准比较
        comparison = self.monitor.compare_with_benchmark(portfolio_returns, benchmark_returns)
        
        # 验证结果
        self.assertIsInstance(comparison, dict)
        self.assertIn('alpha', comparison)
        self.assertIn('beta', comparison)
        self.assertIn('information_ratio', comparison)
        self.assertIn('tracking_error', comparison)
    
    def test_drawdown_analysis(self):
        """测试回撤分析"""
        # 准备测试数据
        cumulative_returns = pd.Series(np.cumsum(np.random.normal(0.001, 0.02, 252)))
        
        # 分析回撤
        drawdown_analysis = self.monitor.analyze_drawdown(cumulative_returns)
        
        # 验证结果
        self.assertIsInstance(drawdown_analysis, dict)
        self.assertIn('max_drawdown', drawdown_analysis)
        self.assertIn('drawdown_duration', drawdown_analysis)
        self.assertIn('recovery_time', drawdown_analysis)
        self.assertIn('current_drawdown', drawdown_analysis)
    
    def test_alert_generation(self):
        """测试告警生成"""
        # 准备测试数据
        alert_type = 'performance_degradation'
        message = 'Portfolio performance below threshold'
        severity = 'high'
        
        # 生成告警
        alert = self.monitor.alert_manager.generate_alert(alert_type, message, severity)
        
        # 验证结果
        self.assertIsInstance(alert, dict)
        self.assertIn('id', alert)
        self.assertIn('type', alert)
        self.assertIn('message', alert)
        self.assertIn('severity', alert)
        self.assertIn('timestamp', alert)
    
    def test_performance_attribution(self):
        """测试性能归因"""
        # 准备测试数据
        portfolio_returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        factor_returns = pd.DataFrame({
            'market': np.random.normal(0.0008, 0.015, 252),
            'size': np.random.normal(0.0002, 0.008, 252),
            'value': np.random.normal(0.0001, 0.006, 252)
        })
        
        # 执行性能归因
        attribution = self.monitor.performance_attribution(portfolio_returns, factor_returns)
        
        # 验证结果
        self.assertIsInstance(attribution, dict)
        self.assertIn('factor_exposures', attribution)
        self.assertIn('factor_contributions', attribution)
        self.assertIn('specific_return', attribution)
    
    def test_rolling_performance_analysis(self):
        """测试滚动性能分析"""
        # 准备测试数据
        returns = pd.Series(
            np.random.normal(0.001, 0.02, 252),
            index=pd.date_range('2023-01-01', periods=252, freq='D')
        )
        
        # 分析滚动性能
        rolling_performance = self.monitor.analyze_rolling_performance(returns, window=60)
        
        # 验证结果
        self.assertIsInstance(rolling_performance, pd.DataFrame)
        self.assertIn('rolling_return', rolling_performance.columns)
        self.assertIn('rolling_volatility', rolling_performance.columns)
        self.assertIn('rolling_sharpe', rolling_performance.columns)
    
    def test_portfolio_composition_monitoring(self):
        """测试组合构成监控"""
        # 准备测试数据
        weights = pd.Series([0.4, 0.3, 0.3], index=['AAPL', 'GOOGL', 'MSFT'])
        sector_mapping = {'AAPL': 'technology', 'GOOGL': 'technology', 'MSFT': 'technology'}
        
        # 监控组合构成
        composition_analysis = self.monitor.monitor_portfolio_composition(weights, sector_mapping)
        
        # 验证结果
        self.assertIsInstance(composition_analysis, dict)
        self.assertIn('sector_weights', composition_analysis)
        self.assertIn('concentration_metrics', composition_analysis)
        self.assertIn('diversification_score', composition_analysis)
    
    def test_transaction_cost_monitoring(self):
        """测试交易成本监控"""
        # 准备测试数据
        trades = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL', 'MSFT'],
            'quantity': [100, 50, -75],
            'price': [150.0, 2800.0, 300.0],
            'commission': [0.15, 2.8, 0.3],
            'market_impact': [0.05, 0.1, 0.02]
        })
        
        # 监控交易成本
        cost_analysis = self.monitor.monitor_transaction_costs(trades)
        
        # 验证结果
        self.assertIsInstance(cost_analysis, dict)
        self.assertIn('total_cost', cost_analysis)
        self.assertIn('cost_breakdown', cost_analysis)
        self.assertIn('cost_per_trade', cost_analysis)
    
    def test_stress_testing_monitoring(self):
        """测试压力测试监控"""
        # 准备测试数据
        portfolio_weights = pd.Series([0.4, 0.3, 0.3], index=['AAPL', 'GOOGL', 'MSFT'])
        stress_scenarios = {
            'market_crash': pd.Series([-0.2, -0.25, -0.15], index=['AAPL', 'GOOGL', 'MSFT']),
            'tech_selloff': pd.Series([-0.3, -0.35, -0.2], index=['AAPL', 'GOOGL', 'MSFT'])
        }
        
        # 执行压力测试监控
        stress_results = self.monitor.monitor_stress_scenarios(portfolio_weights, stress_scenarios)
        
        # 验证结果
        self.assertIsInstance(stress_results, dict)
        self.assertIn('scenario_losses', stress_results)
        self.assertIn('worst_case_loss', stress_results)
        self.assertIn('stress_alerts', stress_results)
    
    def test_performance_report_generation(self):
        """测试性能报告生成"""
        # 准备测试数据
        portfolio_returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        benchmark_returns = pd.Series(np.random.normal(0.0008, 0.015, 252))
        
        # 生成性能报告
        report = self.monitor.generate_performance_report(
            portfolio_returns, benchmark_returns, 
            start_date='2023-01-01', end_date='2023-12-31'
        )
        
        # 验证结果
        self.assertIsInstance(report, dict)
        self.assertIn('summary', report)
        self.assertIn('detailed_metrics', report)
        self.assertIn('charts', report)
    
    def test_factor_exposure_monitoring(self):
        """测试因子暴露监控"""
        # 准备测试数据
        portfolio_weights = pd.Series([0.4, 0.3, 0.3], index=['AAPL', 'GOOGL', 'MSFT'])
        factor_exposures = pd.DataFrame({
            'market_beta': [1.0, 1.2, 0.8],
            'size_factor': [0.5, -0.2, 0.3],
            'value_factor': [-0.1, 0.4, 0.2]
        }, index=['AAPL', 'GOOGL', 'MSFT'])
        
        # 监控因子暴露
        exposure_analysis = self.monitor.monitor_factor_exposures(portfolio_weights, factor_exposures)
        
        # 验证结果
        self.assertIsInstance(exposure_analysis, dict)
        self.assertIn('portfolio_exposures', exposure_analysis)
        self.assertIn('exposure_changes', exposure_analysis)
        self.assertIn('risk_alerts', exposure_analysis)
    
    def test_liquidity_monitoring(self):
        """测试流动性监控"""
        # 准备测试数据
        portfolio_weights = pd.Series([0.4, 0.3, 0.3], index=['AAPL', 'GOOGL', 'MSFT'])
        liquidity_metrics = pd.DataFrame({
            'avg_volume': [1000000, 500000, 800000],
            'bid_ask_spread': [0.01, 0.02, 0.015],
            'market_cap': [3000000000, 2000000000, 2500000000]
        }, index=['AAPL', 'GOOGL', 'MSFT'])
        
        # 监控流动性
        liquidity_analysis = self.monitor.monitor_liquidity(portfolio_weights, liquidity_metrics)
        
        # 验证结果
        self.assertIsInstance(liquidity_analysis, dict)
        self.assertIn('portfolio_liquidity_score', liquidity_analysis)
        self.assertIn('liquidity_risk_alerts', liquidity_analysis)
        self.assertIn('days_to_liquidate', liquidity_analysis)
    
    def test_regime_change_monitoring(self):
        """测试市场状态变化监控"""
        # 准备测试数据
        market_returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        
        # 监控市场状态变化
        regime_analysis = self.monitor.monitor_regime_changes(market_returns)
        
        # 验证结果
        self.assertIsInstance(regime_analysis, dict)
        self.assertIn('current_regime', regime_analysis)
        self.assertIn('regime_probability', regime_analysis)
        self.assertIn('regime_change_alerts', regime_analysis)


if __name__ == '__main__':
    unittest.main() 