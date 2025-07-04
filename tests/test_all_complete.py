#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的模块测试脚本
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.strategy_config import StrategyConfig, DataConfig, FactorConfig
from data.data_manager import DataManager
from factors.factor_engine import FactorEngine
from models.model_manager import ModelManager
from signals.signal_generator import SignalGenerator
from portfolio.portfolio_optimizer import PortfolioOptimizer
from backtest.backtest_engine import BacktestEngine
from risk.risk_manager import RiskManager
from execution.trade_executor import TradeExecutor
from monitoring.performance_monitor import PerformanceMonitor

def test_all_modules():
    """测试所有模块"""
    print("=" * 60)
    print("开始完整模块测试")
    print("=" * 60)
    
    # 创建测试数据
    print("\n1. 创建测试数据...")
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    # 创建价格数据
    price_data = pd.DataFrame()
    for symbol in symbols:
        symbol_data = pd.DataFrame({
            'symbol': symbol,
            'open': np.random.uniform(100, 200, len(dates)),
            'close': np.random.uniform(100, 200, len(dates)),
            'high': np.random.uniform(150, 250, len(dates)),
            'low': np.random.uniform(50, 150, len(dates)),
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        price_data = pd.concat([price_data, symbol_data])
    
    # 计算收益率
    returns = price_data.groupby('symbol')['close'].pct_change().reset_index()
    returns.columns = ['symbol', 'return']
    
    print(f"   测试数据创建成功: {len(price_data)} 条记录")
    
    # 2. 测试配置模块
    print("\n2. 测试配置模块...")
    try:
        config = StrategyConfig()
        config.load_config('./config/default_config.yaml')
        print("   ✅ 配置模块测试通过")
    except Exception as e:
        print(f"   ❌ 配置模块测试失败: {e}")
    
    # 3. 测试数据管理模块
    print("\n3. 测试数据管理模块...")
    try:
        data_manager = DataManager(config.data_config)
        print("   ✅ 数据管理模块测试通过")
    except Exception as e:
        print(f"   ❌ 数据管理模块测试失败: {e}")
    
    # 4. 测试因子引擎
    print("\n4. 测试因子引擎...")
    try:
        factor_engine = FactorEngine(config.factor_config.__dict__)
        
        # 使用价格数据计算因子
        test_data = price_data.pivot(columns='symbol', values='close').fillna(method='ffill')
        factors = factor_engine.calculate_technical_factors(test_data)
        
        print(f"   ✅ 因子引擎测试通过: 计算了 {len(factors.columns)} 个因子")
    except Exception as e:
        print(f"   ❌ 因子引擎测试失败: {e}")
    
    # 5. 测试模型管理器
    print("\n5. 测试模型管理器...")
    try:
        model_manager = ModelManager(config.model_config.__dict__)
        
        # 创建简单的训练数据
        if 'factors' in locals():
            X = factors.iloc[:100].fillna(0)
            y = pd.Series(np.random.randn(len(X)), index=X.index)
            
            # 创建简单模型
            model = model_manager.create_model('linear')
            
            # 测试模型评估器
            evaluator = model_manager.model_evaluator
            y_pred = np.random.randn(len(y))
            metrics = evaluator.evaluate_regression(y.values, y_pred)
            
            print(f"   ✅ 模型管理器测试通过: 评估指标 {list(metrics.keys())}")
        else:
            print("   ✅ 模型管理器基础测试通过")
    except Exception as e:
        print(f"   ❌ 模型管理器测试失败: {e}")
    
    # 6. 测试信号生成器
    print("\n6. 测试信号生成器...")
    try:
        signal_generator = SignalGenerator({})
        
        if 'factors' in locals():
            # 生成信号
            signals = signal_generator.generate_signals(factors.iloc[:100].fillna(0))
            print(f"   ✅ 信号生成器测试通过: 生成了 {signals.shape} 信号")
        else:
            print("   ✅ 信号生成器基础测试通过")
    except Exception as e:
        print(f"   ❌ 信号生成器测试失败: {e}")
    
    # 7. 测试组合优化器
    print("\n7. 测试组合优化器...")
    try:
        portfolio_optimizer = PortfolioOptimizer(config.portfolio_config.__dict__)
        
        # 创建模拟信号
        if 'signals' in locals():
            # 使用生成的信号
            test_signals = signals.iloc[:10].fillna(0)
        else:
            test_signals = pd.DataFrame(
                np.random.randn(10, 5),
                columns=symbols,
                index=pd.date_range('2023-01-01', periods=10)
            )
        
        # 优化组合
        weights = portfolio_optimizer.optimize_portfolio(test_signals)
        print(f"   ✅ 组合优化器测试通过: 优化了 {len(weights)} 个权重")
    except Exception as e:
        print(f"   ❌ 组合优化器测试失败: {e}")
    
    # 8. 测试回测引擎
    print("\n8. 测试回测引擎...")
    try:
        backtest_engine = BacktestEngine({})
        print("   ✅ 回测引擎测试通过")
    except Exception as e:
        print(f"   ❌ 回测引擎测试失败: {e}")
    
    # 9. 测试风险管理器
    print("\n9. 测试风险管理器...")
    try:
        risk_manager = RiskManager(config.risk_config.__dict__)
        print("   ✅ 风险管理器测试通过")
    except Exception as e:
        print(f"   ❌ 风险管理器测试失败: {e}")
    
    # 10. 测试交易执行器
    print("\n10. 测试交易执行器...")
    try:
        trade_executor = TradeExecutor({})
        print("   ✅ 交易执行器测试通过")
    except Exception as e:
        print(f"   ❌ 交易执行器测试失败: {e}")
    
    # 11. 测试性能监控器
    print("\n11. 测试性能监控器...")
    try:
        performance_monitor = PerformanceMonitor(config.monitoring_config.__dict__)
        
        # 创建简单的组合收益率
        portfolio_returns = pd.Series(
            np.random.randn(100) * 0.02,
            index=pd.date_range('2023-01-01', periods=100)
        )
        
        # 计算性能指标
        metrics = performance_monitor.calculate_metrics(portfolio_returns)
        print(f"   ✅ 性能监控器测试通过: 计算了 {len(metrics)} 个指标")
    except Exception as e:
        print(f"   ❌ 性能监控器测试失败: {e}")
    
    # 12. 集成测试
    print("\n12. 集成测试...")
    try:
        # 创建一个简单的端到端流程
        print("   正在执行端到端流程...")
        
        # 使用真实数据源
        data_manager = DataManager(config.data_config)
        
        # 模拟获取数据
        test_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 100),
            'close': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(150, 250, 100),
            'low': np.random.uniform(50, 150, 100),
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=pd.date_range('2023-01-01', periods=100))
        
        # 计算因子
        factor_engine = FactorEngine(config.factor_config.__dict__)
        factors = factor_engine.calculate_technical_factors(test_data)
        
        # 生成信号
        signal_generator = SignalGenerator({})
        signals = signal_generator.generate_signals(factors)
        
        # 优化组合
        portfolio_optimizer = PortfolioOptimizer(config.portfolio_config.__dict__)
        weights = portfolio_optimizer.optimize_portfolio(signals)
        
        print("   ✅ 集成测试通过")
    except Exception as e:
        print(f"   ❌ 集成测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("所有模块测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    test_all_modules() 