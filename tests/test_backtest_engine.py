#!/usr/bin/env python3
"""
回测引擎模块测试
"""

def test_backtest_engine_module():
    """测试回测引擎模块基本功能"""
    print("🚀 开始测试回测引擎模块...")
    
    try:
        # 测试导入
        print("1. 测试导入回测引擎模块...")
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from backtest.backtest_engine import (
            BacktestEngine, EventDrivenBacktest, VectorizedBacktest,
            BacktestResult, PerformanceAnalyzer
        )
        print("✅ 回测引擎模块导入成功")
        
        # 测试初始化
        print("\n2. 测试回测引擎初始化...")
        import pandas as pd
        import numpy as np
        
        # 创建测试数据
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        n_assets = 5
        
        # 生成价格数据
        price_data = pd.DataFrame(
            index=dates,
            columns=[f'asset_{i}' for i in range(n_assets)]
        )
        
        base_prices = np.random.uniform(50, 150, n_assets)
        for i, asset in enumerate(price_data.columns):
            returns = np.random.normal(0.0005, 0.02, len(dates))
            prices = base_prices[i] * np.cumprod(1 + returns)
            price_data[asset] = prices
        
        # 生成信号数据
        signal_data = pd.DataFrame(
            index=dates,
            columns=price_data.columns,
            data=np.random.choice([-1, 0, 1], size=(len(dates), n_assets), p=[0.3, 0.4, 0.3])
        )
        
        print("✅ 测试数据生成成功")
        
        # 测试向量化回测
        print("\n3. 测试向量化回测...")
        vectorized_bt = VectorizedBacktest()
        vectorized_result = vectorized_bt.run(
            price_data=price_data,
            signal_data=signal_data,
            initial_capital=1000000,
            commission=0.001
        )
        
        if isinstance(vectorized_result, dict) and 'portfolio_value' in vectorized_result:
            final_value = vectorized_result['portfolio_value'].iloc[-1]
            print(f"✅ 向量化回测成功: 最终资产={final_value:.2f}")
        else:
            print("❌ 向量化回测失败")
            return False
        
        # 测试事件驱动回测
        print("\n4. 测试事件驱动回测...")
        event_bt = EventDrivenBacktest()
        event_result = event_bt.run(
            price_data=price_data,
            signal_data=signal_data,
            initial_capital=1000000,
            commission=0.001
        )
        
        if isinstance(event_result, dict) and 'portfolio_value' in event_result:
            final_value = event_result['portfolio_value'].iloc[-1]
            print(f"✅ 事件驱动回测成功: 最终资产={final_value:.2f}")
        else:
            print("❌ 事件驱动回测失败")
            return False
        
        # 测试回测引擎主类
        print("\n5. 测试主回测引擎...")
        config = {
            'initial_capital': 1000000,
            'commission': 0.001,
            'slippage': 0.0005,
            'benchmark': 'equal_weight',
            'rebalance_freq': 'daily'
        }
        
        backtest_engine = BacktestEngine(config)
        main_result = backtest_engine.run_backtest(
            price_data=price_data,
            signal_data=signal_data,
            strategy_name="test_strategy"
        )
        
        if isinstance(main_result, BacktestResult):
            print(f"✅ 主回测引擎成功: 最终资产={main_result.final_value:.2f}")
        else:
            print("❌ 主回测引擎失败")
            return False
        
        # 测试性能分析
        print("\n6. 测试性能分析...")
        analyzer = PerformanceAnalyzer()
        performance_metrics = analyzer.analyze(main_result)
        
        if isinstance(performance_metrics, dict) and 'total_return' in performance_metrics:
            print(f"✅ 性能分析成功: 总收益率={performance_metrics['total_return']:.4f}")
        else:
            print("❌ 性能分析失败")
            return False
        
        # 测试风险指标计算
        print("\n7. 测试风险指标计算...")
        risk_metrics = analyzer.calculate_risk_metrics(
            returns=main_result.returns,
            benchmark_returns=price_data.pct_change().mean(axis=1)
        )
        
        if isinstance(risk_metrics, dict) and 'sharpe_ratio' in risk_metrics:
            print(f"✅ 风险指标计算成功: 夏普比率={risk_metrics['sharpe_ratio']:.4f}")
        else:
            print("❌ 风险指标计算失败")
            return False
        
        # 测试交易分析
        print("\n8. 测试交易分析...")
        trade_analysis = analyzer.analyze_trades(main_result.trades)
        
        if isinstance(trade_analysis, dict) and 'total_trades' in trade_analysis:
            print(f"✅ 交易分析成功: 总交易数={trade_analysis['total_trades']}")
        else:
            print("❌ 交易分析失败")
            return False
        
        # 测试回撤分析
        print("\n9. 测试回撤分析...")
        drawdown_analysis = analyzer.analyze_drawdowns(main_result.portfolio_value)
        
        if isinstance(drawdown_analysis, dict) and 'max_drawdown' in drawdown_analysis:
            print(f"✅ 回撤分析成功: 最大回撤={drawdown_analysis['max_drawdown']:.4f}")
        else:
            print("❌ 回撤分析失败")
            return False
        
        # 测试基准比较
        print("\n10. 测试基准比较...")
        benchmark_comparison = analyzer.compare_to_benchmark(
            strategy_returns=main_result.returns,
            benchmark_returns=price_data.pct_change().mean(axis=1)
        )
        
        if isinstance(benchmark_comparison, dict) and 'alpha' in benchmark_comparison:
            print(f"✅ 基准比较成功: Alpha={benchmark_comparison['alpha']:.4f}")
        else:
            print("❌ 基准比较失败")
            return False
        
        print("\n🎉 回测引擎模块核心功能测试通过！")
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_backtest_engine_module()
    if success:
        print("🎉 回测引擎模块测试完成！核心功能正常。")
    else:
        print("⚠️ 回测引擎模块测试失败，需要检查问题。") 