#!/usr/bin/env python3
"""
性能监控模块测试
"""

def test_monitoring_module():
    """测试性能监控模块基本功能"""
    print("🚀 开始测试性能监控模块...")
    
    try:
        # 测试导入
        print("1. 测试导入性能监控模块...")
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from monitoring.performance_monitor import (
            PerformanceMonitor, RealTimeMonitor, AlertManager,
            MetricsCalculator, ReportGenerator
        )
        print("✅ 性能监控模块导入成功")
        
        # 测试初始化
        print("\n2. 测试性能监控器初始化...")
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # 创建测试数据
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # 生成组合收益率数据
        portfolio_returns = pd.Series(
            np.random.normal(0.001, 0.02, len(dates)),
            index=dates
        )
        
        # 生成基准收益率数据
        benchmark_returns = pd.Series(
            np.random.normal(0.0008, 0.015, len(dates)),
            index=dates
        )
        
        # 生成交易数据
        trades = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='3D'),
            'symbol': np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'TSLA'], 100),
            'side': np.random.choice(['BUY', 'SELL'], 100),
            'quantity': np.random.randint(100, 1000, 100),
            'price': np.random.uniform(100, 500, 100),
            'commission': np.random.uniform(1, 10, 100)
        })
        
        print("✅ 测试数据生成成功")
        
        # 测试指标计算器
        print("\n3. 测试指标计算器...")
        metrics_calc = MetricsCalculator()
        
        # 计算基本性能指标
        basic_metrics = metrics_calc.calculate_basic_metrics(portfolio_returns)
        
        if isinstance(basic_metrics, dict) and 'total_return' in basic_metrics:
            print(f"✅ 基本指标计算成功: 总收益率={basic_metrics['total_return']:.4f}")
        else:
            print("❌ 基本指标计算失败")
            return False
        
        # 测试风险指标计算
        print("\n4. 测试风险指标计算...")
        risk_metrics = metrics_calc.calculate_risk_metrics(
            portfolio_returns,
            benchmark_returns
        )
        
        if isinstance(risk_metrics, dict) and 'sharpe_ratio' in risk_metrics:
            print(f"✅ 风险指标计算成功: 夏普比率={risk_metrics['sharpe_ratio']:.4f}")
        else:
            print("❌ 风险指标计算失败")
            return False
        
        # 测试性能归因
        print("\n5. 测试性能归因...")
        attribution = metrics_calc.performance_attribution(
            portfolio_returns,
            benchmark_returns,
            sector_weights=pd.Series([0.3, 0.25, 0.25, 0.2], index=['Tech', 'Finance', 'Healthcare', 'Energy'])
        )
        
        if isinstance(attribution, dict) and 'alpha' in attribution:
            print(f"✅ 性能归因成功: Alpha={attribution['alpha']:.4f}")
        else:
            print("❌ 性能归因失败")
            return False
        
        # 测试实时监控器
        print("\n6. 测试实时监控器...")
        realtime_monitor = RealTimeMonitor()
        
        # 添加实时数据点
        for i in range(10):
            current_time = datetime.now() + timedelta(seconds=i)
            portfolio_value = 1000000 * (1 + portfolio_returns.iloc[:i+1].sum())
            
            realtime_monitor.update_portfolio_value(current_time, portfolio_value)
        
        current_metrics = realtime_monitor.get_current_metrics()
        
        if isinstance(current_metrics, dict) and 'current_value' in current_metrics:
            print(f"✅ 实时监控成功: 当前价值={current_metrics['current_value']:.2f}")
        else:
            print("❌ 实时监控失败")
            return False
        
        # 测试告警管理器
        print("\n7. 测试告警管理器...")
        alert_config = {
            'max_drawdown_threshold': 0.05,
            'volatility_threshold': 0.25,
            'loss_threshold': 0.02,
            'position_concentration_threshold': 0.3
        }
        
        alert_manager = AlertManager(alert_config)
        
        # 检查告警
        alerts = alert_manager.check_alerts(
            portfolio_returns=portfolio_returns,
            current_positions=pd.Series([0.4, 0.3, 0.2, 0.1], index=['AAPL', 'MSFT', 'GOOGL', 'TSLA'])
        )
        
        if isinstance(alerts, list):
            print(f"✅ 告警检查成功: 发现{len(alerts)}个告警")
        else:
            print("❌ 告警检查失败")
            return False
        
        # 测试性能监控器主类
        print("\n8. 测试主性能监控器...")
        config = {
            'update_frequency': '1T',  # 1分钟
            'benchmark': 'SPY',
            'risk_free_rate': 0.02,
            'alert_thresholds': alert_config
        }
        
        performance_monitor = PerformanceMonitor(config)
        
        # 更新数据
        performance_monitor.update_data(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            trades=trades
        )
        
        # 获取当前状态
        status = performance_monitor.get_current_status()
        
        if isinstance(status, dict) and 'portfolio_metrics' in status:
            print(f"✅ 性能监控器更新成功: 状态正常")
        else:
            print("❌ 性能监控器更新失败")
            return False
        
        # 测试报告生成器
        print("\n9. 测试报告生成器...")
        report_generator = ReportGenerator()
        
        # 生成日报
        daily_report = report_generator.generate_daily_report(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            trades=trades.tail(10),  # 最近10笔交易
            date=datetime(2023, 12, 31)
        )
        
        if isinstance(daily_report, dict) and 'summary' in daily_report:
            print(f"✅ 日报生成成功: 日收益率={daily_report['summary']['daily_return']:.4f}")
        else:
            print("❌ 日报生成失败")
            return False
        
        # 测试月度报告
        print("\n10. 测试月度报告...")
        monthly_report = report_generator.generate_monthly_report(
            portfolio_returns=portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1),
            benchmark_returns=benchmark_returns.resample('M').apply(lambda x: (1 + x).prod() - 1),
            year=2023,
            month=12
        )
        
        if isinstance(monthly_report, dict) and 'performance_summary' in monthly_report:
            print(f"✅ 月报生成成功: 月收益率={monthly_report['performance_summary']['monthly_return']:.4f}")
        else:
            print("❌ 月报生成失败")
            return False
        
        # 测试年度报告
        print("\n11. 测试年度报告...")
        annual_report = report_generator.generate_annual_report(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            trades=trades,
            year=2023
        )
        
        if isinstance(annual_report, dict) and 'annual_performance' in annual_report:
            print(f"✅ 年报生成成功: 年收益率={annual_report['annual_performance']['annual_return']:.4f}")
        else:
            print("❌ 年报生成失败")
            return False
        
        # 测试风险监控
        print("\n12. 测试风险监控...")
        risk_monitoring = performance_monitor.monitor_risk(
            portfolio_returns=portfolio_returns,
            positions=pd.Series([0.4, 0.3, 0.2, 0.1], index=['AAPL', 'MSFT', 'GOOGL', 'TSLA'])
        )
        
        if isinstance(risk_monitoring, dict) and 'risk_score' in risk_monitoring:
            print(f"✅ 风险监控成功: 风险评分={risk_monitoring['risk_score']:.2f}")
        else:
            print("❌ 风险监控失败")
            return False
        
        print("\n🎉 性能监控模块核心功能测试通过！")
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
    success = test_monitoring_module()
    if success:
        print("🎉 性能监控模块测试完成！核心功能正常。")
    else:
        print("⚠️ 性能监控模块测试失败，需要检查问题。") 