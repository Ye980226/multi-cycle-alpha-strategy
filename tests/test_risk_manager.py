#!/usr/bin/env python3
"""
风险管理模块测试
"""

def test_risk_manager_module():
    """测试风险管理模块基本功能"""
    print("🚀 开始测试风险管理模块...")
    
    try:
        # 测试导入
        print("1. 测试导入风险管理模块...")
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from risk.risk_manager import (
            RiskManager, VaRCalculator, StressTestEngine,
            PositionSizer, RiskMonitor
        )
        print("✅ 风险管理模块导入成功")
        
        # 测试初始化
        print("\n2. 测试风险管理器初始化...")
        import pandas as pd
        import numpy as np
        
        # 创建测试数据
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        n_assets = 10
        
        # 生成收益率数据
        returns = pd.DataFrame(
            np.random.multivariate_normal(
                mean=np.random.uniform(-0.001, 0.002, n_assets),
                cov=0.0001 * np.eye(n_assets) + 0.00005,
                size=len(dates)
            ),
            columns=[f'asset_{i}' for i in range(n_assets)],
            index=dates
        )
        
        # 生成组合权重
        weights = pd.Series(
            np.random.dirichlet(np.ones(n_assets)),
            index=returns.columns
        )
        
        print("✅ 测试数据生成成功")
        
        # 测试VaR计算器
        print("\n3. 测试VaR计算器...")
        var_calculator = VaRCalculator()
        var_results = var_calculator.calculate(
            returns=returns,
            weights=weights,
            confidence_levels=[0.95, 0.99],
            methods=['parametric', 'historical', 'monte_carlo']
        )
        
        if isinstance(var_results, dict) and 'parametric' in var_results:
            print(f"✅ VaR计算成功: 95% VaR={var_results['parametric']['0.95']:.4f}")
        else:
            print("❌ VaR计算失败")
            return False
        
        # 测试压力测试引擎
        print("\n4. 测试压力测试引擎...")
        stress_engine = StressTestEngine()
        stress_scenarios = {
            'market_crash': {'factor': 'market', 'shock': -0.2},
            'volatility_spike': {'factor': 'volatility', 'shock': 2.0},
            'correlation_breakdown': {'factor': 'correlation', 'shock': 0.8}
        }
        
        stress_results = stress_engine.run_scenarios(
            portfolio_returns=returns @ weights,
            scenarios=stress_scenarios
        )
        
        if isinstance(stress_results, dict) and 'market_crash' in stress_results:
            print(f"✅ 压力测试成功: 市场崩盘损失={stress_results['market_crash']['portfolio_loss']:.4f}")
        else:
            print("❌ 压力测试失败")
            return False
        
        # 测试头寸规模器
        print("\n5. 测试头寸规模器...")
        position_sizer = PositionSizer()
        position_sizes = position_sizer.calculate_positions(
            signals=pd.Series([1, -1, 0, 1, -1], index=returns.columns[:5]),
            risk_budget=0.02,
            volatilities=returns.std(),
            correlations=returns.corr()
        )
        
        if isinstance(position_sizes, pd.Series) and len(position_sizes) == 5:
            print(f"✅ 头寸规模计算成功: 最大权重={position_sizes.abs().max():.4f}")
        else:
            print("❌ 头寸规模计算失败")
            return False
        
        # 测试风险管理器主类
        print("\n6. 测试主风险管理器...")
        config = {
            'max_portfolio_risk': 0.15,
            'max_position_weight': 0.1,
            'var_confidence': 0.95,
            'stress_test_frequency': 'daily',
            'risk_budget': 0.02
        }
        
        risk_manager = RiskManager(config)
        
        # 测试预交易风险检查
        trade_proposal = {
            'asset_1': 0.05,
            'asset_2': -0.03,
            'asset_3': 0.02
        }
        
        pre_trade_check = risk_manager.pre_trade_risk_check(
            proposed_trades=trade_proposal,
            current_positions=weights,
            market_data=returns.iloc[-1]
        )
        
        if isinstance(pre_trade_check, dict) and 'approved' in pre_trade_check:
            print(f"✅ 预交易风险检查成功: 批准={pre_trade_check['approved']}")
        else:
            print("❌ 预交易风险检查失败")
            return False
        
        # 测试风险监控
        print("\n7. 测试风险监控...")
        risk_monitor = RiskMonitor()
        portfolio_returns = returns @ weights
        
        risk_alerts = risk_monitor.monitor_portfolio(
            portfolio_returns=portfolio_returns,
            positions=weights,
            risk_limits=config
        )
        
        if isinstance(risk_alerts, list):
            print(f"✅ 风险监控成功: 发现{len(risk_alerts)}个风险警报")
        else:
            print("❌ 风险监控失败")
            return False
        
        # 测试风险归因
        print("\n8. 测试风险归因...")
        risk_attribution = risk_manager.risk_attribution(
            portfolio_returns=portfolio_returns,
            factor_returns=returns.iloc[:, :3],  # 前3个资产作为因子
            positions=weights
        )
        
        if isinstance(risk_attribution, dict) and 'factor_contributions' in risk_attribution:
            print(f"✅ 风险归因成功: 因子数={len(risk_attribution['factor_contributions'])}")
        else:
            print("❌ 风险归因失败")
            return False
        
        # 测试流动性风险评估
        print("\n9. 测试流动性风险评估...")
        liquidity_assessment = risk_manager.assess_liquidity_risk(
            positions=weights,
            daily_volumes=pd.Series(np.random.uniform(1000000, 10000000, n_assets), index=weights.index),
            market_impact_model='linear'
        )
        
        if isinstance(liquidity_assessment, dict) and 'liquidity_score' in liquidity_assessment:
            print(f"✅ 流动性风险评估成功: 流动性分数={liquidity_assessment['liquidity_score']:.4f}")
        else:
            print("❌ 流动性风险评估失败")
            return False
        
        # 测试风险报告生成
        print("\n10. 测试风险报告生成...")
        risk_report = risk_manager.generate_risk_report(
            portfolio_returns=portfolio_returns,
            positions=weights,
            benchmark_returns=returns.mean(axis=1)
        )
        
        if isinstance(risk_report, dict) and 'summary' in risk_report:
            print(f"✅ 风险报告生成成功: 总风险={risk_report['summary']['total_risk']:.4f}")
        else:
            print("❌ 风险报告生成失败")
            return False
        
        print("\n🎉 风险管理模块核心功能测试通过！")
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
    success = test_risk_manager_module()
    if success:
        print("🎉 风险管理模块测试完成！核心功能正常。")
    else:
        print("⚠️ 风险管理模块测试失败，需要检查问题。") 