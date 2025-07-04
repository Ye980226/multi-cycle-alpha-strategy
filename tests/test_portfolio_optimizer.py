#!/usr/bin/env python3
"""
组合优化模块测试
"""

def test_portfolio_optimizer_module():
    """测试组合优化模块基本功能"""
    print("🚀 开始测试组合优化模块...")
    
    try:
        # 测试导入
        print("1. 测试导入组合优化模块...")
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from portfolio.portfolio_optimizer import (
            PortfolioOptimizer, MinVarianceOptimizer, MaxSharpeOptimizer,
            RiskParityOptimizer, MeanReversionOptimizer, BlackLittermanOptimizer
        )
        print("✅ 组合优化模块导入成功")
        
        # 测试初始化
        print("\n2. 测试组合优化器初始化...")
        import pandas as pd
        import numpy as np
        
        # 创建测试数据
        np.random.seed(42)
        n_assets = 10
        n_periods = 252
        
        # 生成资产收益率数据
        returns = pd.DataFrame(
            np.random.multivariate_normal(
                mean=np.random.uniform(-0.001, 0.002, n_assets),
                cov=np.random.uniform(0.0001, 0.001, (n_assets, n_assets)) * 
                    np.eye(n_assets) + 0.00005,
                size=n_periods
            ),
            columns=[f'asset_{i}' for i in range(n_assets)],
            index=pd.date_range('2023-01-01', periods=n_periods, freq='D')
        )
        
        # 生成因子暴露数据
        factor_exposure = pd.DataFrame(
            np.random.randn(n_assets, 5),
            columns=['factor_1', 'factor_2', 'factor_3', 'factor_4', 'factor_5'],
            index=returns.columns
        )
        
        # 生成预期收益
        expected_returns = pd.Series(
            np.random.uniform(0.0005, 0.002, n_assets),
            index=returns.columns
        )
        
        print("✅ 测试数据生成成功")
        
        # 测试最小方差优化器
        print("\n3. 测试最小方差优化器...")
        min_var_optimizer = MinVarianceOptimizer()
        min_var_weights = min_var_optimizer.optimize(returns)
        if len(min_var_weights) == n_assets and abs(min_var_weights.sum() - 1.0) < 1e-6:
            print(f"✅ 最小方差优化成功: 权重和={min_var_weights.sum():.6f}")
        else:
            print("❌ 最小方差优化失败")
            return False
        
        # 测试最大夏普比优化器
        print("\n4. 测试最大夏普比优化器...")
        max_sharpe_optimizer = MaxSharpeOptimizer()
        max_sharpe_weights = max_sharpe_optimizer.optimize(returns, expected_returns)
        if len(max_sharpe_weights) == n_assets and abs(max_sharpe_weights.sum() - 1.0) < 1e-6:
            print(f"✅ 最大夏普比优化成功: 权重和={max_sharpe_weights.sum():.6f}")
        else:
            print("❌ 最大夏普比优化失败")
            return False
        
        # 测试风险平价优化器
        print("\n5. 测试风险平价优化器...")
        risk_parity_optimizer = RiskParityOptimizer()
        risk_parity_weights = risk_parity_optimizer.optimize(returns)
        if len(risk_parity_weights) == n_assets and abs(risk_parity_weights.sum() - 1.0) < 1e-6:
            print(f"✅ 风险平价优化成功: 权重和={risk_parity_weights.sum():.6f}")
        else:
            print("❌ 风险平价优化失败")
            return False
        
        # 测试组合优化器主类
        print("\n6. 测试主组合优化器...")
        config = {
            'risk_aversion': 1.0,
            'max_weight': 0.2,
            'min_weight': 0.01,
            'turnover_penalty': 0.01,
            'sector_constraints': {}
        }
        
        portfolio_optimizer = PortfolioOptimizer(config)
        main_weights = portfolio_optimizer.optimize(
            expected_returns=expected_returns,
            covariance_matrix=returns.cov(),
            constraints={'max_weight': 0.15, 'min_weight': 0.02}
        )
        
        if len(main_weights) == n_assets and abs(main_weights.sum() - 1.0) < 1e-6:
            print(f"✅ 主组合优化成功: 权重和={main_weights.sum():.6f}")
        else:
            print("❌ 主组合优化失败")
            return False
        
        # 测试组合回测
        print("\n7. 测试组合回测...")
        backtest_results = portfolio_optimizer.backtest(
            returns=returns,
            weights=main_weights,
            rebalance_freq='monthly'
        )
        
        if isinstance(backtest_results, dict) and 'portfolio_returns' in backtest_results:
            print(f"✅ 组合回测成功: 年化收益={backtest_results.get('annual_return', 0):.4f}")
        else:
            print("❌ 组合回测失败")
            return False
        
        # 测试风险分析
        print("\n8. 测试风险分析...")
        risk_analysis = portfolio_optimizer.analyze_risk(
            weights=main_weights,
            factor_exposure=factor_exposure,
            returns=returns
        )
        
        if isinstance(risk_analysis, dict) and 'portfolio_volatility' in risk_analysis:
            print(f"✅ 风险分析成功: 组合波动率={risk_analysis.get('portfolio_volatility', 0):.4f}")
        else:
            print("❌ 风险分析失败")
            return False
        
        # 测试组合归因
        print("\n9. 测试组合归因...")
        attribution = portfolio_optimizer.performance_attribution(
            portfolio_returns=backtest_results['portfolio_returns'],
            benchmark_returns=returns.mean(axis=1),  # 等权基准
            factor_returns=returns.iloc[:, :3]  # 前3个资产作为因子
        )
        
        if isinstance(attribution, dict) and 'total_return' in attribution:
            print(f"✅ 组合归因成功: 总收益={attribution.get('total_return', 0):.4f}")
        else:
            print("❌ 组合归因失败")
            return False
        
        # 测试多策略组合
        print("\n10. 测试多策略组合...")
        strategies = {
            'min_var': min_var_weights,
            'max_sharpe': max_sharpe_weights,
            'risk_parity': risk_parity_weights
        }
        
        multi_strategy_weights = portfolio_optimizer.combine_strategies(
            strategies=strategies,
            method='equal_weight'
        )
        
        if len(multi_strategy_weights) == n_assets and abs(multi_strategy_weights.sum() - 1.0) < 1e-6:
            print(f"✅ 多策略组合成功: 权重和={multi_strategy_weights.sum():.6f}")
        else:
            print("❌ 多策略组合失败")
            return False
        
        print("\n🎉 组合优化模块核心功能测试通过！")
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
    success = test_portfolio_optimizer_module()
    if success:
        print("🎉 组合优化模块测试完成！核心功能正常。")
    else:
        print("⚠️ 组合优化模块测试失败，需要检查问题。") 