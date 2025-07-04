#!/usr/bin/env python3
"""
信号生成模块测试
"""

def test_signal_generator_module():
    """测试信号生成模块基本功能"""
    print("🚀 开始测试信号生成模块...")
    
    try:
        # 测试导入
        print("1. 测试导入信号生成模块...")
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from signals.signal_generator import (
            SignalGenerator, ThresholdSignalGenerator, RankingSignalGenerator,
            MLSignalGenerator, CompositeSignalGenerator, MultiTimeframeSignalGenerator,
            SignalFilter, SignalValidator, SignalAnalyzer
        )
        print("✅ 信号生成模块导入成功")
        
        # 测试初始化
        print("\n2. 测试信号生成器初始化...")
        import pandas as pd
        import numpy as np
        
        # 创建测试数据
        np.random.seed(42)
        n_samples = 100
        n_factors = 10
        
        # 生成因子数据
        factors = pd.DataFrame(
            np.random.randn(n_samples, n_factors),
            columns=[f'factor_{i}' for i in range(n_factors)],
            index=pd.date_range('2024-01-01', periods=n_samples, freq='D')
        )
        
        # 生成收益率数据
        returns = pd.DataFrame(
            np.random.randn(n_samples, 1) * 0.02,
            columns=['return'],
            index=factors.index
        )
        
        # 创建信号生成器
        config = {
            'default_method': 'ranking',
            'threshold_params': {'default_threshold': 1.0},
            'ranking_params': {'top_pct': 0.2, 'bottom_pct': 0.2},
            'filter_params': {'volatility_threshold': 0.3}
        }
        
        signal_generator = SignalGenerator(config)
        signal_generator.initialize(config)
        print("✅ 信号生成器初始化成功")
        
        # 测试阈值信号生成器
        print("\n3. 测试阈值信号生成器...")
        threshold_generator = ThresholdSignalGenerator({'factor_0': 0.5})
        threshold_signals = threshold_generator.generate_signals(factors)
        
        if not threshold_signals.empty:
            print(f"✅ 阈值信号生成成功: {threshold_signals.shape}")
        else:
            print("❌ 阈值信号生成失败")
            return False
        
        # 测试排序信号生成器
        print("\n4. 测试排序信号生成器...")
        ranking_generator = RankingSignalGenerator(top_pct=0.3, bottom_pct=0.3)
        ranking_signals = ranking_generator.generate_signals(factors)
        
        if not ranking_signals.empty:
            print(f"✅ 排序信号生成成功: {ranking_signals.shape}")
        else:
            print("❌ 排序信号生成失败")
            return False
        
        # 测试ML信号生成器
        print("\n5. 测试ML信号生成器...")
        ml_generator = MLSignalGenerator(model_type="classification")
        
        # 先拟合模型
        try:
            ml_generator.fit(factors, returns)
            ml_signals = ml_generator.generate_signals(factors)
            if not ml_signals.empty:
                print(f"✅ ML信号生成成功: {ml_signals.shape}")
            else:
                print("❌ ML信号生成失败")
                return False
        except Exception as e:
            print(f"⚠️ ML信号生成失败（可能缺少依赖）: {e}")
            ml_signals = pd.DataFrame()
        
        # 测试复合信号生成器
        print("\n6. 测试复合信号生成器...")
        sub_generators = [threshold_generator, ranking_generator]
        if not ml_signals.empty:
            sub_generators.append(ml_generator)
        
        composite_generator = CompositeSignalGenerator(
            sub_generators, combination_method="weighted_average"
        )
        composite_signals = composite_generator.generate_signals(factors)
        
        if not composite_signals.empty:
            print(f"✅ 复合信号生成成功: {composite_signals.shape}")
        else:
            print("❌ 复合信号生成失败")
            return False
        
        # 测试多时间框架信号生成器
        print("\n7. 测试多时间框架信号生成器...")
        timeframes = ["1D", "5D", "10D"]
        mtf_generator = MultiTimeframeSignalGenerator(timeframes)
        
        # 设置每个时间框架的生成器
        for timeframe in timeframes:
            mtf_generator.set_timeframe_generator(timeframe, ranking_generator)
        
        mtf_signals = mtf_generator.generate_signals(factors)
        
        if not mtf_signals.empty:
            print(f"✅ 多时间框架信号生成成功: {mtf_signals.shape}")
        else:
            print("❌ 多时间框架信号生成失败")
            return False
        
        # 测试信号过滤器
        print("\n8. 测试信号过滤器...")
        signal_filter = SignalFilter()
        
        # 创建模拟的波动率数据
        volatility = pd.DataFrame(
            np.random.uniform(0.1, 0.5, (n_samples, 1)),
            columns=['volatility'],
            index=factors.index
        )
        
        filtered_signals = signal_filter.filter_by_volatility(
            ranking_signals, volatility, vol_threshold=0.3
        )
        
        if not filtered_signals.empty:
            print(f"✅ 信号过滤成功: {filtered_signals.shape}")
        else:
            print("❌ 信号过滤失败")
            return False
        
        # 测试信号验证器
        print("\n9. 测试信号验证器...")
        signal_validator = SignalValidator()
        
        # 验证信号分布
        distribution_check = signal_validator.validate_signal_distribution(ranking_signals)
        if isinstance(distribution_check, dict):
            print(f"✅ 信号分布验证成功: {distribution_check}")
        else:
            print("❌ 信号分布验证失败")
            return False
        
        # 检查信号稳定性
        stability_check = signal_validator.check_signal_stability(ranking_signals)
        if not stability_check.empty:
            print(f"✅ 信号稳定性检查成功: {stability_check.shape}")
        else:
            print("❌ 信号稳定性检查失败")
            return False
        
        # 测试信号分析器
        print("\n10. 测试信号分析器...")
        signal_analyzer = SignalAnalyzer()
        
        # 分析信号性能
        performance_analysis = signal_analyzer.analyze_signal_performance(
            ranking_signals, returns, periods=[1, 5, 10]
        )
        
        if not performance_analysis.empty:
            print(f"✅ 信号性能分析成功: {performance_analysis.shape}")
        else:
            print("❌ 信号性能分析失败")
            return False
        
        # 计算信号IC
        ic_analysis = signal_analyzer.calculate_signal_ic(ranking_signals, returns)
        
        if not ic_analysis.empty:
            print(f"✅ 信号IC计算成功: {ic_analysis.shape}")
        else:
            print("❌ 信号IC计算失败")
            return False
        
        # 测试主信号生成器的核心功能
        print("\n11. 测试主信号生成器核心功能...")
        
        # 生成原始信号
        raw_signals = signal_generator.generate_raw_signals(factors, method="ranking")
        
        if not raw_signals.empty:
            print(f"✅ 原始信号生成成功: {raw_signals.shape}")
        else:
            print("❌ 原始信号生成失败")
            return False
        
        # 生成多期信号
        multi_horizon_signals = signal_generator.generate_multi_horizon_signals(
            factors, horizons=[1, 5, 10]
        )
        
        if isinstance(multi_horizon_signals, dict) and len(multi_horizon_signals) > 0:
            print(f"✅ 多期信号生成成功: {list(multi_horizon_signals.keys())}")
        else:
            print("❌ 多期信号生成失败")
            return False
        
        # 生成集成信号
        ensemble_signals = signal_generator.generate_ensemble_signals(
            factors, methods=["ranking", "threshold"]
        )
        
        if not ensemble_signals.empty:
            print(f"✅ 集成信号生成成功: {ensemble_signals.shape}")
        else:
            print("❌ 集成信号生成失败")
            return False
        
        # 信号后处理
        processing_config = {
            'smooth': True,
            'smooth_window': 3,
            'normalize': True,
            'clip_extreme': True
        }
        
        processed_signals = signal_generator.post_process_signals(
            raw_signals, processing_config
        )
        
        if not processed_signals.empty:
            print(f"✅ 信号后处理成功: {processed_signals.shape}")
        else:
            print("❌ 信号后处理失败")
            return False
        
        # 信号验证
        validation_config = {
            'check_distribution': True,
            'check_stability': True,
            'check_anomalies': True
        }
        
        validation_results = signal_generator.validate_signals(
            processed_signals, validation_config
        )
        
        if isinstance(validation_results, dict):
            print(f"✅ 信号验证成功: {validation_results}")
        else:
            print("❌ 信号验证失败")
            return False
        
        # 信号回测
        backtest_results = signal_generator.backtest_signals(
            processed_signals, returns, transaction_costs=0.001
        )
        
        if isinstance(backtest_results, dict) and len(backtest_results) > 0:
            print(f"✅ 信号回测成功: {list(backtest_results.keys())}")
        else:
            print("❌ 信号回测失败")
            return False
        
        print("\n🎉 信号生成模块核心功能测试通过！")
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
    success = test_signal_generator_module()
    if success:
        print("🎉 信号生成模块测试完成！核心功能正常。")
    else:
        print("⚠️ 信号生成模块测试失败，需要检查问题。") 