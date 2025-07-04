#!/usr/bin/env python3
"""
因子引擎模块测试
"""

def test_factor_engine_module():
    """测试因子引擎模块基本功能"""
    print("🚀 开始测试因子引擎模块...")
    
    try:
        # 测试导入
        print("1. 测试导入因子引擎模块...")
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from factors.factor_engine import FactorEngine
        from factors.technical_factors import TechnicalFactors
        from factors.fundamental_factors import FundamentalFactors
        from factors.sentiment_factors import SentimentFactors
        from factors.factor_utils import FactorProcessor, FactorValidator, FactorSelector
        print("✅ 因子引擎模块导入成功")
        
        # 测试初始化
        print("\n2. 测试因子引擎初始化...")
        import pandas as pd
        import numpy as np
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='D'),
            'symbol': ['TEST'] * 100,
            'open': np.random.uniform(90, 110, 100),
            'high': np.random.uniform(95, 115, 100),
            'low': np.random.uniform(85, 105, 100),
            'close': np.random.uniform(90, 110, 100),
            'volume': np.random.randint(1000, 10000, 100)
        })
        test_data.set_index('datetime', inplace=True)
        
        # 创建因子引擎实例
        config = {
            'technical_periods': [5, 10, 20],
            'factor_groups': ['technical', 'fundamental', 'sentiment'],
            'max_workers': 2,
            'use_multiprocessing': False,
            'winsorize_quantiles': (0.01, 0.99),
            'standardize_method': 'zscore',
            'correlation_threshold': 0.8
        }
        
        factor_engine = FactorEngine()
        factor_engine.initialize(config)
        print("✅ 因子引擎初始化成功")
        
        # 测试技术因子计算
        print("\n3. 测试技术因子计算...")
        tech_factors = factor_engine.calculate_technical_factors(test_data, [5, 10, 20])
        if len(tech_factors.columns) > 0:
            print(f"✅ 技术因子计算成功: {len(tech_factors.columns)}个因子")
        else:
            print("❌ 技术因子计算失败")
            return False
        
        # 测试基本面因子计算
        print("\n4. 测试基本面因子计算...")
        fundamental_data = factor_engine._generate_mock_fundamental_data(test_data)
        fund_factors = factor_engine.calculate_fundamental_factors(test_data, fundamental_data)
        if len(fund_factors.columns) > 0:
            print(f"✅ 基本面因子计算成功: {len(fund_factors.columns)}个因子")
        else:
            print("❌ 基本面因子计算失败")
            return False
        
        # 测试情绪因子计算
        print("\n5. 测试情绪因子计算...")
        sentiment_factors = factor_engine.calculate_sentiment_factors(test_data)
        if len(sentiment_factors.columns) > 0:
            print(f"✅ 情绪因子计算成功: {len(sentiment_factors.columns)}个因子")
        else:
            print("❌ 情绪因子计算失败")
            return False
        
        # 测试全部因子计算
        print("\n6. 测试全部因子计算...")
        all_factors = factor_engine.calculate_all_factors(test_data)
        if len(all_factors.columns) > 0:
            print(f"✅ 全部因子计算成功: {len(all_factors.columns)}个因子")
        else:
            print("❌ 全部因子计算失败")
            return False
        
        # 测试因子预处理
        print("\n7. 测试因子预处理...")
        processing_config = {
            'winsorize': True,
            'standardize': True,
            'neutralize': False
        }
        processed_factors = factor_engine.process_factors(all_factors, processing_config)
        if len(processed_factors.columns) > 0:
            print(f"✅ 因子预处理成功: {len(processed_factors.columns)}个因子")
        else:
            print("❌ 因子预处理失败")
            return False
        
        # 测试因子验证
        print("\n8. 测试因子验证...")
        validation_results = factor_engine.validate_factors(processed_factors)
        if isinstance(validation_results, dict) and len(validation_results) > 0:
            print(f"✅ 因子验证成功: {validation_results}")
        else:
            print("❌ 因子验证失败")
            return False
        
        # 测试因子相关性矩阵
        print("\n9. 测试因子相关性矩阵...")
        correlation_matrix = factor_engine.get_factor_correlation_matrix(processed_factors)
        if correlation_matrix.shape[0] > 0 and correlation_matrix.shape[1] > 0:
            print(f"✅ 因子相关性矩阵计算成功: {correlation_matrix.shape}")
        else:
            print("❌ 因子相关性矩阵计算失败")
            return False
        
        # 测试因子选择
        print("\n10. 测试因子选择...")
        # 生成模拟收益率数据
        returns_df = pd.DataFrame({
            'return_1d': np.random.normal(0, 0.02, len(test_data)),
            'return_5d': np.random.normal(0, 0.05, len(test_data)),
            'return_10d': np.random.normal(0, 0.08, len(test_data))
        }, index=test_data.index)
        
        # 使用单列收益率进行因子选择
        returns = returns_df['return_1d']
        
        selected_factors = factor_engine.select_factors(processed_factors, returns, method="ic", top_k=10)
        if len(selected_factors) > 0:
            print(f"✅ 因子选择成功: 选择了{len(selected_factors)}个因子")
        else:
            print("❌ 因子选择失败")
            return False
        
        # 测试因子IC计算
        print("\n11. 测试因子IC计算...")
        ic_results = factor_engine.calculate_factor_ic(processed_factors, returns, [1, 5, 10])
        if len(ic_results) > 0:
            print(f"✅ 因子IC计算成功: {len(ic_results)}条记录")
        else:
            print("❌ 因子IC计算失败")
            return False
        
        # 测试因子统计
        print("\n12. 测试因子统计...")
        factor_stats = factor_engine.get_factor_statistics(processed_factors)
        if len(factor_stats) > 0:
            print(f"✅ 因子统计计算成功: {len(factor_stats)}个因子")
        else:
            print("❌ 因子统计计算失败")
            return False
        
        # 测试复合因子
        print("\n13. 测试复合因子...")
        composite_factor = factor_engine.calculate_composite_factor(processed_factors, method="equal_weight")
        if len(composite_factor) > 0:
            print(f"✅ 复合因子计算成功: {len(composite_factor)}个数据点")
        else:
            print("❌ 复合因子计算失败")
            return False
        
        # 测试因子回测
        print("\n14. 测试因子回测...")
        backtest_results = factor_engine.backtest_factor(composite_factor, returns_df, [1, 5, 10])
        if isinstance(backtest_results, dict) and len(backtest_results) > 0:
            print(f"✅ 因子回测成功: {list(backtest_results.keys())}")
        else:
            print("❌ 因子回测失败")
            return False
        
        print("\n🎉 因子引擎模块核心功能测试通过！")
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
    success = test_factor_engine_module()
    if success:
        print("🎉 因子引擎模块测试完成！核心功能正常。")
    else:
        print("⚠️ 因子引擎模块测试失败，需要检查问题。") 