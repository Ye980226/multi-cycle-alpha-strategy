#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内联配置测试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_config_test():
    """运行配置测试"""
    print("=" * 60)
    print("开始测试配置模块功能")
    print("=" * 60)
    
    test_results = []
    
    # 测试1: 导入模块
    print("\n1. 测试导入配置模块...")
    try:
        from config.strategy_config import StrategyConfig
        print("✓ 成功导入配置模块")
        test_results.append(("导入配置模块", True))
    except Exception as e:
        print(f"✗ 导入配置模块失败: {e}")
        test_results.append(("导入配置模块", False))
        return False
    
    # 测试2: 创建配置实例
    print("\n2. 测试创建配置实例...")
    try:
        config = StrategyConfig()
        print("✓ 成功创建配置实例")
        test_results.append(("创建配置实例", True))
    except Exception as e:
        print(f"✗ 创建配置实例失败: {e}")
        test_results.append(("创建配置实例", False))
        return False
    
    # 测试3: 验证配置
    print("\n3. 测试配置验证...")
    try:
        is_valid = config.validate_config()
        print(f"✓ 配置验证完成: {'通过' if is_valid else '失败'}")
        test_results.append(("配置验证", is_valid))
    except Exception as e:
        print(f"✗ 配置验证失败: {e}")
        test_results.append(("配置验证", False))
    
    # 测试4: 获取配置
    print("\n4. 测试获取各项配置...")
    try:
        data_config = config.get_data_config()
        factor_config = config.get_factor_config()
        model_config = config.get_model_config()
        portfolio_config = config.get_portfolio_config()
        risk_config = config.get_risk_config()
        
        print(f"数据源: {data_config.data_source}")
        print(f"频率: {data_config.frequency}")
        print(f"因子组数量: {len(factor_config.factor_groups)}")
        print(f"模型类型数量: {len(model_config.model_types)}")
        print(f"最大仓位: {portfolio_config.max_position}")
        print(f"最大回撤: {risk_config.max_drawdown}")
        
        print("✓ 成功获取各项配置")
        test_results.append(("获取配置", True))
    except Exception as e:
        print(f"✗ 获取配置失败: {e}")
        test_results.append(("获取配置", False))
    
    # 测试5: 配置更新
    print("\n5. 测试配置更新...")
    try:
        config.update_config("data", data_source="tushare", frequency="5min")
        config.update_config("portfolio", max_position=0.08)
        
        # 验证更新后的值
        updated_data_config = config.get_data_config()
        updated_portfolio_config = config.get_portfolio_config()
        
        assert updated_data_config.data_source == "tushare"
        assert updated_data_config.frequency == "5min"
        assert updated_portfolio_config.max_position == 0.08
        
        print("✓ 配置更新成功")
        test_results.append(("配置更新", True))
    except Exception as e:
        print(f"✗ 配置更新失败: {e}")
        test_results.append(("配置更新", False))
    
    # 测试6: 配置序列化
    print("\n6. 测试配置序列化...")
    try:
        all_config = config.get_all_config()
        print(f"配置字典包含 {len(all_config)} 个部分")
        
        # 测试各个配置的to_dict方法
        data_dict = config.data_config.to_dict()
        factor_dict = config.factor_config.to_dict()
        
        print(f"数据配置包含 {len(data_dict)} 个参数")
        print(f"因子配置包含 {len(factor_dict)} 个参数")
        
        print("✓ 配置序列化成功")
        test_results.append(("配置序列化", True))
    except Exception as e:
        print(f"✗ 配置序列化失败: {e}")
        test_results.append(("配置序列化", False))
    
    # 测试结果汇总
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过！配置模块功能正常。")
        return True
    else:
        print("⚠️  部分测试失败，需要修复配置模块。")
        return False

# 直接运行测试
if __name__ == "__main__":
    success = run_config_test()
    print(f"\n📋 配置模块测试结果: {'✅ 通过' if success else '❌ 失败'}") 