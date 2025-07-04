#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置模块验证脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.strategy_config import StrategyConfig
import traceback

def test_config_functionality():
    """测试配置模块功能"""
    print("=" * 60)
    print("开始测试配置模块功能")
    print("=" * 60)
    
    test_results = []
    
    # 测试1: 默认配置创建
    print("\n1. 测试默认配置创建...")
    try:
        config = StrategyConfig()
        print("✓ 默认配置创建成功")
        test_results.append(("默认配置创建", True))
    except Exception as e:
        print(f"✗ 默认配置创建失败: {e}")
        test_results.append(("默认配置创建", False))
        traceback.print_exc()
    
    # 测试2: 配置验证
    print("\n2. 测试配置验证...")
    try:
        is_valid = config.validate_config()
        print(f"✓ 配置验证完成: {'通过' if is_valid else '失败'}")
        test_results.append(("配置验证", is_valid))
    except Exception as e:
        print(f"✗ 配置验证失败: {e}")
        test_results.append(("配置验证", False))
        traceback.print_exc()
    
    # 测试3: 配置更新
    print("\n3. 测试配置更新...")
    try:
        config.update_config("data", data_source="tushare", frequency="5min")
        config.update_config("portfolio", max_position=0.08, max_turnover=0.3)
        print("✓ 配置更新成功")
        test_results.append(("配置更新", True))
    except Exception as e:
        print(f"✗ 配置更新失败: {e}")
        test_results.append(("配置更新", False))
        traceback.print_exc()
    
    # 测试4: 配置获取
    print("\n4. 测试配置获取...")
    try:
        data_config = config.get_data_config()
        portfolio_config = config.get_portfolio_config()
        risk_config = config.get_risk_config()
        
        print(f"数据源: {data_config.data_source}")
        print(f"频率: {data_config.frequency}")
        print(f"最大仓位: {portfolio_config.max_position}")
        print(f"最大换手率: {portfolio_config.max_turnover}")
        print(f"最大回撤: {risk_config.max_drawdown}")
        
        print("✓ 配置获取成功")
        test_results.append(("配置获取", True))
    except Exception as e:
        print(f"✗ 配置获取失败: {e}")
        test_results.append(("配置获取", False))
        traceback.print_exc()
    
    # 测试5: 配置序列化
    print("\n5. 测试配置序列化...")
    try:
        all_config = config.get_all_config()
        print(f"配置字典包含 {len(all_config)} 个部分")
        
        # 测试各个配置的to_dict方法
        data_dict = config.data_config.to_dict()
        factor_dict = config.factor_config.to_dict()
        model_dict = config.model_config.to_dict()
        
        print(f"数据配置包含 {len(data_dict)} 个参数")
        print(f"因子配置包含 {len(factor_dict)} 个参数")
        print(f"模型配置包含 {len(model_dict)} 个参数")
        
        print("✓ 配置序列化成功")
        test_results.append(("配置序列化", True))
    except Exception as e:
        print(f"✗ 配置序列化失败: {e}")
        test_results.append(("配置序列化", False))
        traceback.print_exc()
    
    # 测试6: 配置保存和加载
    print("\n6. 测试配置保存和加载...")
    try:
        # 保存配置
        test_config_path = "test_config_temp.yaml"
        config.save_config(test_config_path)
        print("✓ 配置保存成功")
        
        # 加载配置
        new_config = StrategyConfig(test_config_path)
        print("✓ 配置加载成功")
        
        # 验证配置值
        assert new_config.data_config.data_source == "tushare"
        assert new_config.data_config.frequency == "5min"
        assert new_config.portfolio_config.max_position == 0.08
        print("✓ 配置值验证成功")
        
        # 清理临时文件
        if os.path.exists(test_config_path):
            os.remove(test_config_path)
        
        test_results.append(("配置保存和加载", True))
    except Exception as e:
        print(f"✗ 配置保存和加载失败: {e}")
        test_results.append(("配置保存和加载", False))
        traceback.print_exc()
    
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

if __name__ == "__main__":
    success = test_config_functionality()
    sys.exit(0 if success else 1) 