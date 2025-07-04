#!/usr/bin/env python3
"""
简化的配置模块测试
"""

def test_config_module():
    """测试配置模块基本功能"""
    print("🚀 开始测试配置模块...")
    
    try:
        # 测试导入
        print("1. 测试导入配置模块...")
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from config.strategy_config import StrategyConfig
        print("✅ 配置模块导入成功")
        
        # 测试创建实例
        print("\n2. 测试创建配置实例...")
        config = StrategyConfig()
        print("✅ 配置实例创建成功")
        
        # 测试基本功能
        print("\n3. 测试配置基本功能...")
        
        # 获取配置
        data_config = config.get_data_config()
        portfolio_config = config.get_portfolio_config()
        
        print(f"✅ 数据源: {data_config.data_source}")
        print(f"✅ 频率: {data_config.frequency}")
        print(f"✅ 最大仓位: {portfolio_config.max_position}")
        
        # 更新配置
        config.update_config("data", data_source="test", frequency="5min")
        updated_data_config = config.get_data_config()
        
        if updated_data_config.data_source == "test" and updated_data_config.frequency == "5min":
            print("✅ 配置更新功能正常")
        else:
            print("❌ 配置更新功能异常")
            return False
        
        # 验证配置
        is_valid = config.validate_config()
        if is_valid:
            print("✅ 配置验证通过")
        else:
            print("❌ 配置验证失败")
            return False
        
        # 序列化测试
        all_config = config.get_all_config()
        if isinstance(all_config, dict) and len(all_config) > 0:
            print(f"✅ 配置序列化成功，包含 {len(all_config)} 个部分")
        else:
            print("❌ 配置序列化失败")
            return False
        
        print("\n🎉 配置模块所有测试通过！")
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
    success = test_config_module()
    if success:
        print("🎉 配置模块测试完成！功能正常。")
    else:
        print("⚠️ 配置模块测试失败，需要检查问题。") 