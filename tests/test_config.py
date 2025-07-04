#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置模块测试脚本
"""

import sys
import os
import tempfile
import shutil
# 添加上级目录到路径，以便导入主模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.strategy_config import (
    StrategyConfig, DataConfig, FactorConfig, ModelConfig, 
    PortfolioConfig, RiskConfig, BacktestConfig, LoggingConfig, MonitoringConfig
)

def test_config_module():
    """测试配置模块功能"""
    
    print("============== 配置模块测试 ==============")
    
    # 1. 测试默认配置创建
    print("\n1. 测试默认配置创建...")
    config = StrategyConfig()
    print("✓ 默认配置创建成功")
    
    # 2. 测试配置验证
    print("\n2. 测试配置验证...")
    is_valid = config.validate_config()
    print(f"配置验证结果: {'✓ 通过' if is_valid else '✗ 失败'}")
    
    # 3. 测试配置打印
    print("\n3. 测试配置打印...")
    config.print_config()
    print("✓ 配置打印完成")
    
    # 4. 测试配置更新
    print("\n4. 测试配置更新...")
    config.update_config("data", data_source="tushare", frequency="5min")
    config.update_config("portfolio", max_position=0.08, max_turnover=0.3)
    config.update_config("risk", max_drawdown=0.08, var_confidence=0.99)
    print("✓ 配置更新完成")
    
    # 5. 测试配置保存
    print("\n5. 测试配置保存...")
    temp_dir = tempfile.mkdtemp()
    try:
        test_config_path = os.path.join(temp_dir, "test_config.yaml")
        config.save_config(test_config_path)
        print("✓ 配置保存成功")
        
        # 6. 测试配置加载
        print("\n6. 测试配置加载...")
        new_config = StrategyConfig(test_config_path)
        print("✓ 配置加载成功")
        
        # 验证加载的配置
        assert new_config.data_config.data_source == "tushare"
        assert new_config.data_config.frequency == "5min"
        assert new_config.portfolio_config.max_position == 0.08
        assert new_config.portfolio_config.max_turnover == 0.3
        assert new_config.risk_config.max_drawdown == 0.08
        assert new_config.risk_config.var_confidence == 0.99
        print("✓ 配置值验证成功")
        
        # 7. 测试配置备份
        print("\n7. 测试配置备份...")
        backup_path = new_config.create_backup()
        print(f"✓ 配置备份创建: {backup_path}")
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)
    
    # 8. 测试各个配置类的to_dict和from_dict方法
    print("\n8. 测试配置类序列化...")
    
    # 测试DataConfig
    data_config = DataConfig(data_source="test", frequency="1h")
    data_dict = data_config.to_dict()
    data_config_new = DataConfig.from_dict(data_dict)
    assert data_config_new.data_source == "test"
    assert data_config_new.frequency == "1h"
    print("✓ DataConfig序列化测试成功")
    
    # 测试FactorConfig
    factor_config = FactorConfig(factor_groups=["technical"], lookback_periods=[10, 20])
    factor_dict = factor_config.to_dict()
    factor_config_new = FactorConfig.from_dict(factor_dict)
    assert factor_config_new.factor_groups == ["technical"]
    assert factor_config_new.lookback_periods == [10, 20]
    print("✓ FactorConfig序列化测试成功")
    
    # 测试ModelConfig
    model_config = ModelConfig(model_types=["lightgbm"], max_features=30)
    model_dict = model_config.to_dict()
    model_config_new = ModelConfig.from_dict(model_dict)
    assert model_config_new.model_types == ["lightgbm"]
    assert model_config_new.max_features == 30
    print("✓ ModelConfig序列化测试成功")
    
    # 测试PortfolioConfig
    portfolio_config = PortfolioConfig(max_position=0.1, cycles=["daily"])
    portfolio_dict = portfolio_config.to_dict()
    portfolio_config_new = PortfolioConfig.from_dict(portfolio_dict)
    assert portfolio_config_new.max_position == 0.1
    assert portfolio_config_new.cycles == ["daily"]
    print("✓ PortfolioConfig序列化测试成功")
    
    # 测试RiskConfig
    risk_config = RiskConfig(max_drawdown=0.05, beta_range=(0.5, 1.5))
    risk_dict = risk_config.to_dict()
    risk_config_new = RiskConfig.from_dict(risk_dict)
    assert risk_config_new.max_drawdown == 0.05
    assert risk_config_new.beta_range == (0.5, 1.5)
    print("✓ RiskConfig序列化测试成功")
    
    # 测试BacktestConfig
    backtest_config = BacktestConfig(initial_capital=500000, transaction_cost=0.002)
    backtest_dict = backtest_config.to_dict()
    backtest_config_new = BacktestConfig.from_dict(backtest_dict)
    assert backtest_config_new.initial_capital == 500000
    assert backtest_config_new.transaction_cost == 0.002
    print("✓ BacktestConfig序列化测试成功")
    
    # 测试LoggingConfig
    logging_config = LoggingConfig(level="DEBUG", max_file_size="5MB")
    logging_dict = logging_config.to_dict()
    logging_config_new = LoggingConfig.from_dict(logging_dict)
    assert logging_config_new.level == "DEBUG"
    assert logging_config_new.max_file_size == "5MB"
    print("✓ LoggingConfig序列化测试成功")
    
    # 测试MonitoringConfig
    monitoring_config = MonitoringConfig()
    monitoring_dict = monitoring_config.to_dict()
    monitoring_config_new = MonitoringConfig.from_dict(monitoring_dict)
    assert monitoring_config_new.alert_thresholds is not None
    assert monitoring_config_new.notification is not None
    print("✓ MonitoringConfig序列化测试成功")
    
    # 9. 测试配置验证错误情况
    print("\n9. 测试配置验证错误情况...")
    invalid_config = StrategyConfig()
    invalid_config.portfolio_config.max_position = 1.5  # 无效值
    invalid_config.risk_config.max_drawdown = -0.1  # 无效值
    invalid_config.backtest_config.initial_capital = -1000  # 无效值
    
    is_valid = invalid_config.validate_config()
    assert not is_valid, "配置验证应该失败"
    print("✓ 配置验证错误检测成功")
    
    # 10. 测试获取所有配置
    print("\n10. 测试获取所有配置...")
    all_config = config.get_all_config()
    assert isinstance(all_config, dict)
    assert 'data' in all_config
    assert 'factors' in all_config
    assert 'models' in all_config
    assert 'portfolio' in all_config
    assert 'risk' in all_config
    assert 'backtest' in all_config
    assert 'logging' in all_config
    assert 'monitoring' in all_config
    print("✓ 获取所有配置成功")
    
    print("\n============== 配置模块测试完成 ==============")
    print("所有测试都通过！配置模块功能正常。")

if __name__ == "__main__":
    test_config_module() 