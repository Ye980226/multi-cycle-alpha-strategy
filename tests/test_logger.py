#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志模块测试脚本
"""

import sys
import os
# 添加上级目录到路径，以便导入主模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import Logger, PerformanceLogger, TradeLogger, ErrorLogger, AlertLogger
from datetime import datetime
import time

def test_logger_module():
    """测试日志模块功能"""
    
    # 测试主日志管理器
    print("测试主日志管理器...")
    logger_config = {
        'level': 'INFO',
        'file_path': './logs/test.log',
        'max_file_size': '10MB',
        'backup_count': 3
    }
    
    main_logger = Logger(logger_config)
    main_logger.log_strategy_start("TestStrategy", {"test": "config"})
    print("✓ 主日志记录器测试完成")
    
    # 测试性能日志记录器
    print("\n测试性能日志记录器...")
    perf_logger = PerformanceLogger()
    perf_logger.log_execution_time("test_function", 0.123)
    perf_logger.log_memory_usage("test_function", 50.5)
    perf_logger.log_performance_metrics({
        'sharpe_ratio': 1.5,
        'max_drawdown': 0.08,
        'annual_return': 0.15
    })
    print("✓ 性能日志记录器测试完成")
    
    # 测试交易日志记录器
    print("\n测试交易日志记录器...")
    trade_logger = TradeLogger()
    trade_logger.log_trade_signal("000001.SZ", 0.8, datetime.now())
    trade_logger.log_trade_execution({
        'symbol': '000001.SZ',
        'side': 'buy',
        'quantity': 100,
        'price': 12.34
    })
    trade_logger.log_portfolio_update({
        'total_value': 1000000,
        'cash': 100000,
        'positions': {'000001.SZ': 100}
    })
    print("✓ 交易日志记录器测试完成")
    
    # 测试错误日志记录器
    print("\n测试错误日志记录器...")
    error_logger = ErrorLogger()
    try:
        # 故意创建一个错误
        1 / 0
    except Exception as e:
        error_logger.log_exception(e, {'context': 'test_division'})
    
    error_logger.log_data_quality_issue("missing_data", {
        'symbol': '000001.SZ',
        'missing_fields': ['volume', 'turnover']
    })
    
    error_logger.log_model_error("lightgbm_model", {
        'error_type': 'training_failure',
        'details': 'Insufficient data'
    })
    print("✓ 错误日志记录器测试完成")
    
    # 测试告警日志记录器
    print("\n测试告警日志记录器...")
    alert_logger = AlertLogger()
    alert_logger.log_risk_alert("high_drawdown", {
        'current_drawdown': 0.09,
        'threshold': 0.08,
        'critical': True
    })
    alert_logger.log_performance_alert({
        'metric': 'sharpe_ratio',
        'current_value': 0.3,
        'threshold': 0.5
    })
    alert_logger.log_system_alert({
        'alert_type': 'high_memory_usage',
        'usage_percent': 85
    })
    print("✓ 告警日志记录器测试完成")
    
    # 测试更多主日志功能
    print("\n测试更多主日志功能...")
    main_logger.log_data_update("stock_data", {
        'symbols': ['000001.SZ', '000002.SZ'],
        'date_range': '2024-01-01 to 2024-01-31'
    })
    main_logger.log_model_training({
        'model_name': 'test_model',
        'training_samples': 1000,
        'validation_score': 0.85
    })
    main_logger.log_signal_generation({
        'total_signals': 50,
        'long_signals': 30,
        'short_signals': 20
    })
    main_logger.log_portfolio_optimization({
        'method': 'mean_variance',
        'num_assets': 30,
        'optimization_time': 2.5
    })
    print("✓ 更多主日志功能测试完成")
    
    print("\n所有日志模块测试完成！")
    print("请查看 ./logs/test.log 文件以验证日志输出")

if __name__ == "__main__":
    test_logger_module() 