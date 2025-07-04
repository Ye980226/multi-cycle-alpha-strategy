#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具模块测试文件
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import tempfile
import os

# 导入被测试的模块
from utils import Logger


class TestLogger(unittest.TestCase):
    """测试日志管理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.logger = Logger()
    
    def test_logger_initialization(self):
        """测试日志管理器初始化"""
        self.assertIsNotNone(self.logger)
        self.assertIsNotNone(self.logger.main_logger)
        self.assertIsNotNone(self.logger.performance_logger)
        self.assertIsNotNone(self.logger.trade_logger)
        self.assertIsNotNone(self.logger.error_logger)
    
    def test_main_logger_functionality(self):
        """测试主日志记录器功能"""
        # 获取主日志记录器
        main_logger = self.logger.get_logger("main")
        
        # 验证日志记录器
        self.assertIsNotNone(main_logger)
        
        # 测试日志记录
        main_logger.info("Test info message")
        main_logger.warning("Test warning message")
        main_logger.error("Test error message")
        
        # 验证日志记录成功（通过检查处理器）
        self.assertGreater(len(main_logger.handlers), 0)
    
    def test_performance_logger_functionality(self):
        """测试性能日志记录器功能"""
        # 记录性能指标
        performance_data = {
            'timestamp': datetime.now(),
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.05,
            'volatility': 0.12
        }
        
        # 记录性能日志
        result = self.logger.log_performance(performance_data)
        
        # 验证结果
        self.assertIsInstance(result, bool)
        self.assertTrue(result)
    
    def test_trade_logger_functionality(self):
        """测试交易日志记录器功能"""
        # 记录交易信息
        trade_data = {
            'timestamp': datetime.now(),
            'symbol': 'AAPL',
            'quantity': 100,
            'price': 150.0,
            'order_type': 'market',
            'status': 'filled'
        }
        
        # 记录交易日志
        result = self.logger.log_trade(trade_data)
        
        # 验证结果
        self.assertIsInstance(result, bool)
        self.assertTrue(result)
    
    def test_error_logger_functionality(self):
        """测试错误日志记录器功能"""
        # 记录错误信息
        error_data = {
            'timestamp': datetime.now(),
            'error_type': 'DataError',
            'error_message': 'Failed to fetch market data',
            'stack_trace': 'Mock stack trace',
            'severity': 'high'
        }
        
        # 记录错误日志
        result = self.logger.log_error(error_data)
        
        # 验证结果
        self.assertIsInstance(result, bool)
        self.assertTrue(result)
    
    def test_alert_logger_functionality(self):
        """测试告警日志记录器功能"""
        # 记录告警信息
        alert_data = {
            'timestamp': datetime.now(),
            'alert_type': 'performance_degradation',
            'message': 'Portfolio performance below threshold',
            'severity': 'high',
            'component': 'portfolio_monitor'
        }
        
        # 记录告警日志
        result = self.logger.log_alert(alert_data)
        
        # 验证结果
        self.assertIsInstance(result, bool)
        self.assertTrue(result)
    
    def test_structured_logging(self):
        """测试结构化日志记录"""
        # 准备结构化数据
        structured_data = {
            'event_type': 'strategy_execution',
            'timestamp': datetime.now(),
            'strategy_name': 'multi_cycle_alpha',
            'execution_time': 0.5,
            'success': True,
            'details': {
                'assets_processed': 100,
                'signals_generated': 50,
                'trades_executed': 10
            }
        }
        
        # 记录结构化日志
        result = self.logger.log_structured(structured_data)
        
        # 验证结果
        self.assertIsInstance(result, bool)
        self.assertTrue(result)
    
    def test_log_rotation(self):
        """测试日志轮转"""
        # 测试日志轮转功能
        rotation_result = self.logger.rotate_logs()
        
        # 验证结果
        self.assertIsInstance(rotation_result, bool)
        self.assertTrue(rotation_result)
    
    def test_log_cleanup(self):
        """测试日志清理"""
        # 测试清理旧日志
        cleanup_result = self.logger.cleanup_old_logs(days=30)
        
        # 验证结果
        self.assertIsInstance(cleanup_result, bool)
        self.assertTrue(cleanup_result)
    
    def test_log_analysis(self):
        """测试日志分析"""
        # 分析日志统计
        analysis_result = self.logger.analyze_logs(
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        # 验证结果
        self.assertIsInstance(analysis_result, dict)
        self.assertIn('log_counts', analysis_result)
        self.assertIn('error_summary', analysis_result)
        self.assertIn('performance_summary', analysis_result)
    
    def test_log_export(self):
        """测试日志导出"""
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = os.path.join(temp_dir, 'exported_logs.csv')
            
            # 导出日志
            export_result = self.logger.export_logs(
                export_path=export_path,
                start_date='2023-01-01',
                end_date='2023-12-31'
            )
            
            # 验证结果
            self.assertIsInstance(export_result, bool)
    
    def test_log_filtering(self):
        """测试日志过滤"""
        # 过滤日志
        filtered_logs = self.logger.filter_logs(
            log_type='error',
            severity='high',
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        # 验证结果
        self.assertIsInstance(filtered_logs, list)
    
    def test_log_aggregation(self):
        """测试日志聚合"""
        # 聚合日志统计
        aggregated_stats = self.logger.aggregate_log_stats(
            group_by='hour',
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        # 验证结果
        self.assertIsInstance(aggregated_stats, dict)
        self.assertIn('aggregated_counts', aggregated_stats)
        self.assertIn('time_series', aggregated_stats)
    
    def test_performance_metrics_logging(self):
        """测试性能指标日志记录"""
        # 记录详细性能指标
        metrics = {
            'timestamp': datetime.now(),
            'portfolio_value': 1000000,
            'daily_return': 0.01,
            'volatility': 0.15,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.03,
            'positions': {
                'AAPL': 0.4,
                'GOOGL': 0.3,
                'MSFT': 0.3
            }
        }
        
        # 记录性能日志
        result = self.logger.log_performance_metrics(metrics)
        
        # 验证结果
        self.assertIsInstance(result, bool)
        self.assertTrue(result)
    
    def test_trading_activity_logging(self):
        """测试交易活动日志记录"""
        # 记录交易活动
        trading_activity = {
            'timestamp': datetime.now(),
            'session_id': 'session_123',
            'trades_executed': 5,
            'total_volume': 10000,
            'execution_time': 2.5,
            'success_rate': 0.8,
            'trade_details': [
                {'symbol': 'AAPL', 'quantity': 100, 'price': 150.0},
                {'symbol': 'GOOGL', 'quantity': 50, 'price': 2800.0}
            ]
        }
        
        # 记录交易活动日志
        result = self.logger.log_trading_activity(trading_activity)
        
        # 验证结果
        self.assertIsInstance(result, bool)
        self.assertTrue(result)
    
    def test_system_health_logging(self):
        """测试系统健康日志记录"""
        # 记录系统健康状态
        health_data = {
            'timestamp': datetime.now(),
            'cpu_usage': 0.75,
            'memory_usage': 0.6,
            'disk_usage': 0.4,
            'network_latency': 0.05,
            'active_connections': 10,
            'system_status': 'healthy'
        }
        
        # 记录系统健康日志
        result = self.logger.log_system_health(health_data)
        
        # 验证结果
        self.assertIsInstance(result, bool)
        self.assertTrue(result)
    
    def test_log_compression(self):
        """测试日志压缩"""
        # 测试日志压缩功能
        compression_result = self.logger.compress_logs(
            target_date='2023-01-01',
            compression_format='gzip'
        )
        
        # 验证结果
        self.assertIsInstance(compression_result, bool)
    
    def test_log_backup(self):
        """测试日志备份"""
        # 创建临时备份目录
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_path = os.path.join(temp_dir, 'backup')
            
            # 备份日志
            backup_result = self.logger.backup_logs(backup_path)
            
            # 验证结果
            self.assertIsInstance(backup_result, bool)
    
    def test_log_restoration(self):
        """测试日志恢复"""
        # 创建临时恢复目录
        with tempfile.TemporaryDirectory() as temp_dir:
            restore_path = os.path.join(temp_dir, 'restore')
            
            # 恢复日志
            restore_result = self.logger.restore_logs(restore_path)
            
            # 验证结果
            self.assertIsInstance(restore_result, bool)
    
    def test_real_time_log_monitoring(self):
        """测试实时日志监控"""
        # 启动实时日志监控
        monitoring_result = self.logger.start_real_time_monitoring()
        
        # 验证结果
        self.assertIsInstance(monitoring_result, bool)
        
        # 停止实时日志监控
        stop_result = self.logger.stop_real_time_monitoring()
        self.assertIsInstance(stop_result, bool)
    
    def test_log_configuration_update(self):
        """测试日志配置更新"""
        # 更新日志配置
        new_config = {
            'log_level': 'DEBUG',
            'max_file_size': '100MB',
            'backup_count': 10,
            'compression_enabled': True
        }
        
        # 更新配置
        update_result = self.logger.update_configuration(new_config)
        
        # 验证结果
        self.assertIsInstance(update_result, bool)
        self.assertTrue(update_result)


if __name__ == '__main__':
    unittest.main() 