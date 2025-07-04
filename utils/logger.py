#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志工具模块
"""

import logging
import logging.handlers
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional
import traceback


class JSONFormatter(logging.Formatter):
    """JSON格式化器"""
    
    def __init__(self):
        """初始化JSON格式化器"""
        super().__init__()
    
    def format(self, record):
        """格式化日志记录"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'process': record.process
        }
        
        # 添加额外的上下文信息
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)
        
        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry, ensure_ascii=False, indent=2)


class PerformanceLogger:
    """性能日志记录器"""
    
    def __init__(self, logger_name: str = "performance"):
        """初始化性能日志记录器"""
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # 如果没有处理器，添加一个控制台处理器
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(JSONFormatter())
            self.logger.addHandler(handler)
    
    def log_execution_time(self, func_name: str, execution_time: float):
        """记录执行时间"""
        self.logger.info(
            f"Function execution time: {func_name}",
            extra={
                'extra_data': {
                    'event_type': 'execution_time',
                    'function_name': func_name,
                    'execution_time_seconds': execution_time,
                    'execution_time_ms': execution_time * 1000
                }
            }
        )
    
    def log_memory_usage(self, func_name: str, memory_usage: float):
        """记录内存使用"""
        self.logger.info(
            f"Memory usage: {func_name}",
            extra={
                'extra_data': {
                    'event_type': 'memory_usage',
                    'function_name': func_name,
                    'memory_usage_mb': memory_usage,
                    'memory_usage_bytes': memory_usage * 1024 * 1024
                }
            }
        )
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """记录性能指标"""
        self.logger.info(
            "Performance metrics updated",
            extra={
                'extra_data': {
                    'event_type': 'performance_metrics',
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                }
            }
        )


class TradeLogger:
    """交易日志记录器"""
    
    def __init__(self, logger_name: str = "trade"):
        """初始化交易日志记录器"""
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # 如果没有处理器，添加一个控制台处理器
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(JSONFormatter())
            self.logger.addHandler(handler)
    
    def log_trade_signal(self, symbol: str, signal: float, timestamp: datetime):
        """记录交易信号"""
        self.logger.info(
            f"Trade signal generated: {symbol}",
            extra={
                'extra_data': {
                    'event_type': 'trade_signal',
                    'symbol': symbol,
                    'signal': signal,
                    'signal_timestamp': timestamp.isoformat(),
                    'signal_direction': 'long' if signal > 0 else 'short' if signal < 0 else 'neutral'
                }
            }
        )
    
    def log_trade_execution(self, trade_info: Dict[str, Any]):
        """记录交易执行"""
        self.logger.info(
            f"Trade executed: {trade_info.get('symbol', 'Unknown')}",
            extra={
                'extra_data': {
                    'event_type': 'trade_execution',
                    'trade_info': trade_info,
                    'execution_timestamp': datetime.now().isoformat()
                }
            }
        )
    
    def log_portfolio_update(self, portfolio_info: Dict[str, Any]):
        """记录组合更新"""
        self.logger.info(
            "Portfolio updated",
            extra={
                'extra_data': {
                    'event_type': 'portfolio_update',
                    'portfolio_info': portfolio_info,
                    'update_timestamp': datetime.now().isoformat()
                }
            }
        )
    
    def log_rebalance(self, rebalance_info: Dict[str, Any]):
        """记录再平衡"""
        self.logger.info(
            "Portfolio rebalanced",
            extra={
                'extra_data': {
                    'event_type': 'rebalance',
                    'rebalance_info': rebalance_info,
                    'rebalance_timestamp': datetime.now().isoformat()
                }
            }
        )


class ErrorLogger:
    """错误日志记录器"""
    
    def __init__(self, logger_name: str = "error"):
        """初始化错误日志记录器"""
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.ERROR)
        
        # 如果没有处理器，添加一个控制台处理器
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(JSONFormatter())
            self.logger.addHandler(handler)
    
    def log_exception(self, exception: Exception, context: Dict[str, Any] = None):
        """记录异常"""
        self.logger.error(
            f"Exception occurred: {type(exception).__name__}",
            exc_info=True,
            extra={
                'extra_data': {
                    'event_type': 'exception',
                    'exception_type': type(exception).__name__,
                    'exception_message': str(exception),
                    'context': context or {},
                    'error_timestamp': datetime.now().isoformat()
                }
            }
        )
    
    def log_data_quality_issue(self, issue_type: str, details: Dict[str, Any]):
        """记录数据质量问题"""
        self.logger.warning(
            f"Data quality issue: {issue_type}",
            extra={
                'extra_data': {
                    'event_type': 'data_quality_issue',
                    'issue_type': issue_type,
                    'details': details,
                    'timestamp': datetime.now().isoformat()
                }
            }
        )
    
    def log_model_error(self, model_name: str, error_details: Dict[str, Any]):
        """记录模型错误"""
        self.logger.error(
            f"Model error: {model_name}",
            extra={
                'extra_data': {
                    'event_type': 'model_error',
                    'model_name': model_name,
                    'error_details': error_details,
                    'timestamp': datetime.now().isoformat()
                }
            }
        )


class AlertLogger:
    """告警日志记录器"""
    
    def __init__(self, logger_name: str = "alert"):
        """初始化告警日志记录器"""
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.WARNING)
        
        # 如果没有处理器，添加一个控制台处理器
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(JSONFormatter())
            self.logger.addHandler(handler)
    
    def log_risk_alert(self, alert_type: str, alert_details: Dict[str, Any]):
        """记录风险告警"""
        self.logger.warning(
            f"Risk alert: {alert_type}",
            extra={
                'extra_data': {
                    'event_type': 'risk_alert',
                    'alert_type': alert_type,
                    'alert_details': alert_details,
                    'alert_timestamp': datetime.now().isoformat(),
                    'severity': 'high' if alert_details.get('critical', False) else 'medium'
                }
            }
        )
    
    def log_performance_alert(self, alert_details: Dict[str, Any]):
        """记录性能告警"""
        self.logger.warning(
            "Performance alert triggered",
            extra={
                'extra_data': {
                    'event_type': 'performance_alert',
                    'alert_details': alert_details,
                    'alert_timestamp': datetime.now().isoformat()
                }
            }
        )
    
    def log_system_alert(self, alert_details: Dict[str, Any]):
        """记录系统告警"""
        self.logger.warning(
            "System alert triggered",
            extra={
                'extra_data': {
                    'event_type': 'system_alert',
                    'alert_details': alert_details,
                    'alert_timestamp': datetime.now().isoformat()
                }
            }
        )


class Logger:
    """主日志管理器"""
    
    def __init__(self, config: Dict = None):
        """初始化日志管理器"""
        self.config = config or {}
        self.loggers = {}
        self.setup_logging(self.config)
        
        # 创建专用日志记录器
        self.performance_logger = PerformanceLogger()
        self.trade_logger = TradeLogger()
        self.error_logger = ErrorLogger()
        self.alert_logger = AlertLogger()
        
        self.main_logger = self.get_logger("main")
    
    def setup_logging(self, config: Dict):
        """设置日志配置"""
        # 设置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, config.get('level', 'INFO')))
        
        # 清除现有的处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 创建日志目录
        log_dir = os.path.dirname(config.get('file_path', './logs/strategy.log'))
        os.makedirs(log_dir, exist_ok=True)
        
        # 添加文件处理器
        if config.get('file_path'):
            file_handler = self.create_rotating_handler(
                config['file_path'],
                max_bytes=self._parse_size(config.get('max_file_size', '10MB')),
                backup_count=config.get('backup_count', 5)
            )
            root_logger.addHandler(file_handler)
        
        # 添加控制台处理器
        console_handler = self.create_console_handler()
        root_logger.addHandler(console_handler)
    
    def _parse_size(self, size_str: str) -> int:
        """解析文件大小字符串"""
        if isinstance(size_str, int):
            return size_str
        
        size_str = size_str.upper().strip()
        # 按长度排序，优先匹配较长的后缀
        multipliers = [
            ('GB', 1024 * 1024 * 1024),
            ('MB', 1024 * 1024),
            ('KB', 1024),
            ('B', 1)
        ]
        
        for suffix, multiplier in multipliers:
            if size_str.endswith(suffix):
                number_part = size_str[:-len(suffix)].strip()
                if number_part:
                    return int(number_part) * multiplier
        
        return int(size_str)
    
    def create_file_handler(self, log_file: str, level: int = logging.INFO) -> logging.FileHandler:
        """创建文件处理器"""
        handler = logging.FileHandler(log_file, encoding='utf-8')
        handler.setLevel(level)
        handler.setFormatter(JSONFormatter())
        return handler
    
    def create_console_handler(self, level: int = logging.INFO) -> logging.StreamHandler:
        """创建控制台处理器"""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        # 控制台使用简单格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        return handler
    
    def create_rotating_handler(self, log_file: str, 
                              max_bytes: int = 10*1024*1024,
                              backup_count: int = 5) -> logging.handlers.RotatingFileHandler:
        """创建轮转文件处理器"""
        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(JSONFormatter())
        return handler
    
    def get_logger(self, name: str) -> logging.Logger:
        """获取日志记录器"""
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        return self.loggers[name]
    
    def log_strategy_start(self, strategy_name: str, config: Dict):
        """记录策略启动"""
        self.main_logger.info(
            f"Strategy started: {strategy_name}",
            extra={
                'extra_data': {
                    'event_type': 'strategy_start',
                    'strategy_name': strategy_name,
                    'config': config,
                    'start_timestamp': datetime.now().isoformat()
                }
            }
        )
    
    def log_strategy_stop(self, strategy_name: str, summary: Dict):
        """记录策略停止"""
        self.main_logger.info(
            f"Strategy stopped: {strategy_name}",
            extra={
                'extra_data': {
                    'event_type': 'strategy_stop',
                    'strategy_name': strategy_name,
                    'summary': summary,
                    'stop_timestamp': datetime.now().isoformat()
                }
            }
        )
    
    def log_data_update(self, data_type: str, update_info: Dict):
        """记录数据更新"""
        self.main_logger.info(
            f"Data updated: {data_type}",
            extra={
                'extra_data': {
                    'event_type': 'data_update',
                    'data_type': data_type,
                    'update_info': update_info,
                    'update_timestamp': datetime.now().isoformat()
                }
            }
        )
    
    def log_model_training(self, model_info: Dict):
        """记录模型训练"""
        self.main_logger.info(
            f"Model training: {model_info.get('model_name', 'Unknown')}",
            extra={
                'extra_data': {
                    'event_type': 'model_training',
                    'model_info': model_info,
                    'training_timestamp': datetime.now().isoformat()
                }
            }
        )
    
    def log_signal_generation(self, signal_info: Dict):
        """记录信号生成"""
        self.main_logger.info(
            "Signal generation completed",
            extra={
                'extra_data': {
                    'event_type': 'signal_generation',
                    'signal_info': signal_info,
                    'generation_timestamp': datetime.now().isoformat()
                }
            }
        )
    
    def log_portfolio_optimization(self, optimization_info: Dict):
        """记录组合优化"""
        self.main_logger.info(
            "Portfolio optimization completed",
            extra={
                'extra_data': {
                    'event_type': 'portfolio_optimization',
                    'optimization_info': optimization_info,
                    'optimization_timestamp': datetime.now().isoformat()
                }
            }
        )
    
    def info(self, message: str, **kwargs):
        """记录信息级别日志"""
        self.main_logger.info(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """记录错误级别日志"""
        self.main_logger.error(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """记录警告级别日志"""
        self.main_logger.warning(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """记录调试级别日志"""
        self.main_logger.debug(message, **kwargs)