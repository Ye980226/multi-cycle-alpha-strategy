#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多周期Alpha策略主入口文件
"""

import sys
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.strategy_config import StrategyConfig
from data.data_manager import DataManager
from factors.factor_engine import FactorEngine
from models.model_manager import ModelManager
from signals.signal_generator import SignalGenerator
from portfolio.portfolio_optimizer import PortfolioOptimizer
from backtest.backtest_engine import BacktestEngine
from risk.risk_manager import RiskManager
from execution.trade_executor import TradeExecutor
from monitoring.performance_monitor import PerformanceMonitor
from utils.logger import Logger


class MultiCycleAlphaStrategy:
    """多周期Alpha策略主类"""
    
    def __init__(self, config_path: str = None):
        """初始化策略"""
        self.config = StrategyConfig(config_path)
        self.logger = Logger().get_logger("strategy")
        
        # 初始化各个组件
        self.data_manager = None
        self.factor_engine = None
        self.model_manager = None
        self.signal_generator = None
        self.portfolio_optimizer = None
        self.backtest_engine = None
        self.risk_manager = None
        self.trade_executor = None
        self.performance_monitor = None
        
        # 策略状态
        self.is_initialized = False
        self.is_running = False
        self.current_positions = pd.Series(dtype=float)
        self.current_signals = pd.DataFrame()
        
        self.logger.info("多周期Alpha策略初始化完成")
    
    def initialize(self):
        """初始化所有组件"""
        try:
            self.logger.info("开始初始化策略组件...")
            
            # 初始化数据管理器
            self.data_manager = DataManager(
                data_source=self.config.data_config.data_source,
                cache_enabled=self.config.data_config.cache_enabled
            )
            self.data_manager.initialize(self.config.data_config.__dict__)
            
            # 初始化因子引擎
            self.factor_engine = FactorEngine(self.config.factor_config.__dict__)
            self.factor_engine.initialize()
            
            # 初始化模型管理器
            self.model_manager = ModelManager(self.config.model_config.__dict__)
            self.model_manager.initialize(self.config.model_config.__dict__)
            
            # 初始化信号生成器
            self.signal_generator = SignalGenerator(self.config.factor_config.__dict__)
            self.signal_generator.initialize(self.config.factor_config.__dict__)
            
            # 初始化组合优化器
            self.portfolio_optimizer = PortfolioOptimizer(self.config.portfolio_config.__dict__)
            self.portfolio_optimizer.initialize(self.config.portfolio_config.__dict__)
            
            # 初始化回测引擎
            self.backtest_engine = BacktestEngine(self.config.backtest_config.__dict__)
            self.backtest_engine.initialize(self.config.backtest_config.__dict__)
            
            # 初始化风险管理器
            self.risk_manager = RiskManager(self.config.risk_config.__dict__)
            self.risk_manager.initialize(self.config.risk_config.__dict__)
            
            # 初始化交易执行器
            self.trade_executor = TradeExecutor(self.config.portfolio_config.__dict__)
            self.trade_executor.initialize(self.config.portfolio_config.__dict__)
            
            # 初始化性能监控器
            self.performance_monitor = PerformanceMonitor(self.config.monitoring_config.__dict__)
            self.performance_monitor.initialize(self.config.monitoring_config.__dict__)
            
            self.is_initialized = True
            self.logger.info("策略组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"策略组件初始化失败: {e}")
            raise
    
    def run_backtest(self, start_date: str, end_date: str):
        """运行回测"""
        try:
            if not self.is_initialized:
                self.initialize()
            
            self.logger.info(f"开始运行回测: {start_date} 到 {end_date}")
            
            # 获取股票池
            universe = self.data_manager.get_universe(
                self.config.data_config.universe_name
            )
            
            # 获取历史数据
            data = self.data_manager.get_stock_data(
                symbols=universe,
                start_date=start_date,
                end_date=end_date,
                frequency=self.config.data_config.frequency
            )
            
            if data.empty:
                self.logger.warning("没有获取到数据，回测终止")
                return None
            
            # 计算因子
            factors = self.factor_engine.calculate_factors(data)
            
            # 生成信号
            signals = self.signal_generator.generate_signals(
                factors=factors,
                method=self.config.factor_config.signal_method
            )
            
            # 运行回测
            backtest_results = self.backtest_engine.run_backtest(
                signals=signals,
                universe=universe,
                start_date=start_date,
                end_date=end_date
            )
            
            # 生成回测报告
            self.generate_backtest_report(backtest_results)
            
            self.logger.info("回测完成")
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"回测运行失败: {e}")
            raise
    
    def run_live_trading(self):
        """运行实盘交易"""
        try:
            if not self.is_initialized:
                self.initialize()
            
            self.logger.info("开始实盘交易")
            self.is_running = True
            
            while self.is_running:
                try:
                    # 获取当前时间
                    current_time = datetime.now()
                    
                    # 检查是否在交易时间
                    if not self._is_trading_time(current_time):
                        continue
                    
                    # 获取最新数据
                    universe = self.data_manager.get_universe(
                        self.config.data_config.universe_name
                    )
                    
                    # 获取最新因子数据
                    latest_data = self._get_latest_data(universe)
                    
                    if latest_data.empty:
                        continue
                    
                    # 计算因子
                    factors = self.factor_engine.calculate_factors(latest_data)
                    
                    # 生成信号
                    signals = self.signal_generator.real_time_signal_generation(factors)
                    
                    # 组合优化
                    target_weights = self.portfolio_optimizer.optimize_portfolio(
                        signals=signals,
                        current_weights=self.current_positions
                    )
                    
                    # 风险检查
                    risk_check = self.risk_manager.pre_trade_risk_check(
                        target_weights=target_weights,
                        current_weights=self.current_positions,
                        market_data=latest_data
                    )
                    
                    if risk_check.get('all_checks_passed', False):
                        # 执行交易
                        execution_results = self.trade_executor.real_time_execution(
                            signals=signals,
                            current_positions=self.current_positions
                        )
                        
                        # 更新持仓
                        self.current_positions = execution_results.get('new_positions', self.current_positions)
                        
                        # 监控性能
                        self.performance_monitor.real_time_monitoring(execution_results)
                        
                    else:
                        self.logger.warning(f"风险检查未通过: {risk_check}")
                    
                    # 等待下一个周期
                    self._wait_for_next_cycle()
                    
                except Exception as e:
                    self.logger.error(f"实盘交易过程中发生错误: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"实盘交易失败: {e}")
            raise
    
    def generate_research_report(self):
        """生成研究报告"""
        try:
            self.logger.info("生成研究报告")
            
            # 获取性能数据
            performance_data = self.performance_monitor.generate_performance_summary()
            
            # 生成报告
            report = {
                'strategy_name': '多周期Alpha策略',
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'performance_summary': performance_data,
                'risk_analysis': self.risk_manager.generate_risk_report(performance_data),
                'factor_analysis': self.factor_engine.generate_factor_report(),
                'model_performance': self.model_manager.get_model_performance_summary(),
                'execution_analysis': self.trade_executor.get_execution_summary()
            }
            
            # 保存报告
            report_path = f"./results/research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            import json
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"研究报告已生成: {report_path}")
            return report
            
        except Exception as e:
            self.logger.error(f"生成研究报告失败: {e}")
            raise
    
    def update_models(self):
        """更新模型"""
        try:
            self.logger.info("更新模型")
            
            # 获取最新训练数据
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=252)).strftime('%Y-%m-%d')
            
            universe = self.data_manager.get_universe(
                self.config.data_config.universe_name
            )
            
            data = self.data_manager.get_stock_data(
                symbols=universe,
                start_date=start_date,
                end_date=end_date,
                frequency=self.config.data_config.frequency
            )
            
            if not data.empty:
                # 计算因子
                factors = self.factor_engine.calculate_factors(data)
                
                # 计算前向收益率
                returns = self.factor_engine.calculate_forward_returns(data)
                
                # 重新训练模型
                self.model_manager.retrain_models(factors, returns)
                
                self.logger.info("模型更新完成")
            else:
                self.logger.warning("没有获取到数据，模型更新跳过")
                
        except Exception as e:
            self.logger.error(f"模型更新失败: {e}")
            raise
    
    def rebalance_portfolio(self):
        """组合再平衡"""
        try:
            self.logger.info("执行组合再平衡")
            
            # 获取当前信号
            universe = self.data_manager.get_universe(
                self.config.data_config.universe_name
            )
            
            latest_data = self._get_latest_data(universe)
            
            if not latest_data.empty:
                # 计算因子
                factors = self.factor_engine.calculate_factors(latest_data)
                
                # 生成信号
                signals = self.signal_generator.generate_signals(
                    factors=factors,
                    method=self.config.factor_config.signal_method
                )
                
                # 组合优化
                target_weights = self.portfolio_optimizer.optimize_portfolio(
                    signals=signals,
                    current_weights=self.current_positions
                )
                
                # 执行再平衡
                rebalance_results = self.trade_executor.execute_rebalance(
                    target_weights=target_weights,
                    current_weights=self.current_positions,
                    market_data=latest_data
                )
                
                # 更新持仓
                self.current_positions = rebalance_results.get('new_positions', self.current_positions)
                
                self.logger.info("组合再平衡完成")
                return rebalance_results
            else:
                self.logger.warning("没有获取到数据，组合再平衡跳过")
                
        except Exception as e:
            self.logger.error(f"组合再平衡失败: {e}")
            raise
    
    def generate_backtest_report(self, backtest_results: Dict):
        """生成回测报告"""
        try:
            if not backtest_results:
                return
            
            # 生成详细报告
            report = {
                'backtest_summary': backtest_results,
                'risk_metrics': self.risk_manager.calculate_risk_metrics(backtest_results),
                'performance_attribution': self.performance_monitor.calculate_performance_attribution(backtest_results),
                'factor_analysis': self.factor_engine.analyze_factor_performance(backtest_results),
                'execution_analysis': self.trade_executor.analyze_execution_costs(backtest_results)
            }
            
            # 保存报告
            report_path = f"./results/backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            import json
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"回测报告已生成: {report_path}")
            
        except Exception as e:
            self.logger.error(f"生成回测报告失败: {e}")
    
    def _is_trading_time(self, current_time: datetime) -> bool:
        """检查是否在交易时间"""
        # 简化实现，实际应根据交易所时间判断
        weekday = current_time.weekday()
        hour = current_time.hour
        
        # 周一到周五，9:30-15:00
        if 0 <= weekday <= 4 and 9 <= hour <= 15:
            return True
        return False
    
    def _get_latest_data(self, universe: List[str]) -> pd.DataFrame:
        """获取最新数据"""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
            
            data = self.data_manager.get_stock_data(
                symbols=universe,
                start_date=start_date,
                end_date=end_date,
                frequency=self.config.data_config.frequency
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"获取最新数据失败: {e}")
            return pd.DataFrame()
    
    def _wait_for_next_cycle(self):
        """等待下一个周期"""
        import time
        # 根据频率决定等待时间
        frequency = self.config.data_config.frequency
        if frequency == '1min':
            time.sleep(60)
        elif frequency == '5min':
            time.sleep(300)
        elif frequency == '15min':
            time.sleep(900)
        elif frequency == '30min':
            time.sleep(1800)
        elif frequency == '1h':
            time.sleep(3600)
        else:
            time.sleep(3600)  # 默认1小时
    
    def stop_trading(self):
        """停止交易"""
        self.is_running = False
        self.logger.info("停止实盘交易")
    
    def get_current_status(self) -> Dict:
        """获取当前状态"""
        return {
            'is_initialized': self.is_initialized,
            'is_running': self.is_running,
            'current_positions': self.current_positions.to_dict() if not self.current_positions.empty else {},
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


if __name__ == "__main__":
    # 示例用法
    strategy = MultiCycleAlphaStrategy()
    strategy.initialize()
    
    # 运行回测
    backtest_results = strategy.run_backtest("2023-01-01", "2024-01-01")
    
    # 生成研究报告
    strategy.generate_research_report()
    
    # 更新模型
    strategy.update_models()
    
    # 执行组合再平衡
    strategy.rebalance_portfolio()
    
    print("策略运行完成！") 