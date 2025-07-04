#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易执行模块测试文件
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# 导入被测试的模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.trade_executor import (
    TradeExecutor, Order, OrderStatus, OrderType, SimulatedExecutor, OrderManager
)


class TestOrder(unittest.TestCase):
    """测试订单类"""
    
    def setUp(self):
        """设置测试环境"""
        self.order = Order(
            symbol='AAPL',
            quantity=100,
            order_type=OrderType.MARKET,
            price=150.0
        )
    
    def test_order_creation(self):
        """测试订单创建"""
        self.assertEqual(self.order.symbol, 'AAPL')
        self.assertEqual(self.order.quantity, 100)
        self.assertEqual(self.order.price, 150.0)
        self.assertEqual(self.order.status, OrderStatus.PENDING)
    
    def test_order_status_update(self):
        """测试订单状态更新"""
        # 更新为已提交
        self.order.update_status(OrderStatus.SUBMITTED)
        self.assertEqual(self.order.status, OrderStatus.SUBMITTED)
        
        # 更新为已成交
        self.order.update_status(OrderStatus.FILLED, fill_quantity=100, fill_price=149.50)
        self.assertEqual(self.order.status, OrderStatus.FILLED)
        self.assertEqual(self.order.fill_quantity, 100)
        self.assertEqual(self.order.fill_price, 149.50)
    
    def test_order_to_dict(self):
        """测试订单转换为字典"""
        order_dict = self.order.to_dict()
        self.assertIsInstance(order_dict, dict)
        self.assertEqual(order_dict['symbol'], 'AAPL')
        self.assertEqual(order_dict['quantity'], 100)
        self.assertEqual(order_dict['price'], 150.0)


class TestSimulatedExecutor(unittest.TestCase):
    """测试模拟执行器"""
    
    def setUp(self):
        """设置测试环境"""
        self.executor = SimulatedExecutor()
    
    def test_simulated_executor_initialization(self):
        """测试模拟执行器初始化"""
        self.assertIsNotNone(self.executor)
        self.assertEqual(self.executor.commission_rate, 0.001)
        self.assertEqual(self.executor.slippage_factor, 0.0005)
    
    def test_single_order_execution(self):
        """测试单订单执行"""
        # 创建测试订单
        order = Order(
            symbol='AAPL',
            quantity=100,
            order_type=OrderType.MARKET,
            price=150.0
        )
        
        # 执行订单
        results = self.executor.execute_orders([order])
        
        # 验证结果
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['symbol'], 'AAPL')
        self.assertEqual(results[0]['status'], 'filled')
        self.assertIn('executed_price', results[0])
        self.assertIn('commission', results[0])
    
    def test_multiple_orders_execution(self):
        """测试多订单执行"""
        # 创建多个测试订单
        orders = [
            Order(symbol='AAPL', quantity=100, order_type=OrderType.MARKET, price=150.0),
            Order(symbol='GOOGL', quantity=50, order_type=OrderType.MARKET, price=2800.0),
            Order(symbol='MSFT', quantity=-75, order_type=OrderType.MARKET, price=300.0)
        ]
        
        # 执行订单
        results = self.executor.execute_orders(orders)
        
        # 验证结果
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn('symbol', result)
            self.assertIn('status', result)
    
    def test_order_cancellation(self):
        """测试订单取消"""
        # 创建测试订单
        order = Order(
            symbol='AAPL',
            quantity=100,
            order_type=OrderType.LIMIT,
            price=150.0
        )
        
        # 添加到待处理订单
        self.executor.pending_orders[order.order_id] = order
        
        # 取消订单
        success = self.executor.cancel_order(order.order_id)
        
        # 验证结果
        self.assertTrue(success)
        self.assertEqual(order.status, OrderStatus.CANCELLED)
    
    def test_market_impact_simulation(self):
        """测试市场冲击模拟"""
        # 创建测试订单
        order = Order(
            symbol='AAPL',
            quantity=10000,  # 大订单
            order_type=OrderType.MARKET,
            price=150.0
        )
        
        # 模拟交易量数据
        volume_data = pd.Series({'AAPL': 1000000})
        
        # 计算市场冲击
        impact = self.executor.simulate_market_impact(order, volume_data)
        
        # 验证结果
        self.assertIsInstance(impact, float)
        self.assertGreater(impact, 0)  # 买入订单正冲击


class TestOrderManager(unittest.TestCase):
    """测试订单管理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.order_manager = OrderManager()
    
    def test_order_manager_initialization(self):
        """测试订单管理器初始化"""
        self.assertIsNotNone(self.order_manager)
        self.assertEqual(len(self.order_manager.active_orders), 0)
    
    def test_order_creation_from_weights(self):
        """测试根据权重创建订单"""
        # 准备测试数据
        target_weights = pd.Series([0.4, 0.3, 0.3], index=['AAPL', 'GOOGL', 'MSFT'])
        current_weights = pd.Series([0.5, 0.2, 0.3], index=['AAPL', 'GOOGL', 'MSFT'])
        portfolio_value = 1000000
        prices = pd.Series([150.0, 2800.0, 300.0], index=['AAPL', 'GOOGL', 'MSFT'])
        
        # 创建订单
        orders = self.order_manager.create_orders_from_weights(
            target_weights, current_weights, portfolio_value, prices
        )
        
        # 验证结果
        self.assertIsInstance(orders, list)
        self.assertGreater(len(orders), 0)
        for order in orders:
            self.assertIsInstance(order, Order)
    
    def test_order_validation(self):
        """测试订单验证"""
        # 创建测试订单
        orders = [
            Order(symbol='AAPL', quantity=100, order_type=OrderType.MARKET, price=150.0),
            Order(symbol='GOOGL', quantity=50, order_type=OrderType.MARKET, price=2800.0)
        ]
        
        # 模拟账户信息
        account_info = {
            'cash': 500000,
            'buying_power': 1000000
        }
        
        # 验证订单
        validated_orders = self.order_manager.validate_orders(orders, account_info)
        
        # 验证结果
        self.assertIsInstance(validated_orders, list)
        self.assertEqual(len(validated_orders), 2)
    
    def test_order_lifecycle_management(self):
        """测试订单生命周期管理"""
        # 创建测试订单
        orders = [
            Order(symbol='AAPL', quantity=100, order_type='market', price=150.0),
            Order(symbol='GOOGL', quantity=50, order_type='market', price=2800.0)
        ]
        
        # 管理订单生命周期
        lifecycle_stats = self.order_manager.manage_order_lifecycle(orders)
        
        # 验证结果
        self.assertIsInstance(lifecycle_stats, dict)
        self.assertIn('submitted', lifecycle_stats)
        self.assertIn('filled', lifecycle_stats)
        self.assertIn('cancelled', lifecycle_stats)
    
    def test_partial_fill_handling(self):
        """测试部分成交处理"""
        # 创建测试订单
        order = Order(
            symbol='AAPL',
            quantity=1000,
            order_type='limit',
            price=150.0
        )
        
        # 部分成交信息
        fill_info = {
            'quantity': 300,
            'price': 149.50
        }
        
        # 处理部分成交
        result = self.order_manager.handle_partial_fills(order, fill_info)
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertEqual(result['total_filled'], 300)
        self.assertEqual(result['remaining'], 700)
        self.assertEqual(order.status, OrderStatus.PARTIAL_FILLED)


class TestTradeExecutor(unittest.TestCase):
    """测试交易执行器主类"""
    
    def setUp(self):
        """设置测试环境"""
        self.executor = TradeExecutor()
    
    def test_trade_executor_initialization(self):
        """测试交易执行器初始化"""
        self.assertIsNotNone(self.executor)
        self.assertIsNotNone(self.executor.order_manager)
        self.assertIsNotNone(self.executor.simulated_executor)
    
    def test_rebalance_execution(self):
        """测试再平衡执行"""
        # 准备测试数据
        target_weights = pd.Series([0.4, 0.3, 0.3], index=['AAPL', 'GOOGL', 'MSFT'])
        current_weights = pd.Series([0.5, 0.2, 0.3], index=['AAPL', 'GOOGL', 'MSFT'])
        
        market_data = {
            'portfolio_value': 1000000,
            'prices': pd.Series([150.0, 2800.0, 300.0], index=['AAPL', 'GOOGL', 'MSFT']),
            'volume_data': {'AAPL': 1000000, 'GOOGL': 500000, 'MSFT': 800000},
            'volatility_data': {'AAPL': 0.2, 'GOOGL': 0.25, 'MSFT': 0.18}
        }
        
        # 执行再平衡
        result = self.executor.execute_rebalance(target_weights, current_weights, market_data)
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        if result['status'] == 'success':
            self.assertIn('orders_executed', result)
            self.assertIn('execution_results', result)
    
    def test_trade_list_generation(self):
        """测试交易清单生成"""
        # 准备测试数据
        target_weights = pd.Series([0.4, 0.3, 0.3], index=['AAPL', 'GOOGL', 'MSFT'])
        current_weights = pd.Series([0.5, 0.2, 0.3], index=['AAPL', 'GOOGL', 'MSFT'])
        portfolio_value = 1000000
        prices = pd.Series([150.0, 2800.0, 300.0], index=['AAPL', 'GOOGL', 'MSFT'])
        
        # 生成交易清单
        orders = self.executor.generate_trade_list(
            target_weights, current_weights, portfolio_value, prices
        )
        
        # 验证结果
        self.assertIsInstance(orders, list)
        for order in orders:
            self.assertIsInstance(order, Order)
    
    def test_execution_schedule_optimization(self):
        """测试执行计划优化"""
        # 创建测试订单
        orders = [
            Order(symbol='AAPL', quantity=100, order_type='market', price=150.0),
            Order(symbol='GOOGL', quantity=50, order_type='market', price=2800.0),
            Order(symbol='MSFT', quantity=-75, order_type='market', price=300.0)
        ]
        
        # 市场条件
        market_conditions = {
            'volume_data': {'AAPL': 1000000, 'GOOGL': 500000, 'MSFT': 800000},
            'volatility_data': {'AAPL': 0.2, 'GOOGL': 0.25, 'MSFT': 0.18}
        }
        
        # 优化执行计划
        optimized_orders = self.executor.optimize_execution_schedule(orders, market_conditions)
        
        # 验证结果
        self.assertIsInstance(optimized_orders, list)
        self.assertEqual(len(optimized_orders), 3)
    
    def test_execution_progress_monitoring(self):
        """测试执行进度监控"""
        # 创建测试订单
        orders = [
            Order(symbol='AAPL', quantity=100, order_type='market', price=150.0),
            Order(symbol='GOOGL', quantity=50, order_type='market', price=2800.0)
        ]
        
        # 模拟订单状态
        orders[0].update_status(OrderStatus.FILLED, 100, 149.50)
        orders[1].update_status(OrderStatus.PARTIAL_FILLED, 25, 2795.00)
        
        # 监控执行进度
        progress = self.executor.monitor_execution_progress(orders)
        
        # 验证结果
        self.assertIsInstance(progress, dict)
        self.assertIn('total_orders', progress)
        self.assertIn('completed_orders', progress)
        self.assertIn('pending_orders', progress)
        self.assertIn('completion_rate', progress)
    
    def test_execution_error_handling(self):
        """测试执行错误处理"""
        # 创建失败的测试订单
        failed_orders = [
            Order(symbol='AAPL', quantity=100, order_type='market', price=150.0),
            Order(symbol='GOOGL', quantity=50, order_type='market', price=2800.0)
        ]
        
        # 设置失败状态
        failed_orders[0].update_status(OrderStatus.REJECTED)
        failed_orders[1].update_status(OrderStatus.PARTIAL_FILLED, 25, 2795.00)
        
        # 处理执行错误
        retry_orders = self.executor.handle_execution_errors(failed_orders)
        
        # 验证结果
        self.assertIsInstance(retry_orders, list)
    
    def test_real_time_execution(self):
        """测试实时执行"""
        # 准备测试数据
        signals = pd.DataFrame({
            'AAPL': [0.4, 0.3, 0.5],
            'GOOGL': [0.3, 0.4, 0.2],
            'MSFT': [0.3, 0.3, 0.3]
        })
        
        current_positions = pd.Series([0.5, 0.2, 0.3], index=['AAPL', 'GOOGL', 'MSFT'])
        
        # 执行实时交易
        result = self.executor.real_time_execution(signals, current_positions)
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        self.assertIn('timestamp', result)


if __name__ == '__main__':
    unittest.main() 