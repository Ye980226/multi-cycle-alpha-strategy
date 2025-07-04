#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易执行模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TWAP = "twap"
    VWAP = "vwap"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class Order:
    """订单类"""
    
    def __init__(self, symbol: str, quantity: float, order_type: OrderType,
                 price: float = None, timestamp: datetime = None):
        """初始化订单"""
        self.symbol = symbol
        self.quantity = quantity
        self.order_type = order_type
        self.price = price
        self.timestamp = timestamp or datetime.now()
        self.status = OrderStatus.PENDING
        self.fill_quantity = 0.0
        self.fill_price = 0.0
        self.order_id = f"{symbol}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"
        self.commission = 0.0
        self.slippage = 0.0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'order_type': self.order_type.value if hasattr(self.order_type, 'value') else self.order_type,
            'price': self.price,
            'timestamp': self.timestamp,
            'status': self.status.value if hasattr(self.status, 'value') else self.status,
            'fill_quantity': self.fill_quantity,
            'fill_price': self.fill_price,
            'commission': self.commission,
            'slippage': self.slippage
        }
    
    def update_status(self, status: OrderStatus, fill_quantity: float = 0,
                     fill_price: float = 0):
        """更新订单状态"""
        self.status = status
        self.fill_quantity += fill_quantity
        if fill_price > 0:
            # 加权平均填充价格
            if self.fill_quantity > fill_quantity:
                total_value = self.fill_price * (self.fill_quantity - fill_quantity) + fill_price * fill_quantity
                self.fill_price = total_value / self.fill_quantity
            else:
                self.fill_price = fill_price


class BaseExecutor(ABC):
    """执行器基类"""
    
    def __init__(self, name: str):
        """初始化执行器"""
        pass
    
    @abstractmethod
    def execute_orders(self, orders: List[Order]) -> List[Dict]:
        """执行订单"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        pass


class SimulatedExecutor(BaseExecutor):
    """模拟执行器"""
    
    def __init__(self, slippage_model: str = "linear",
                 delay_model: str = "constant"):
        """初始化模拟执行器"""
        super().__init__("simulated_executor")
        self.slippage_model = slippage_model
        self.delay_model = delay_model
        self.commission_rate = 0.001  # 0.1%手续费
        self.slippage_factor = 0.0005  # 滑点因子
        self.market_impact_factor = 0.01  # 市场冲击因子
        self.pending_orders = {}
        self.executed_orders = []
    
    def execute_orders(self, orders: List[Order]) -> List[Dict]:
        """模拟执行订单"""
        execution_results = []
        
        for order in orders:
            try:
                # 模拟执行延迟
                execution_delay = self._simulate_execution_delay(order)
                
                # 计算执行价格
                execution_price = self._calculate_execution_price(order)
                
                # 计算手续费
                commission = abs(order.quantity) * execution_price * self.commission_rate
                
                # 更新订单状态
                order.update_status(OrderStatus.FILLED, order.quantity, execution_price)
                order.commission = commission
                
                # 记录执行结果
                result = {
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'executed_quantity': order.quantity,
                    'executed_price': execution_price,
                    'commission': commission,
                    'execution_time': order.timestamp + timedelta(seconds=execution_delay),
                    'status': 'filled'
                }
                
                execution_results.append(result)
                self.executed_orders.append(order)
                
            except Exception as e:
                # 执行失败
                order.update_status(OrderStatus.REJECTED)
                result = {
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'status': 'rejected',
                    'error': str(e)
                }
                execution_results.append(result)
        
        return execution_results
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order.update_status(OrderStatus.CANCELLED)
            del self.pending_orders[order_id]
            return True
        return False
    
    def simulate_market_impact(self, order: Order, volume_data: pd.Series) -> float:
        """模拟市场冲击"""
        if order.symbol not in volume_data.index:
            return 0.0
        
        avg_volume = volume_data.get(order.symbol, 1000000)  # 默认日均量
        
        # 市场冲击与交易量成比例
        volume_ratio = abs(order.quantity) / avg_volume
        market_impact = volume_ratio * self.market_impact_factor
        
        # 买入正冲击，卖出负冲击
        return market_impact if order.quantity > 0 else -market_impact
    
    def simulate_slippage(self, order: Order, volatility: float) -> float:
        """模拟滑点"""
        if self.slippage_model == "linear":
            # 线性滑点模型
            base_slippage = self.slippage_factor
            volatility_slippage = volatility * 0.5
            total_slippage = base_slippage + volatility_slippage
        elif self.slippage_model == "square_root":
            # 平方根滑点模型
            total_slippage = self.slippage_factor * np.sqrt(abs(order.quantity) / 10000)
        else:
            # 固定滑点
            total_slippage = self.slippage_factor
        
        # 买入正滑点，卖出负滑点
        return total_slippage if order.quantity > 0 else -total_slippage
    
    def _simulate_execution_delay(self, order: Order) -> float:
        """模拟执行延迟"""
        if self.delay_model == "constant":
            return 1.0  # 1秒延迟
        elif self.delay_model == "random":
            return np.random.exponential(2.0)  # 指数分布延迟
        else:
            return 0.5
    
    def _calculate_execution_price(self, order: Order) -> float:
        """计算执行价格"""
        base_price = order.price if order.price else 100.0  # 如果没有价格，使用默认值
        
        # 添加滑点
        volatility = 0.02  # 假设2%的波动率
        slippage = self.simulate_slippage(order, volatility)
        
        # 添加市场冲击
        volume_data = pd.Series({order.symbol: 1000000})  # 假设默认交易量
        market_impact = self.simulate_market_impact(order, volume_data)
        
        # 计算最终执行价格
        execution_price = base_price * (1 + slippage + market_impact)
        
        return max(execution_price, 0.01)  # 确保价格为正


class TWAPExecutor(BaseExecutor):
    """TWAP执行器"""
    
    def __init__(self, time_window: int = 60, slice_count: int = 10):
        """初始化TWAP执行器"""
        super().__init__("twap_executor")
        self.time_window = time_window  # 执行时间窗口（分钟）
        self.slice_count = slice_count  # 切片数量
        self.commission_rate = 0.001
        self.min_slice_size = 100  # 最小切片大小
        self.executed_orders = []
        self.active_children = {}
    
    def execute_orders(self, orders: List[Order]) -> List[Dict]:
        """TWAP执行"""
        execution_results = []
        
        for order in orders:
            try:
                # 生成子订单
                child_orders = self.generate_child_orders(order)
                
                # 记录活跃子订单
                self.active_children[order.order_id] = child_orders
                
                # 模拟执行所有子订单
                total_executed_quantity = 0
                total_executed_value = 0
                
                for child_order in child_orders:
                    # 模拟执行延迟
                    execution_delay = child_order.timestamp - order.timestamp
                    
                    # 计算执行价格（带随机波动）
                    price_variation = np.random.normal(0, 0.001)  # 小幅价格波动
                    execution_price = order.price * (1 + price_variation)
                    
                    # 更新子订单状态
                    child_order.update_status(OrderStatus.FILLED, child_order.quantity, execution_price)
                    
                    # 累计执行结果
                    total_executed_quantity += child_order.quantity
                    total_executed_value += child_order.quantity * execution_price
                
                # 计算平均执行价格
                avg_execution_price = total_executed_value / total_executed_quantity if total_executed_quantity > 0 else order.price
                
                # 计算总手续费
                total_commission = abs(total_executed_quantity) * avg_execution_price * self.commission_rate
                
                # 更新父订单状态
                order.update_status(OrderStatus.FILLED, total_executed_quantity, avg_execution_price)
                order.commission = total_commission
                
                # 记录执行结果
                result = {
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'executed_quantity': total_executed_quantity,
                    'executed_price': avg_execution_price,
                    'commission': total_commission,
                    'execution_time': order.timestamp + timedelta(minutes=self.time_window),
                    'status': 'filled',
                    'child_orders_count': len(child_orders),
                    'twap_variance': self._calculate_twap_variance(child_orders)
                }
                
                execution_results.append(result)
                self.executed_orders.append(order)
                
            except Exception as e:
                # 执行失败
                order.update_status(OrderStatus.REJECTED)
                result = {
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'status': 'rejected',
                    'error': str(e)
                }
                execution_results.append(result)
        
        return execution_results
    
    def generate_child_orders(self, parent_order: Order) -> List[Order]:
        """生成子订单"""
        child_orders = []
        
        # 计算切片大小
        slice_sizes = self.calculate_slice_sizes(parent_order.quantity, self.slice_count)
        
        # 计算时间间隔
        time_interval = self.time_window / self.slice_count
        
        # 生成子订单
        for i, slice_size in enumerate(slice_sizes):
            if abs(slice_size) >= self.min_slice_size:
                child_timestamp = parent_order.timestamp + timedelta(minutes=i * time_interval)
                
                child_order = Order(
                    symbol=parent_order.symbol,
                    quantity=slice_size,
                    order_type=OrderType.MARKET,  # 子订单使用市价单
                    price=parent_order.price,
                    timestamp=child_timestamp
                )
                
                # 设置子订单ID
                child_order.order_id = f"{parent_order.order_id}_child_{i}"
                
                child_orders.append(child_order)
        
        return child_orders
    
    def calculate_slice_sizes(self, total_quantity: float, slice_count: int) -> List[float]:
        """计算切片大小"""
        # 基础等分
        base_slice_size = total_quantity / slice_count
        
        # 添加随机变化以避免过于规律
        slice_sizes = []
        remaining_quantity = total_quantity
        
        for i in range(slice_count - 1):
            # 在基础大小附近加入20%的随机变化
            variation = np.random.uniform(-0.2, 0.2)
            slice_size = base_slice_size * (1 + variation)
            
            # 确保不超过剩余数量
            slice_size = min(abs(slice_size), abs(remaining_quantity)) * np.sign(total_quantity)
            
            slice_sizes.append(slice_size)
            remaining_quantity -= slice_size
        
        # 最后一个切片包含剩余数量
        slice_sizes.append(remaining_quantity)
        
        return slice_sizes
    
    def _calculate_twap_variance(self, child_orders: List[Order]) -> float:
        """计算TWAP执行的价格方差"""
        if len(child_orders) < 2:
            return 0.0
        
        prices = [order.fill_price for order in child_orders if order.fill_price > 0]
        if len(prices) < 2:
            return 0.0
        
        return np.var(prices)
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id in self.active_children:
            # 取消所有未执行的子订单
            child_orders = self.active_children[order_id]
            cancelled_count = 0
            
            for child_order in child_orders:
                if child_order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                    child_order.update_status(OrderStatus.CANCELLED)
                    cancelled_count += 1
            
            del self.active_children[order_id]
            return cancelled_count > 0
        
        return False


class VWAPExecutor(BaseExecutor):
    """VWAP执行器"""
    
    def __init__(self, volume_forecast_window: int = 20):
        """初始化VWAP执行器"""
        super().__init__("vwap_executor")
        self.volume_forecast_window = volume_forecast_window  # 成交量预测窗口（天）
        self.commission_rate = 0.001
        self.min_slice_size = 100
        self.executed_orders = []
        self.active_children = {}
        self.volume_cache = {}  # 缓存历史成交量数据
    
    def execute_orders(self, orders: List[Order]) -> List[Dict]:
        """VWAP执行"""
        execution_results = []
        
        for order in orders:
            try:
                # 获取或预测成交量分布
                volume_profile = self._get_volume_profile(order.symbol)
                
                # 生成VWAP切片订单
                child_orders = self.calculate_vwap_slices(order, volume_profile)
                
                # 记录活跃子订单
                self.active_children[order.order_id] = child_orders
                
                # 模拟执行所有子订单
                total_executed_quantity = 0
                total_executed_value = 0
                
                for child_order in child_orders:
                    # 计算基于成交量的执行价格
                    execution_price = self._calculate_vwap_price(child_order, volume_profile)
                    
                    # 更新子订单状态
                    child_order.update_status(OrderStatus.FILLED, child_order.quantity, execution_price)
                    
                    # 累计执行结果
                    total_executed_quantity += child_order.quantity
                    total_executed_value += child_order.quantity * execution_price
                
                # 计算成交量加权平均价格
                vwap_price = total_executed_value / total_executed_quantity if total_executed_quantity > 0 else order.price
                
                # 计算总手续费
                total_commission = abs(total_executed_quantity) * vwap_price * self.commission_rate
                
                # 更新父订单状态
                order.update_status(OrderStatus.FILLED, total_executed_quantity, vwap_price)
                order.commission = total_commission
                
                # 记录执行结果
                result = {
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'executed_quantity': total_executed_quantity,
                    'executed_price': vwap_price,
                    'commission': total_commission,
                    'execution_time': order.timestamp + timedelta(hours=8),  # 假设一个交易日
                    'status': 'filled',
                    'child_orders_count': len(child_orders),
                    'vwap_efficiency': self._calculate_vwap_efficiency(child_orders, volume_profile)
                }
                
                execution_results.append(result)
                self.executed_orders.append(order)
                
            except Exception as e:
                # 执行失败
                order.update_status(OrderStatus.REJECTED)
                result = {
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'status': 'rejected',
                    'error': str(e)
                }
                execution_results.append(result)
        
        return execution_results
    
    def forecast_volume_profile(self, symbol: str, 
                              historical_volume: pd.DataFrame) -> pd.Series:
        """预测成交量分布"""
        if symbol not in historical_volume.columns:
            # 如果没有历史数据，使用标准的日内成交量分布模式
            return self._default_volume_profile()
        
        # 获取历史成交量数据
        historical_data = historical_volume[symbol].dropna()
        
        if len(historical_data) < self.volume_forecast_window:
            return self._default_volume_profile()
        
        # 使用简单移动平均预测
        recent_volumes = historical_data.tail(self.volume_forecast_window)
        
        # 计算日内成交量分布（假设分为24个时段）
        volume_profile = pd.Series(index=range(24), dtype=float)
        
        # 模拟U型成交量分布（开盘和收盘时成交量较大）
        for hour in range(24):
            if 9 <= hour <= 15:  # 交易时段
                if hour in [9, 10, 14, 15]:  # 开盘和收盘时段
                    volume_profile[hour] = recent_volumes.mean() * 1.5
                else:
                    volume_profile[hour] = recent_volumes.mean() * 0.8
            else:
                volume_profile[hour] = 0.0
        
        # 标准化使总和为1
        volume_profile = volume_profile / volume_profile.sum()
        
        return volume_profile
    
    def calculate_vwap_slices(self, order: Order, volume_profile: pd.Series) -> List[Order]:
        """计算VWAP切片"""
        child_orders = []
        
        # 根据成交量分布计算切片大小
        total_quantity = order.quantity
        
        for hour, volume_weight in volume_profile.items():
            if volume_weight > 0:
                # 根据成交量权重分配订单数量
                slice_quantity = total_quantity * volume_weight
                
                if abs(slice_quantity) >= self.min_slice_size:
                    child_timestamp = order.timestamp + timedelta(hours=hour)
                    
                    child_order = Order(
                        symbol=order.symbol,
                        quantity=slice_quantity,
                        order_type=OrderType.MARKET,
                        price=order.price,
                        timestamp=child_timestamp
                    )
                    
                    # 设置子订单ID和成交量权重
                    child_order.order_id = f"{order.order_id}_vwap_{hour}"
                    child_order.volume_weight = volume_weight
                    
                    child_orders.append(child_order)
        
        return child_orders
    
    def _get_volume_profile(self, symbol: str) -> pd.Series:
        """获取成交量分布"""
        if symbol in self.volume_cache:
            return self.volume_cache[symbol]
        
        # 模拟历史成交量数据
        mock_volume_data = pd.DataFrame({
            symbol: np.random.lognormal(10, 0.5, self.volume_forecast_window)
        })
        
        volume_profile = self.forecast_volume_profile(symbol, mock_volume_data)
        self.volume_cache[symbol] = volume_profile
        
        return volume_profile
    
    def _default_volume_profile(self) -> pd.Series:
        """默认的日内成交量分布"""
        # U型分布：开盘和收盘时段成交量较大
        volume_weights = [0.0] * 24
        
        # 设置交易时段的成交量权重
        volume_weights[9] = 0.2   # 开盘
        volume_weights[10] = 0.15
        volume_weights[11] = 0.1
        volume_weights[12] = 0.1
        volume_weights[13] = 0.1
        volume_weights[14] = 0.15
        volume_weights[15] = 0.2  # 收盘
        
        return pd.Series(volume_weights, index=range(24))
    
    def _calculate_vwap_price(self, child_order: Order, volume_profile: pd.Series) -> float:
        """计算基于成交量加权的执行价格"""
        base_price = child_order.price
        
        # 根据成交量权重调整价格影响
        volume_weight = getattr(child_order, 'volume_weight', 0.1)
        
        # 高成交量时段市场冲击较小
        market_impact = 0.001 / (volume_weight + 0.01)
        
        # 添加随机价格变动
        price_variation = np.random.normal(0, 0.001)
        
        # 计算最终执行价格
        direction = 1 if child_order.quantity > 0 else -1
        execution_price = base_price * (1 + direction * market_impact + price_variation)
        
        return execution_price
    
    def _calculate_vwap_efficiency(self, child_orders: List[Order], volume_profile: pd.Series) -> float:
        """计算VWAP执行效率"""
        if not child_orders:
            return 0.0
        
        # 计算实际执行与理论VWAP的偏差
        total_value = sum(order.fill_quantity * order.fill_price for order in child_orders)
        total_quantity = sum(order.fill_quantity for order in child_orders)
        
        if total_quantity == 0:
            return 0.0
        
        actual_vwap = total_value / total_quantity
        
        # 计算理论VWAP（假设在各时段都能获得基准价格）
        base_price = child_orders[0].price
        theoretical_vwap = base_price
        
        # 效率 = 1 - |实际偏差|/基准价格
        efficiency = 1 - abs(actual_vwap - theoretical_vwap) / theoretical_vwap
        
        return max(0.0, min(1.0, efficiency))
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id in self.active_children:
            # 取消所有未执行的子订单
            child_orders = self.active_children[order_id]
            cancelled_count = 0
            
            for child_order in child_orders:
                if child_order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                    child_order.update_status(OrderStatus.CANCELLED)
                    cancelled_count += 1
            
            del self.active_children[order_id]
            return cancelled_count > 0
        
        return False


class SmartOrderRouter:
    """智能订单路由"""
    
    def __init__(self):
        """初始化智能订单路由"""
        self.twap_executor = TWAPExecutor()
        self.vwap_executor = VWAPExecutor()
        self.simulated_executor = SimulatedExecutor()
        
        # 算法选择阈值
        self.large_order_threshold = 50000  # 大订单阈值（美元）
        self.high_volatility_threshold = 0.03  # 高波动率阈值
        self.low_liquidity_threshold = 100000  # 低流动性阈值（日均成交量）
    
    def route_order(self, order: Order, market_conditions: Dict) -> str:
        """路由订单"""
        # 选择最优执行算法
        selected_executor = self.select_execution_algorithm(order, market_conditions)
        
        # 返回选择的算法名称
        return selected_executor.name if hasattr(selected_executor, 'name') else "unknown"
    
    def select_execution_algorithm(self, order: Order, 
                                 market_data: Dict) -> BaseExecutor:
        """选择执行算法"""
        # 获取市场数据
        volume = market_data.get('volume', {}).get(order.symbol, 1000000)
        volatility = market_data.get('volatility', {}).get(order.symbol, 0.02)
        spread = market_data.get('spread', {}).get(order.symbol, 0.001)
        
        # 计算订单价值
        order_value = abs(order.quantity * order.price) if order.price else 0
        
        # 决策逻辑
        if order_value >= self.large_order_threshold:
            # 大订单：根据市场条件选择算法
            if volume < self.low_liquidity_threshold:
                # 低流动性市场：使用TWAP以减少市场冲击
                return self.twap_executor
            elif volatility > self.high_volatility_threshold:
                # 高波动性市场：使用VWAP跟随成交量
                return self.vwap_executor
            else:
                # 正常市场：使用TWAP
                return self.twap_executor
        else:
            # 小订单：使用简单的模拟执行
            return self.simulated_executor
    
    def optimize_execution_timing(self, orders: List[Order],
                                market_conditions: Dict) -> List[Order]:
        """优化执行时机"""
        optimized_orders = []
        
        # 根据市场条件调整执行时间
        market_volatility = market_conditions.get('market_volatility', 0.02)
        trading_volume = market_conditions.get('trading_volume', 1.0)
        
        for order in orders:
            optimized_order = self._optimize_single_order_timing(order, market_conditions)
            optimized_orders.append(optimized_order)
        
        # 按优先级排序
        optimized_orders.sort(key=lambda x: self._calculate_execution_priority(x), reverse=True)
        
        return optimized_orders
    
    def _optimize_single_order_timing(self, order: Order, market_conditions: Dict) -> Order:
        """优化单个订单的执行时机"""
        # 复制订单以避免修改原始订单
        optimized_order = Order(
            symbol=order.symbol,
            quantity=order.quantity,
            order_type=order.order_type,
            price=order.price,
            timestamp=order.timestamp
        )
        
        # 获取市场条件
        volatility = market_conditions.get('volatility', {}).get(order.symbol, 0.02)
        volume = market_conditions.get('volume', {}).get(order.symbol, 1000000)
        
        # 根据市场条件调整执行时间
        current_hour = order.timestamp.hour
        
        if volatility > self.high_volatility_threshold:
            # 高波动率：延迟到成交量较大的时段
            if current_hour < 10:
                optimized_order.timestamp = order.timestamp.replace(hour=10)
            elif current_hour > 14:
                optimized_order.timestamp = order.timestamp.replace(hour=14)
        
        elif volume < self.low_liquidity_threshold:
            # 低流动性：分散到多个时段
            if order.order_type in [OrderType.TWAP, OrderType.VWAP]:
                # 对于算法订单，延长执行窗口
                optimized_order.execution_window = getattr(order, 'execution_window', 60) * 1.5
        
        return optimized_order
    
    def _calculate_execution_priority(self, order: Order) -> float:
        """计算执行优先级"""
        base_priority = 1.0
        
        # 订单价值因子
        order_value = abs(order.quantity * order.price) if order.price else 0
        value_factor = min(order_value / 100000, 3.0)  # 最大3倍权重
        
        # 订单类型因子
        type_factor = 1.0
        if order.order_type == OrderType.MARKET:
            type_factor = 1.5  # 市价单优先级更高
        elif order.order_type in [OrderType.TWAP, OrderType.VWAP]:
            type_factor = 1.2  # 算法订单中等优先级
        
        # 时间因子（卖出订单优先级稍高）
        direction_factor = 1.1 if order.quantity < 0 else 1.0
        
        return base_priority * value_factor * type_factor * direction_factor
    
    def analyze_execution_venues(self, order: Order, available_venues: List[str]) -> Dict:
        """分析执行场所"""
        venue_analysis = {}
        
        for venue in available_venues:
            # 模拟场所分析
            venue_score = self._calculate_venue_score(order, venue)
            venue_analysis[venue] = {
                'score': venue_score,
                'estimated_cost': self._estimate_venue_cost(order, venue),
                'estimated_speed': self._estimate_execution_speed(order, venue),
                'liquidity_score': self._estimate_venue_liquidity(order, venue)
            }
        
        return venue_analysis
    
    def _calculate_venue_score(self, order: Order, venue: str) -> float:
        """计算执行场所评分"""
        # 基础评分（模拟）
        base_scores = {
            'NYSE': 0.8,
            'NASDAQ': 0.85,
            'BATS': 0.75,
            'IEX': 0.7,
            'DARK_POOL': 0.6
        }
        
        base_score = base_scores.get(venue, 0.5)
        
        # 根据订单特征调整评分
        order_value = abs(order.quantity * order.price) if order.price else 0
        
        if order_value > 100000:  # 大订单
            if venue == 'DARK_POOL':
                base_score += 0.2  # 大订单偏好暗池
        else:  # 小订单
            if venue in ['NYSE', 'NASDAQ']:
                base_score += 0.1  # 小订单偏好主要交易所
        
        return min(1.0, base_score)
    
    def _estimate_venue_cost(self, order: Order, venue: str) -> float:
        """估算执行场所成本"""
        # 基础成本（点差的倍数）
        base_costs = {
            'NYSE': 0.5,
            'NASDAQ': 0.4,
            'BATS': 0.3,
            'IEX': 0.6,
            'DARK_POOL': 0.2
        }
        
        return base_costs.get(venue, 0.5)
    
    def _estimate_execution_speed(self, order: Order, venue: str) -> float:
        """估算执行速度（秒）"""
        # 基础执行时间
        base_speeds = {
            'NYSE': 0.1,
            'NASDAQ': 0.08,
            'BATS': 0.05,
            'IEX': 0.35,  # IEX有速度降低机制
            'DARK_POOL': 0.2
        }
        
        return base_speeds.get(venue, 0.1)
    
    def _estimate_venue_liquidity(self, order: Order, venue: str) -> float:
        """估算执行场所流动性"""
        # 流动性评分（0-1）
        liquidity_scores = {
            'NYSE': 0.9,
            'NASDAQ': 0.95,
            'BATS': 0.7,
            'IEX': 0.6,
            'DARK_POOL': 0.4
        }
        
        return liquidity_scores.get(venue, 0.5)


class OrderManager:
    """订单管理器"""
    
    def __init__(self):
        """初始化订单管理器"""
        self.active_orders = {}
        self.order_history = []
        self.min_order_size = 100  # 最小订单金额
        self.max_order_size = 1000000  # 最大订单金额
    
    def create_orders_from_weights(self, target_weights: pd.Series,
                                 current_weights: pd.Series,
                                 portfolio_value: float,
                                 prices: pd.Series) -> List[Order]:
        """从权重创建订单"""
        orders = []
        
        # 对齐索引
        all_symbols = target_weights.index.union(current_weights.index)
        target_aligned = target_weights.reindex(all_symbols, fill_value=0)
        current_aligned = current_weights.reindex(all_symbols, fill_value=0)
        
        for symbol in all_symbols:
            if symbol not in prices.index:
                continue
                
            target_weight = target_aligned[symbol]
            current_weight = current_aligned[symbol]
            price = prices[symbol]
            
            # 计算权重变化
            weight_change = target_weight - current_weight
            
            # 跳过微小变化
            if abs(weight_change) < 0.001:
                continue
            
            # 计算交易数量（以股数为单位）
            target_value = target_weight * portfolio_value
            current_value = current_weight * portfolio_value
            trade_value = target_value - current_value
            
            # 转换为股数（向下取整到100股的倍数）
            shares = int(trade_value / price / 100) * 100
            
            if abs(shares * price) >= self.min_order_size:
                # 确定订单类型
                order_type = OrderType.MARKET  # 默认使用市价单
                
                # 创建订单
                order = Order(
                    symbol=symbol,
                    quantity=shares,
                    order_type=order_type,
                    price=price,
                    timestamp=datetime.now()
                )
                
                orders.append(order)
        
        return orders
    
    def validate_orders(self, orders: List[Order],
                       account_info: Dict) -> List[Order]:
        """验证订单"""
        valid_orders = []
        available_cash = account_info.get('cash', 0)
        
        for order in orders:
            # 验证订单大小
            order_value = abs(order.quantity * order.price) if order.price else 0
            
            if order_value < self.min_order_size:
                continue
            
            if order_value > self.max_order_size:
                continue
            
            # 验证现金充足性（仅对买入订单）
            if order.quantity > 0:
                if order_value > available_cash:
                    # 调整订单大小到可用现金
                    adjusted_quantity = int(available_cash / order.price / 100) * 100
                    if adjusted_quantity > 0:
                        order.quantity = adjusted_quantity
                        available_cash -= adjusted_quantity * order.price
                        valid_orders.append(order)
                else:
                    available_cash -= order_value
                    valid_orders.append(order)
            else:
                # 卖出订单不需要验证现金
                valid_orders.append(order)
        
        return valid_orders
    
    def manage_order_lifecycle(self, orders: List[Order]) -> Dict:
        """管理订单生命周期"""
        lifecycle_stats = {
            'submitted': 0,
            'filled': 0,
            'partially_filled': 0,
            'cancelled': 0,
            'rejected': 0
        }
        
        for order in orders:
            # 添加到活跃订单
            self.active_orders[order.order_id] = order
            
            # 更新状态为已提交
            order.update_status(OrderStatus.SUBMITTED)
            lifecycle_stats['submitted'] += 1
            
            # 模拟订单处理过程
            self._process_order(order, lifecycle_stats)
        
        return lifecycle_stats
    
    def _process_order(self, order: Order, stats: Dict):
        """处理单个订单"""
        # 简化的订单处理逻辑
        import random
        
        # 90%概率完全成交
        if random.random() < 0.9:
            order.update_status(OrderStatus.FILLED, order.quantity, order.price)
            stats['filled'] += 1
            self.order_history.append(order)
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]
        
        # 5%概率部分成交
        elif random.random() < 0.95:
            partial_quantity = order.quantity * random.uniform(0.3, 0.8)
            order.update_status(OrderStatus.PARTIAL_FILLED, partial_quantity, order.price)
            stats['partially_filled'] += 1
        
        # 5%概率被拒绝
        else:
            order.update_status(OrderStatus.REJECTED)
            stats['rejected'] += 1
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]
    
    def handle_partial_fills(self, order: Order, fill_info: Dict):
        """处理部分成交"""
        fill_quantity = fill_info.get('quantity', 0)
        fill_price = fill_info.get('price', order.price)
        
        # 更新订单
        order.update_status(OrderStatus.PARTIAL_FILLED, fill_quantity, fill_price)
        
        # 检查是否完全成交
        if order.fill_quantity >= order.quantity:
            order.update_status(OrderStatus.FILLED)
            self.order_history.append(order)
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]
        
        return {
            'order_id': order.order_id,
            'total_filled': order.fill_quantity,
            'remaining': order.quantity - order.fill_quantity,
            'avg_fill_price': order.fill_price
        }


class ExecutionCostAnalyzer:
    """执行成本分析器"""
    
    def __init__(self):
        """初始化执行成本分析器"""
        pass
    
    def calculate_implementation_shortfall(self, orders: List[Order],
                                         decision_prices: pd.Series) -> Dict:
        """计算执行偏差"""
        pass
    
    def analyze_market_impact(self, trades: pd.DataFrame,
                            volume_data: pd.DataFrame) -> pd.DataFrame:
        """分析市场冲击"""
        pass
    
    def calculate_timing_costs(self, execution_times: pd.Series,
                             price_moves: pd.Series) -> pd.Series:
        """计算时机成本"""
        pass
    
    def benchmark_execution_quality(self, actual_trades: pd.DataFrame,
                                  benchmark_prices: pd.DataFrame) -> Dict:
        """基准执行质量"""
        pass


class RiskControls:
    """风险控制"""
    
    def __init__(self, config: Dict):
        """初始化风险控制"""
        pass
    
    def pre_trade_checks(self, orders: List[Order],
                        current_positions: pd.Series) -> List[bool]:
        """交易前检查"""
        pass
    
    def position_limit_check(self, order: Order,
                           current_position: float,
                           limit: float) -> bool:
        """仓位限制检查"""
        pass
    
    def concentration_check(self, orders: List[Order],
                          current_weights: pd.Series) -> bool:
        """集中度检查"""
        pass
    
    def leverage_check(self, orders: List[Order],
                      current_leverage: float,
                      max_leverage: float) -> bool:
        """杠杆检查"""
        pass


class TradeExecutor:
    """交易执行主类"""
    
    def __init__(self, config: Dict = None):
        """初始化交易执行器"""
        self.config = config or {}
        self.order_manager = OrderManager()
        self.simulated_executor = SimulatedExecutor()
        self.execution_cost_analyzer = ExecutionCostAnalyzer()
        self.risk_controls = RiskControls(self.config)
        self.execution_history = []
        self.performance_metrics = {}
        
    def initialize(self, config: Dict):
        """初始化配置"""
        self.config.update(config)
        self.risk_controls.config.update(config)
        
    def execute_rebalance(self, target_weights: pd.Series,
                         current_weights: pd.Series,
                         market_data: Dict) -> Dict:
        """执行再平衡"""
        try:
            # 生成交易列表
            portfolio_value = market_data.get('portfolio_value', 1000000)
            prices = market_data.get('prices', pd.Series())
            
            if len(prices) == 0:
                return {'status': 'failed', 'error': 'No price data available'}
            
            orders = self.generate_trade_list(target_weights, current_weights, portfolio_value, prices)
            
            if len(orders) == 0:
                return {'status': 'success', 'message': 'No trades needed'}
            
            # 风险检查
            risk_checks = self.risk_controls.pre_trade_checks(orders, current_weights)
            if not all(risk_checks):
                failed_checks = [check for check, passed in risk_checks.items() if not passed]
                return {'status': 'failed', 'error': f'Risk checks failed: {failed_checks}'}
            
            # 优化执行计划
            optimized_orders = self.optimize_execution_schedule(orders, market_data)
            
            # 执行订单
            execution_results = self.simulated_executor.execute_orders(optimized_orders)
            
            # 监控执行进度
            execution_progress = self.monitor_execution_progress(optimized_orders)
            
            # 计算执行指标
            trades_df = pd.DataFrame([order.to_dict() for order in optimized_orders])
            execution_metrics = self.calculate_execution_metrics(trades_df)
            
            # 记录执行历史
            execution_record = {
                'timestamp': datetime.now(),
                'target_weights': target_weights,
                'current_weights': current_weights,
                'orders': [order.to_dict() for order in optimized_orders],
                'execution_results': execution_results,
                'execution_metrics': execution_metrics
            }
            self.execution_history.append(execution_record)
            
            return {
                'status': 'success',
                'orders_executed': len(optimized_orders),
                'execution_results': execution_results,
                'execution_metrics': execution_metrics,
                'execution_progress': execution_progress
            }
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    def generate_trade_list(self, target_weights: pd.Series,
                          current_weights: pd.Series,
                          portfolio_value: float,
                          prices: pd.Series) -> List[Order]:
        """生成交易清单"""
        # 使用订单管理器生成订单
        orders = self.order_manager.create_orders_from_weights(
            target_weights, current_weights, portfolio_value, prices
        )
        
        # 验证订单
        account_info = {'cash': portfolio_value * 0.1}  # 假设10%现金可用
        validated_orders = self.order_manager.validate_orders(orders, account_info)
        
        return validated_orders
    
    def optimize_execution_schedule(self, orders: List[Order],
                                  market_conditions: Dict) -> List[Order]:
        """优化执行计划"""
        # 按执行优先级排序
        optimized_orders = sorted(orders, key=self._calculate_execution_priority, reverse=True)
        
        # 根据市场条件调整执行时机
        volume_data = market_conditions.get('volume_data', {})
        volatility_data = market_conditions.get('volatility_data', {})
        
        for order in optimized_orders:
            # 根据流动性调整订单类型
            if order.symbol in volume_data:
                volume = volume_data[order.symbol]
                if volume < 100000:  # 低流动性
                    order.order_type = OrderType.LIMIT  # 使用限价单
                    
            # 根据波动率调整执行策略
            if order.symbol in volatility_data:
                volatility = volatility_data[order.symbol]
                if volatility > 0.05:  # 高波动率
                    # 可以考虑分批执行
                    pass
        
        return optimized_orders
    
    def _calculate_execution_priority(self, order: Order) -> float:
        """计算执行优先级"""
        # 基于订单大小、流动性等因素计算优先级
        base_priority = 1.0
        
        # 订单越大，优先级越高
        size_factor = min(abs(order.quantity) / 10000, 2.0)
        
        # 卖出订单优先级稍高（释放资金）
        direction_factor = 1.1 if order.quantity < 0 else 1.0
        
        return base_priority * size_factor * direction_factor
    
    def monitor_execution_progress(self, orders: List[Order]) -> Dict:
        """监控执行进度"""
        progress = {
            'total_orders': len(orders),
            'completed_orders': 0,
            'pending_orders': 0,
            'failed_orders': 0,
            'total_value': 0,
            'executed_value': 0
        }
        
        for order in orders:
            progress['total_value'] += abs(order.quantity * order.price) if order.price else 0
            
            if order.status == OrderStatus.FILLED:
                progress['completed_orders'] += 1
                progress['executed_value'] += abs(order.fill_quantity * order.fill_price)
            elif order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED]:
                progress['pending_orders'] += 1
                if order.status == OrderStatus.PARTIAL_FILLED:
                    progress['executed_value'] += abs(order.fill_quantity * order.fill_price)
            else:
                progress['failed_orders'] += 1
        
        progress['completion_rate'] = progress['completed_orders'] / progress['total_orders'] if progress['total_orders'] > 0 else 0
        progress['execution_rate'] = progress['executed_value'] / progress['total_value'] if progress['total_value'] > 0 else 0
        
        return progress
    
    def handle_execution_errors(self, failed_orders: List[Order]) -> List[Order]:
        """处理执行错误"""
        retry_orders = []
        
        for order in failed_orders:
            if order.status == OrderStatus.REJECTED:
                # 尝试降低订单大小重试
                if abs(order.quantity) > 100:
                    retry_order = Order(
                        symbol=order.symbol,
                        quantity=order.quantity * 0.5,  # 减半重试
                        order_type=OrderType.LIMIT,     # 改为限价单
                        price=order.price * 0.99 if order.quantity > 0 else order.price * 1.01,  # 调整价格
                        timestamp=datetime.now()
                    )
                    retry_orders.append(retry_order)
            
            elif order.status == OrderStatus.PARTIAL_FILLED:
                # 处理剩余数量
                remaining_quantity = order.quantity - order.fill_quantity
                if abs(remaining_quantity) > 100:
                    remaining_order = Order(
                        symbol=order.symbol,
                        quantity=remaining_quantity,
                        order_type=order.order_type,
                        price=order.price,
                        timestamp=datetime.now()
                    )
                    retry_orders.append(remaining_order)
        
        return retry_orders
    
    def calculate_execution_metrics(self, trades: pd.DataFrame) -> Dict:
        """计算执行指标"""
        if len(trades) == 0:
            return {}
        
        metrics = {}
        
        # 基本统计
        metrics['total_trades'] = len(trades)
        metrics['total_volume'] = trades['quantity'].abs().sum() if 'quantity' in trades.columns else 0
        metrics['total_value'] = (trades['quantity'].abs() * trades['price']).sum() if 'price' in trades.columns else 0
        
        # 执行成本
        if 'commission' in trades.columns:
            metrics['total_commission'] = trades['commission'].sum()
            metrics['avg_commission_rate'] = metrics['total_commission'] / metrics['total_value'] if metrics['total_value'] > 0 else 0
        
        # 成功率
        filled_trades = trades[trades['status'] == 'filled'] if 'status' in trades.columns else trades
        metrics['fill_rate'] = len(filled_trades) / len(trades) if len(trades) > 0 else 0
        
        # 价格指标
        if 'price' in trades.columns and len(trades) > 0:
            metrics['avg_execution_price'] = trades['price'].mean()
            metrics['price_std'] = trades['price'].std()
        
        return metrics
    
    def real_time_execution(self, signals: pd.DataFrame,
                          current_positions: pd.Series) -> Dict:
        """实时执行"""
        execution_results = []
        
        # 获取最新信号
        if len(signals) == 0:
            return {'status': 'no_signals', 'results': []}
        
        latest_signals = signals.iloc[-1]
        
        # 转换信号为权重
        target_weights = latest_signals / latest_signals.abs().sum() if latest_signals.abs().sum() > 0 else latest_signals
        
        # 获取市场数据（模拟）
        market_data = {
            'portfolio_value': 1000000,
            'prices': pd.Series(100, index=target_weights.index),  # 假设价格都是100
            'volume_data': {},
            'volatility_data': {}
        }
        
        # 执行再平衡
        result = self.execute_rebalance(target_weights, current_positions, market_data)
        
        return {
            'status': 'completed',
            'execution_result': result,
            'timestamp': datetime.now()
        } 