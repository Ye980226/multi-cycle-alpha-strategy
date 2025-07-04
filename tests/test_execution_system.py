#!/usr/bin/env python3
"""
交易执行模块测试
"""

def test_execution_module():
    """测试交易执行模块基本功能"""
    print("🚀 开始测试交易执行模块...")
    
    try:
        # 测试导入
        print("1. 测试导入交易执行模块...")
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from execution.execution_engine import (
            ExecutionEngine, OrderManager, OrderType,
            Order, Trade, ExecutionAlgorithm
        )
        print("✅ 交易执行模块导入成功")
        
        # 测试初始化
        print("\n2. 测试交易执行引擎初始化...")
        import pandas as pd
        import numpy as np
        from datetime import datetime
        
        # 创建测试数据
        np.random.seed(42)
        
        # 生成市场数据
        market_data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'] * 100,
            'price': np.random.uniform(100, 500, 500),
            'volume': np.random.uniform(1000, 10000, 500),
            'timestamp': pd.date_range('2023-01-01', periods=500, freq='T')
        })
        
        print("✅ 测试数据生成成功")
        
        # 测试订单创建
        print("\n3. 测试订单创建...")
        order = Order(
            symbol='AAPL',
            order_type=OrderType.MARKET,
            side='BUY',
            quantity=100,
            price=None,
            timestamp=datetime.now()
        )
        
        if order.symbol == 'AAPL' and order.quantity == 100:
            print(f"✅ 订单创建成功: {order.symbol} {order.side} {order.quantity}")
        else:
            print("❌ 订单创建失败")
            return False
        
        # 测试订单管理器
        print("\n4. 测试订单管理器...")
        order_manager = OrderManager()
        
        # 添加多个订单
        orders = [
            Order('AAPL', OrderType.MARKET, 'BUY', 100, None, datetime.now()),
            Order('MSFT', OrderType.LIMIT, 'SELL', 50, 300.0, datetime.now()),
            Order('GOOGL', OrderType.STOP, 'BUY', 25, 2500.0, datetime.now())
        ]
        
        for order in orders:
            order_manager.submit_order(order)
        
        if len(order_manager.get_pending_orders()) == 3:
            print(f"✅ 订单管理成功: {len(order_manager.get_pending_orders())}个待处理订单")
        else:
            print("❌ 订单管理失败")
            return False
        
        # 测试执行引擎
        print("\n5. 测试执行引擎...")
        config = {
            'commission_rate': 0.001,
            'slippage_model': 'linear',
            'market_impact_model': 'sqrt',
            'execution_delay': 0.1
        }
        
        execution_engine = ExecutionEngine(config)
        
        # 执行市价单
        market_order = orders[0]
        execution_result = execution_engine.execute_order(
            order=market_order,
            market_data=market_data[market_data['symbol'] == 'AAPL'].iloc[0]
        )
        
        if isinstance(execution_result, Trade):
            print(f"✅ 市价单执行成功: 成交价格={execution_result.executed_price:.2f}")
        else:
            print("❌ 市价单执行失败")
            return False
        
        # 测试限价单执行
        print("\n6. 测试限价单执行...")
        limit_order = orders[1]
        limit_execution = execution_engine.execute_order(
            order=limit_order,
            market_data=market_data[market_data['symbol'] == 'MSFT'].iloc[0]
        )
        
        if limit_execution is not None:
            print(f"✅ 限价单处理成功")
        else:
            print("✅ 限价单未触发（正常）")
        
        # 测试执行算法
        print("\n7. 测试执行算法...")
        algo_config = {
            'algorithm': 'VWAP',
            'participation_rate': 0.1,
            'time_horizon': 3600  # 1小时
        }
        
        execution_algo = ExecutionAlgorithm(algo_config)
        
        # 创建大订单进行算法执行
        large_order = Order('AAPL', OrderType.MARKET, 'BUY', 10000, None, datetime.now())
        
        algo_execution = execution_algo.execute_algorithmic_order(
            order=large_order,
            market_data=market_data[market_data['symbol'] == 'AAPL']
        )
        
        if isinstance(algo_execution, list) and len(algo_execution) > 0:
            print(f"✅ 算法执行成功: 分解为{len(algo_execution)}个子订单")
        else:
            print("❌ 算法执行失败")
            return False
        
        # 测试滑点计算
        print("\n8. 测试滑点计算...")
        slippage = execution_engine.calculate_slippage(
            order_size=1000,
            average_volume=5000,
            volatility=0.02,
            spread=0.01
        )
        
        if isinstance(slippage, float) and slippage >= 0:
            print(f"✅ 滑点计算成功: 滑点={slippage:.4f}")
        else:
            print("❌ 滑点计算失败")
            return False
        
        # 测试市场冲击成本
        print("\n9. 测试市场冲击成本...")
        market_impact = execution_engine.calculate_market_impact(
            order_size=1000,
            daily_volume=100000,
            price=150.0,
            volatility=0.02
        )
        
        if isinstance(market_impact, float):
            print(f"✅ 市场冲击计算成功: 冲击成本={market_impact:.4f}")
        else:
            print("❌ 市场冲击计算失败")
            return False
        
        # 测试执行成本分析
        print("\n10. 测试执行成本分析...")
        execution_costs = execution_engine.analyze_execution_costs(
            trades=[execution_result] + algo_execution[:3],  # 取前几个交易
            benchmark_price=150.0
        )
        
        if isinstance(execution_costs, dict) and 'total_cost' in execution_costs:
            print(f"✅ 执行成本分析成功: 总成本={execution_costs['total_cost']:.4f}")
        else:
            print("❌ 执行成本分析失败")
            return False
        
        # 测试执行绩效报告
        print("\n11. 测试执行绩效报告...")
        performance_report = execution_engine.generate_execution_report(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31)
        )
        
        if isinstance(performance_report, dict) and 'summary' in performance_report:
            print(f"✅ 执行绩效报告生成成功: 总交易数={performance_report['summary']['total_trades']}")
        else:
            print("❌ 执行绩效报告生成失败")
            return False
        
        print("\n🎉 交易执行模块核心功能测试通过！")
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
    success = test_execution_module()
    if success:
        print("🎉 交易执行模块测试完成！核心功能正常。")
    else:
        print("⚠️ 交易执行模块测试失败，需要检查问题。") 