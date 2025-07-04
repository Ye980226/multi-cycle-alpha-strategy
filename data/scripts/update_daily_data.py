#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据更新脚本
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.data_manager import DataManager
from config.strategy_config import StrategyConfig

def update_daily_data():
    """更新日线数据"""
    print(f"开始更新日线数据 - {datetime.now()}")
    
    try:
        config = StrategyConfig()
        config.load_config()
        data_manager = DataManager(config.data_config)
        
        # 获取股票列表
        symbols = data_manager.get_stock_list()
        print(f"获取到 {len(symbols)} 只股票")
        
        # 获取最近一个交易日的数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)  # 获取最近5天
        
        for symbol in symbols[:10]:  # 先测试前10只
            try:
                data = data_manager.get_stock_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    frequency='daily'
                )
                
                if not data.empty:
                    # 保存数据
                    save_path = f"data/raw/akshare/daily/{symbol}.parquet"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    data.to_parquet(save_path)
                    print(f"  ✅ {symbol}: {len(data)} 条记录")
                else:
                    print(f"  ❌ {symbol}: 无数据")
                    
            except Exception as e:
                print(f"  ❌ {symbol}: {e}")
                
    except Exception as e:
        print(f"更新失败: {e}")
        
    print(f"数据更新完成 - {datetime.now()}")

if __name__ == "__main__":
    update_daily_data() 