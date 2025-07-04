#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据目录结构规划脚本
"""

import os
import json
from datetime import datetime

def create_data_directory_structure():
    """创建完整的数据目录结构"""
    
    # 定义目录结构
    data_structure = {
        'data/': {
            'raw/': {
                'akshare/': {
                    'daily/': '日线数据',
                    'minute/': '分钟数据',
                    'tick/': '逐笔数据',
                    'fundamental/': '基本面数据',
                    'index/': '指数数据'
                },
                'tushare/': {
                    'daily/': '日线数据',
                    'minute/': '分钟数据',
                    'fundamental/': '基本面数据',
                    'index/': '指数数据'
                },
                'external/': {
                    'news/': '新闻数据',
                    'sentiment/': '情绪数据',
                    'macro/': '宏观经济数据'
                }
            },
            'processed/': {
                'clean/': '清洗后的数据',
                'aligned/': '对齐后的数据',
                'normalized/': '标准化后的数据'
            },
            'factors/': {
                'technical/': {
                    'momentum/': '动量因子',
                    'mean_reversion/': '均值回归因子',
                    'volatility/': '波动率因子',
                    'volume/': '成交量因子'
                },
                'fundamental/': {
                    'quality/': '质量因子',
                    'growth/': '成长因子',
                    'value/': '价值因子',
                    'profitability/': '盈利能力因子'
                },
                'sentiment/': {
                    'market/': '市场情绪因子',
                    'news/': '新闻情绪因子',
                    'social/': '社交媒体情绪因子'
                },
                'alternative/': {
                    'satellite/': '卫星数据因子',
                    'credit/': '信用数据因子',
                    'macro/': '宏观因子'
                }
            },
            'signals/': {
                'daily/': {
                    'individual/': '个股日信号',
                    'sector/': '行业信号',
                    'market/': '市场信号'
                },
                'intraday/': {
                    'minute/': '分钟级信号',
                    'hour/': '小时级信号'
                },
                'multi_horizon/': {
                    'short_term/': '短期信号(1-5天)',
                    'medium_term/': '中期信号(1-4周)',
                    'long_term/': '长期信号(1-3月)'
                }
            },
            'portfolios/': {
                'weights/': {
                    'daily/': '日度权重',
                    'weekly/': '周度权重',
                    'monthly/': '月度权重'
                },
                'positions/': {
                    'target/': '目标仓位',
                    'actual/': '实际仓位',
                    'history/': '历史仓位'
                },
                'performance/': {
                    'returns/': '收益率数据',
                    'metrics/': '性能指标',
                    'attribution/': '归因分析'
                }
            },
            'backtest/': {
                'results/': {
                    'single_factor/': '单因子回测',
                    'multi_factor/': '多因子回测',
                    'strategy/': '策略回测'
                },
                'cache/': '回测缓存',
                'reports/': '回测报告'
            },
            'models/': {
                'trained/': {
                    'factor_models/': '因子模型',
                    'signal_models/': '信号模型',
                    'portfolio_models/': '组合模型'
                },
                'configs/': '模型配置',
                'features/': '特征数据'
            },
            'cache/': {
                'factor_cache/': '因子缓存',
                'signal_cache/': '信号缓存',
                'model_cache/': '模型缓存'
            },
            'logs/': {
                'data_processing/': '数据处理日志',
                'factor_computation/': '因子计算日志',
                'signal_generation/': '信号生成日志',
                'portfolio_optimization/': '组合优化日志',
                'backtest/': '回测日志',
                'execution/': '执行日志'
            }
        }
    }
    
    def create_directories(structure, base_path=''):
        """递归创建目录"""
        for name, content in structure.items():
            current_path = os.path.join(base_path, name)
            
            if isinstance(content, dict):
                # 如果是字典，创建目录并递归
                os.makedirs(current_path, exist_ok=True)
                create_directories(content, current_path)
                
                # 创建README文件
                readme_path = os.path.join(current_path, 'README.md')
                if not os.path.exists(readme_path):
                    with open(readme_path, 'w', encoding='utf-8') as f:
                        f.write(f"# {name}\n\n")
                        f.write(f"创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        f.write("## 目录说明\n\n")
                        for sub_name, sub_content in content.items():
                            if isinstance(sub_content, str):
                                f.write(f"- `{sub_name}`: {sub_content}\n")
                            else:
                                f.write(f"- `{sub_name}`: 子目录\n")
            else:
                # 如果是字符串，创建目录
                os.makedirs(current_path, exist_ok=True)
                
                # 创建说明文件
                readme_path = os.path.join(current_path, 'README.md')
                if not os.path.exists(readme_path):
                    with open(readme_path, 'w', encoding='utf-8') as f:
                        f.write(f"# {name}\n\n")
                        f.write(f"{content}\n\n")
                        f.write(f"创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 创建目录结构
    create_directories(data_structure)
    
    # 创建配置文件
    config_path = 'data/data_config.json'
    data_config = {
        "data_sources": {
            "primary": "akshare",
            "backup": "tushare",
            "external": ["news", "sentiment", "macro"]
        },
        "update_frequency": {
            "daily": ["daily_price", "fundamental"],
            "intraday": ["minute_price", "tick"],
            "weekly": ["sentiment", "news"],
            "monthly": ["macro"]
        },
        "storage_config": {
            "format": "parquet",
            "compression": "snappy",
            "partitioning": ["date", "symbol"]
        },
        "factor_config": {
            "computation_frequency": "daily",
            "lookback_periods": [5, 10, 20, 60, 120],
            "universe_size": 500
        },
        "signal_config": {
            "generation_frequency": "daily",
            "horizons": [1, 5, 10, 20],
            "signal_types": ["ranking", "threshold", "ml"]
        },
        "portfolio_config": {
            "rebalance_frequency": "daily",
            "max_position_size": 0.05,
            "min_position_size": 0.001
        }
    }
    
    os.makedirs('data', exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(data_config, f, indent=2, ensure_ascii=False)
    
    print("=" * 60)
    print("数据目录结构创建完成！")
    print("=" * 60)
    
    # 打印目录结构
    def print_structure(structure, level=0):
        for name, content in structure.items():
            indent = "  " * level
            if isinstance(content, dict):
                print(f"{indent}{name}")
                print_structure(content, level + 1)
            else:
                print(f"{indent}{name} - {content}")
    
    print("\n目录结构:")
    print_structure(data_structure)
    
    print(f"\n配置文件已创建: {config_path}")
    
    # 创建数据管理工具
    create_data_management_tools()

def create_data_management_tools():
    """创建数据管理工具"""
    
    # 创建数据更新脚本
    update_script = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据更新脚本
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
'''
    
    with open('data/scripts/update_daily_data.py', 'w', encoding='utf-8') as f:
        f.write(update_script)
    
    os.makedirs('data/scripts', exist_ok=True)
    
    print("数据管理工具创建完成:")
    print("- data/scripts/update_daily_data.py")

if __name__ == "__main__":
    create_data_directory_structure() 