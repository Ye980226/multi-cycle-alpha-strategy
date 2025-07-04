#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用真实数据的完整测试脚本
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.strategy_config import StrategyConfig
from data.data_manager import DataManager
from factors.factor_engine import FactorEngine
from signals.signal_generator import SignalGenerator
from portfolio.portfolio_optimizer import PortfolioOptimizer
from models.model_manager import ModelManager

def test_real_data_complete():
    """使用真实数据的完整测试"""
    print("=" * 80)
    print("开始真实数据完整测试")
    print("=" * 80)
    
    # 1. 初始化配置
    print("\n1. 初始化配置...")
    try:
        config = StrategyConfig()
        config.load_config('./config/default_config.yaml')
        print("   ✅ 配置加载成功")
    except Exception as e:
        print(f"   ❌ 配置加载失败: {e}")
        return
    
    # 2. 初始化数据管理器
    print("\n2. 初始化数据管理器...")
    try:
        data_manager = DataManager(config.data_config)
        print("   ✅ 数据管理器初始化成功")
    except Exception as e:
        print(f"   ❌ 数据管理器初始化失败: {e}")
        return
    
    # 3. 获取股票列表
    print("\n3. 获取股票列表...")
    try:
        # 使用配置中的测试股票
        if hasattr(config.data_config, 'test') and 'symbols' in config.data_config.test:
            test_symbols = config.data_config.test['symbols']
        else:
            # 使用默认测试股票
            test_symbols = ['000001', '000002', '600000', '600036', '000858']
        
        print(f"   ✅ 使用测试股票: {test_symbols}")
        
    except Exception as e:
        print(f"   ❌ 获取股票列表失败: {e}")
        return
    
    # 4. 获取真实数据
    print("\n4. 获取真实价格数据...")
    try:
        # 获取最近3个月的数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        all_data = pd.DataFrame()
        
        for symbol in test_symbols:
            try:
                # 获取日线数据
                daily_data = data_manager.get_stock_data(
                    symbols=[symbol],
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    frequency='1d'
                )
                
                if not daily_data.empty:
                    daily_data['symbol'] = symbol
                    all_data = pd.concat([all_data, daily_data])
                    print(f"   ✅ {symbol}: {len(daily_data)} 条日线数据")
                    
                    # 保存数据到对应目录
                    save_path = f"data/raw/akshare/daily/{symbol}.parquet"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    daily_data.to_parquet(save_path)
                    
                else:
                    print(f"   ❌ {symbol}: 无数据")
                    
            except Exception as e:
                print(f"   ❌ {symbol}: {e}")
                continue
        
        if all_data.empty:
            print("   ❌ 未获取到任何数据")
            return
        
        print(f"   ✅ 总共获取到 {len(all_data)} 条数据记录")
        
    except Exception as e:
        print(f"   ❌ 获取数据失败: {e}")
        return
    
    # 5. 数据预处理
    print("\n5. 数据预处理...")
    try:
        # 透视表格式化
        price_data = all_data.reset_index()
        # 检查索引名称
        if 'datetime' in price_data.columns:
            price_pivot = price_data.pivot(index='datetime', columns='symbol', values='close')
        elif 'date' in price_data.columns:
            price_pivot = price_data.pivot(index='date', columns='symbol', values='close')
        else:
            # 如果没有找到日期列，使用原索引
            price_pivot = all_data.pivot_table(index=all_data.index, columns='symbol', values='close', aggfunc='first')
        
        # 填充缺失值
        price_pivot = price_pivot.ffill().bfill()
        
        # 保存处理后的数据
        os.makedirs('data/processed/clean', exist_ok=True)
        price_pivot.to_parquet('data/processed/clean/daily_prices.parquet')
        
        print(f"   ✅ 数据预处理完成: {price_pivot.shape}")
        
    except Exception as e:
        print(f"   ❌ 数据预处理失败: {e}")
        return
    
    # 6. 计算因子
    print("\n6. 计算技术因子...")
    try:
        factor_engine = FactorEngine(config.factor_config.__dict__)
        
        # 使用原始数据计算技术因子（选择第一只股票作为示例）
        sample_stock = test_symbols[0]
        sample_data = all_data[all_data['symbol'] == sample_stock].copy()
        
        # 确保数据格式正确
        sample_data = sample_data.reset_index().set_index('datetime') if 'datetime' in sample_data.columns else sample_data
        
        factors = factor_engine.calculate_technical_factors(sample_data)
        
        if not factors.empty:
            print(f"   ✅ 计算了 {len(factors.columns)} 个技术因子")
            
            # 保存因子数据
            os.makedirs('data/factors/technical', exist_ok=True)
            factors.to_parquet('data/factors/technical/technical_factors.parquet')
            
            # 显示因子统计信息
            print(f"   因子数据形状: {factors.shape}")
            print(f"   因子数据时间范围: {factors.index.min()} - {factors.index.max()}")
            
        else:
            print("   ❌ 未计算到任何因子")
            return
        
    except Exception as e:
        print(f"   ❌ 计算因子失败: {e}")
        return
    
    # 7. 生成交易信号
    print("\n7. 生成交易信号...")
    try:
        signal_generator = SignalGenerator({})
        
        # 只选择数值列进行信号生成
        numeric_columns = factors.select_dtypes(include=[np.number]).columns
        factor_data = factors[numeric_columns]
        
        # 生成因子信号
        factor_signals = signal_generator.generate_signals(factor_data, method='ranking')
        
        # 将因子信号转换为股票信号（简化版本：使用因子信号的均值作为股票信号）
        if not factor_signals.empty:
            # 创建股票信号DataFrame，列名为股票代码
            stock_symbols = test_symbols[:4]  # 使用前4只股票
            signals = pd.DataFrame(index=factor_signals.index, columns=stock_symbols)
            
            # 简化处理：将因子信号的均值作为每只股票的信号
            # 在实际应用中，这里应该是更复杂的因子到股票的映射
            for stock in stock_symbols:
                signals[stock] = factor_signals.mean(axis=1)
                
            print(f"   ✅ 生成了股票信号: {signals.shape}")
        else:
            signals = pd.DataFrame()
        
        if not signals.empty:
            print(f"   ✅ 生成了 {signals.shape} 信号")
            
            # 保存信号数据
            os.makedirs('data/signals/daily/individual', exist_ok=True)
            signals.to_parquet('data/signals/daily/individual/ranking_signals.parquet')
            
            # 显示信号统计信息
            print(f"   信号统计:")
            print(f"   - 买入信号(>0): {(signals > 0).sum().sum()}")
            print(f"   - 卖出信号(<0): {(signals < 0).sum().sum()}")
            print(f"   - 中性信号(=0): {(signals == 0).sum().sum()}")
            
        else:
            print("   ❌ 未生成任何信号")
            return
        
    except Exception as e:
        print(f"   ❌ 生成信号失败: {e}")
        return
    
    # 8. 组合优化
    print("\n8. 组合优化...")
    try:
        portfolio_optimizer = PortfolioOptimizer(config.portfolio_config.__dict__)
        
        # 构建股票收益率数据
        stock_returns = pd.DataFrame()
        
        # 从价格透视表计算收益率
        if 'price_pivot' in locals() and not price_pivot.empty:
            stock_returns = price_pivot.pct_change().dropna()
            print(f"   ✅ 从价格透视表计算收益率: {stock_returns.shape}")
        else:
            # 从原始数据构建收益率
            for symbol in stock_symbols:
                symbol_data = all_data[all_data['symbol'] == symbol].copy()
                if not symbol_data.empty:
                    # 计算收益率
                    returns = symbol_data['close'].pct_change().dropna()
                    stock_returns[symbol] = returns
        
        # 对齐时间索引 - 确保股票收益率和信号数据的时间索引匹配
        if not stock_returns.empty:
            stock_returns = stock_returns.dropna()
            
            # 与信号数据对齐时间索引
            common_dates = stock_returns.index.intersection(signals.index)
            if len(common_dates) > 0:
                stock_returns = stock_returns.loc[common_dates]
                aligned_signals = signals.loc[common_dates]
                print(f"   ✅ 构建了股票收益率数据: {stock_returns.shape}")
                print(f"   ✅ 对齐后的信号数据: {aligned_signals.shape}")
                print(f"   ✅ 共同时间点: {len(common_dates)} 个")
                
                # 使用最近10天的数据进行组合优化
                recent_returns = stock_returns.tail(10)
                recent_signals = aligned_signals.tail(10)
                
                print(f"   用于优化的数据:")
                print(f"   - 收益率数据: {recent_returns.shape}")
                print(f"   - 信号数据: {recent_signals.shape}")
                
                # 使用Barra模型进行组合优化
                weights = portfolio_optimizer.optimize_portfolio(
                    recent_signals, 
                    returns_data=recent_returns,
                    method='barra'
                )
            else:
                print("   ❌ 股票收益率和信号数据没有共同的时间点")
                return
        else:
            print("   ❌ 无法构建股票收益率数据")
            return
        
        if weights is not None and len(weights) > 0:
            print(f"   ✅ 组合优化完成: {len(weights)} 个权重")
            
            # 保存权重数据
            os.makedirs('data/portfolios/weights/daily', exist_ok=True)
            try:
                if isinstance(weights, pd.Series):
                    weights_df = weights.to_frame('weight')
                    weights_df.to_parquet('data/portfolios/weights/daily/latest_weights.parquet')
                else:
                    weights.to_parquet('data/portfolios/weights/daily/latest_weights.parquet')
                print(f"   ✅ 权重数据已保存")
            except Exception as save_error:
                print(f"   ⚠️ 权重保存失败: {save_error}")
                # 使用CSV作为备选方案
                try:
                    if isinstance(weights, pd.Series):
                        weights.to_csv('data/portfolios/weights/daily/latest_weights.csv')
                    else:
                        weights.to_csv('data/portfolios/weights/daily/latest_weights.csv')
                    print(f"   ✅ 权重数据已保存为CSV格式")
                except Exception as csv_error:
                    print(f"   ❌ CSV保存也失败: {csv_error}")
            
            # 显示权重统计信息
            print(f"   权重统计:")
            print(f"   - 总权重: {weights.sum():.4f}")
            print(f"   - 最大权重: {weights.max():.4f}")
            print(f"   - 最小权重: {weights.min():.4f}")
            
            # 显示具体权重
            print(f"   具体权重:")
            for asset, weight in weights.items():
                print(f"   - {asset}: {weight:.4f}")
                
        else:
            print("   ❌ 组合优化失败")
            return
        
    except Exception as e:
        print(f"   ❌ 组合优化失败: {e}")
        return
    
    # 9. 计算组合收益率
    print("\n9. 计算组合收益率...")
    try:
        # 使用股票收益率数据
        if 'stock_returns' in locals() and not stock_returns.empty:
            returns = stock_returns
        else:
            # 从价格数据计算收益率
            returns = price_pivot.pct_change().dropna()
        
        # 计算组合收益率
        portfolio_returns = pd.Series(index=returns.index, dtype=float)
        
        # 使用对齐的信号数据
        signal_data = aligned_signals if 'aligned_signals' in locals() else signals
        
        for date in returns.index:
            if date in signal_data.index:
                signal_row = signal_data.loc[date]
                return_row = returns.loc[date]
                
                # 计算加权收益率
                portfolio_return = (signal_row * return_row).sum() / len(signal_row)
                portfolio_returns.loc[date] = portfolio_return
        
        portfolio_returns = portfolio_returns.dropna()
        
        if not portfolio_returns.empty:
            print(f"   ✅ 计算了 {len(portfolio_returns)} 个组合收益率")
            
            # 保存收益率数据
            os.makedirs('data/portfolios/performance/returns', exist_ok=True)
            try:
                if isinstance(portfolio_returns, pd.Series):
                    portfolio_returns.to_frame('return').to_parquet('data/portfolios/performance/returns/portfolio_returns.parquet')
                else:
                    portfolio_returns.to_parquet('data/portfolios/performance/returns/portfolio_returns.parquet')
                print(f"   ✅ 收益率数据已保存")
            except Exception as save_error:
                print(f"   ⚠️ 收益率保存失败: {save_error}")
                # 使用CSV作为备选方案
                try:
                    portfolio_returns.to_csv('data/portfolios/performance/returns/portfolio_returns.csv')
                    print(f"   ✅ 收益率数据已保存为CSV格式")
                except Exception as csv_error:
                    print(f"   ❌ CSV保存也失败: {csv_error}")
            
            # 计算基本统计信息
            cumulative_return = (1 + portfolio_returns).cumprod() - 1
            total_return = cumulative_return.iloc[-1]
            annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            print(f"   性能统计:")
            print(f"   - 总收益率: {total_return:.4f}")
            print(f"   - 年化收益率: {annualized_return:.4f}")
            print(f"   - 年化波动率: {volatility:.4f}")
            print(f"   - 夏普比率: {sharpe_ratio:.4f}")
            
        else:
            print("   ❌ 未计算到组合收益率")
            
    except Exception as e:
        print(f"   ❌ 计算组合收益率失败: {e}")
    
    # 10. 模型训练和测试
    print("\n10. 模型训练和测试...")
    try:
        model_manager = ModelManager(config.model_config.__dict__)
        
        # 准备训练数据
        X = factors.iloc[:-5].fillna(0)  # 特征
        y = returns.iloc[5:].mean(axis=1)  # 目标：下一期的平均收益率
        
        # 确保数据长度一致
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len]
        y = y.iloc[:min_len]
        
        if len(X) > 20:  # 确保有足够的数据
            # 创建线性模型
            model = model_manager.create_model('linear')
            
            # 训练模型
            model.fit(X, y)
            
            # 预测
            predictions = model.predict(X)
            
            # 评估模型
            evaluator = model_manager.model_evaluator
            metrics = evaluator.evaluate_regression(y.values, predictions)
            
            print(f"   ✅ 模型训练完成")
            print(f"   模型评估指标:")
            for metric, value in metrics.items():
                print(f"   - {metric}: {value:.4f}")
                
            # 保存模型
            os.makedirs('data/models/trained/signal_models', exist_ok=True)
            # 这里可以添加模型保存逻辑
            
        else:
            print("   ❌ 数据不足，无法训练模型")
            
    except Exception as e:
        print(f"   ❌ 模型训练失败: {e}")
    
    # 11. 生成报告
    print("\n11. 生成测试报告...")
    try:
        report = {
            'test_date': datetime.now().isoformat(),
            'data_range': f"{start_date.date()} - {end_date.date()}",
            'symbols_tested': test_symbols,
            'data_points': len(all_data),
            'factors_computed': len(factors.columns) if 'factors' in locals() else 0,
            'signals_generated': signals.shape if 'signals' in locals() else (0, 0),
            'portfolio_performance': {
                'total_return': total_return if 'total_return' in locals() else 0,
                'annualized_return': annualized_return if 'annualized_return' in locals() else 0,
                'volatility': volatility if 'volatility' in locals() else 0,
                'sharpe_ratio': sharpe_ratio if 'sharpe_ratio' in locals() else 0
            }
        }
        
        os.makedirs('data/backtest/reports', exist_ok=True)
        import json
        with open('data/backtest/reports/real_data_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"   ✅ 测试报告已生成")
        
    except Exception as e:
        print(f"   ❌ 生成报告失败: {e}")
    
    print("\n" + "=" * 80)
    print("真实数据完整测试完成！")
    print("=" * 80)
    
    # 输出目录结构
    print("\n生成的数据文件:")
    data_files = [
        'data/raw/akshare/daily/*.parquet',
        'data/processed/clean/daily_prices.parquet',
        'data/factors/technical/technical_factors.parquet',
        'data/signals/daily/individual/ranking_signals.parquet',
        'data/portfolios/weights/daily/latest_weights.parquet',
        'data/portfolios/performance/returns/portfolio_returns.parquet',
        'data/backtest/reports/real_data_test_report.json'
    ]
    
    for file_path in data_files:
        if '*' in file_path:
            print(f"- {file_path}")
        else:
            if os.path.exists(file_path):
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path}")

if __name__ == "__main__":
    test_real_data_complete() 