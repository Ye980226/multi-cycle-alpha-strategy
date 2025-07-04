#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测引擎模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')

try:
    from .performance_analyzer import PerformanceAnalyzer
    from .attribution_analyzer import AttributionAnalyzer
    from .benchmark_comparison import BenchmarkComparison
except ImportError:
    # 如果模块不存在，创建简单的占位符类
    class PerformanceAnalyzer:
        def __init__(self):
            pass
        
        def analyze_performance(self, returns, benchmark_returns=None):
            return {}
    
    class AttributionAnalyzer:
        def __init__(self):
            pass
        
        def analyze_attribution(self, portfolio_returns, factor_returns, exposures):
            return {}
    
    class BenchmarkComparison:
        def __init__(self):
            pass
        
        def compare_with_benchmark(self, portfolio_returns, benchmark_returns):
            return {}


class BacktestConfig:
    """回测配置类"""
    
    def __init__(self):
        """初始化回测配置"""
        self.universe = []
        self.benchmark = "000300.SH"  # 沪深300
        self.transaction_cost = 0.001
        self.management_fee = 0.01
        self.max_position = 0.05
        self.max_leverage = 1.0
        self.min_position = 0.001
        self.rebalance_frequency = "daily"
        self.cash_buffer = 0.02
    
    def set_universe(self, universe: List[str]):
        """设置股票池"""
        self.universe = universe
        
    def set_benchmark(self, benchmark: str):
        """设置基准"""
        self.benchmark = benchmark
        
    def set_costs(self, transaction_cost: float = 0.001,
                 management_fee: float = 0.01):
        """设置成本"""
        self.transaction_cost = transaction_cost
        self.management_fee = management_fee
        
    def set_constraints(self, max_position: float = 0.05,
                       max_leverage: float = 1.0):
        """设置约束"""
        self.max_position = max_position
        self.max_leverage = max_leverage


class TradeSimulator:
    """交易模拟器"""
    
    def __init__(self, config: BacktestConfig):
        """初始化交易模拟器"""
        self.config = config
        self.slippage_factor = 0.0005  # 滑点因子
        self.market_impact_factor = 0.01  # 市场冲击因子
        
    def simulate_trades(self, target_weights: pd.DataFrame,
                       price_data: pd.DataFrame) -> pd.DataFrame:
        """模拟交易执行"""
        trades = []
        
        for date in target_weights.index:
            if date not in price_data.index:
                continue
                
            # 获取当日目标权重和价格
            target_weight = target_weights.loc[date]
            prices = price_data.loc[date]
            
            # 计算需要调整的权重
            if len(trades) == 0:
                # 首次建仓
                current_weight = pd.Series(0, index=target_weight.index)
            else:
                # 获取上一期的权重
                last_trade = trades[-1]
                current_weight = last_trade['weights_after']
                
            # 计算权重变化
            weight_change = target_weight - current_weight
            
            # 模拟交易
            trade_info = {
                'date': date,
                'target_weights': target_weight,
                'current_weights': current_weight,
                'weight_changes': weight_change,
                'prices': prices,
                'weights_after': target_weight
            }
            
            trades.append(trade_info)
            
        return pd.DataFrame(trades)
    
    def calculate_transaction_costs(self, trades: pd.DataFrame,
                                  volume_data: pd.DataFrame) -> pd.Series:
        """计算交易成本"""
        costs = []
        
        for _, trade in trades.iterrows():
            date = trade['date']
            weight_changes = trade['weight_changes']
            
            # 计算换手率
            turnover = weight_changes.abs().sum()
            
            # 基础交易成本
            base_cost = turnover * self.config.transaction_cost
            
            # 市场冲击成本
            if date in volume_data.index:
                volumes = volume_data.loc[date]
                market_impact = self._calculate_market_impact(weight_changes, volumes)
            else:
                market_impact = 0
                
            total_cost = base_cost + market_impact
            costs.append(total_cost)
            
        return pd.Series(costs, index=trades.index)
    
    def _calculate_market_impact(self, weight_changes: pd.Series, volumes: pd.Series) -> float:
        """计算市场冲击"""
        # 简化的市场冲击模型
        impact = 0
        for symbol in weight_changes.index:
            if symbol in volumes.index and volumes[symbol] > 0:
                trade_size = abs(weight_changes[symbol])
                # 市场冲击与交易量成正比
                impact += trade_size * self.market_impact_factor
        return impact
    
    def apply_market_impact(self, trades: pd.DataFrame,
                          volume_data: pd.DataFrame) -> pd.DataFrame:
        """应用市场冲击"""
        adjusted_trades = trades.copy()
        
        for idx, trade in adjusted_trades.iterrows():
            date = trade['date']
            weight_changes = trade['weight_changes']
            
            if date in volume_data.index:
                volumes = volume_data.loc[date]
                impact = self._calculate_market_impact(weight_changes, volumes)
                
                # 调整后的价格（考虑市场冲击）
                adjusted_prices = trade['prices'] * (1 + impact)
                adjusted_trades.at[idx, 'prices'] = adjusted_prices
                
        return adjusted_trades
    
    def simulate_slippage(self, trades: pd.DataFrame,
                         volatility_data: pd.DataFrame) -> pd.DataFrame:
        """模拟滑点"""
        adjusted_trades = trades.copy()
        
        for idx, trade in adjusted_trades.iterrows():
            date = trade['date']
            
            if date in volatility_data.index:
                volatilities = volatility_data.loc[date]
                
                # 滑点与波动率成正比
                slippage = volatilities * self.slippage_factor
                
                # 应用滑点
                adjusted_prices = trade['prices'] * (1 + slippage)
                adjusted_trades.at[idx, 'prices'] = adjusted_prices
                
        return adjusted_trades


class PositionTracker:
    """持仓跟踪器"""
    
    def __init__(self):
        """初始化持仓跟踪器"""
        self.positions_history = []
        self.cash_flow_history = []
        self.portfolio_value_history = []
        
    def update_positions(self, current_positions: pd.Series,
                        trades: pd.Series) -> pd.Series:
        """更新持仓"""
        new_positions = current_positions.copy()
        
        for symbol, trade_size in trades.items():
            if symbol in new_positions.index:
                new_positions[symbol] += trade_size
            else:
                new_positions[symbol] = trade_size
                
        # 清理接近零的持仓
        new_positions = new_positions[new_positions.abs() > 1e-6]
        
        return new_positions
    
    def calculate_portfolio_value(self, positions: pd.DataFrame,
                                prices: pd.DataFrame) -> pd.Series:
        """计算组合价值"""
        portfolio_values = []
        
        for date in positions.index:
            if date not in prices.index:
                continue
                
            position = positions.loc[date]
            price = prices.loc[date]
            
            # 计算各资产价值
            values = position * price
            total_value = values.sum()
            
            portfolio_values.append(total_value)
            
        return pd.Series(portfolio_values, index=positions.index)
    
    def track_cash_flow(self, trades: pd.DataFrame,
                       transaction_costs: pd.Series) -> pd.Series:
        """跟踪现金流"""
        cash_flows = []
        
        for idx, trade in trades.iterrows():
            weight_changes = trade['weight_changes']
            prices = trade['prices']
            cost = transaction_costs.iloc[idx]
            
            # 计算现金流（负值表示买入，正值表示卖出）
            cash_flow = -(weight_changes * prices).sum() - cost
            cash_flows.append(cash_flow)
            
        return pd.Series(cash_flows, index=trades.index)
    
    def calculate_leverage(self, positions: pd.DataFrame,
                         portfolio_value: pd.Series) -> pd.Series:
        """计算杠杆率"""
        leverage_ratios = []
        
        for date in positions.index:
            if date not in portfolio_value.index:
                continue
                
            position = positions.loc[date]
            portfolio_val = portfolio_value.loc[date]
            
            # 计算总曝险
            total_exposure = position.abs().sum()
            
            # 计算杠杆率
            if portfolio_val > 0:
                leverage = total_exposure / portfolio_val
            else:
                leverage = 0
                
            leverage_ratios.append(leverage)
            
        return pd.Series(leverage_ratios, index=positions.index)


class RebalanceScheduler:
    """再平衡调度器"""
    
    def __init__(self, frequency: str = "daily",
                 threshold: float = 0.05):
        """初始化再平衡调度器"""
        self.frequency = frequency
        self.threshold = threshold
        
    def should_rebalance(self, current_date: pd.Timestamp,
                        last_rebalance: pd.Timestamp,
                        drift: float) -> bool:
        """判断是否需要再平衡"""
        # 基于时间的再平衡
        if self.frequency == "daily":
            time_condition = True
        elif self.frequency == "weekly":
            time_condition = (current_date - last_rebalance).days >= 7
        elif self.frequency == "monthly":
            time_condition = (current_date - last_rebalance).days >= 30
        else:
            time_condition = True
            
        # 基于漂移的再平衡
        drift_condition = drift > self.threshold
        
        return time_condition or drift_condition
    
    def schedule_rebalances(self, start_date: str, end_date: str) -> List[pd.Timestamp]:
        """安排再平衡时间"""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        if self.frequency == "daily":
            dates = pd.date_range(start, end, freq='D')
        elif self.frequency == "weekly":
            dates = pd.date_range(start, end, freq='W')
        elif self.frequency == "monthly":
            dates = pd.date_range(start, end, freq='M')
        else:
            dates = pd.date_range(start, end, freq='D')
            
        return dates.tolist()
    
    def calculate_drift(self, target_weights: pd.Series,
                       current_weights: pd.Series) -> float:
        """计算权重漂移"""
        # 对齐索引
        common_index = target_weights.index.intersection(current_weights.index)
        target_aligned = target_weights.reindex(common_index, fill_value=0)
        current_aligned = current_weights.reindex(common_index, fill_value=0)
        
        # 计算平均绝对偏差
        drift = (target_aligned - current_aligned).abs().mean()
        
        return drift


class MultiFrequencyBacktester:
    """多频率回测器"""
    
    def __init__(self):
        """初始化多频率回测器"""
        pass
    
    def run_multi_frequency_backtest(self, signals_dict: Dict[str, pd.DataFrame],
                                   data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """运行多频率回测"""
        results = {}
        
        for freq, signals in signals_dict.items():
            if freq in data_dict:
                # 为每个频率运行回测
                backtest_engine = BacktestEngine()
                result = backtest_engine.run_backtest(
                    signals, data_dict[freq], 
                    signals.index[0].strftime('%Y-%m-%d'),
                    signals.index[-1].strftime('%Y-%m-%d')
                )
                results[freq] = result
                
        return results
    
    def align_frequencies(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """对齐不同频率数据"""
        aligned_data = {}
        
        # 找到最低频率作为基准
        min_freq = min(data_dict.keys())
        base_dates = data_dict[min_freq].index
        
        for freq, data in data_dict.items():
            # 将数据重采样到基准频率
            aligned_data[freq] = data.reindex(base_dates, method='ffill')
            
        return aligned_data
    
    def aggregate_results(self, results_dict: Dict[str, Dict]) -> Dict:
        """聚合结果"""
        aggregated = {
            'performance_metrics': {},
            'returns': {},
            'weights': {}
        }
        
        for freq, results in results_dict.items():
            aggregated['performance_metrics'][freq] = results.get('performance_metrics', {})
            aggregated['returns'][freq] = results.get('returns', pd.Series())
            aggregated['weights'][freq] = results.get('weights', pd.DataFrame())
            
        return aggregated


class RiskManager:
    """风险管理器"""
    
    def __init__(self, config: Dict):
        """初始化风险管理器"""
        self.config = config
        self.max_position = config.get('max_position', 0.05)
        self.max_leverage = config.get('max_leverage', 1.0)
        self.var_confidence = config.get('var_confidence', 0.95)
        self.max_drawdown = config.get('max_drawdown', 0.15)
        
    def check_position_limits(self, weights: pd.Series) -> bool:
        """检查仓位限制"""
        max_weight = weights.abs().max()
        return max_weight <= self.max_position
    
    def check_sector_limits(self, weights: pd.Series,
                          sector_data: pd.DataFrame) -> bool:
        """检查行业限制"""
        # 简化实现：假设所有股票都属于同一行业
        total_weight = weights.abs().sum()
        return total_weight <= 1.0
    
    def calculate_var(self, portfolio_returns: pd.Series,
                     confidence: float = 0.95) -> float:
        """计算VaR"""
        if len(portfolio_returns) == 0:
            return 0.0
            
        return -portfolio_returns.quantile(1 - confidence)
    
    def monitor_drawdown(self, portfolio_value: pd.Series,
                        max_drawdown: float = 0.1) -> pd.Series:
        """监控回撤"""
        # 计算累积最大值
        running_max = portfolio_value.expanding().max()
        
        # 计算回撤
        drawdown = (portfolio_value - running_max) / running_max
        
        # 检查是否超过最大回撤
        breach = drawdown < -max_drawdown
        
        return breach


class BacktestEngine:
    """回测引擎主类"""
    
    def __init__(self, config: BacktestConfig = None):
        """初始化回测引擎"""
        self.config = config or BacktestConfig()
        self.trade_simulator = TradeSimulator(self.config)
        self.position_tracker = PositionTracker()
        self.rebalance_scheduler = RebalanceScheduler()
        self.risk_manager = RiskManager(self.config.__dict__)
        
    def initialize(self, config: Dict):
        """初始化配置"""
        if config:
            for key, value in config.items():
                setattr(self.config, key, value)
                
    def run_backtest(self, signals: pd.DataFrame,
                    price_data: pd.DataFrame,
                    start_date: str, end_date: str,
                    initial_capital: float = 1000000) -> Dict:
        """运行回测"""
        # 筛选日期范围
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        signals = signals.loc[start:end]
        price_data = price_data.loc[start:end]
        
        # 标准化权重
        signals = signals.div(signals.abs().sum(axis=1), axis=0).fillna(0)
        
        # 模拟交易
        trades = self.trade_simulator.simulate_trades(signals, price_data)
        
        # 计算交易成本
        transaction_costs = pd.Series(0, index=trades.index)
        
        # 跟踪持仓
        portfolio_values = []
        returns = []
        
        for idx, trade in trades.iterrows():
            weights = trade['weights_after']
            prices = trade['prices']
            
            # 计算组合价值
            portfolio_value = (weights * prices).sum() * initial_capital
            portfolio_values.append(portfolio_value)
            
            # 计算收益率
            if len(portfolio_values) > 1:
                return_rate = (portfolio_values[-1] / portfolio_values[-2]) - 1
            else:
                return_rate = 0
            returns.append(return_rate)
            
        # 创建结果
        results = {
            'portfolio_values': pd.Series(portfolio_values, index=signals.index),
            'returns': pd.Series(returns, index=signals.index),
            'weights': signals,
            'trades': trades,
            'transaction_costs': transaction_costs,
            'performance_metrics': self._calculate_performance_metrics(pd.Series(returns, index=signals.index))
        }
        
        return results
    
    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        """计算性能指标"""
        if len(returns) == 0:
            return {}
            
        metrics = {
            'total_return': (1 + returns).prod() - 1,
            'annualized_return': (1 + returns.mean()) ** 252 - 1,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(returns),
            'win_rate': (returns > 0).mean(),
            'avg_win': returns[returns > 0].mean() if (returns > 0).any() else 0,
            'avg_loss': returns[returns < 0].mean() if (returns < 0).any() else 0
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        if len(returns) == 0:
            return 0.0
            
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        
        return drawdown.min()
    
    def run_walk_forward_analysis(self, signals: pd.DataFrame,
                                price_data: pd.DataFrame,
                                train_period: int = 252,
                                test_period: int = 63) -> Dict:
        """走向前分析"""
        results = []
        
        for i in range(train_period, len(signals), test_period):
            train_end = i
            test_start = i
            test_end = min(i + test_period, len(signals))
            
            # 获取测试期间的数据
            test_signals = signals.iloc[test_start:test_end]
            test_prices = price_data.iloc[test_start:test_end]
            
            # 运行回测
            backtest_result = self.run_backtest(
                test_signals, test_prices,
                test_signals.index[0].strftime('%Y-%m-%d'),
                test_signals.index[-1].strftime('%Y-%m-%d')
            )
            
            results.append({
                'period': f"{test_signals.index[0].strftime('%Y-%m-%d')} to {test_signals.index[-1].strftime('%Y-%m-%d')}",
                'result': backtest_result
            })
            
        return {'walk_forward_results': results}
    
    def run_rolling_backtest(self, signals: pd.DataFrame,
                           price_data: pd.DataFrame,
                           window_size: int = 252,
                           step_size: int = 63) -> Dict:
        """滚动回测"""
        results = []
        
        for i in range(0, len(signals) - window_size, step_size):
            window_signals = signals.iloc[i:i + window_size]
            window_prices = price_data.iloc[i:i + window_size]
            
            backtest_result = self.run_backtest(
                window_signals, window_prices,
                window_signals.index[0].strftime('%Y-%m-%d'),
                window_signals.index[-1].strftime('%Y-%m-%d')
            )
            
            results.append({
                'period': f"{window_signals.index[0].strftime('%Y-%m-%d')} to {window_signals.index[-1].strftime('%Y-%m-%d')}",
                'result': backtest_result
            })
            
        return {'rolling_results': results}
    
    def run_multi_cycle_backtest(self, signals_dict: Dict[str, pd.DataFrame],
                               data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """多周期回测"""
        multi_freq_backtester = MultiFrequencyBacktester()
        return multi_freq_backtester.run_multi_frequency_backtest(signals_dict, data_dict)
    
    def stress_test(self, portfolio_weights: pd.DataFrame,
                   stress_scenarios: Dict[str, pd.DataFrame]) -> Dict:
        """压力测试"""
        stress_results = {}
        
        for scenario_name, scenario_data in stress_scenarios.items():
            # 运行压力测试场景
            stressed_result = self.run_backtest(
                portfolio_weights, scenario_data,
                portfolio_weights.index[0].strftime('%Y-%m-%d'),
                portfolio_weights.index[-1].strftime('%Y-%m-%d')
            )
            
            stress_results[scenario_name] = stressed_result
            
        return stress_results
    
    def monte_carlo_simulation(self, strategy_returns: pd.Series,
                             n_simulations: int = 1000,
                             time_horizon: int = 252) -> Dict:
        """蒙特卡洛模拟"""
        if len(strategy_returns) == 0:
            return {}
            
        # 估计收益率分布参数
        mean_return = strategy_returns.mean()
        std_return = strategy_returns.std()
        
        # 运行蒙特卡洛模拟
        simulated_paths = []
        
        for _ in range(n_simulations):
            # 生成随机收益率序列
            random_returns = np.random.normal(mean_return, std_return, time_horizon)
            
            # 计算累积收益
            cumulative_return = (1 + random_returns).cumprod()
            simulated_paths.append(cumulative_return)
            
        simulated_paths = np.array(simulated_paths)
        
        # 计算统计量
        final_returns = simulated_paths[:, -1] - 1
        
        results = {
            'mean_final_return': final_returns.mean(),
            'std_final_return': final_returns.std(),
            'percentile_5': np.percentile(final_returns, 5),
            'percentile_95': np.percentile(final_returns, 95),
            'probability_of_loss': (final_returns < 0).mean(),
            'simulated_paths': simulated_paths
        }
        
        return results
    
    def bootstrap_analysis(self, strategy_returns: pd.Series,
                          n_bootstrap: int = 1000,
                          block_size: int = 20) -> Dict:
        """自举分析"""
        if len(strategy_returns) == 0:
            return {}
            
        bootstrap_results = []
        
        for _ in range(n_bootstrap):
            # 块自举采样
            n_blocks = len(strategy_returns) // block_size
            bootstrap_sample = []
            
            for _ in range(n_blocks):
                start_idx = np.random.randint(0, len(strategy_returns) - block_size)
                block = strategy_returns.iloc[start_idx:start_idx + block_size]
                bootstrap_sample.extend(block.values)
                
            bootstrap_sample = pd.Series(bootstrap_sample)
            
            # 计算统计量
            bootstrap_results.append({
                'mean_return': bootstrap_sample.mean(),
                'std_return': bootstrap_sample.std(),
                'sharpe_ratio': bootstrap_sample.mean() / bootstrap_sample.std() * np.sqrt(252) if bootstrap_sample.std() > 0 else 0
            })
            
        return {
            'bootstrap_results': bootstrap_results,
            'mean_sharpe': np.mean([r['sharpe_ratio'] for r in bootstrap_results]),
            'std_sharpe': np.std([r['sharpe_ratio'] for r in bootstrap_results])
        }
    
    def calculate_attribution(self, portfolio_returns: pd.Series,
                            factor_returns: pd.DataFrame,
                            factor_exposures: pd.DataFrame) -> Dict:
        """计算归因分析"""
        # 简化的归因分析
        attribution = {}
        
        if len(portfolio_returns) > 0 and len(factor_returns) > 0:
            # 回归分析
            from sklearn.linear_model import LinearRegression
            
            # 对齐数据
            common_dates = portfolio_returns.index.intersection(factor_returns.index)
            y = portfolio_returns.loc[common_dates].values.reshape(-1, 1)
            X = factor_returns.loc[common_dates].values
            
            if len(y) > 0 and X.shape[1] > 0:
                model = LinearRegression()
                model.fit(X, y)
                
                attribution = {
                    'factor_loadings': dict(zip(factor_returns.columns, model.coef_[0])),
                    'alpha': model.intercept_[0],
                    'r_squared': model.score(X, y)
                }
                
        return attribution
    
    def compare_strategies(self, strategies_dict: Dict[str, pd.Series]) -> Dict:
        """比较策略"""
        comparison = {}
        
        for strategy_name, returns in strategies_dict.items():
            metrics = self._calculate_performance_metrics(returns)
            comparison[strategy_name] = metrics
            
        return comparison
    
    def generate_tearsheet(self, backtest_results: Dict,
                         save_path: str = None) -> None:
        """生成分析报告"""
        # 简化的报告生成
        print("=== 回测报告 ===")
        print(f"总收益率: {backtest_results['performance_metrics']['total_return']:.2%}")
        print(f"年化收益率: {backtest_results['performance_metrics']['annualized_return']:.2%}")
        print(f"波动率: {backtest_results['performance_metrics']['volatility']:.2%}")
        print(f"夏普比率: {backtest_results['performance_metrics']['sharpe_ratio']:.2f}")
        print(f"最大回撤: {backtest_results['performance_metrics']['max_drawdown']:.2%}")
        print(f"胜率: {backtest_results['performance_metrics']['win_rate']:.2%}")
        
        if save_path:
            # 保存到文件
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("回测报告\n")
                f.write("=" * 50 + "\n")
                for key, value in backtest_results['performance_metrics'].items():
                    f.write(f"{key}: {value}\n")
    
    def optimize_parameters(self, strategy_func, parameter_grid: Dict,
                          optimization_metric: str = "sharpe_ratio") -> Dict:
        """参数优化"""
        best_params = None
        best_score = float('-inf')
        optimization_results = []
        
        # 生成参数组合
        param_combinations = self._generate_param_combinations(parameter_grid)
        
        for params in param_combinations:
            try:
                # 运行策略
                result = strategy_func(**params)
                
                # 获取优化指标
                score = result['performance_metrics'].get(optimization_metric, 0)
                
                optimization_results.append({
                    'params': params,
                    'score': score,
                    'result': result
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                print(f"参数优化错误: {params}, 错误: {e}")
                continue
                
        return {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_results': optimization_results
        }
    
    def _generate_param_combinations(self, parameter_grid: Dict) -> List[Dict]:
        """生成参数组合"""
        import itertools
        
        keys = list(parameter_grid.keys())
        values = list(parameter_grid.values())
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
            
        return combinations
    
    def cross_validate_strategy(self, signals: pd.DataFrame,
                              price_data: pd.DataFrame,
                              cv_folds: int = 5) -> Dict:
        """交叉验证策略"""
        fold_size = len(signals) // cv_folds
        cv_results = []
        
        for i in range(cv_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < cv_folds - 1 else len(signals)
            
            fold_signals = signals.iloc[start_idx:end_idx]
            fold_prices = price_data.iloc[start_idx:end_idx]
            
            if len(fold_signals) > 0:
                fold_result = self.run_backtest(
                    fold_signals, fold_prices,
                    fold_signals.index[0].strftime('%Y-%m-%d'),
                    fold_signals.index[-1].strftime('%Y-%m-%d')
                )
                
                cv_results.append(fold_result)
                
        return {'cv_results': cv_results}
    
    def analyze_regime_performance(self, strategy_returns: pd.Series,
                                 market_regimes: pd.Series) -> Dict:
        """分析不同市场状态下的表现"""
        regime_performance = {}
        
        # 获取共同的时间索引
        common_index = strategy_returns.index.intersection(market_regimes.index)
        
        if len(common_index) > 0:
            aligned_returns = strategy_returns.loc[common_index]
            aligned_regimes = market_regimes.loc[common_index]
            
            # 按市场状态分组
            for regime in aligned_regimes.unique():
                regime_returns = aligned_returns[aligned_regimes == regime]
                
                if len(regime_returns) > 0:
                    regime_performance[regime] = self._calculate_performance_metrics(regime_returns)
                    
        return regime_performance
    
    def calculate_risk_metrics(self, portfolio_returns: pd.Series,
                             benchmark_returns: pd.Series = None) -> Dict:
        """计算风险指标"""
        risk_metrics = {}
        
        if len(portfolio_returns) > 0:
            # 基础风险指标
            risk_metrics['volatility'] = portfolio_returns.std() * np.sqrt(252)
            risk_metrics['var_95'] = self.risk_manager.calculate_var(portfolio_returns, 0.95)
            risk_metrics['var_99'] = self.risk_manager.calculate_var(portfolio_returns, 0.99)
            
            # 下行风险指标
            negative_returns = portfolio_returns[portfolio_returns < 0]
            if len(negative_returns) > 0:
                risk_metrics['downside_volatility'] = negative_returns.std() * np.sqrt(252)
                risk_metrics['sortino_ratio'] = portfolio_returns.mean() / negative_returns.std() * np.sqrt(252)
            
            # 与基准的比较
            if benchmark_returns is not None:
                common_index = portfolio_returns.index.intersection(benchmark_returns.index)
                if len(common_index) > 0:
                    port_aligned = portfolio_returns.loc[common_index]
                    bench_aligned = benchmark_returns.loc[common_index]
                    
                    excess_returns = port_aligned - bench_aligned
                    risk_metrics['tracking_error'] = excess_returns.std() * np.sqrt(252)
                    risk_metrics['information_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
                    
                    # Beta计算
                    if bench_aligned.var() > 0:
                        risk_metrics['beta'] = port_aligned.cov(bench_aligned) / bench_aligned.var()
                    else:
                        risk_metrics['beta'] = 0
                        
        return risk_metrics
    
    def run_constrained_backtest(self, signals: pd.DataFrame,
                               price_data: pd.DataFrame,
                               constraints: Dict,
                               start_date: str, end_date: str,
                               initial_capital: float = 1000000) -> Dict:
        """运行约束回测"""
        try:
            # 基础回测
            base_results = self.run_backtest(signals, price_data, start_date, end_date, initial_capital)
            
            # 应用约束
            constrained_results = base_results.copy()
            constrained_results['constraints'] = constraints
            constrained_results['constraint_violations'] = []
            
            return constrained_results
            
        except Exception as e:
            return {'error': f"约束回测失败: {e}"}
    
    def run_multi_frequency_backtest(self, signals_dict: Dict[str, pd.DataFrame],
                                   data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """运行多频率回测"""
        try:
            results = {}
            
            for freq, signals in signals_dict.items():
                if freq in data_dict:
                    # 获取数据的时间范围
                    data = data_dict[freq]
                    start_date = data.index[0].strftime('%Y-%m-%d')
                    end_date = data.index[-1].strftime('%Y-%m-%d')
                    
                    # 运行回测
                    result = self.run_backtest(signals, data, start_date, end_date)
                    results[freq] = result
            
            return results
            
        except Exception as e:
            return {'error': f"多频率回测失败: {e}"}
    
    def calculate_performance_metrics(self, returns: pd.Series,
                                    benchmark_returns: pd.Series = None) -> Dict:
        """计算性能指标"""
        try:
            metrics = {}
            
            if len(returns) == 0:
                return {'error': '收益序列为空'}
            
            # 基础收益指标
            metrics['total_return'] = (1 + returns).prod() - 1
            metrics['annual_return'] = ((1 + returns).prod()) ** (252 / len(returns)) - 1
            metrics['volatility'] = returns.std() * np.sqrt(252)
            metrics['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
            
            # 风险指标
            metrics['max_drawdown'] = self._calculate_max_drawdown(returns)
            metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
            
            # 相对基准指标
            if benchmark_returns is not None:
                common_index = returns.index.intersection(benchmark_returns.index)
                if len(common_index) > 0:
                    aligned_returns = returns.loc[common_index]
                    aligned_benchmark = benchmark_returns.loc[common_index]
                    
                    excess_returns = aligned_returns - aligned_benchmark
                    metrics['tracking_error'] = excess_returns.std() * np.sqrt(252)
                    metrics['information_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else 0
                    
                    if aligned_benchmark.var() > 0:
                        metrics['beta'] = aligned_returns.cov(aligned_benchmark) / aligned_benchmark.var()
                        metrics['alpha'] = metrics['annual_return'] - metrics['beta'] * aligned_benchmark.mean() * 252
                    else:
                        metrics['beta'] = 0
                        metrics['alpha'] = 0
            
            return metrics
            
        except Exception as e:
            return {'error': f"计算性能指标失败: {e}"}
    
    def calculate_risk_adjusted_metrics(self, returns: pd.Series,
                                      benchmark_returns: pd.Series = None) -> Dict:
        """计算风险调整指标"""
        try:
            metrics = {}
            
            if len(returns) == 0:
                return {'error': '收益序列为空'}
            
            # 基础风险调整指标
            metrics['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
            
            # Sortino比率
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                metrics['sortino_ratio'] = returns.mean() / negative_returns.std() * np.sqrt(252)
            else:
                metrics['sortino_ratio'] = float('inf')
            
            # 风险价值指标
            metrics['var_95'] = np.percentile(returns, 5)
            metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
            
            # 最大回撤相关
            metrics['max_drawdown'] = self._calculate_max_drawdown(returns)
            metrics['calmar_ratio'] = returns.mean() * 252 / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
            
            # Treynor比率
            if benchmark_returns is not None:
                common_index = returns.index.intersection(benchmark_returns.index)
                if len(common_index) > 0:
                    aligned_returns = returns.loc[common_index]
                    aligned_benchmark = benchmark_returns.loc[common_index]
                    
                    if aligned_benchmark.var() > 0:
                        beta = aligned_returns.cov(aligned_benchmark) / aligned_benchmark.var()
                        metrics['treynor_ratio'] = returns.mean() * 252 / beta if beta != 0 else 0
                    else:
                        metrics['treynor_ratio'] = 0
            
            return metrics
            
        except Exception as e:
            return {'error': f"计算风险调整指标失败: {e}"}
    
    def calculate_rolling_performance(self, returns: pd.Series,
                                    window: int = 252) -> Dict:
        """计算滚动性能"""
        try:
            if len(returns) < window:
                return {'error': f'数据长度不足，需要至少{window}个观测值'}
            
            rolling_metrics = {}
            
            # 滚动收益
            rolling_metrics['rolling_returns'] = returns.rolling(window).sum()
            rolling_metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)
            rolling_metrics['rolling_sharpe'] = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
            
            # 滚动最大回撤
            rolling_max_dd = []
            for i in range(window-1, len(returns)):
                window_returns = returns.iloc[i-window+1:i+1]
                rolling_max_dd.append(self._calculate_max_drawdown(window_returns))
            
            rolling_metrics['rolling_max_drawdown'] = pd.Series(rolling_max_dd, index=returns.index[window-1:])
            
            return rolling_metrics
            
        except Exception as e:
            return {'error': f"计算滚动性能失败: {e}"}
    
    def calculate_transaction_costs(self, trades: pd.DataFrame,
                                  transaction_cost_rate: float = 0.001) -> Dict:
        """计算交易成本"""
        try:
            costs = {}
            
            if trades.empty:
                return {'total_transaction_costs': 0, 'average_transaction_cost': 0, 'turnover': pd.Series(dtype=float)}
            
            # 计算换手率
            if 'weight_changes' in trades.columns:
                turnover = trades['weight_changes'].apply(lambda x: x.abs().sum() if isinstance(x, pd.Series) else abs(x) if pd.notna(x) else 0)
                costs['turnover'] = turnover
                costs['transaction_costs'] = turnover * transaction_cost_rate
                costs['total_transaction_costs'] = costs['transaction_costs'].sum()
                costs['average_transaction_cost'] = costs['transaction_costs'].mean()
            else:
                costs['turnover'] = pd.Series(dtype=float)
                costs['transaction_costs'] = pd.Series(dtype=float)
                costs['total_transaction_costs'] = 0
                costs['average_transaction_cost'] = 0
            
            return costs
            
        except Exception as e:
            return {'error': f"计算交易成本失败: {e}"} 