#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能监控模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings


class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self):
        """初始化指标计算器"""
        pass
    
    def calculate_returns(self, portfolio_value: pd.Series) -> pd.Series:
        """计算收益率"""
        if len(portfolio_value) < 2:
            return pd.Series()
        
        returns = portfolio_value.pct_change().dropna()
        return returns
    
    def calculate_cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """计算累计收益率"""
        if len(returns) == 0:
            return pd.Series()
        
        cumulative_returns = (1 + returns).cumprod() - 1
        return cumulative_returns
    
    def calculate_sharpe_ratio(self, returns: pd.Series, 
                             risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        # 转换为年化
        excess_returns = returns.mean() - risk_free_rate / 252  # 日度风险无风险收益率
        annualized_excess_return = excess_returns * 252
        annualized_volatility = returns.std() * np.sqrt(252)
        
        sharpe_ratio = annualized_excess_return / annualized_volatility
        return sharpe_ratio
    
    def calculate_sortino_ratio(self, returns: pd.Series,
                              risk_free_rate: float = 0.02) -> float:
        """计算索提诺比率"""
        if len(returns) == 0:
            return 0.0
        
        # 下行偏差
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = downside_returns.std() * np.sqrt(252)
        if downside_std == 0:
            return 0.0
        
        excess_returns = returns.mean() - risk_free_rate / 252
        annualized_excess_return = excess_returns * 252
        
        sortino_ratio = annualized_excess_return / downside_std
        return sortino_ratio
    
    def calculate_max_drawdown(self, portfolio_value: pd.Series) -> float:
        """计算最大回撤"""
        if len(portfolio_value) == 0:
            return 0.0
        
        # 计算累计最高点（高水位）
        running_max = portfolio_value.cummax()
        
        # 计算回撤
        drawdown = (portfolio_value - running_max) / running_max
        
        # 返回最大回撤（负值）
        max_drawdown = drawdown.min()
        
        return max_drawdown
    
    def calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """计算卡玛比率"""
        if len(returns) == 0:
            return 0.0
        
        annualized_return = (1 + returns.mean()) ** 252 - 1
        max_drawdown = self.calculate_max_drawdown((1 + returns).cumprod())
        
        if max_drawdown == 0:
            return float('inf')
        
        calmar_ratio = annualized_return / abs(max_drawdown)
        return calmar_ratio
    
    def calculate_information_ratio(self, portfolio_returns: pd.Series,
                                  benchmark_returns: pd.Series) -> float:
        """计算信息比率"""
        # 对齐数据
        common_index = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_index) == 0:
            return 0.0
        
        port_aligned = portfolio_returns.reindex(common_index)
        bench_aligned = benchmark_returns.reindex(common_index)
        
        # 计算超额收益
        excess_returns = port_aligned - bench_aligned
        
        if excess_returns.std() == 0:
            return 0.0
        
        # 年化信息比率
        information_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        return information_ratio
    
    def calculate_beta(self, portfolio_returns: pd.Series,
                      market_returns: pd.Series) -> float:
        """计算Beta"""
        # 对齐数据
        common_index = portfolio_returns.index.intersection(market_returns.index)
        if len(common_index) < 2:
            return 1.0
        
        port_aligned = portfolio_returns.reindex(common_index)
        market_aligned = market_returns.reindex(common_index)
        
        # 计算Beta
        covariance = port_aligned.cov(market_aligned)
        market_variance = market_aligned.var()
        
        if market_variance == 0:
            return 1.0
        
        beta = covariance / market_variance
        return beta
    
    def calculate_alpha(self, portfolio_returns: pd.Series,
                       market_returns: pd.Series,
                       risk_free_rate: float = 0.02) -> float:
        """计算Alpha"""
        # 对齐数据
        common_index = portfolio_returns.index.intersection(market_returns.index)
        if len(common_index) < 2:
            return 0.0
        
        port_aligned = portfolio_returns.reindex(common_index)
        market_aligned = market_returns.reindex(common_index)
        
        # 计算Beta
        beta = self.calculate_beta(port_aligned, market_aligned)
        
        # 计算Alpha
        risk_free_daily = risk_free_rate / 252
        portfolio_excess = port_aligned.mean() - risk_free_daily
        market_excess = market_aligned.mean() - risk_free_daily
        
        alpha_daily = portfolio_excess - beta * market_excess
        alpha_annualized = alpha_daily * 252
        
        return alpha_annualized
    
    def calculate_tracking_error(self, portfolio_returns: pd.Series,
                               benchmark_returns: pd.Series) -> float:
        """计算跟踪误差"""
        # 对齐数据
        common_index = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_index) == 0:
            return 0.0
        
        port_aligned = portfolio_returns.reindex(common_index)
        bench_aligned = benchmark_returns.reindex(common_index)
        
        # 计算跟踪误差
        excess_returns = port_aligned - bench_aligned
        tracking_error = excess_returns.std() * np.sqrt(252)  # 年化
        
        return tracking_error


class RealTimeMonitor:
    """实时监控器"""
    
    def __init__(self, alert_thresholds: Dict = None):
        """初始化实时监控器"""
        self.alert_thresholds = alert_thresholds or {
            'value_change_threshold': 0.05,  # 5%变动
            'drawdown_threshold': -0.10,     # 10%回撤
            'volatility_threshold': 0.25,    # 25%年化波动率
            'max_position_threshold': 0.20,  # 20%单一仓位
            'concentration_threshold': 0.50, # 50%前5大仓位
            'leverage_threshold': 2.0        # 2倍杠杆
        }
    
    def monitor_portfolio_value(self, current_value: float,
                              previous_value: float) -> Dict:
        """监控组合价值"""
        result = {
            'current_value': current_value,
            'previous_value': previous_value,
            'change': current_value - previous_value,
            'change_pct': (current_value - previous_value) / previous_value if previous_value != 0 else 0,
            'alert': False,
            'alert_message': ''
        }
        
        # 检查是否有大幅变动
        if self.alert_thresholds and 'value_change_threshold' in self.alert_thresholds:
            threshold = self.alert_thresholds['value_change_threshold']
            if abs(result['change_pct']) > threshold:
                result['alert'] = True
                result['alert_message'] = f"组合价值变动超过阈值: {result['change_pct']:.2%} > {threshold:.2%}"
        
        return result
    
    def monitor_drawdown(self, current_drawdown: float) -> Dict:
        """监控回撤"""
        result = {
            'current_drawdown': current_drawdown,
            'alert': False,
            'alert_message': ''
        }
        
        # 检查回撤警报
        if self.alert_thresholds and 'drawdown_threshold' in self.alert_thresholds:
            threshold = self.alert_thresholds['drawdown_threshold']
            if current_drawdown < threshold:  # 回撤是负值
                result['alert'] = True
                result['alert_message'] = f"回撤超过阈值: {current_drawdown:.2%} < {threshold:.2%}"
        
        return result
    
    def monitor_volatility(self, recent_returns: pd.Series) -> Dict:
        """监控波动率"""
        if len(recent_returns) == 0:
            return {'volatility': 0, 'alert': False}
        
        # 计算年化波动率
        volatility = recent_returns.std() * np.sqrt(252)
        
        result = {
            'volatility': volatility,
            'alert': False,
            'alert_message': ''
        }
        
        # 检查波动率警报
        if self.alert_thresholds and 'volatility_threshold' in self.alert_thresholds:
            threshold = self.alert_thresholds['volatility_threshold']
            if volatility > threshold:
                result['alert'] = True
                result['alert_message'] = f"波动率超过阈值: {volatility:.2%} > {threshold:.2%}"
        
        return result
    
    def monitor_position_concentration(self, weights: pd.Series) -> Dict:
        """监控仓位集中度"""
        if len(weights) == 0:
            return {'max_weight': 0, 'top5_concentration': 0, 'alert': False}
        
        abs_weights = weights.abs()
        max_weight = abs_weights.max()
        top5_concentration = abs_weights.nlargest(5).sum()
        
        result = {
            'max_weight': max_weight,
            'top5_concentration': top5_concentration,
            'num_positions': len(weights[weights.abs() > 0.001]),
            'alert': False,
            'alert_message': ''
        }
        
        # 检查集中度警报
        if self.alert_thresholds:
            if 'max_position_threshold' in self.alert_thresholds:
                threshold = self.alert_thresholds['max_position_threshold']
                if max_weight > threshold:
                    result['alert'] = True
                    result['alert_message'] = f"单一仓位过大: {max_weight:.2%} > {threshold:.2%}"
            
            if 'concentration_threshold' in self.alert_thresholds:
                threshold = self.alert_thresholds['concentration_threshold']
                if top5_concentration > threshold:
                    result['alert'] = True
                    result['alert_message'] = f"前5大仓位集中度过高: {top5_concentration:.2%} > {threshold:.2%}"
        
        return result
    
    def monitor_leverage(self, current_leverage: float) -> Dict:
        """监控杠杆率"""
        result = {
            'leverage': current_leverage,
            'alert': False,
            'alert_message': ''
        }
        
        # 检查杠杆率警报
        if self.alert_thresholds and 'leverage_threshold' in self.alert_thresholds:
            threshold = self.alert_thresholds['leverage_threshold']
            if current_leverage > threshold:
                result['alert'] = True
                result['alert_message'] = f"杠杆率超过阈值: {current_leverage:.2f} > {threshold:.2f}"
        
        return result
    
    def check_risk_limits(self, current_metrics: Dict) -> Dict[str, bool]:
        """检查风险限制"""
        checks = {}
        
        if not self.alert_thresholds:
            return checks
        
        # 检查各项指标
        if 'volatility' in current_metrics and 'volatility_threshold' in self.alert_thresholds:
            checks['volatility_check'] = current_metrics['volatility'] <= self.alert_thresholds['volatility_threshold']
        
        if 'drawdown' in current_metrics and 'drawdown_threshold' in self.alert_thresholds:
            checks['drawdown_check'] = abs(current_metrics['drawdown']) <= self.alert_thresholds['drawdown_threshold']
        
        if 'leverage' in current_metrics and 'leverage_threshold' in self.alert_thresholds:
            checks['leverage_check'] = current_metrics['leverage'] <= self.alert_thresholds['leverage_threshold']
        
        if 'var' in current_metrics and 'var_threshold' in self.alert_thresholds:
            checks['var_check'] = current_metrics['var'] <= self.alert_thresholds['var_threshold']
        
        return checks


class AlertManager:
    """告警管理器"""
    
    def __init__(self, notification_config: Dict = None):
        """初始化告警管理器"""
        self.notification_config = notification_config or {
            'email_enabled': False,
            'sms_enabled': False,
            'log_enabled': True,
            'escalation_enabled': True,
            'escalation_threshold': 'high'
        }
        self.alert_history = []
        self.alert_counters = {}
        
    def generate_alert(self, alert_type: str, message: str,
                      severity: str = "medium") -> Dict:
        """生成告警"""
        alert_info = {
            'id': f"alert_{len(self.alert_history) + 1}",
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'status': 'active'
        }
        
        # 更新告警计数
        if alert_type not in self.alert_counters:
            self.alert_counters[alert_type] = 0
        self.alert_counters[alert_type] += 1
        
        # 记录告警
        self.alert_history.append(alert_info)
        
        # 根据配置发送告警
        if self.notification_config.get('log_enabled', True):
            self.log_alert(alert_info)
        
        if self.notification_config.get('email_enabled', False):
            self.send_email_alert(alert_info)
        
        if self.notification_config.get('sms_enabled', False) and severity == 'high':
            self.send_sms_alert(alert_info)
        
        if (self.notification_config.get('escalation_enabled', True) and 
            severity == self.notification_config.get('escalation_threshold', 'high')):
            self.escalate_alert(alert_info)
            
        return alert_info
    
    def send_email_alert(self, alert_info: Dict):
        """发送邮件告警"""
        try:
            # 这里应该集成实际的邮件发送服务
            # 示例：使用SMTP或第三方邮件服务
            subject = f"[{alert_info['severity'].upper()}] {alert_info['type']}"
            body = f"""
            告警类型: {alert_info['type']}
            严重程度: {alert_info['severity']}
            时间: {alert_info['timestamp']}
            消息: {alert_info['message']}
            """
            
            # 模拟邮件发送
            print(f"发送邮件告警: {subject}")
            
        except Exception as e:
            print(f"邮件告警发送失败: {e}")
    
    def send_sms_alert(self, alert_info: Dict):
        """发送短信告警"""
        try:
            # 这里应该集成实际的短信发送服务
            message = f"[{alert_info['severity'].upper()}] {alert_info['type']}: {alert_info['message']}"
            
            # 模拟短信发送
            print(f"发送短信告警: {message}")
            
        except Exception as e:
            print(f"短信告警发送失败: {e}")
    
    def log_alert(self, alert_info: Dict):
        """记录告警"""
        try:
            # 使用日志记录
            import logging
            logger = logging.getLogger(__name__)
            
            log_message = f"[{alert_info['severity'].upper()}] {alert_info['type']}: {alert_info['message']}"
            
            if alert_info['severity'] == 'high':
                logger.error(log_message)
            elif alert_info['severity'] == 'medium':
                logger.warning(log_message)
            else:
                logger.info(log_message)
                
        except Exception as e:
            print(f"告警日志记录失败: {e}")
    
    def escalate_alert(self, alert_info: Dict):
        """告警升级"""
        try:
            # 告警升级逻辑
            escalated_alert = alert_info.copy()
            escalated_alert['id'] = f"escalated_{alert_info['id']}"
            escalated_alert['status'] = 'escalated'
            escalated_alert['escalation_timestamp'] = datetime.now().isoformat()
            
            # 记录升级告警
            self.alert_history.append(escalated_alert)
            
            # 可以在这里添加更多升级逻辑，如通知管理员
            print(f"告警升级: {escalated_alert['id']}")
            
        except Exception as e:
            print(f"告警升级失败: {e}")
    
    def get_alert_summary(self) -> Dict:
        """获取告警摘要"""
        return {
            'total_alerts': len(self.alert_history),
            'alert_counters': self.alert_counters,
            'recent_alerts': self.alert_history[-10:] if self.alert_history else []
        }
    
    def clear_old_alerts(self, days: int = 30):
        """清理旧告警"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        self.alert_history = [
            alert for alert in self.alert_history 
            if datetime.fromisoformat(alert['timestamp']) > cutoff_date
        ]


class PerformanceReporter:
    """性能报告器"""
    
    def __init__(self):
        """初始化性能报告器"""
        self.metrics_calculator = MetricsCalculator()
        self.report_templates = {
            'daily': self._get_daily_template(),
            'weekly': self._get_weekly_template(),
            'monthly': self._get_monthly_template()
        }
    
    def generate_daily_report(self, portfolio_data: Dict) -> Dict:
        """生成日报"""
        try:
            portfolio_value = portfolio_data.get('portfolio_value', pd.Series())
            returns = portfolio_data.get('returns', pd.Series())
            positions = portfolio_data.get('positions', pd.Series())
            
            report = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'type': 'daily',
                'portfolio_value': portfolio_value.iloc[-1] if len(portfolio_value) > 0 else 0,
                'daily_return': returns.iloc[-1] if len(returns) > 0 else 0,
                'positions_count': len(positions[positions.abs() > 0.001]) if len(positions) > 0 else 0,
                'largest_position': positions.abs().max() if len(positions) > 0 else 0,
                'portfolio_concentration': positions.abs().nlargest(5).sum() if len(positions) > 0 else 0
            }
            
            # 计算最近一周的指标
            if len(returns) >= 5:
                recent_returns = returns.tail(5)
                report['weekly_volatility'] = recent_returns.std() * np.sqrt(252)
                report['weekly_return'] = recent_returns.sum()
            
            return report
            
        except Exception as e:
            return {'error': f"生成日报失败: {e}"}
    
    def generate_weekly_report(self, portfolio_data: Dict) -> Dict:
        """生成周报"""
        try:
            portfolio_value = portfolio_data.get('portfolio_value', pd.Series())
            returns = portfolio_data.get('returns', pd.Series())
            benchmark_returns = portfolio_data.get('benchmark_returns', pd.Series())
            
            report = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'type': 'weekly',
                'portfolio_value': portfolio_value.iloc[-1] if len(portfolio_value) > 0 else 0,
                'weekly_return': returns.tail(5).sum() if len(returns) >= 5 else 0,
                'weekly_volatility': returns.tail(5).std() * np.sqrt(252) if len(returns) >= 5 else 0
            }
            
            # 计算性能指标
            if len(returns) >= 20:
                recent_returns = returns.tail(20)
                report['sharpe_ratio'] = self.metrics_calculator.calculate_sharpe_ratio(recent_returns)
                report['max_drawdown'] = self.metrics_calculator.calculate_max_drawdown(
                    (1 + recent_returns).cumprod()
                )
                
                # 与基准比较
                if len(benchmark_returns) > 0:
                    recent_benchmark = benchmark_returns.tail(20)
                    report['information_ratio'] = self.metrics_calculator.calculate_information_ratio(
                        recent_returns, recent_benchmark
                    )
                    report['tracking_error'] = self.metrics_calculator.calculate_tracking_error(
                        recent_returns, recent_benchmark
                    )
            
            return report
            
        except Exception as e:
            return {'error': f"生成周报失败: {e}"}
    
    def generate_monthly_report(self, portfolio_data: Dict) -> Dict:
        """生成月报"""
        try:
            portfolio_value = portfolio_data.get('portfolio_value', pd.Series())
            returns = portfolio_data.get('returns', pd.Series())
            benchmark_returns = portfolio_data.get('benchmark_returns', pd.Series())
            trades = portfolio_data.get('trades', pd.DataFrame())
            
            report = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'type': 'monthly',
                'portfolio_value': portfolio_value.iloc[-1] if len(portfolio_value) > 0 else 0,
                'monthly_return': returns.tail(20).sum() if len(returns) >= 20 else 0,
                'ytd_return': returns.sum() if len(returns) > 0 else 0
            }
            
            # 计算完整的性能指标
            if len(returns) >= 30:
                report['sharpe_ratio'] = self.metrics_calculator.calculate_sharpe_ratio(returns)
                report['sortino_ratio'] = self.metrics_calculator.calculate_sortino_ratio(returns)
                report['max_drawdown'] = self.metrics_calculator.calculate_max_drawdown(
                    (1 + returns).cumprod()
                )
                report['volatility'] = returns.std() * np.sqrt(252)
                
                # 与基准比较
                if len(benchmark_returns) > 0:
                    common_index = returns.index.intersection(benchmark_returns.index)
                    if len(common_index) > 0:
                        port_aligned = returns.reindex(common_index)
                        bench_aligned = benchmark_returns.reindex(common_index)
                        
                        report['alpha'] = self.metrics_calculator.calculate_alpha(
                            port_aligned, bench_aligned
                        )
                        report['beta'] = self.metrics_calculator.calculate_beta(
                            port_aligned, bench_aligned
                        )
                        report['information_ratio'] = self.metrics_calculator.calculate_information_ratio(
                            port_aligned, bench_aligned
                        )
            
            # 交易统计
            if not trades.empty:
                report['total_trades'] = len(trades)
                report['average_trade_size'] = trades.get('quantity', pd.Series()).abs().mean()
                report['turnover'] = trades.get('quantity', pd.Series()).abs().sum()
            
            return report
            
        except Exception as e:
            return {'error': f"生成月报失败: {e}"}
    
    def create_performance_charts(self, portfolio_data: Dict,
                                benchmark_data: Dict = None) -> Dict:
        """创建性能图表"""
        try:
            charts = {}
            
            portfolio_value = portfolio_data.get('portfolio_value', pd.Series())
            returns = portfolio_data.get('returns', pd.Series())
            
            # 净值曲线图
            if len(portfolio_value) > 0:
                plt.figure(figsize=(12, 6))
                plt.plot(portfolio_value.index, portfolio_value.values, label='组合净值', linewidth=2)
                
                if benchmark_data and 'portfolio_value' in benchmark_data:
                    benchmark_value = benchmark_data['portfolio_value']
                    plt.plot(benchmark_value.index, benchmark_value.values, 
                            label='基准净值', linewidth=2, alpha=0.7)
                
                plt.title('组合净值曲线')
                plt.xlabel('日期')
                plt.ylabel('净值')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # 保存图表
                chart_path = f"performance_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                charts['performance_chart'] = chart_path
            
            # 收益率分布图
            if len(returns) > 0:
                plt.figure(figsize=(10, 6))
                plt.hist(returns.values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                plt.title('收益率分布')
                plt.xlabel('日收益率')
                plt.ylabel('频数')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                chart_path = f"returns_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                charts['returns_distribution'] = chart_path
            
            return charts
            
        except Exception as e:
            return {'error': f"创建性能图表失败: {e}"}
    
    def create_risk_analysis_charts(self, portfolio_data: Dict) -> Dict:
        """创建风险分析图表"""
        try:
            charts = {}
            
            portfolio_value = portfolio_data.get('portfolio_value', pd.Series())
            returns = portfolio_data.get('returns', pd.Series())
            
            # 回撤图
            if len(portfolio_value) > 0:
                running_max = portfolio_value.cummax()
                drawdown = (portfolio_value - running_max) / running_max
                
                plt.figure(figsize=(12, 8))
                
                # 上图：净值曲线
                plt.subplot(2, 1, 1)
                plt.plot(portfolio_value.index, portfolio_value.values, label='组合净值', linewidth=2)
                plt.plot(running_max.index, running_max.values, label='历史最高', 
                        linestyle='--', alpha=0.7)
                plt.title('组合净值与历史最高')
                plt.ylabel('净值')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # 下图：回撤
                plt.subplot(2, 1, 2)
                plt.fill_between(drawdown.index, drawdown.values, 0, 
                               color='red', alpha=0.3, label='回撤')
                plt.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
                plt.title('回撤分析')
                plt.xlabel('日期')
                plt.ylabel('回撤')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                chart_path = f"drawdown_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                charts['drawdown_analysis'] = chart_path
            
            # 滚动风险指标
            if len(returns) >= 60:
                window = 20
                rolling_vol = returns.rolling(window).std() * np.sqrt(252)
                rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
                
                plt.figure(figsize=(12, 8))
                
                # 滚动波动率
                plt.subplot(2, 1, 1)
                plt.plot(rolling_vol.index, rolling_vol.values, label='滚动波动率', linewidth=2)
                plt.title('滚动风险指标')
                plt.ylabel('年化波动率')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # 滚动夏普比率
                plt.subplot(2, 1, 2)
                plt.plot(rolling_sharpe.index, rolling_sharpe.values, label='滚动夏普比率', linewidth=2)
                plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                plt.xlabel('日期')
                plt.ylabel('夏普比率')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                chart_path = f"rolling_risk_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                charts['rolling_risk_metrics'] = chart_path
            
            return charts
            
        except Exception as e:
            return {'error': f"创建风险分析图表失败: {e}"}
    
    def export_report(self, report_data: Dict, format: str = "pdf",
                     save_path: str = None):
        """导出报告"""
        try:
            if save_path is None:
                save_path = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
            
            if format.lower() == "json":
                import json
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
                    
            elif format.lower() == "csv":
                # 将报告数据转换为DataFrame并保存
                df = pd.DataFrame([report_data])
                df.to_csv(save_path, index=False, encoding='utf-8')
                
            elif format.lower() == "excel":
                df = pd.DataFrame([report_data])
                df.to_excel(save_path, index=False)
                
            elif format.lower() == "pdf":
                # 这里需要一个PDF生成库，如reportlab
                # 简化实现：保存为文本文件
                with open(save_path.replace('.pdf', '.txt'), 'w', encoding='utf-8') as f:
                    f.write("性能报告\n")
                    f.write("=" * 50 + "\n\n")
                    for key, value in report_data.items():
                        f.write(f"{key}: {value}\n")
                        
            return {"success": True, "file_path": save_path}
            
        except Exception as e:
            return {"success": False, "error": f"导出报告失败: {e}"}
    
    def _get_daily_template(self) -> Dict:
        """获取日报模板"""
        return {
            'sections': ['portfolio_value', 'daily_return', 'positions', 'alerts'],
            'charts': ['performance_chart'],
            'metrics': ['daily_return', 'portfolio_value', 'positions_count']
        }
    
    def _get_weekly_template(self) -> Dict:
        """获取周报模板"""
        return {
            'sections': ['portfolio_value', 'weekly_return', 'risk_metrics', 'comparison'],
            'charts': ['performance_chart', 'returns_distribution'],
            'metrics': ['weekly_return', 'sharpe_ratio', 'max_drawdown', 'volatility']
        }
    
    def _get_monthly_template(self) -> Dict:
        """获取月报模板"""
        return {
            'sections': ['portfolio_value', 'performance_summary', 'risk_analysis', 'attribution'],
            'charts': ['performance_chart', 'drawdown_analysis', 'rolling_risk_metrics'],
            'metrics': ['monthly_return', 'ytd_return', 'sharpe_ratio', 'alpha', 'beta']
        }


class BenchmarkComparator:
    """基准比较器"""
    
    def __init__(self, benchmark_returns: pd.Series):
        """初始化基准比较器"""
        self.benchmark_returns = benchmark_returns
        self.metrics_calculator = MetricsCalculator()
        
    def compare_returns(self, portfolio_returns: pd.Series) -> Dict:
        """比较收益率"""
        try:
            # 对齐数据
            common_index = portfolio_returns.index.intersection(self.benchmark_returns.index)
            if len(common_index) == 0:
                return {'error': '无法对齐组合和基准数据'}
            
            port_aligned = portfolio_returns.reindex(common_index)
            bench_aligned = self.benchmark_returns.reindex(common_index)
            
            # 计算累计收益率
            port_cumulative = (1 + port_aligned).cumprod() - 1
            bench_cumulative = (1 + bench_aligned).cumprod() - 1
            
            # 计算年化收益率
            trading_days = len(port_aligned)
            years = trading_days / 252
            
            port_annual_return = (1 + port_cumulative.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
            bench_annual_return = (1 + bench_cumulative.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
            
            # 超额收益
            excess_returns = port_aligned - bench_aligned
            excess_cumulative = (1 + excess_returns).cumprod() - 1
            excess_annual_return = (1 + excess_cumulative.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
            
            # 胜率
            win_rate = (excess_returns > 0).sum() / len(excess_returns)
            
            comparison = {
                'portfolio_annual_return': port_annual_return,
                'benchmark_annual_return': bench_annual_return,
                'excess_annual_return': excess_annual_return,
                'portfolio_total_return': port_cumulative.iloc[-1],
                'benchmark_total_return': bench_cumulative.iloc[-1],
                'excess_total_return': excess_cumulative.iloc[-1],
                'win_rate': win_rate,
                'outperformance_periods': (excess_returns > 0).sum(),
                'underperformance_periods': (excess_returns < 0).sum(),
                'neutral_periods': (excess_returns == 0).sum(),
                'max_outperformance': excess_returns.max(),
                'max_underperformance': excess_returns.min(),
                'average_outperformance': excess_returns[excess_returns > 0].mean() if (excess_returns > 0).any() else 0,
                'average_underperformance': excess_returns[excess_returns < 0].mean() if (excess_returns < 0).any() else 0
            }
            
            return comparison
            
        except Exception as e:
            return {'error': f"比较收益率失败: {e}"}
    
    def compare_risk_metrics(self, portfolio_returns: pd.Series) -> Dict:
        """比较风险指标"""
        try:
            # 对齐数据
            common_index = portfolio_returns.index.intersection(self.benchmark_returns.index)
            if len(common_index) == 0:
                return {'error': '无法对齐组合和基准数据'}
            
            port_aligned = portfolio_returns.reindex(common_index)
            bench_aligned = self.benchmark_returns.reindex(common_index)
            
            # 计算各项风险指标
            port_volatility = port_aligned.std() * np.sqrt(252)
            bench_volatility = bench_aligned.std() * np.sqrt(252)
            
            port_sharpe = self.metrics_calculator.calculate_sharpe_ratio(port_aligned)
            bench_sharpe = self.metrics_calculator.calculate_sharpe_ratio(bench_aligned)
            
            port_sortino = self.metrics_calculator.calculate_sortino_ratio(port_aligned)
            bench_sortino = self.metrics_calculator.calculate_sortino_ratio(bench_aligned)
            
            # 计算最大回撤
            port_cumulative = (1 + port_aligned).cumprod()
            bench_cumulative = (1 + bench_aligned).cumprod()
            
            port_max_drawdown = self.metrics_calculator.calculate_max_drawdown(port_cumulative)
            bench_max_drawdown = self.metrics_calculator.calculate_max_drawdown(bench_cumulative)
            
            # 计算相关性
            correlation = port_aligned.corr(bench_aligned)
            
            # Beta和Alpha
            beta = self.metrics_calculator.calculate_beta(port_aligned, bench_aligned)
            alpha = self.metrics_calculator.calculate_alpha(port_aligned, bench_aligned)
            
            # 信息比率和跟踪误差
            information_ratio = self.metrics_calculator.calculate_information_ratio(port_aligned, bench_aligned)
            tracking_error = self.metrics_calculator.calculate_tracking_error(port_aligned, bench_aligned)
            
            comparison = {
                'portfolio_volatility': port_volatility,
                'benchmark_volatility': bench_volatility,
                'volatility_ratio': port_volatility / bench_volatility if bench_volatility != 0 else 0,
                'portfolio_sharpe': port_sharpe,
                'benchmark_sharpe': bench_sharpe,
                'sharpe_difference': port_sharpe - bench_sharpe,
                'portfolio_sortino': port_sortino,
                'benchmark_sortino': bench_sortino,
                'sortino_difference': port_sortino - bench_sortino,
                'portfolio_max_drawdown': port_max_drawdown,
                'benchmark_max_drawdown': bench_max_drawdown,
                'max_drawdown_difference': port_max_drawdown - bench_max_drawdown,
                'correlation': correlation,
                'beta': beta,
                'alpha': alpha,
                'information_ratio': information_ratio,
                'tracking_error': tracking_error
            }
            
            return comparison
            
        except Exception as e:
            return {'error': f"比较风险指标失败: {e}"}
    
    def rolling_comparison(self, portfolio_returns: pd.Series,
                          window: int = 252) -> pd.DataFrame:
        """滚动比较"""
        try:
            # 对齐数据
            common_index = portfolio_returns.index.intersection(self.benchmark_returns.index)
            if len(common_index) < window:
                return pd.DataFrame({'error': '数据不足以进行滚动比较'})
            
            port_aligned = portfolio_returns.reindex(common_index)
            bench_aligned = self.benchmark_returns.reindex(common_index)
            
            # 计算滚动指标
            rolling_results = []
            
            for i in range(window, len(port_aligned) + 1):
                port_window = port_aligned.iloc[i-window:i]
                bench_window = bench_aligned.iloc[i-window:i]
                
                # 滚动收益率
                port_return = (1 + port_window).prod() - 1
                bench_return = (1 + bench_window).prod() - 1
                excess_return = port_return - bench_return
                
                # 滚动波动率
                port_vol = port_window.std() * np.sqrt(252)
                bench_vol = bench_window.std() * np.sqrt(252)
                
                # 滚动夏普比率
                port_sharpe = self.metrics_calculator.calculate_sharpe_ratio(port_window)
                bench_sharpe = self.metrics_calculator.calculate_sharpe_ratio(bench_window)
                
                # 滚动相关性
                correlation = port_window.corr(bench_window)
                
                # 滚动Beta
                beta = self.metrics_calculator.calculate_beta(port_window, bench_window)
                
                # 滚动信息比率
                information_ratio = self.metrics_calculator.calculate_information_ratio(port_window, bench_window)
                
                rolling_results.append({
                    'date': port_aligned.index[i-1],
                    'portfolio_return': port_return,
                    'benchmark_return': bench_return,
                    'excess_return': excess_return,
                    'portfolio_volatility': port_vol,
                    'benchmark_volatility': bench_vol,
                    'portfolio_sharpe': port_sharpe,
                    'benchmark_sharpe': bench_sharpe,
                    'correlation': correlation,
                    'beta': beta,
                    'information_ratio': information_ratio
                })
            
            result_df = pd.DataFrame(rolling_results)
            result_df.set_index('date', inplace=True)
            
            return result_df
            
        except Exception as e:
            return pd.DataFrame({'error': f"滚动比较失败: {e}"})
    
    def regime_analysis(self, portfolio_returns: pd.Series,
                       market_regimes: pd.Series) -> Dict:
        """状态分析"""
        try:
            # 对齐数据
            common_index = portfolio_returns.index.intersection(self.benchmark_returns.index).intersection(market_regimes.index)
            if len(common_index) == 0:
                return {'error': '无法对齐所有数据'}
            
            port_aligned = portfolio_returns.reindex(common_index)
            bench_aligned = self.benchmark_returns.reindex(common_index)
            regimes_aligned = market_regimes.reindex(common_index)
            
            # 按市场状态分组分析
            regime_analysis = {}
            
            for regime in regimes_aligned.unique():
                if pd.isna(regime):
                    continue
                    
                mask = regimes_aligned == regime
                port_regime = port_aligned[mask]
                bench_regime = bench_aligned[mask]
                
                if len(port_regime) == 0:
                    continue
                
                # 计算该状态下的指标
                port_return = port_regime.mean() * 252  # 年化收益率
                bench_return = bench_regime.mean() * 252
                excess_return = port_return - bench_return
                
                port_vol = port_regime.std() * np.sqrt(252)
                bench_vol = bench_regime.std() * np.sqrt(252)
                
                port_sharpe = self.metrics_calculator.calculate_sharpe_ratio(port_regime)
                bench_sharpe = self.metrics_calculator.calculate_sharpe_ratio(bench_regime)
                
                # 胜率
                win_rate = (port_regime > bench_regime).sum() / len(port_regime)
                
                # 相关性
                correlation = port_regime.corr(bench_regime)
                
                # Beta
                beta = self.metrics_calculator.calculate_beta(port_regime, bench_regime)
                
                regime_analysis[str(regime)] = {
                    'periods': len(port_regime),
                    'portfolio_return': port_return,
                    'benchmark_return': bench_return,
                    'excess_return': excess_return,
                    'portfolio_volatility': port_vol,
                    'benchmark_volatility': bench_vol,
                    'portfolio_sharpe': port_sharpe,
                    'benchmark_sharpe': bench_sharpe,
                    'win_rate': win_rate,
                    'correlation': correlation,
                    'beta': beta
                }
            
            # 整体统计
            regime_summary = {
                'total_periods': len(port_aligned),
                'regime_count': len(regime_analysis),
                'regime_breakdown': {k: v['periods'] for k, v in regime_analysis.items()},
                'best_regime': max(regime_analysis.items(), key=lambda x: x[1]['excess_return'])[0] if regime_analysis else None,
                'worst_regime': min(regime_analysis.items(), key=lambda x: x[1]['excess_return'])[0] if regime_analysis else None
            }
            
            return {
                'regime_analysis': regime_analysis,
                'regime_summary': regime_summary
            }
            
        except Exception as e:
            return {'error': f"状态分析失败: {e}"}


class FactorAttributionMonitor:
    """因子归因监控器"""
    
    def __init__(self):
        """初始化因子归因监控器"""
        self.factor_exposures_history = []
        self.attribution_history = []
        
    def calculate_factor_exposure(self, weights: pd.Series,
                                factor_loadings: pd.DataFrame) -> pd.Series:
        """计算因子暴露"""
        try:
            # 对齐权重和因子载荷
            common_assets = weights.index.intersection(factor_loadings.index)
            if len(common_assets) == 0:
                return pd.Series()
            
            weights_aligned = weights.reindex(common_assets)
            loadings_aligned = factor_loadings.reindex(common_assets)
            
            # 计算因子暴露：权重 × 因子载荷
            factor_exposures = weights_aligned.values @ loadings_aligned.values
            
            exposure_series = pd.Series(
                factor_exposures,
                index=factor_loadings.columns,
                name='factor_exposure'
            )
            
            return exposure_series
            
        except Exception as e:
            return pd.Series()
    
    def monitor_factor_drift(self, current_exposures: pd.Series,
                           target_exposures: pd.Series) -> Dict:
        """监控因子漂移"""
        try:
            # 对齐因子
            common_factors = current_exposures.index.intersection(target_exposures.index)
            if len(common_factors) == 0:
                return {'error': '无法对齐因子数据'}
            
            current_aligned = current_exposures.reindex(common_factors)
            target_aligned = target_exposures.reindex(common_factors)
            
            # 计算漂移
            drift = current_aligned - target_aligned
            drift_pct = drift / target_aligned.abs()
            
            # 统计
            max_drift = drift.abs().max()
            max_drift_factor = drift.abs().idxmax()
            avg_drift = drift.abs().mean()
            
            # 检查是否需要调整
            drift_threshold = 0.1  # 10%漂移阈值
            significant_drifts = drift_pct.abs() > drift_threshold
            
            result = {
                'drift_vector': drift.to_dict(),
                'drift_pct': drift_pct.to_dict(),
                'max_drift': max_drift,
                'max_drift_factor': max_drift_factor,
                'avg_drift': avg_drift,
                'significant_drifts': significant_drifts.sum(),
                'drift_factors': drift_pct[significant_drifts].index.tolist(),
                'rebalance_needed': significant_drifts.any()
            }
            
            return result
            
        except Exception as e:
            return {'error': f"监控因子漂移失败: {e}"}
    
    def analyze_factor_contribution(self, portfolio_returns: pd.Series,
                                  factor_returns: pd.DataFrame,
                                  factor_exposures: pd.DataFrame) -> pd.DataFrame:
        """分析因子贡献"""
        try:
            # 对齐时间序列
            common_dates = portfolio_returns.index.intersection(factor_returns.index).intersection(factor_exposures.index)
            if len(common_dates) == 0:
                return pd.DataFrame({'error': '无法对齐时间序列数据'})
            
            port_returns = portfolio_returns.reindex(common_dates)
            factor_rets = factor_returns.reindex(common_dates)
            exposures = factor_exposures.reindex(common_dates)
            
            # 计算因子贡献
            contributions = pd.DataFrame(index=common_dates, columns=factor_rets.columns)
            
            for date in common_dates:
                for factor in factor_rets.columns:
                    if factor in exposures.columns:
                        # 因子贡献 = 因子暴露 × 因子收益
                        contributions.loc[date, factor] = (
                            exposures.loc[date, factor] * factor_rets.loc[date, factor]
                        )
            
            # 计算残差
            total_factor_contribution = contributions.sum(axis=1)
            residual_returns = port_returns - total_factor_contribution
            contributions['residual'] = residual_returns
            
            return contributions.fillna(0)
            
        except Exception as e:
            return pd.DataFrame({'error': f"分析因子贡献失败: {e}"})


class TradingMetricsMonitor:
    """交易指标监控器"""
    
    def __init__(self):
        """初始化交易指标监控器"""
        self.trading_history = []
        self.cost_history = []
        
    def monitor_turnover(self, trades: pd.DataFrame,
                        portfolio_value: float) -> float:
        """监控换手率"""
        try:
            if trades.empty or portfolio_value <= 0:
                return 0.0
            
            # 计算交易总额
            if 'quantity' in trades.columns and 'price' in trades.columns:
                trade_value = (trades['quantity'].abs() * trades['price']).sum()
            elif 'value' in trades.columns:
                trade_value = trades['value'].abs().sum()
            else:
                return 0.0
            
            # 换手率 = 交易总额 / 组合价值
            turnover = trade_value / portfolio_value
            
            return turnover
            
        except Exception as e:
            return 0.0
    
    def monitor_transaction_costs(self, trades: pd.DataFrame) -> Dict:
        """监控交易成本"""
        try:
            if trades.empty:
                return {'total_costs': 0, 'avg_cost': 0, 'cost_breakdown': {}}
            
            costs = {
                'commission': 0,
                'spread': 0,
                'market_impact': 0,
                'total': 0
            }
            
            # 计算不同类型的成本
            if 'commission' in trades.columns:
                costs['commission'] = trades['commission'].sum()
            
            if 'spread_cost' in trades.columns:
                costs['spread'] = trades['spread_cost'].sum()
                
            if 'market_impact' in trades.columns:
                costs['market_impact'] = trades['market_impact'].sum()
            
            # 总成本
            costs['total'] = sum(costs.values())
            
            # 平均成本
            avg_cost = costs['total'] / len(trades) if len(trades) > 0 else 0
            
            # 成本率（相对于交易金额）
            if 'value' in trades.columns:
                total_value = trades['value'].abs().sum()
                cost_rate = costs['total'] / total_value if total_value > 0 else 0
            else:
                cost_rate = 0
            
            result = {
                'total_costs': costs['total'],
                'avg_cost': avg_cost,
                'cost_rate': cost_rate,
                'cost_breakdown': costs,
                'trade_count': len(trades)
            }
            
            return result
            
        except Exception as e:
            return {'error': f"监控交易成本失败: {e}"}
    
    def monitor_execution_quality(self, trades: pd.DataFrame,
                                benchmark_prices: pd.DataFrame) -> Dict:
        """监控执行质量"""
        try:
            if trades.empty or benchmark_prices.empty:
                return {'execution_quality': 0, 'slippage': 0}
            
            quality_metrics = {
                'total_slippage': 0,
                'avg_slippage': 0,
                'execution_shortfall': 0,
                'fill_rate': 0,
                'time_weighted_slippage': 0
            }
            
            slippages = []
            
            for _, trade in trades.iterrows():
                symbol = trade.get('symbol', '')
                trade_price = trade.get('price', 0)
                trade_time = trade.get('timestamp', None)
                
                # 获取基准价格
                if symbol in benchmark_prices.columns and trade_time is not None:
                    try:
                        benchmark_price = benchmark_prices.loc[trade_time, symbol]
                        slippage = (trade_price - benchmark_price) / benchmark_price
                        slippages.append(slippage)
                    except:
                        continue
            
            if slippages:
                quality_metrics['total_slippage'] = sum(slippages)
                quality_metrics['avg_slippage'] = np.mean(slippages)
                quality_metrics['execution_shortfall'] = np.mean([abs(s) for s in slippages])
            
            # 成交率
            if 'quantity' in trades.columns and 'target_quantity' in trades.columns:
                filled_quantity = trades['quantity'].sum()
                target_quantity = trades['target_quantity'].sum()
                quality_metrics['fill_rate'] = filled_quantity / target_quantity if target_quantity != 0 else 0
            else:
                quality_metrics['fill_rate'] = 1.0  # 假设全部成交
            
            return quality_metrics
            
        except Exception as e:
            return {'error': f"监控执行质量失败: {e}"}
    
    def calculate_win_rate(self, trades: pd.DataFrame) -> float:
        """计算胜率"""
        try:
            if trades.empty:
                return 0.0
            
            # 计算每笔交易的盈亏
            if 'pnl' in trades.columns:
                winning_trades = (trades['pnl'] > 0).sum()
                total_trades = len(trades)
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
            elif 'entry_price' in trades.columns and 'exit_price' in trades.columns:
                # 根据买卖价格计算盈亏
                buy_trades = trades[trades['side'] == 'buy']
                sell_trades = trades[trades['side'] == 'sell']
                
                # 简化计算：假设每个买入对应一个卖出
                if len(buy_trades) > 0 and len(sell_trades) > 0:
                    # 计算平均买入价和卖出价
                    avg_buy_price = buy_trades['price'].mean()
                    avg_sell_price = sell_trades['price'].mean()
                    
                    win_rate = 1.0 if avg_sell_price > avg_buy_price else 0.0
                else:
                    win_rate = 0.0
            else:
                win_rate = 0.0
            
            return win_rate
            
        except Exception as e:
            return 0.0


class PerformanceMonitor:
    """性能监控主类"""
    
    def __init__(self, config: Dict = None):
        """初始化性能监控器"""
        self.config = config or {}
        self.metrics_calculator = MetricsCalculator()
        self.real_time_monitor = RealTimeMonitor(self.config.get('alert_thresholds'))
        self.alert_manager = AlertManager(self.config.get('notification_config'))
        self.performance_reporter = PerformanceReporter()
        self.factor_attribution_monitor = FactorAttributionMonitor()
        self.trading_metrics_monitor = TradingMetricsMonitor()
        
        # 数据存储
        self.portfolio_history = pd.DataFrame()
        self.benchmark_history = pd.DataFrame()
        self.metrics_history = []
        self.current_benchmark_comparator = None
        
    def initialize(self, config: Dict):
        """初始化配置"""
        self.config.update(config)
        
        # 初始化基准比较器
        if 'benchmark_returns' in config and config['benchmark_returns'] is not None:
            self.current_benchmark_comparator = BenchmarkComparator(config['benchmark_returns'])
        
        # 更新告警阈值
        if 'alert_thresholds' in config:
            self.real_time_monitor.alert_thresholds.update(config['alert_thresholds'])
    
    def update_portfolio_metrics(self, portfolio_value: pd.Series,
                               benchmark_value: pd.Series = None) -> Dict:
        """更新组合指标"""
        try:
            # 计算基础指标
            returns = self.metrics_calculator.calculate_returns(portfolio_value)
            cumulative_returns = self.metrics_calculator.calculate_cumulative_returns(returns)
            
            # 计算性能指标
            current_metrics = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': portfolio_value.iloc[-1] if len(portfolio_value) > 0 else 0,
                'daily_return': returns.iloc[-1] if len(returns) > 0 else 0,
                'total_return': cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else 0,
                'volatility': returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
                'sharpe_ratio': self.metrics_calculator.calculate_sharpe_ratio(returns),
                'sortino_ratio': self.metrics_calculator.calculate_sortino_ratio(returns),
                'max_drawdown': self.metrics_calculator.calculate_max_drawdown(portfolio_value),
                'calmar_ratio': self.metrics_calculator.calculate_calmar_ratio(returns)
            }
            
            # 与基准比较
            if benchmark_value is not None and len(benchmark_value) > 0:
                benchmark_returns = self.metrics_calculator.calculate_returns(benchmark_value)
                if len(benchmark_returns) > 0:
                    current_metrics.update({
                        'benchmark_return': benchmark_returns.iloc[-1],
                        'excess_return': current_metrics['daily_return'] - benchmark_returns.iloc[-1],
                        'information_ratio': self.metrics_calculator.calculate_information_ratio(returns, benchmark_returns),
                        'tracking_error': self.metrics_calculator.calculate_tracking_error(returns, benchmark_returns),
                        'beta': self.metrics_calculator.calculate_beta(returns, benchmark_returns),
                        'alpha': self.metrics_calculator.calculate_alpha(returns, benchmark_returns)
                    })
            
            # 更新历史记录
            self.metrics_history.append(current_metrics)
            
            # 保持最近1000条记录
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            return current_metrics
            
        except Exception as e:
            return {'error': f"更新组合指标失败: {e}"}
    
    def real_time_monitoring(self, current_data: Dict) -> Dict:
        """实时监控"""
        try:
            monitoring_results = {}
            
            # 监控组合价值变化
            if 'portfolio_value' in current_data and 'previous_value' in current_data:
                value_result = self.real_time_monitor.monitor_portfolio_value(
                    current_data['portfolio_value'],
                    current_data['previous_value']
                )
                monitoring_results['portfolio_value'] = value_result
                
                # 生成告警
                if value_result.get('alert', False):
                    self.alert_manager.generate_alert(
                        'portfolio_value_change',
                        value_result['alert_message'],
                        'medium'
                    )
            
            # 监控回撤
            if 'current_drawdown' in current_data:
                drawdown_result = self.real_time_monitor.monitor_drawdown(
                    current_data['current_drawdown']
                )
                monitoring_results['drawdown'] = drawdown_result
                
                if drawdown_result.get('alert', False):
                    self.alert_manager.generate_alert(
                        'drawdown',
                        drawdown_result['alert_message'],
                        'high'
                    )
            
            # 监控波动率
            if 'recent_returns' in current_data:
                volatility_result = self.real_time_monitor.monitor_volatility(
                    current_data['recent_returns']
                )
                monitoring_results['volatility'] = volatility_result
                
                if volatility_result.get('alert', False):
                    self.alert_manager.generate_alert(
                        'volatility',
                        volatility_result['alert_message'],
                        'medium'
                    )
            
            # 监控持仓集中度
            if 'weights' in current_data:
                concentration_result = self.real_time_monitor.monitor_position_concentration(
                    current_data['weights']
                )
                monitoring_results['concentration'] = concentration_result
                
                if concentration_result.get('alert', False):
                    self.alert_manager.generate_alert(
                        'concentration',
                        concentration_result['alert_message'],
                        'medium'
                    )
            
            # 监控杠杆率
            if 'leverage' in current_data:
                leverage_result = self.real_time_monitor.monitor_leverage(
                    current_data['leverage']
                )
                monitoring_results['leverage'] = leverage_result
                
                if leverage_result.get('alert', False):
                    self.alert_manager.generate_alert(
                        'leverage',
                        leverage_result['alert_message'],
                        'high'
                    )
            
            # 检查整体风险限制
            if monitoring_results:
                risk_check_metrics = {
                    'volatility': monitoring_results.get('volatility', {}).get('volatility', 0),
                    'drawdown': monitoring_results.get('drawdown', {}).get('current_drawdown', 0),
                    'leverage': monitoring_results.get('leverage', {}).get('leverage', 0)
                }
                
                risk_limits = self.real_time_monitor.check_risk_limits(risk_check_metrics)
                monitoring_results['risk_limits'] = risk_limits
                
                # 检查是否有风险限制违规
                failed_checks = [check for check, passed in risk_limits.items() if not passed]
                if failed_checks:
                    self.alert_manager.generate_alert(
                        'risk_limits',
                        f"风险限制检查失败: {failed_checks}",
                        'high'
                    )
            
            monitoring_results['timestamp'] = datetime.now().isoformat()
            return monitoring_results
            
        except Exception as e:
            return {'error': f"实时监控失败: {e}"}
    
    def generate_performance_summary(self, start_date: str = None,
                                   end_date: str = None) -> Dict:
        """生成性能摘要"""
        try:
            # 准备数据
            if self.metrics_history:
                metrics_df = pd.DataFrame(self.metrics_history)
                
                # 过滤时间范围
                if start_date or end_date:
                    metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
                    if start_date:
                        metrics_df = metrics_df[metrics_df['timestamp'] >= start_date]
                    if end_date:
                        metrics_df = metrics_df[metrics_df['timestamp'] <= end_date]
                
                # 计算汇总统计
                summary = {
                    'period': {
                        'start_date': start_date or metrics_df['timestamp'].min().strftime('%Y-%m-%d'),
                        'end_date': end_date or metrics_df['timestamp'].max().strftime('%Y-%m-%d'),
                        'total_periods': len(metrics_df)
                    },
                    'returns': {
                        'total_return': metrics_df['total_return'].iloc[-1] if len(metrics_df) > 0 else 0,
                        'annualized_return': metrics_df['daily_return'].mean() * 252 if len(metrics_df) > 0 else 0,
                        'volatility': metrics_df['volatility'].iloc[-1] if len(metrics_df) > 0 else 0,
                        'best_day': metrics_df['daily_return'].max() if len(metrics_df) > 0 else 0,
                        'worst_day': metrics_df['daily_return'].min() if len(metrics_df) > 0 else 0
                    },
                    'risk_metrics': {
                        'sharpe_ratio': metrics_df['sharpe_ratio'].iloc[-1] if len(metrics_df) > 0 else 0,
                        'sortino_ratio': metrics_df['sortino_ratio'].iloc[-1] if len(metrics_df) > 0 else 0,
                        'max_drawdown': metrics_df['max_drawdown'].min() if len(metrics_df) > 0 else 0,
                        'calmar_ratio': metrics_df['calmar_ratio'].iloc[-1] if len(metrics_df) > 0 else 0
                    }
                }
                
                # 基准比较
                if 'benchmark_return' in metrics_df.columns:
                    summary['benchmark_comparison'] = {
                        'excess_return': metrics_df['excess_return'].sum() if len(metrics_df) > 0 else 0,
                        'information_ratio': metrics_df['information_ratio'].iloc[-1] if len(metrics_df) > 0 else 0,
                        'tracking_error': metrics_df['tracking_error'].iloc[-1] if len(metrics_df) > 0 else 0,
                        'beta': metrics_df['beta'].iloc[-1] if len(metrics_df) > 0 else 0,
                        'alpha': metrics_df['alpha'].iloc[-1] if len(metrics_df) > 0 else 0
                    }
                
                return summary
            else:
                return {'error': '无历史数据'}
                
        except Exception as e:
            return {'error': f"生成性能摘要失败: {e}"}
    
    def monitor_strategy_health(self, portfolio_data: Dict,
                              signal_data: Dict) -> Dict:
        """监控策略健康度"""
        try:
            health_metrics = {
                'overall_health': 'unknown',
                'portfolio_health': {},
                'signal_health': {},
                'trading_health': {},
                'risk_health': {}
            }
            
            # 组合健康度
            portfolio_value = portfolio_data.get('portfolio_value', pd.Series())
            if len(portfolio_value) > 0:
                returns = self.metrics_calculator.calculate_returns(portfolio_value)
                
                # 计算健康度指标
                recent_returns = returns.tail(20) if len(returns) >= 20 else returns
                health_metrics['portfolio_health'] = {
                    'recent_performance': recent_returns.sum() if len(recent_returns) > 0 else 0,
                    'stability': 1 - (recent_returns.std() / recent_returns.mean()) if len(recent_returns) > 0 and recent_returns.mean() != 0 else 0,
                    'consistency': (recent_returns > 0).sum() / len(recent_returns) if len(recent_returns) > 0 else 0,
                    'drawdown_recovery': abs(self.metrics_calculator.calculate_max_drawdown(portfolio_value))
                }
            
            # 信号健康度
            if signal_data:
                signal_strength = signal_data.get('signal_strength', pd.Series())
                if len(signal_strength) > 0:
                    health_metrics['signal_health'] = {
                        'signal_coverage': (signal_strength.abs() > 0.1).sum() / len(signal_strength),
                        'signal_strength': signal_strength.abs().mean(),
                        'signal_stability': 1 - signal_strength.std() / signal_strength.abs().mean() if signal_strength.abs().mean() != 0 else 0
                    }
            
            # 交易健康度
            trades = portfolio_data.get('trades', pd.DataFrame())
            if not trades.empty:
                portfolio_value_current = portfolio_data.get('portfolio_value', pd.Series()).iloc[-1] if len(portfolio_data.get('portfolio_value', pd.Series())) > 0 else 1000000
                turnover = self.trading_metrics_monitor.monitor_turnover(trades, portfolio_value_current)
                costs = self.trading_metrics_monitor.monitor_transaction_costs(trades)
                
                health_metrics['trading_health'] = {
                    'turnover': turnover,
                    'cost_efficiency': 1 - costs.get('cost_rate', 0),
                    'execution_quality': 1 - costs.get('avg_cost', 0) / portfolio_value_current if portfolio_value_current > 0 else 0
                }
            
            # 风险健康度
            current_metrics = self.metrics_history[-1] if self.metrics_history else {}
            if current_metrics:
                health_metrics['risk_health'] = {
                    'volatility_control': 1 - min(current_metrics.get('volatility', 0) / 0.25, 1),  # 假设25%是最大容忍波动率
                    'drawdown_control': 1 - min(abs(current_metrics.get('max_drawdown', 0)) / 0.15, 1),  # 假设15%是最大容忍回撤
                    'risk_adjusted_return': current_metrics.get('sharpe_ratio', 0) / 2 if current_metrics.get('sharpe_ratio', 0) > 0 else 0
                }
            
            # 计算整体健康度
            health_scores = []
            for category, metrics in health_metrics.items():
                if isinstance(metrics, dict) and category != 'overall_health':
                    category_score = sum(metrics.values()) / len(metrics) if metrics else 0
                    health_scores.append(max(0, min(1, category_score)))  # 限制在0-1之间
            
            overall_score = sum(health_scores) / len(health_scores) if health_scores else 0
            
            if overall_score >= 0.8:
                health_metrics['overall_health'] = 'excellent'
            elif overall_score >= 0.6:
                health_metrics['overall_health'] = 'good'
            elif overall_score >= 0.4:
                health_metrics['overall_health'] = 'fair'
            else:
                health_metrics['overall_health'] = 'poor'
            
            health_metrics['overall_score'] = overall_score
            
            return health_metrics
            
        except Exception as e:
            return {'error': f"监控策略健康度失败: {e}"}
    
    def track_model_performance(self, predictions: pd.DataFrame,
                              actual_returns: pd.DataFrame) -> Dict:
        """跟踪模型表现"""
        try:
            # 对齐预测和实际收益
            common_index = predictions.index.intersection(actual_returns.index)
            if len(common_index) == 0:
                return {'error': '无法对齐预测和实际数据'}
            
            pred_aligned = predictions.reindex(common_index)
            actual_aligned = actual_returns.reindex(common_index)
            
            model_performance = {}
            
            # 对每个模型计算性能指标
            for model_name in pred_aligned.columns:
                if model_name in actual_aligned.columns:
                    pred_series = pred_aligned[model_name]
                    actual_series = actual_aligned[model_name]
                    
                    # 计算相关性
                    correlation = pred_series.corr(actual_series)
                    
                    # 计算信息系数（IC）
                    ic = correlation
                    
                    # 计算RMSE
                    rmse = np.sqrt(((pred_series - actual_series) ** 2).mean())
                    
                    # 计算方向准确率
                    direction_accuracy = ((pred_series > 0) == (actual_series > 0)).mean()
                    
                    # 计算分位数收益
                    quintiles = pd.qcut(pred_series, 5, labels=False, duplicates='drop')
                    quintile_returns = actual_series.groupby(quintiles).mean()
                    
                    model_performance[model_name] = {
                        'ic': ic,
                        'rmse': rmse,
                        'direction_accuracy': direction_accuracy,
                        'quintile_returns': quintile_returns.to_dict() if len(quintile_returns) > 0 else {},
                        'rank_correlation': pred_series.corr(actual_series, method='spearman')
                    }
            
            # 计算整体模型性能
            if model_performance:
                avg_ic = sum(perf['ic'] for perf in model_performance.values()) / len(model_performance)
                avg_direction_accuracy = sum(perf['direction_accuracy'] for perf in model_performance.values()) / len(model_performance)
                
                model_performance['overall'] = {
                    'avg_ic': avg_ic,
                    'avg_direction_accuracy': avg_direction_accuracy,
                    'model_count': len(model_performance) - 1,  # 排除overall
                    'best_model': max(model_performance.items(), key=lambda x: x[1]['ic'] if isinstance(x[1], dict) and 'ic' in x[1] else 0)[0]
                }
            
            return model_performance
            
        except Exception as e:
            return {'error': f"跟踪模型表现失败: {e}"}
    
    def monitor_factor_performance(self, factor_returns: pd.DataFrame,
                                 weights: pd.DataFrame) -> Dict:
        """监控因子表现"""
        try:
            # 使用因子归因监控器
            factor_performance = {}
            
            # 计算因子贡献
            if len(factor_returns) > 0 and len(weights) > 0:
                # 构建组合收益
                common_dates = factor_returns.index.intersection(weights.index)
                if len(common_dates) == 0:
                    return {'error': '无法对齐因子收益和权重数据'}
                
                # 计算组合收益（简化计算）
                portfolio_returns = (factor_returns.reindex(common_dates) * weights.reindex(common_dates)).sum(axis=1)
                
                # 分析因子贡献
                contributions = self.factor_attribution_monitor.analyze_factor_contribution(
                    portfolio_returns, factor_returns, weights
                )
                
                if not contributions.empty and 'error' not in contributions.columns:
                    # 计算每个因子的统计信息
                    for factor in factor_returns.columns:
                        if factor in contributions.columns:
                            factor_contrib = contributions[factor]
                            
                            factor_performance[factor] = {
                                'total_contribution': factor_contrib.sum(),
                                'avg_contribution': factor_contrib.mean(),
                                'volatility': factor_contrib.std(),
                                'sharpe_ratio': factor_contrib.mean() / factor_contrib.std() if factor_contrib.std() != 0 else 0,
                                'hit_rate': (factor_contrib > 0).sum() / len(factor_contrib)
                            }
                    
                    # 计算残差性能
                    if 'residual' in contributions.columns:
                        residual_contrib = contributions['residual']
                        factor_performance['residual'] = {
                            'total_contribution': residual_contrib.sum(),
                            'avg_contribution': residual_contrib.mean(),
                            'volatility': residual_contrib.std(),
                            'residual_ratio': residual_contrib.std() / portfolio_returns.std() if portfolio_returns.std() != 0 else 0
                        }
                
                # 计算因子暴露
                if len(weights) > 0:
                    latest_weights = weights.iloc[-1] if len(weights) > 0 else pd.Series()
                    if len(latest_weights) > 0:
                        # 假设因子载荷等于因子收益（简化）
                        factor_loadings = factor_returns.iloc[-1:].T if len(factor_returns) > 0 else pd.DataFrame()
                        if not factor_loadings.empty:
                            exposures = self.factor_attribution_monitor.calculate_factor_exposure(
                                latest_weights, factor_loadings
                            )
                            
                            if not exposures.empty:
                                factor_performance['current_exposures'] = exposures.to_dict()
            
            return factor_performance
            
        except Exception as e:
            return {'error': f"监控因子表现失败: {e}"}
    
    def create_dashboard_data(self) -> Dict:
        """创建仪表板数据"""
        try:
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'summary': {},
                'charts': {},
                'alerts': {},
                'metrics': {}
            }
            
            # 获取最新指标
            if self.metrics_history:
                latest_metrics = self.metrics_history[-1]
                dashboard_data['summary'] = {
                    'portfolio_value': latest_metrics.get('portfolio_value', 0),
                    'daily_return': latest_metrics.get('daily_return', 0),
                    'total_return': latest_metrics.get('total_return', 0),
                    'sharpe_ratio': latest_metrics.get('sharpe_ratio', 0),
                    'max_drawdown': latest_metrics.get('max_drawdown', 0),
                    'volatility': latest_metrics.get('volatility', 0)
                }
                
                # 历史指标
                recent_metrics = self.metrics_history[-30:] if len(self.metrics_history) >= 30 else self.metrics_history
                dashboard_data['metrics'] = {
                    'recent_returns': [m.get('daily_return', 0) for m in recent_metrics],
                    'recent_values': [m.get('portfolio_value', 0) for m in recent_metrics],
                    'recent_timestamps': [m.get('timestamp', '') for m in recent_metrics]
                }
            
            # 获取告警信息
            alert_summary = self.alert_manager.get_alert_summary()
            dashboard_data['alerts'] = {
                'total_alerts': alert_summary.get('total_alerts', 0),
                'recent_alerts': alert_summary.get('recent_alerts', []),
                'alert_types': alert_summary.get('alert_counters', {})
            }
            
            # 图表数据（简化）
            dashboard_data['charts'] = {
                'performance_chart': 'performance_chart.png',
                'drawdown_chart': 'drawdown_chart.png',
                'risk_metrics_chart': 'risk_metrics_chart.png'
            }
            
            return dashboard_data
            
        except Exception as e:
            return {'error': f"创建仪表板数据失败: {e}"}
    
    def export_metrics(self, format: str = "csv", save_path: str = None):
        """导出指标"""
        try:
            if not self.metrics_history:
                return {'error': '无指标数据可导出'}
            
            # 转换为DataFrame
            metrics_df = pd.DataFrame(self.metrics_history)
            
            # 生成文件名
            if save_path is None:
                save_path = f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
            
            # 导出
            if format.lower() == 'csv':
                metrics_df.to_csv(save_path, index=False, encoding='utf-8')
            elif format.lower() == 'excel':
                metrics_df.to_excel(save_path, index=False)
            elif format.lower() == 'json':
                metrics_df.to_json(save_path, orient='records', indent=2)
            else:
                return {'error': f'不支持的格式: {format}'}
            
            return {'success': True, 'file_path': save_path, 'records': len(metrics_df)}
            
        except Exception as e:
            return {'error': f"导出指标失败: {e}"} 