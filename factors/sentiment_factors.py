#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情绪因子模块
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional

# 导入日志模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import Logger


class SentimentFactors:
    """情绪因子计算类"""
    
    def __init__(self, logger: Logger = None):
        """初始化情绪因子计算器"""
        if logger is not None:
            self.logger = logger.get_logger("sentiment_factors")
        else:
            self.logger = Logger().get_logger("sentiment_factors")
    
    def calculate_all_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算所有情绪因子"""
        try:
            result = data.copy()
            
            # 基于价格的情绪因子
            price_sentiment = self.calculate_price_sentiment_factors(data)
            result = pd.concat([result, price_sentiment], axis=1)
            
            # 基于成交量的情绪因子
            volume_sentiment = self.calculate_volume_sentiment_factors(data)
            result = pd.concat([result, volume_sentiment], axis=1)
            
            # 基于资金流的情绪因子
            money_flow_sentiment = self.calculate_money_flow_factors(data)
            result = pd.concat([result, money_flow_sentiment], axis=1)
            
            # 综合情绪指标
            composite_sentiment = self.calculate_composite_sentiment_factors(data)
            result = pd.concat([result, composite_sentiment], axis=1)
            
            self.logger.info(f"计算情绪因子完成，新增{len([col for col in result.columns if col not in data.columns])}个因子")
            return result
            
        except Exception as e:
            self.logger.error(f"计算情绪因子失败: {e}")
            return data
    
    def calculate_price_sentiment_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算基于价格的情绪因子"""
        try:
            result = pd.DataFrame(index=data.index)
            
            # RSI背离
            result['rsi_divergence'] = self.calculate_rsi_divergence(data['close'], data['volume'])
            
            # 价量相关性
            result['price_volume_correlation'] = self.calculate_price_volume_correlation(data['close'], data['volume'])
            
            # 价格动量
            for period in [5, 10, 20]:
                if 'symbol' in data.columns:
                    result[f'price_momentum_{period}'] = data.groupby('symbol')['close'].pct_change(period)
                else:
                    result[f'price_momentum_{period}'] = data['close'].pct_change(period)
            
            # 价格加速度
            returns = data['close'].pct_change()
            result['price_acceleration'] = returns.diff()
            
            # 相对强度
            if 'symbol' in data.columns:
                for symbol in data['symbol'].unique():
                    symbol_mask = data['symbol'] == symbol
                    symbol_returns = data.loc[symbol_mask, 'close'].pct_change()
                    market_returns = data.groupby(data.index)['close'].mean().pct_change()
                    market_returns = market_returns.reindex(data.loc[symbol_mask].index)
                    result.loc[symbol_mask, 'relative_strength'] = (symbol_returns - market_returns).rolling(20).mean()
            
            return result
            
        except Exception as e:
            self.logger.error(f"计算价格情绪因子失败: {e}")
            return pd.DataFrame(index=data.index)
    
    def calculate_volume_sentiment_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算基于成交量的情绪因子"""
        try:
            result = pd.DataFrame(index=data.index)
            
            # 成交量激增
            result['volume_surge'] = self.calculate_volume_surge(data['volume'])
            
            # 买盘压力
            result['buying_pressure'] = self.calculate_buying_pressure(data['close'], data['high'], data['low'], data['volume'])
            
            # 成交量价格趋势
            result['sentiment_volume_trend'] = self.calculate_volume_price_trend_sentiment(data['close'], data['volume'])
            
            # 成交量相对强度
            vol_ma = data['volume'].rolling(20).mean()
            result['volume_relative_strength'] = data['volume'] / vol_ma
            
            # 成交量标准化
            vol_std = data['volume'].rolling(20).std()
            result['volume_zscore'] = (data['volume'] - vol_ma) / vol_std
            
            # 成交量动量
            for period in [5, 10, 20]:
                result[f'volume_momentum_{period}'] = data['volume'].pct_change(period)
            
            return result
            
        except Exception as e:
            self.logger.error(f"计算成交量情绪因子失败: {e}")
            return pd.DataFrame(index=data.index)
    
    def calculate_money_flow_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算基于资金流的情绪因子"""
        try:
            result = pd.DataFrame(index=data.index)
            
            # 资金流量指数
            result['money_flow_index'] = self.calculate_money_flow_index(
                data['high'], data['low'], data['close'], data['volume']
            )
            
            # 聚散指标
            result['accumulation_distribution'] = self.calculate_accumulation_distribution(
                data['high'], data['low'], data['close'], data['volume']
            )
            
            # 简化的净流入计算
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            money_flow = typical_price * data['volume']
            
            # 上涨时的资金流入
            price_up = data['close'] > data['close'].shift(1)
            inflow = money_flow.where(price_up, 0)
            outflow = money_flow.where(~price_up, 0)
            
            result['net_inflow'] = inflow - outflow
            result['inflow_ratio'] = inflow.rolling(20).sum() / (inflow.rolling(20).sum() + outflow.rolling(20).sum())
            
            # 资金流动量
            result['money_flow_momentum'] = money_flow.pct_change(10)
            
            return result
            
        except Exception as e:
            self.logger.error(f"计算资金流情绪因子失败: {e}")
            return pd.DataFrame(index=data.index)
    
    def calculate_composite_sentiment_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算综合情绪因子"""
        try:
            result = pd.DataFrame(index=data.index)
            
            # 恐惧贪婪指数
            result['fear_greed_index'] = self.calculate_fear_greed_index(
                data['close'], data['volume'], data['high'] - data['low']
            )
            
            # 市场情绪强度
            returns = data['close'].pct_change()
            volume_ratio = data['volume'] / data['volume'].rolling(20).mean()
            result['sentiment_intensity'] = returns.abs() * volume_ratio
            
            # 情绪一致性
            price_direction = np.sign(returns)
            volume_direction = np.sign(data['volume'] - data['volume'].rolling(5).mean())
            result['sentiment_consistency'] = (price_direction * volume_direction).rolling(10).mean()
            
            # 极端情绪识别
            returns_zscore = (returns - returns.rolling(20).mean()) / returns.rolling(20).std()
            volume_zscore = (data['volume'] - data['volume'].rolling(20).mean()) / data['volume'].rolling(20).std()
            result['extreme_sentiment'] = np.abs(returns_zscore) + np.abs(volume_zscore)
            
            return result
            
        except Exception as e:
            self.logger.error(f"计算综合情绪因子失败: {e}")
            return pd.DataFrame(index=data.index)
    
    # 基于价格的情绪因子
    def calculate_rsi_divergence(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """RSI背离"""
        try:
            # 计算RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # 价格高点和RSI高点的背离
            price_peaks = close.rolling(10, center=True).max() == close
            rsi_peaks = rsi.rolling(10, center=True).max() == rsi
            
            # 简化的背离计算
            divergence = pd.Series(0, index=close.index)
            
            for i in range(20, len(close)):
                recent_price_peak = price_peaks.iloc[i-20:i].any()
                recent_rsi_peak = rsi_peaks.iloc[i-20:i].any()
                
                if recent_price_peak and recent_rsi_peak:
                    price_change = close.iloc[i] - close.iloc[i-20:i][price_peaks.iloc[i-20:i]].iloc[-1]
                    rsi_change = rsi.iloc[i] - rsi.iloc[i-20:i][rsi_peaks.iloc[i-20:i]].iloc[-1]
                    
                    # 价格上涨但RSI下降，形成背离
                    if price_change > 0 and rsi_change < 0:
                        divergence.iloc[i] = -1
                    elif price_change < 0 and rsi_change > 0:
                        divergence.iloc[i] = 1
            
            return divergence
            
        except Exception as e:
            self.logger.error(f"计算RSI背离失败: {e}")
            return pd.Series(index=close.index, dtype=float)
    
    def calculate_price_volume_correlation(self, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
        """价量相关性"""
        try:
            price_returns = close.pct_change()
            volume_change = volume.pct_change()
            
            correlation = price_returns.rolling(window=period).corr(volume_change)
            return correlation.fillna(0)
            
        except Exception as e:
            self.logger.error(f"计算价量相关性失败: {e}")
            return pd.Series(index=close.index, dtype=float)
    
    def calculate_volatility_premium(self, realized_vol: pd.Series, implied_vol: pd.Series) -> pd.Series:
        """波动率溢价"""
        try:
            premium = implied_vol - realized_vol
            return premium
        except Exception as e:
            self.logger.error(f"计算波动率溢价失败: {e}")
            return pd.Series(index=realized_vol.index, dtype=float)
    
    # 基于成交量的情绪因子
    def calculate_volume_surge(self, volume: pd.Series, period: int = 20) -> pd.Series:
        """成交量激增"""
        try:
            volume_ma = volume.rolling(period).mean()
            volume_std = volume.rolling(period).std()
            surge = (volume - volume_ma) / volume_std
            return surge.fillna(0)
        except Exception as e:
            self.logger.error(f"计算成交量激增失败: {e}")
            return pd.Series(index=volume.index, dtype=float)
    
    def calculate_buying_pressure(self, close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
        """买盘压力"""
        try:
            # 收盘价在当日价格区间中的位置
            price_position = (close - low) / (high - low)
            price_position = price_position.fillna(0.5)  # 当high=low时设为中性
            
            # 买盘压力 = 价格位置 * 成交量
            buying_pressure = price_position * volume
            
            # 标准化
            bp_ma = buying_pressure.rolling(20).mean()
            bp_std = buying_pressure.rolling(20).std()
            normalized_bp = (buying_pressure - bp_ma) / bp_std
            
            return normalized_bp.fillna(0)
            
        except Exception as e:
            self.logger.error(f"计算买盘压力失败: {e}")
            return pd.Series(index=close.index, dtype=float)
    
    def calculate_accumulation_distribution(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """聚散指标"""
        try:
            clv = ((close - low) - (high - close)) / (high - low)
            clv = clv.fillna(0)  # 当high=low时
            ad = (clv * volume).cumsum()
            return ad
        except Exception as e:
            self.logger.error(f"计算聚散指标失败: {e}")
            return pd.Series(index=close.index, dtype=float)
    
    def calculate_volume_price_trend_sentiment(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """成交量价格趋势情绪版本"""
        try:
            price_change_pct = close.pct_change()
            
            # 成交量权重的价格变化
            weighted_change = price_change_pct * volume
            
            # 累积成交量价格趋势
            vpt_sentiment = weighted_change.cumsum()
            
            # 标准化
            vpt_ma = vpt_sentiment.rolling(50).mean()
            vpt_std = vpt_sentiment.rolling(50).std()
            normalized_vpt = (vpt_sentiment - vpt_ma) / vpt_std
            
            return normalized_vpt.fillna(0)
            
        except Exception as e:
            self.logger.error(f"计算VPT情绪失败: {e}")
            return pd.Series(index=close.index, dtype=float)
    
    # 基于资金流的情绪因子
    def calculate_money_flow_index(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """资金流量指数"""
        try:
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            positive_flow = pd.Series(index=close.index, dtype=float)
            negative_flow = pd.Series(index=close.index, dtype=float)
            
            price_change = typical_price.diff()
            
            positive_flow = money_flow.where(price_change > 0, 0)
            negative_flow = money_flow.where(price_change < 0, 0)
            
            positive_mf = positive_flow.rolling(window=period).sum()
            negative_mf = negative_flow.rolling(window=period).sum()
            
            money_ratio = positive_mf / negative_mf
            mfi = 100 - (100 / (1 + money_ratio))
            
            return mfi.fillna(50)  # 中性值
            
        except Exception as e:
            self.logger.error(f"计算资金流量指数失败: {e}")
            return pd.Series(index=close.index, dtype=float)
    
    def calculate_net_inflow(self, buy_volume: pd.Series, sell_volume: pd.Series) -> pd.Series:
        """净流入"""
        try:
            net_inflow = buy_volume - sell_volume
            return net_inflow
        except Exception as e:
            self.logger.error(f"计算净流入失败: {e}")
            return pd.Series(index=buy_volume.index, dtype=float)
    
    # 基于技术面的情绪因子
    def calculate_fear_greed_index(self, close: pd.Series, volume: pd.Series, volatility: pd.Series) -> pd.Series:
        """恐惧贪婪指数"""
        try:
            # 标准化各个组件
            returns = close.pct_change()
            
            # 动量组件 (价格变化)
            momentum = returns.rolling(10).mean()
            momentum_norm = (momentum - momentum.rolling(50).mean()) / momentum.rolling(50).std()
            
            # 波动率组件 (波动率越高越恐惧)
            vol_norm = (volatility - volatility.rolling(50).mean()) / volatility.rolling(50).std()
            vol_component = -vol_norm  # 负号表示高波动率对应恐惧
            
            # 成交量组件
            volume_ratio = volume / volume.rolling(20).mean()
            volume_norm = (volume_ratio - volume_ratio.rolling(50).mean()) / volume_ratio.rolling(50).std()
            
            # 综合指数 (0-100, 50为中性)
            fear_greed = 50 + 10 * (momentum_norm + vol_component + volume_norm) / 3
            fear_greed = fear_greed.clip(0, 100)
            
            return fear_greed.fillna(50)
            
        except Exception as e:
            self.logger.error(f"计算恐惧贪婪指数失败: {e}")
            return pd.Series(index=close.index, dtype=float)
    
    def calculate_put_call_ratio(self, put_volume: pd.Series, call_volume: pd.Series) -> pd.Series:
        """看跌看涨比率"""
        try:
            put_call_ratio = put_volume / call_volume
            return put_call_ratio
        except Exception as e:
            self.logger.error(f"计算看跌看涨比率失败: {e}")
            return pd.Series(index=put_volume.index, dtype=float)
    
    # 额外的情绪因子
    def calculate_sentiment_momentum(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """情绪动量因子"""
        try:
            result = pd.DataFrame(index=data.index)
            
            # 价格动量情绪
            returns = data['close'].pct_change()
            result['price_sentiment_momentum'] = returns.rolling(period).mean()
            
            # 成交量动量情绪
            volume_change = data['volume'].pct_change()
            result['volume_sentiment_momentum'] = volume_change.rolling(period).mean()
            
            # 综合动量情绪
            result['composite_momentum_sentiment'] = (
                result['price_sentiment_momentum'] * 0.6 + 
                result['volume_sentiment_momentum'] * 0.4
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"计算情绪动量失败: {e}")
            return pd.DataFrame(index=data.index)
    
    def calculate_contrarian_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """反向指标"""
        try:
            result = pd.DataFrame(index=data.index)
            
            returns = data['close'].pct_change()
            
            # 过度乐观指标
            extreme_positive = returns > returns.rolling(50).quantile(0.95)
            result['excessive_optimism'] = extreme_positive.rolling(10).mean()
            
            # 过度悲观指标
            extreme_negative = returns < returns.rolling(50).quantile(0.05)
            result['excessive_pessimism'] = extreme_negative.rolling(10).mean()
            
            # 反向信号
            result['contrarian_signal'] = result['excessive_pessimism'] - result['excessive_optimism']
            
            return result
            
        except Exception as e:
            self.logger.error(f"计算反向指标失败: {e}")
            return pd.DataFrame(index=data.index)
    
    def calculate_market_stress_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """市场压力指标"""
        try:
            result = pd.DataFrame(index=data.index)
            
            returns = data['close'].pct_change()
            
            # 压力指数
            vol = returns.rolling(20).std()
            downside_vol = returns[returns < 0].rolling(20).std()
            result['stress_index'] = downside_vol / vol
            
            # 恐慌指标
            large_drops = (returns < -0.02).rolling(5).sum()
            result['panic_indicator'] = large_drops / 5
            
            # 流动性压力
            volume_stress = (data['volume'] - data['volume'].rolling(20).mean()) / data['volume'].rolling(20).std()
            price_stress = (vol - vol.rolling(50).mean()) / vol.rolling(50).std()
            result['liquidity_stress'] = (volume_stress + price_stress) / 2
            
            return result
            
        except Exception as e:
            self.logger.error(f"计算市场压力指标失败: {e}")
            return pd.DataFrame(index=data.index)
    
    def calculate_herding_behavior(self, data: pd.DataFrame) -> pd.DataFrame:
        """羊群行为指标"""
        try:
            result = pd.DataFrame(index=data.index)
            
            returns = data['close'].pct_change()
            
            # 羊群行为强度
            # 基于收益率的分散度
            if 'symbol' in data.columns:
                # 多股票情况
                cross_sectional_std = data.groupby(data.index)['close'].pct_change().groupby(level=0).std()
                result['herding_intensity'] = 1 / cross_sectional_std  # 分散度越小，羊群行为越强
            else:
                # 单股票情况，使用滚动窗口内的收益率分散度
                returns_std = returns.rolling(20).std()
                result['herding_intensity'] = 1 / returns_std
            
            # 趋势跟随强度
            trend_direction = np.sign(returns.rolling(5).mean())
            trend_consistency = (np.sign(returns) == trend_direction).rolling(10).mean()
            result['trend_following'] = trend_consistency
            
            return result
            
        except Exception as e:
            self.logger.error(f"计算羊群行为失败: {e}")
            return pd.DataFrame(index=data.index) 