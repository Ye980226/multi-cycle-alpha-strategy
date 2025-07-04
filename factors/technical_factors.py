#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术因子模块
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import warnings
from sklearn.decomposition import PCA

# 导入日志模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import Logger


class TechnicalFactors:
    """技术因子计算类"""
    
    def __init__(self, logger: Logger = None):
        """初始化技术因子计算器"""
        if logger is not None:
            self.logger = logger.get_logger("technical_factors")
        else:
            self.logger = Logger().get_logger("technical_factors")
    
    def calculate_all_factors(self, data: pd.DataFrame, periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """计算所有技术因子"""
        try:
            result = data.copy()
            
            # 计算动量因子
            momentum_factors = self.calculate_momentum_factors(data, periods)
            result = pd.concat([result, momentum_factors], axis=1)
            
            # 计算趋势因子
            trend_factors = self.calculate_trend_factors(data)
            result = pd.concat([result, trend_factors], axis=1)
            
            # 计算波动率因子
            volatility_factors = self.calculate_volatility_factors(data)
            result = pd.concat([result, volatility_factors], axis=1)
            
            # 计算成交量因子
            volume_factors = self.calculate_volume_factors(data)
            result = pd.concat([result, volume_factors], axis=1)
            
            # 计算市场微观结构因子
            if len(data) > 100:  # 只有足够数据时才计算
                microstructure_factors = self.calculate_microstructure_factors(data)
                result = pd.concat([result, microstructure_factors], axis=1)
            
            # 计算日内因子
            if len(data) > 100:  # 只有足够数据时才计算
                intraday_factors = self.calculate_intraday_factors(data)
                result = pd.concat([result, intraday_factors], axis=1)
            
            self.logger.info(f"计算技术因子完成，新增{len([col for col in result.columns if col not in data.columns])}个因子")
            return result
            
        except Exception as e:
            self.logger.error(f"计算技术因子失败: {e}")
            return data
    
    # 价格动量类因子
    def calculate_momentum_factors(self, data: pd.DataFrame, 
                                  periods: List[int] = [1, 5, 10, 20, 60]) -> pd.DataFrame:
        """计算动量因子"""
        try:
            result = data.copy()
            
            # 计算收益率动量
            for period in periods:
                if 'symbol' in data.columns:
                    result[f'momentum_{period}'] = data.groupby('symbol')['close'].pct_change(period)
                else:
                    result[f'momentum_{period}'] = data['close'].pct_change(period)
                
                # 计算价格相对位置
                if 'symbol' in data.columns:
                    price_position = data.groupby('symbol')['close'].apply(
                        lambda x: (x - x.rolling(period).min()) / 
                                  (x.rolling(period).max() - x.rolling(period).min())
                    ).reset_index(level=0, drop=True)
                    result[f'price_position_{period}'] = price_position
                else:
                    result[f'price_position_{period}'] = (
                        (data['close'] - data['close'].rolling(period).min()) / 
                        (data['close'].rolling(period).max() - data['close'].rolling(period).min())
                    )
            
            # RSI
            result['rsi_14'] = self.calculate_rsi(data['close'], 14)
            result['rsi_21'] = self.calculate_rsi(data['close'], 21)
            
            # MACD
            macd_data = self.calculate_macd(data['close'])
            result['macd'] = macd_data['macd']
            result['macd_signal'] = macd_data['signal']
            result['macd_histogram'] = macd_data['histogram']
            
            self.logger.info(f"计算动量因子完成，新增{len([col for col in result.columns if col not in data.columns])}个因子")
            return result
            
        except Exception as e:
            self.logger.error(f"计算动量因子失败: {e}")
            return data
    
    def calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """相对强弱指数"""
        try:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            self.logger.error(f"计算RSI失败: {e}")
            return pd.Series(index=close.index, dtype=float)
    
    def calculate_macd(self, close: pd.Series, fast: int = 12, 
                      slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD指标"""
        try:
            exp1 = close.ewm(span=fast).mean()
            exp2 = close.ewm(span=slow).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal).mean()
            histogram = macd - signal_line
            
            return {
                'macd': macd,
                'signal': signal_line,
                'histogram': histogram
            }
        except Exception as e:
            self.logger.error(f"计算MACD失败: {e}")
            return {
                'macd': pd.Series(index=close.index, dtype=float),
                'signal': pd.Series(index=close.index, dtype=float),
                'histogram': pd.Series(index=close.index, dtype=float)
            }
    
    def calculate_cci(self, high: pd.Series, low: pd.Series, 
                     close: pd.Series, period: int = 20) -> pd.Series:
        """顺势指标"""
        try:
            tp = (high + low + close) / 3
            sma = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
            cci = (tp - sma) / (0.015 * mad)
            return cci
        except Exception as e:
            self.logger.error(f"计算CCI失败: {e}")
            return pd.Series(index=close.index, dtype=float)
    
    def calculate_williams_r(self, high: pd.Series, low: pd.Series,
                           close: pd.Series, period: int = 14) -> pd.Series:
        """威廉指标"""
        try:
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
            return williams_r
        except Exception as e:
            self.logger.error(f"计算Williams %R失败: {e}")
            return pd.Series(index=close.index, dtype=float)
    
    # 趋势类因子
    def calculate_trend_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算趋势因子"""
        try:
            result = data.copy()
            
            # 移动平均线
            ma_data = self.calculate_moving_averages(data['close'])
            result = pd.concat([result, ma_data], axis=1)
            
            # 指数移动平均
            ema_data = self.calculate_ema_factors(data['close'])
            result = pd.concat([result, ema_data], axis=1)
            
            # ADX
            result['adx'] = self.calculate_adx(data['high'], data['low'], data['close'])
            
            # 价格通道
            for period in [10, 20, 30]:
                result[f'price_channel_upper_{period}'] = data['high'].rolling(period).max()
                result[f'price_channel_lower_{period}'] = data['low'].rolling(period).min()
                result[f'price_channel_position_{period}'] = (
                    (data['close'] - result[f'price_channel_lower_{period}']) /
                    (result[f'price_channel_upper_{period}'] - result[f'price_channel_lower_{period}'])
                )
            
            self.logger.info(f"计算趋势因子完成，新增{len([col for col in result.columns if col not in data.columns])}个因子")
            return result
            
        except Exception as e:
            self.logger.error(f"计算趋势因子失败: {e}")
            return data
    
    def calculate_moving_averages(self, close: pd.Series, 
                                periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """移动平均线"""
        try:
            result = pd.DataFrame(index=close.index)
            
            for period in periods:
                result[f'ma_{period}'] = close.rolling(window=period).mean()
                result[f'ma_ratio_{period}'] = close / result[f'ma_{period}']
                
                # 移动平均线斜率
                result[f'ma_slope_{period}'] = result[f'ma_{period}'].diff(5) / 5
            
            return result
        except Exception as e:
            self.logger.error(f"计算移动平均线失败: {e}")
            return pd.DataFrame(index=close.index)
    
    def calculate_ema_factors(self, close: pd.Series,
                            periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """指数移动平均"""
        try:
            result = pd.DataFrame(index=close.index)
            
            for period in periods:
                result[f'ema_{period}'] = close.ewm(span=period).mean()
                result[f'ema_ratio_{period}'] = close / result[f'ema_{period}']
                
                # EMA斜率
                result[f'ema_slope_{period}'] = result[f'ema_{period}'].diff(5) / 5
            
            return result
        except Exception as e:
            self.logger.error(f"计算EMA失败: {e}")
            return pd.DataFrame(index=close.index)
    
    def calculate_adx(self, high: pd.Series, low: pd.Series,
                     close: pd.Series, period: int = 14) -> pd.Series:
        """平均趋向指数"""
        try:
            # 计算+DI和-DI
            high_diff = high.diff()
            low_diff = low.diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            
            plus_di = 100 * pd.Series(plus_dm).rolling(period).sum() / tr.rolling(period).sum()
            minus_di = 100 * pd.Series(minus_dm).rolling(period).sum() / tr.rolling(period).sum()
            
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            adx = dx.rolling(period).mean()
            
            return adx
        except Exception as e:
            self.logger.error(f"计算ADX失败: {e}")
            return pd.Series(index=high.index, dtype=float)
    
    def calculate_aroon(self, high: pd.Series, low: pd.Series,
                       period: int = 25) -> Dict[str, pd.Series]:
        """阿隆指标"""
        try:
            aroon_up = high.rolling(window=period).apply(
                lambda x: (period - (len(x) - 1 - x.argmax())) / period * 100
            )
            aroon_down = low.rolling(window=period).apply(
                lambda x: (period - (len(x) - 1 - x.argmin())) / period * 100
            )
            
            return {
                'aroon_up': aroon_up,
                'aroon_down': aroon_down,
                'aroon_oscillator': aroon_up - aroon_down
            }
        except Exception as e:
            self.logger.error(f"计算Aroon失败: {e}")
            return {
                'aroon_up': pd.Series(index=high.index, dtype=float),
                'aroon_down': pd.Series(index=high.index, dtype=float),
                'aroon_oscillator': pd.Series(index=high.index, dtype=float)
            }
    
    # 波动率因子
    def calculate_volatility_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算波动率因子"""
        try:
            result = data.copy()
            
            # ATR
            result['atr'] = self.calculate_atr(data['high'], data['low'], data['close'])
            
            # 布林带
            bb_data = self.calculate_bollinger_bands(data['close'])
            result['bb_upper'] = bb_data['upper']
            result['bb_lower'] = bb_data['lower']
            result['bb_width'] = bb_data['width']
            result['bb_position'] = bb_data['position']
            
            # 历史波动率
            hv_data = self.calculate_historical_volatility(data['close'])
            result = pd.concat([result, hv_data], axis=1)
            
            # 价格波动率
            for period in [5, 10, 20, 60]:
                if 'symbol' in data.columns:
                    result[f'price_volatility_{period}'] = data.groupby('symbol')['close'].rolling(period).std().reset_index(level=0, drop=True)
                else:
                    result[f'price_volatility_{period}'] = data['close'].rolling(period).std()
                
                # 相对波动率
                result[f'relative_volatility_{period}'] = result[f'price_volatility_{period}'] / data['close']
            
            self.logger.info(f"计算波动率因子完成，新增{len([col for col in result.columns if col not in data.columns])}个因子")
            return result
            
        except Exception as e:
            self.logger.error(f"计算波动率因子失败: {e}")
            return data
    
    def calculate_atr(self, high: pd.Series, low: pd.Series,
                     close: pd.Series, period: int = 14) -> pd.Series:
        """真实波动幅度"""
        try:
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return atr
        except Exception as e:
            self.logger.error(f"计算ATR失败: {e}")
            return pd.Series(index=high.index, dtype=float)
    
    def calculate_bollinger_bands(self, close: pd.Series, period: int = 20,
                                 std_dev: int = 2) -> Dict[str, pd.Series]:
        """布林带"""
        try:
            ma = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            
            upper = ma + (std * std_dev)
            lower = ma - (std * std_dev)
            width = (upper - lower) / ma
            position = (close - lower) / (upper - lower)
            
            return {
                'upper': upper,
                'lower': lower,
                'middle': ma,
                'width': width,
                'position': position
            }
        except Exception as e:
            self.logger.error(f"计算布林带失败: {e}")
            return {
                'upper': pd.Series(index=close.index, dtype=float),
                'lower': pd.Series(index=close.index, dtype=float),
                'middle': pd.Series(index=close.index, dtype=float),
                'width': pd.Series(index=close.index, dtype=float),
                'position': pd.Series(index=close.index, dtype=float)
            }
    
    def calculate_keltner_channels(self, high: pd.Series, low: pd.Series,
                                  close: pd.Series, period: int = 20,
                                  multiplier: float = 2.0) -> Dict[str, pd.Series]:
        """肯特纳通道"""
        try:
            ma = close.rolling(window=period).mean()
            atr = self.calculate_atr(high, low, close, period)
            
            upper = ma + (multiplier * atr)
            lower = ma - (multiplier * atr)
            
            return {
                'upper': upper,
                'lower': lower,
                'middle': ma
            }
        except Exception as e:
            self.logger.error(f"计算肯特纳通道失败: {e}")
            return {
                'upper': pd.Series(index=high.index, dtype=float),
                'lower': pd.Series(index=high.index, dtype=float),
                'middle': pd.Series(index=high.index, dtype=float)
            }
    
    def calculate_historical_volatility(self, close: pd.Series,
                                       periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """历史波动率"""
        try:
            result = pd.DataFrame(index=close.index)
            
            returns = close.pct_change()
            
            for period in periods:
                result[f'historical_volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)
                
                # 波动率分位数
                result[f'volatility_percentile_{period}'] = (
                    returns.rolling(window=period).std().rolling(window=252).rank(pct=True)
                )
            
            return result
        except Exception as e:
            self.logger.error(f"计算历史波动率失败: {e}")
            return pd.DataFrame(index=close.index)
    
    # 成交量因子
    def calculate_volume_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算成交量因子"""
        try:
            result = data.copy()
            
            # OBV
            result['obv'] = self.calculate_obv(data['close'], data['volume'])
            
            # 成交量相关指标
            result['volume_ma_5'] = data['volume'].rolling(5).mean()
            result['volume_ma_20'] = data['volume'].rolling(20).mean()
            result['volume_ratio'] = data['volume'] / result['volume_ma_20']
            
            # VWAP
            result['vwap'] = self.calculate_vwap(data['high'], data['low'], data['close'], data['volume'])
            result['vwap_ratio'] = data['close'] / result['vwap']
            
            # 资金流量指标
            result['mfi'] = self.calculate_mfi(data['high'], data['low'], data['close'], data['volume'])
            
            # 成交量价格趋势
            result['volume_price_trend'] = self.calculate_volume_price_trend(data['close'], data['volume'])
            
            # 成交量震荡指标
            for period in [10, 20, 30]:
                result[f'volume_oscillator_{period}'] = (
                    (data['volume'].rolling(period//2).mean() - data['volume'].rolling(period).mean()) /
                    data['volume'].rolling(period).mean() * 100
                )
            
            self.logger.info(f"计算成交量因子完成，新增{len([col for col in result.columns if col not in data.columns])}个因子")
            return result
            
        except Exception as e:
            self.logger.error(f"计算成交量因子失败: {e}")
            return data
    
    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """成交量平衡指标"""
        try:
            price_change = close.diff()
            obv = pd.Series(index=close.index, dtype=float)
            obv.iloc[0] = volume.iloc[0]
            
            for i in range(1, len(close)):
                if price_change.iloc[i] > 0:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif price_change.iloc[i] < 0:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            return obv
        except Exception as e:
            self.logger.error(f"计算OBV失败: {e}")
            return pd.Series(index=close.index, dtype=float)
    
    def calculate_ad_line(self, high: pd.Series, low: pd.Series,
                         close: pd.Series, volume: pd.Series) -> pd.Series:
        """聚散指标"""
        try:
            clv = ((close - low) - (high - close)) / (high - low)
            clv = clv.fillna(0)  # 当high=low时
            ad = (clv * volume).cumsum()
            return ad
        except Exception as e:
            self.logger.error(f"计算A/D Line失败: {e}")
            return pd.Series(index=close.index, dtype=float)
    
    def calculate_cmf(self, high: pd.Series, low: pd.Series, close: pd.Series,
                     volume: pd.Series, period: int = 20) -> pd.Series:
        """资金流量指标"""
        try:
            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = mfm.fillna(0)
            mfv = mfm * volume
            cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
            return cmf
        except Exception as e:
            self.logger.error(f"计算CMF失败: {e}")
            return pd.Series(index=close.index, dtype=float)
    
    def calculate_vwap(self, high: pd.Series, low: pd.Series, close: pd.Series,
                      volume: pd.Series, period: int = 20) -> pd.Series:
        """成交量加权平均价"""
        try:
            typical_price = (high + low + close) / 3
            vwap = (typical_price * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
            return vwap
        except Exception as e:
            self.logger.error(f"计算VWAP失败: {e}")
            return pd.Series(index=close.index, dtype=float)
    
    def calculate_mfi(self, high: pd.Series, low: pd.Series, close: pd.Series,
                     volume: pd.Series, period: int = 14) -> pd.Series:
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
            
            mfi = 100 - (100 / (1 + positive_mf / negative_mf))
            return mfi
        except Exception as e:
            self.logger.error(f"计算MFI失败: {e}")
            return pd.Series(index=close.index, dtype=float)
    
    def calculate_volume_price_trend(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """成交量价格趋势"""
        try:
            price_change_pct = close.pct_change()
            vpt = (volume * price_change_pct).cumsum()
            return vpt
        except Exception as e:
            self.logger.error(f"计算VPT失败: {e}")
            return pd.Series(index=close.index, dtype=float)
    
    # 市场微观结构因子
    def calculate_microstructure_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算市场微观结构因子"""
        try:
            result = data.copy()
            
            # 价格冲击（简化版）
            if 'symbol' in data.columns:
                result['price_impact'] = data.groupby('symbol').apply(
                    lambda x: x['close'].pct_change().abs() / (x['volume'] / x['volume'].mean())
                ).reset_index(level=0, drop=True)
            else:
                result['price_impact'] = data['close'].pct_change().abs() / (data['volume'] / data['volume'].mean())
            
            # 流动性代理指标
            result['liquidity_proxy'] = data['volume'] / (data['high'] - data['low'])
            
            # 价格效率指标
            for period in [5, 10, 20]:
                if 'symbol' in data.columns:
                    returns = data.groupby('symbol')['close'].pct_change()
                else:
                    returns = data['close'].pct_change()
                
                # 价格延迟（1阶自相关）
                result[f'price_delay_{period}'] = returns.rolling(period).apply(
                    lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 else np.nan
                )
            
            self.logger.info(f"计算微观结构因子完成，新增{len([col for col in result.columns if col not in data.columns])}个因子")
            return result
            
        except Exception as e:
            self.logger.error(f"计算微观结构因子失败: {e}")
            return data
    
    def calculate_bid_ask_spread(self, bid: pd.Series, ask: pd.Series) -> pd.Series:
        """买卖价差"""
        try:
            spread = (ask - bid) / ((ask + bid) / 2)
            return spread
        except Exception as e:
            self.logger.error(f"计算买卖价差失败: {e}")
            return pd.Series(index=bid.index, dtype=float)
    
    def calculate_order_imbalance(self, buy_volume: pd.Series,
                                 sell_volume: pd.Series) -> pd.Series:
        """订单不平衡"""
        try:
            imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
            return imbalance
        except Exception as e:
            self.logger.error(f"计算订单不平衡失败: {e}")
            return pd.Series(index=buy_volume.index, dtype=float)
    
    def calculate_price_impact(self, returns: pd.Series, volume: pd.Series) -> pd.Series:
        """价格冲击"""
        try:
            normalized_volume = volume / volume.rolling(20).mean()
            price_impact = returns.abs() / normalized_volume
            return price_impact
        except Exception as e:
            self.logger.error(f"计算价格冲击失败: {e}")
            return pd.Series(index=returns.index, dtype=float)
    
    # 高频因子（分钟级特有）
    def calculate_intraday_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算日内因子"""
        try:
            result = data.copy()
            
            # 添加时间特征
            if isinstance(data.index, pd.DatetimeIndex):
                result['hour'] = data.index.hour
                result['minute'] = data.index.minute
                result['day_of_week'] = data.index.dayofweek
                
                # 时段效应
                result['is_opening'] = (result['hour'] == 9) & (result['minute'] < 60)
                result['is_closing'] = (result['hour'] == 14) & (result['minute'] >= 30)
                result['is_lunch'] = (result['hour'] == 11) & (result['minute'] >= 30) | (result['hour'] == 13) & (result['minute'] < 0)
            
            # 日内动量
            intraday_momentum = self.calculate_intraday_momentum(data)
            result = pd.concat([result, intraday_momentum], axis=1)
            
            # 隔夜跳空（如果有前一日数据）
            if len(data) > 1440:  # 至少2天的分钟数据
                result['overnight_gap'] = self.calculate_overnight_gap(data)
            
            self.logger.info(f"计算日内因子完成，新增{len([col for col in result.columns if col not in data.columns])}个因子")
            return result
            
        except Exception as e:
            self.logger.error(f"计算日内因子失败: {e}")
            return data
    
    def calculate_overnight_gap(self, data: pd.DataFrame) -> pd.Series:
        """隔夜跳空"""
        try:
            # 假设交易时间是9:30-15:00
            if isinstance(data.index, pd.DatetimeIndex):
                daily_data = data.resample('D').agg({
                    'open': 'first',
                    'close': 'last'
                })
                gap = (daily_data['open'] - daily_data['close'].shift(1)) / daily_data['close'].shift(1)
                
                # 将日度gap扩展到分钟级别
                gap_expanded = pd.Series(index=data.index, dtype=float)
                for date in daily_data.index:
                    if date in data.index:
                        mask = data.index.date == date.date()
                        gap_expanded[mask] = gap.loc[date]
                
                return gap_expanded
            else:
                return pd.Series(index=data.index, dtype=float)
                
        except Exception as e:
            self.logger.error(f"计算隔夜跳空失败: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def calculate_intraday_momentum(self, data: pd.DataFrame,
                                   periods: List[int] = [5, 15, 30, 60]) -> pd.DataFrame:
        """日内动量"""
        try:
            result = pd.DataFrame(index=data.index)
            
            for period in periods:
                if 'symbol' in data.columns:
                    result[f'intraday_momentum_{period}'] = data.groupby('symbol')['close'].pct_change(period)
                else:
                    result[f'intraday_momentum_{period}'] = data['close'].pct_change(period)
                
                # 日内反转
                result[f'intraday_reversal_{period}'] = -result[f'intraday_momentum_{period}']
            
            return result
        except Exception as e:
            self.logger.error(f"计算日内动量失败: {e}")
            return pd.DataFrame(index=data.index)
    
    def calculate_volume_profile(self, price: pd.Series, volume: pd.Series) -> pd.DataFrame:
        """成交量分布"""
        try:
            # 简化的成交量分布计算
            result = pd.DataFrame(index=price.index)
            
            # 价格区间分析
            price_bins = 20
            for window in [60, 240, 480]:  # 1小时、4小时、8小时窗口
                rolling_data = pd.concat([price, volume], axis=1).rolling(window)
                
                def volume_at_price(x):
                    if len(x) < 2:
                        return np.nan
                    hist, bins = np.histogram(x.iloc[:, 0], bins=price_bins, weights=x.iloc[:, 1])
                    max_volume_price = bins[np.argmax(hist)]
                    return max_volume_price
                
                result[f'volume_weighted_price_{window}'] = rolling_data.apply(volume_at_price)
            
            return result
        except Exception as e:
            self.logger.error(f"计算成交量分布失败: {e}")
            return pd.DataFrame(index=price.index)
    
    def calculate_time_of_day_effects(self, data: pd.DataFrame) -> pd.DataFrame:
        """时段效应"""
        try:
            result = pd.DataFrame(index=data.index)
            
            if isinstance(data.index, pd.DatetimeIndex):
                # 小时效应
                hour_returns = data.groupby(data.index.hour)['close'].pct_change().mean()
                result['hour_effect'] = data.index.hour.map(hour_returns)
                
                # 分钟效应
                minute_returns = data.groupby(data.index.minute)['close'].pct_change().mean()
                result['minute_effect'] = data.index.minute.map(minute_returns)
                
                # 星期效应
                dow_returns = data.groupby(data.index.dayofweek)['close'].pct_change().mean()
                result['dow_effect'] = data.index.dayofweek.map(dow_returns)
            
            return result
        except Exception as e:
            self.logger.error(f"计算时段效应失败: {e}")
            return pd.DataFrame(index=data.index)
    
    # 跨周期因子
    def calculate_cross_timeframe_factors(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """计算跨时间框架因子"""
        try:
            if '1min' not in data_dict:
                raise ValueError("需要1分钟数据作为基准")
            
            base_data = data_dict['1min']
            result = base_data.copy()
            
            # 趋势一致性
            if '5min' in data_dict and '1h' in data_dict:
                short_trend = data_dict['5min']['close'].pct_change(12)  # 1小时趋势
                long_trend = data_dict['1h']['close'].pct_change(24)     # 1天趋势
                
                # 对齐到1分钟级别
                short_trend_aligned = short_trend.reindex(base_data.index, method='ffill')
                long_trend_aligned = long_trend.reindex(base_data.index, method='ffill')
                
                result['trend_alignment'] = self.calculate_trend_alignment(short_trend_aligned, long_trend_aligned)
            
            # 波动率比值
            if '5min' in data_dict:
                short_vol = base_data['close'].rolling(60).std()  # 1小时波动率
                long_vol = data_dict['5min']['close'].rolling(288).std().reindex(base_data.index, method='ffill')  # 1天波动率
                
                result['volatility_ratio'] = self.calculate_volatility_ratio(short_vol, long_vol)
            
            self.logger.info(f"计算跨周期因子完成，新增{len([col for col in result.columns if col not in base_data.columns])}个因子")
            return result
            
        except Exception as e:
            self.logger.error(f"计算跨周期因子失败: {e}")
            return data_dict.get('1min', pd.DataFrame())
    
    def calculate_trend_alignment(self, short_term: pd.Series,
                                 long_term: pd.Series) -> pd.Series:
        """趋势一致性"""
        try:
            alignment = np.sign(short_term) * np.sign(long_term)
            return alignment
        except Exception as e:
            self.logger.error(f"计算趋势一致性失败: {e}")
            return pd.Series(index=short_term.index, dtype=float)
    
    def calculate_volatility_ratio(self, short_vol: pd.Series,
                                  long_vol: pd.Series) -> pd.Series:
        """波动率比值"""
        try:
            ratio = short_vol / long_vol
            return ratio
        except Exception as e:
            self.logger.error(f"计算波动率比值失败: {e}")
            return pd.Series(index=short_vol.index, dtype=float)
    
    # 因子组合方法
    def combine_factors(self, factors: pd.DataFrame, method: str = "pca",
                       n_components: int = 5) -> pd.DataFrame:
        """因子组合"""
        try:
            if method == "pca":
                # 只选择数值列
                numeric_factors = factors.select_dtypes(include=[np.number])
                
                # 去除缺失值
                clean_factors = numeric_factors.dropna()
                
                if len(clean_factors) == 0:
                    return pd.DataFrame(index=factors.index)
                
                # PCA降维
                pca = PCA(n_components=min(n_components, clean_factors.shape[1]))
                pca_result = pca.fit_transform(clean_factors)
                
                # 创建结果DataFrame
                result = pd.DataFrame(
                    data=pca_result,
                    index=clean_factors.index,
                    columns=[f'pc_{i+1}' for i in range(pca_result.shape[1])]
                )
                
                # 重新索引到原始索引
                result = result.reindex(factors.index)
                
                self.logger.info(f"PCA因子组合完成，生成{pca_result.shape[1]}个主成分")
                return result
            
            else:
                self.logger.warning(f"不支持的因子组合方法: {method}")
                return pd.DataFrame(index=factors.index)
                
        except Exception as e:
            self.logger.error(f"因子组合失败: {e}")
            return pd.DataFrame(index=factors.index)
    
    def calculate_factor_momentum(self, factors: pd.DataFrame,
                                 period: int = 20) -> pd.DataFrame:
        """因子动量"""
        try:
            result = pd.DataFrame(index=factors.index)
            
            numeric_factors = factors.select_dtypes(include=[np.number])
            
            for col in numeric_factors.columns:
                result[f'{col}_momentum'] = numeric_factors[col].pct_change(period)
            
            return result
        except Exception as e:
            self.logger.error(f"计算因子动量失败: {e}")
            return pd.DataFrame(index=factors.index)
    
    def calculate_factor_mean_reversion(self, factors: pd.DataFrame,
                                      period: int = 20) -> pd.DataFrame:
        """因子均值回归"""
        try:
            result = pd.DataFrame(index=factors.index)
            
            numeric_factors = factors.select_dtypes(include=[np.number])
            
            for col in numeric_factors.columns:
                rolling_mean = numeric_factors[col].rolling(period).mean()
                result[f'{col}_mean_reversion'] = (rolling_mean - numeric_factors[col]) / numeric_factors[col].rolling(period).std()
            
            return result
        except Exception as e:
            self.logger.error(f"计算因子均值回归失败: {e}")
            return pd.DataFrame(index=factors.index) 