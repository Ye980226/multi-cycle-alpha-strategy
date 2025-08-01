#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据管理模块
"""

import pandas as pd
import numpy as np
import os
import pickle
import hashlib
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import warnings

# 导入日志模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import Logger


class DataSource(ABC):
    """数据源抽象基类"""
    
    @abstractmethod
    def get_stock_data(self, symbols: List[str], start_date: str, end_date: str, 
                       frequency: str) -> pd.DataFrame:
        """获取股票数据"""
        pass
    
    @abstractmethod
    def get_index_data(self, index_code: str, start_date: str, end_date: str,
                       frequency: str) -> pd.DataFrame:
        """获取指数数据"""
        pass
    
    @abstractmethod
    def get_fundamental_data(self, symbols: List[str], start_date: str, 
                            end_date: str) -> pd.DataFrame:
        """获取基本面数据"""
        pass


class AkshareDataSource(DataSource):
    """AkShare数据源"""
    
    def __init__(self):
        """初始化AkShare数据源"""
        try:
            import akshare as ak
            self.ak = ak
            self.logger = Logger().get_logger("akshare_data")
        except ImportError:
            raise ImportError("请安装akshare包: pip install akshare")
    
    def get_stock_data(self, symbols: List[str], start_date: str, end_date: str,
                       frequency: str) -> pd.DataFrame:
        """获取股票分钟级数据"""
        all_data = []
        
        for symbol in symbols:
            try:
                # 转换股票代码格式
                symbol_code = self._convert_symbol_format(symbol)
                
                if frequency == "1min":
                    # 获取分钟级数据 (由于akshare分钟数据接口限制，这里使用日线数据模拟)
                    # 实际使用时可以调用: stock_zh_a_hist_min_em
                    data = self.ak.stock_zh_a_hist(
                        symbol=symbol_code,
                        period="daily",
                        start_date=start_date.replace("-", ""),
                        end_date=end_date.replace("-", ""),
                        adjust="qfq"
                    )
                    # 模拟分钟数据
                    if data is not None and not data.empty:
                        data = self._simulate_minute_data(data, symbol)
                elif frequency == "5min":
                    # 使用日线数据模拟5分钟数据
                    data = self.ak.stock_zh_a_hist(
                        symbol=symbol_code,
                        period="daily", 
                        start_date=start_date.replace("-", ""),
                        end_date=end_date.replace("-", ""),
                        adjust="qfq"
                    )
                    if data is not None and not data.empty:
                        data = self._simulate_minute_data(data, symbol, freq='5min')
                elif frequency == "1d":
                    data = self.ak.stock_zh_a_hist(
                        symbol=symbol_code,
                        period="daily",
                        start_date=start_date.replace("-", ""),
                        end_date=end_date.replace("-", ""),
                        adjust="qfq"
                    )
                else:
                    raise ValueError(f"不支持的频率: {frequency}")
                
                if data is not None and not data.empty:
                    # 处理akshare返回的实际列名
                    if '日期' in data.columns:
                        data['datetime'] = pd.to_datetime(data['日期'])
                        
                        # 重命名列
                        column_mapping = {
                            '开盘': 'open',
                            '收盘': 'close', 
                            '最高': 'high',
                            '最低': 'low',
                            '成交量': 'volume',
                            '成交额': 'amount'
                        }
                        
                        # 只重命名存在的列
                        for old_col, new_col in column_mapping.items():
                            if old_col in data.columns:
                                data.rename(columns={old_col: new_col}, inplace=True)
                        
                        # 添加symbol列
                        data['symbol'] = symbol
                        
                        # 数据类型转换
                        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                        for col in numeric_cols:
                            if col in data.columns:
                                data[col] = pd.to_numeric(data[col], errors='coerce')
                        
                        # 删除无效数据
                        data = data.dropna(subset=numeric_cols[:4])  # OHLC不能为空
                        
                        if len(data) > 0:
                            # 选择需要的列
                            available_cols = ['datetime', 'symbol', 'open', 'high', 'low', 'close']
                            if 'volume' in data.columns:
                                available_cols.append('volume')
                            if 'amount' in data.columns:
                                available_cols.append('amount')
                                
                            data = data[available_cols]
                            all_data.append(data)
                            
                            self.logger.info(f"获取{symbol}数据成功，共{len(data)}条")
                        else:
                            self.logger.warning(f"{symbol}: 清理后无有效数据")
                    else:
                        self.logger.error(f"{symbol}: 数据格式不符合预期，列名: {list(data.columns)}")
                
            except Exception as e:
                self.logger.error(f"获取{symbol}数据失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result.set_index('datetime', inplace=True)
            return result
        else:
            return pd.DataFrame()
    
    def _simulate_minute_data(self, daily_data: pd.DataFrame, symbol: str, freq: str = '1min') -> pd.DataFrame:
        """将日线数据模拟成分钟数据（用于演示）"""
        minute_data_list = []
        
        # 确定分钟间隔
        interval = 1 if freq == '1min' else 5
        periods_per_day = 240 // interval  # 4小时交易时间
        
        for _, row in daily_data.iterrows():
            try:
                # 数据类型转换
                date_str = str(row['日期']) if '日期' in daily_data.columns else str(row.name)
                open_price = float(row['开盘'])
                close_price = float(row['收盘'])
                high_price = float(row['最高'])
                low_price = float(row['最低'])
                total_volume = float(row['成交量']) if '成交量' in daily_data.columns else 0
                
                # 基准时间
                base_date = pd.to_datetime(date_str).date()
                base_time = pd.Timestamp.combine(base_date, pd.Timestamp('09:30:00').time())
                
                # 生成分钟时间序列
                minutes = pd.date_range(
                    start=base_time,
                    periods=periods_per_day,
                    freq=f'{interval}min'
                )
                
                # 生成价格路径
                returns = np.random.normal(0, 0.001, len(minutes))
                returns[0] = 0
                if open_price > 0:
                    returns[-1] = (close_price - open_price) / open_price
                else:
                    returns[-1] = 0
                
                prices = open_price * np.cumprod(1 + returns)
                
                # 确保价格在高低范围内
                if high_price > low_price:
                    prices = np.clip(prices, low_price, high_price)
                
                # 生成OHLC数据
                for i, minute_time in enumerate(minutes):
                    if i == 0:
                        open_p = open_price
                    else:
                        open_p = prices[i-1]
                    
                    close_p = prices[i]
                    high_p = max(open_p, close_p) * (1 + abs(np.random.normal(0, 0.001)))
                    low_p = min(open_p, close_p) * (1 - abs(np.random.normal(0, 0.001)))
                    
                    # 确保在日内高低范围内
                    if high_price > low_price:
                        high_p = min(high_p, high_price)
                        low_p = max(low_p, low_price)
                    
                    volume_p = total_volume / len(minutes) * np.random.uniform(0.5, 2.0)
                    
                    minute_data_list.append({
                        '日期': minute_time,
                        '开盘': round(float(open_p), 2),
                        '最高': round(float(high_p), 2),
                        '最低': round(float(low_p), 2),
                        '收盘': round(float(close_p), 2),
                        '成交量': int(float(volume_p))
                    })
                    
            except Exception as e:
                self.logger.warning(f"模拟{symbol}分钟数据失败: {e}")
                continue
        
        if minute_data_list:
            minute_df = pd.DataFrame(minute_data_list)
            return minute_df
        else:
            return pd.DataFrame()
    
    def get_index_data(self, index_code: str, start_date: str, end_date: str,
                       frequency: str) -> pd.DataFrame:
        """获取指数数据"""
        try:
            # 转换指数代码格式
            index_symbol = self._convert_index_format(index_code)
            
            if frequency == "1min":
                data = self.ak.index_zh_a_hist_min_em(
                    symbol=index_symbol,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    period="1"
                )
            elif frequency == "1d":
                data = self.ak.index_zh_a_hist(
                    symbol=index_symbol,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", "")
                )
            else:
                raise ValueError(f"不支持的频率: {frequency}")
            
            if data is not None and not data.empty:
                data['symbol'] = index_code
                data['datetime'] = pd.to_datetime(data['时间'])
                
                # 重命名列
                data.rename(columns={
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high', 
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount'
                }, inplace=True)
                
                # 选择需要的列
                data = data[['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount']]
                data.set_index('datetime', inplace=True)
                return data
            
        except Exception as e:
            self.logger.error(f"获取指数{index_code}数据失败: {e}")
            
        return pd.DataFrame()
    
    def get_fundamental_data(self, symbols: List[str], start_date: str,
                            end_date: str) -> pd.DataFrame:
        """获取基本面数据"""
        all_data = []
        
        for symbol in symbols:
            try:
                symbol_code = self._convert_symbol_format(symbol)
                
                # 获取财务数据
                financial_data = self.ak.stock_financial_analysis_indicator(symbol=symbol_code)
                
                if financial_data is not None and not financial_data.empty:
                    financial_data['symbol'] = symbol
                    all_data.append(financial_data)
                    
                    self.logger.info(f"获取{symbol}基本面数据成功")
                
            except Exception as e:
                self.logger.error(f"获取{symbol}基本面数据失败: {e}")
                continue
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_stock_basic_info(self) -> pd.DataFrame:
        """获取股票基本信息"""
        try:
            data = self.ak.stock_info_a_code_name()
            return data
        except Exception as e:
            self.logger.error(f"获取股票基本信息失败: {e}")
            return pd.DataFrame()
    
    def _convert_symbol_format(self, symbol: str) -> str:
        """转换股票代码格式"""
        # 从 "000001.SZ" 转换为 "000001"
        if "." in symbol:
            return symbol.split(".")[0]
        return symbol
    
    def _convert_index_format(self, index_code: str) -> str:
        """转换指数代码格式"""
        index_map = {
            "000300.SH": "000300",  # 沪深300
            "000905.SH": "000905",  # 中证500
            "000001.SH": "000001",  # 上证指数
            "399001.SZ": "399001",  # 深证成指
            "399006.SZ": "399006"   # 创业板指
        }
        return index_map.get(index_code, index_code)


class TushareDataSource(DataSource):
    """Tushare数据源"""
    
    def __init__(self, token: str):
        """初始化Tushare数据源"""
        try:
            import tushare as ts
            self.ts = ts
            ts.set_token(token)
            self.pro = ts.pro_api()
            self.logger = Logger().get_logger("tushare_data")
        except ImportError:
            raise ImportError("请安装tushare包: pip install tushare")
    
    def get_stock_data(self, symbols: List[str], start_date: str, end_date: str,
                       frequency: str) -> pd.DataFrame:
        """获取股票数据"""
        all_data = []
        
        for symbol in symbols:
            try:
                # 转换股票代码格式
                ts_symbol = self._convert_symbol_format(symbol)
                
                if frequency == "1min":
                    # Tushare分钟级数据需要特殊处理
                    data = self.pro.stk_mins(ts_code=ts_symbol, start_date=start_date, end_date=end_date)
                elif frequency == "1d":
                    data = self.pro.daily(ts_code=ts_symbol, start_date=start_date, end_date=end_date)
                else:
                    raise ValueError(f"不支持的频率: {frequency}")
                
                if data is not None and not data.empty:
                    data['symbol'] = symbol
                    data['datetime'] = pd.to_datetime(data['trade_date'])
                    
                    # 重命名列
                    data.rename(columns={
                        'open': 'open',
                        'close': 'close',
                        'high': 'high',
                        'low': 'low',
                        'vol': 'volume',
                        'amount': 'amount'
                    }, inplace=True)
                    
                    # 选择需要的列
                    data = data[['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount']]
                    all_data.append(data)
                    
                    self.logger.info(f"获取{symbol}数据成功，共{len(data)}条")
                
            except Exception as e:
                self.logger.error(f"获取{symbol}数据失败: {e}")
                continue
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result.set_index('datetime', inplace=True)
            return result
        else:
            return pd.DataFrame()
    
    def get_index_data(self, index_code: str, start_date: str, end_date: str,
                       frequency: str) -> pd.DataFrame:
        """获取指数数据"""
        try:
            ts_code = self._convert_index_format(index_code)
            
            if frequency == "1d":
                data = self.pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            else:
                raise ValueError(f"不支持的频率: {frequency}")
            
            if data is not None and not data.empty:
                data['symbol'] = index_code
                data['datetime'] = pd.to_datetime(data['trade_date'])
                
                # 重命名列
                data.rename(columns={
                    'open': 'open',
                    'close': 'close',
                    'high': 'high',
                    'low': 'low',
                    'vol': 'volume',
                    'amount': 'amount'
                }, inplace=True)
                
                # 选择需要的列
                data = data[['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount']]
                data.set_index('datetime', inplace=True)
                return data
            
        except Exception as e:
            self.logger.error(f"获取指数{index_code}数据失败: {e}")
            
        return pd.DataFrame()
    
    def get_fundamental_data(self, symbols: List[str], start_date: str,
                            end_date: str) -> pd.DataFrame:
        """获取基本面数据"""
        all_data = []
        
        for symbol in symbols:
            try:
                ts_symbol = self._convert_symbol_format(symbol)
                
                # 获取财务数据
                financial_data = self.pro.fina_indicator(ts_code=ts_symbol, start_date=start_date, end_date=end_date)
                
                if financial_data is not None and not financial_data.empty:
                    financial_data['symbol'] = symbol
                    all_data.append(financial_data)
                    
                    self.logger.info(f"获取{symbol}基本面数据成功")
                
            except Exception as e:
                self.logger.error(f"获取{symbol}基本面数据失败: {e}")
                continue
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _convert_symbol_format(self, symbol: str) -> str:
        """转换股票代码格式"""
        # 从 "000001.SZ" 转换为 "000001.SZ"（Tushare格式）
        return symbol
    
    def _convert_index_format(self, index_code: str) -> str:
        """转换指数代码格式"""
        return index_code


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        """初始化预处理器"""
        self.logger = Logger().get_logger("data_preprocessor")
    
    def clean_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """清洗价格数据"""
        try:
            # 复制数据
            clean_data = data.copy()
            
            # 检查价格数据的有效性
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in clean_data.columns:
                    # 去除负值和零值
                    clean_data = clean_data[clean_data[col] > 0]
                    
                    # 去除异常的价格跳跃（超过20%的单日涨跌幅）
                    if len(clean_data) > 1:
                        pct_change = clean_data[col].pct_change().abs()
                        clean_data = clean_data[pct_change <= 0.2]
            
            # 检查高开低收的逻辑关系
            if all(col in clean_data.columns for col in price_columns):
                # high应该是最高价
                clean_data = clean_data[
                    (clean_data['high'] >= clean_data['open']) &
                    (clean_data['high'] >= clean_data['close']) &
                    (clean_data['high'] >= clean_data['low'])
                ]
                
                # low应该是最低价
                clean_data = clean_data[
                    (clean_data['low'] <= clean_data['open']) &
                    (clean_data['low'] <= clean_data['close']) &
                    (clean_data['low'] <= clean_data['high'])
                ]
            
            # 去除成交量为0的数据
            if 'volume' in clean_data.columns:
                clean_data = clean_data[clean_data['volume'] > 0]
            
            self.logger.info(f"数据清洗完成，从{len(data)}条记录清洗到{len(clean_data)}条记录")
            return clean_data
            
        except Exception as e:
            self.logger.error(f"清洗价格数据失败: {e}")
            return data
    
    def handle_missing_data(self, data: pd.DataFrame, method: str = "forward_fill") -> pd.DataFrame:
        """处理缺失数据"""
        try:
            clean_data = data.copy()
            
            if method == "forward_fill":
                # 前向填充
                clean_data = clean_data.ffill()
            elif method == "backward_fill":
                # 后向填充
                clean_data = clean_data.bfill()
            elif method == "interpolate":
                # 线性插值
                clean_data = clean_data.interpolate()
            elif method == "drop":
                # 删除缺失值
                clean_data = clean_data.dropna()
            else:
                raise ValueError(f"不支持的缺失值处理方法: {method}")
            
            missing_before = data.isnull().sum().sum()
            missing_after = clean_data.isnull().sum().sum()
            
            self.logger.info(f"缺失值处理完成，缺失值从{missing_before}个减少到{missing_after}个")
            return clean_data
            
        except Exception as e:
            self.logger.error(f"处理缺失数据失败: {e}")
            return data
    
    def remove_outliers(self, data: pd.DataFrame, method: str = "iqr", 
                       threshold: float = 3.0) -> pd.DataFrame:
        """去除异常值"""
        try:
            clean_data = data.copy()
            
            # 只对数值列进行异常值检测
            numeric_columns = clean_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if method == "iqr":
                    # 四分位数方法
                    Q1 = clean_data[col].quantile(0.25)
                    Q3 = clean_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    clean_data = clean_data[
                        (clean_data[col] >= lower_bound) & 
                        (clean_data[col] <= upper_bound)
                    ]
                    
                elif method == "zscore":
                    # Z分数方法
                    z_scores = np.abs((clean_data[col] - clean_data[col].mean()) / clean_data[col].std())
                    clean_data = clean_data[z_scores < threshold]
                    
                elif method == "percentile":
                    # 百分位数方法
                    lower_bound = clean_data[col].quantile(0.01)
                    upper_bound = clean_data[col].quantile(0.99)
                    
                    clean_data = clean_data[
                        (clean_data[col] >= lower_bound) & 
                        (clean_data[col] <= upper_bound)
                    ]
            
            self.logger.info(f"异常值处理完成，从{len(data)}条记录处理到{len(clean_data)}条记录")
            return clean_data
            
        except Exception as e:
            self.logger.error(f"去除异常值失败: {e}")
            return data
    
    def adjust_for_splits_dividends(self, data: pd.DataFrame) -> pd.DataFrame:
        """复权处理"""
        try:
            # 如果数据源已经提供了复权数据，这里可以进行额外的验证
            # 一般情况下，从数据源获取数据时已经指定了复权参数
            
            clean_data = data.copy()
            
            # 检查价格数据的连续性
            if 'close' in clean_data.columns and len(clean_data) > 1:
                # 检查是否有异常的价格跳跃
                pct_change = clean_data['close'].pct_change()
                large_changes = pct_change.abs() > 0.5  # 50%以上的跳跃
                
                if large_changes.any():
                    self.logger.warning("发现可能的未复权数据，建议检查数据源的复权设置")
            
            return clean_data
            
        except Exception as e:
            self.logger.error(f"复权处理失败: {e}")
            return data
    
    def align_timestamps(self, data: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """时间戳对齐"""
        try:
            clean_data = data.copy()
            
            if not isinstance(clean_data.index, pd.DatetimeIndex):
                # 如果索引不是时间索引，尝试转换
                if 'datetime' in clean_data.columns:
                    clean_data.set_index('datetime', inplace=True)
                else:
                    raise ValueError("数据中没有时间列")
            
            # 根据频率重新采样
            if frequency == "1min":
                clean_data = clean_data.resample('1min').last()
            elif frequency == "5min":
                clean_data = clean_data.resample('5min').last()
            elif frequency == "15min":
                clean_data = clean_data.resample('15min').last()
            elif frequency == "30min":
                clean_data = clean_data.resample('30min').last()
            elif frequency == "1h":
                clean_data = clean_data.resample('1h').last()
            elif frequency == "1d":
                clean_data = clean_data.resample('1d').last()
            
            # 去除空值
            clean_data = clean_data.dropna()
            
            self.logger.info(f"时间戳对齐完成，频率: {frequency}")
            return clean_data
            
        except Exception as e:
            self.logger.error(f"时间戳对齐失败: {e}")
            return data
    
    def calculate_returns(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """计算收益率"""
        try:
            result = data.copy()
            
            if 'close' not in result.columns:
                raise ValueError("数据中没有收盘价列")
            
            # 按symbol分组计算收益率
            if 'symbol' in result.columns:
                for period in periods:
                    result[f'return_{period}'] = result.groupby('symbol')['close'].pct_change(period)
            else:
                # 单个资产的情况
                for period in periods:
                    result[f'return_{period}'] = result['close'].pct_change(period)
            
            self.logger.info(f"计算收益率完成，计算周期: {periods}")
            return result
            
        except Exception as e:
            self.logger.error(f"计算收益率失败: {e}")
            return data


class DataCache:
    """数据缓存管理"""
    
    def __init__(self, cache_dir: str = "./cache"):
        """初始化缓存"""
        self.cache_dir = cache_dir
        self.logger = Logger().get_logger("data_cache")
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cached_data(self, key: str) -> Optional[pd.DataFrame]:
        """获取缓存数据"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{self._hash_key(key)}.pkl")
            
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                self.logger.info(f"从缓存中获取数据: {key}")
                return cached_data['data']
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取缓存数据失败: {e}")
            return None
    
    def cache_data(self, key: str, data: pd.DataFrame, expire_hours: int = 24):
        """缓存数据"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{self._hash_key(key)}.pkl")
            
            cached_data = {
                'data': data,
                'timestamp': datetime.now(),
                'expire_hours': expire_hours
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            
            self.logger.info(f"数据已缓存: {key}")
            
        except Exception as e:
            self.logger.error(f"缓存数据失败: {e}")
    
    def clear_cache(self, pattern: str = None):
        """清理缓存"""
        try:
            if pattern is None:
                # 清理所有缓存
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.pkl'):
                        os.remove(os.path.join(self.cache_dir, file))
                self.logger.info("清理所有缓存完成")
            else:
                # 清理匹配模式的缓存
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.pkl') and pattern in file:
                        os.remove(os.path.join(self.cache_dir, file))
                self.logger.info(f"清理匹配缓存完成: {pattern}")
                
        except Exception as e:
            self.logger.error(f"清理缓存失败: {e}")
    
    def is_cache_valid(self, key: str, expire_hours: int = 24) -> bool:
        """检查缓存是否有效"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{self._hash_key(key)}.pkl")
            
            if not os.path.exists(cache_file):
                return False
            
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # 检查是否过期
            cache_time = cached_data['timestamp']
            expire_time = cache_time + timedelta(hours=expire_hours)
            
            return datetime.now() < expire_time
            
        except Exception as e:
            self.logger.error(f"检查缓存有效性失败: {e}")
            return False
    
    def _hash_key(self, key: str) -> str:
        """生成缓存键的哈希值"""
        return hashlib.md5(key.encode()).hexdigest()


class UniverseManager:
    """股票池管理"""
    
    def __init__(self):
        """初始化股票池管理器"""
        self.logger = Logger().get_logger("universe_manager")
        self.data_source = None
    
    def set_data_source(self, data_source: DataSource):
        """设置数据源"""
        self.data_source = data_source
    
    def get_hs300_constituents(self, date: str = None) -> List[str]:
        """获取沪深300成分股"""
        try:
            if isinstance(self.data_source, AkshareDataSource):
                # 使用akshare获取沪深300成分股
                data = self.data_source.ak.index_stock_cons(symbol="000300")
                if data is not None and not data.empty:
                    # 转换为标准格式
                    symbols = []
                    for code in data['品种代码']:
                        if code.startswith('6'):
                            symbols.append(f"{code}.SH")
                        else:
                            symbols.append(f"{code}.SZ")
                    return symbols
            
            # 默认返回部分成分股
            return [
                "000001.SZ", "000002.SZ", "000858.SZ", "000895.SZ", "000938.SZ",
                "600000.SH", "600036.SH", "600519.SH", "600887.SH", "600900.SH"
            ]
            
        except Exception as e:
            self.logger.error(f"获取沪深300成分股失败: {e}")
            return []
    
    def get_zz500_constituents(self, date: str = None) -> List[str]:
        """获取中证500成分股"""
        try:
            if isinstance(self.data_source, AkshareDataSource):
                # 使用akshare获取中证500成分股
                data = self.data_source.ak.index_stock_cons(symbol="000905")
                if data is not None and not data.empty:
                    # 转换为标准格式
                    symbols = []
                    for code in data['品种代码']:
                        if code.startswith('6'):
                            symbols.append(f"{code}.SH")
                        else:
                            symbols.append(f"{code}.SZ")
                    return symbols
            
            # 默认返回部分成分股
            return [
                "000005.SZ", "000006.SZ", "000007.SZ", "000008.SZ", "000009.SZ",
                "600001.SH", "600002.SH", "600003.SH", "600004.SH", "600005.SH"
            ]
            
        except Exception as e:
            self.logger.error(f"获取中证500成分股失败: {e}")
            return []
    
    def get_custom_universe(self, criteria: Dict) -> List[str]:
        """获取自定义股票池"""
        try:
            # 这里可以根据各种条件筛选股票
            # 例如：市值、流动性、行业等
            
            # 示例：返回一个固定的股票池
            return [
                "000001.SZ", "000002.SZ", "000858.SZ", "000895.SZ", "000938.SZ",
                "600000.SH", "600036.SH", "600519.SH", "600887.SH", "600900.SH",
                "000001.SH", "399001.SZ", "399006.SZ"
            ]
            
        except Exception as e:
            self.logger.error(f"获取自定义股票池失败: {e}")
            return []
    
    def filter_by_liquidity(self, symbols: List[str], min_volume: float) -> List[str]:
        """按流动性筛选"""
        try:
            # 这里应该根据实际的成交量数据进行筛选
            # 简化处理，返回原始列表
            self.logger.info(f"按流动性筛选股票，最小成交量: {min_volume}")
            return symbols
            
        except Exception as e:
            self.logger.error(f"按流动性筛选失败: {e}")
            return symbols
    
    def filter_by_market_cap(self, symbols: List[str], min_cap: float, 
                            max_cap: float = None) -> List[str]:
        """按市值筛选"""
        try:
            # 这里应该根据实际的市值数据进行筛选
            # 简化处理，返回原始列表
            self.logger.info(f"按市值筛选股票，最小市值: {min_cap}, 最大市值: {max_cap}")
            return symbols
            
        except Exception as e:
            self.logger.error(f"按市值筛选失败: {e}")
            return symbols


class DataManager:
    """数据管理主类"""
    
    def __init__(self, config=None, cache_enabled: bool = True):
        """初始化数据管理器"""
        self.logger = Logger().get_logger("data_manager")
        
        # 处理配置参数
        if config is None:
            self.data_source_name = "akshare"
            self.config = {}
        elif isinstance(config, str):
            self.data_source_name = config
            self.config = {}
        else:
            # 处理DataConfig对象
            if hasattr(config, 'data_source'):
                self.data_source_name = config.data_source
            else:
                self.data_source_name = "akshare"
            
            if hasattr(config, '__dict__'):
                self.config = config.__dict__.copy()
            else:
                self.config = {}
        
        self.cache_enabled = cache_enabled
        
        # 初始化组件
        self.preprocessor = DataPreprocessor()
        self.cache = DataCache() if cache_enabled else None
        self.universe_manager = UniverseManager()
        
        # 直接初始化数据源
        try:
            if self.data_source_name == "akshare":
                self.data_source = AkshareDataSource()
            elif self.data_source_name == "tushare":
                # 对于tushare，如果没有token，先设为None，稍后在initialize中设置
                self.data_source = None
            else:
                self.logger.warning(f"不支持的数据源: {self.data_source_name}，使用akshare作为默认数据源")
                self.data_source = AkshareDataSource()
                self.data_source_name = "akshare"
            
            # 设置股票池管理器的数据源
            if self.data_source:
                self.universe_manager.set_data_source(self.data_source)
            
            self.logger.info(f"数据管理器初始化完成，数据源: {self.data_source_name}")
            
        except Exception as e:
            self.logger.error(f"数据管理器初始化失败: {e}")
            self.data_source = None
    
    def initialize(self, config: Dict):
        """初始化配置"""
        try:
            self.config = config
            
            # 初始化数据源
            if self.data_source_name == "akshare":
                self.data_source = AkshareDataSource()
            elif self.data_source_name == "tushare":
                token = config.get("tushare_token")
                if not token:
                    raise ValueError("使用Tushare数据源需要提供token")
                self.data_source = TushareDataSource(token)
            else:
                raise ValueError(f"不支持的数据源: {self.data_source_name}")
            
            # 设置股票池管理器的数据源
            self.universe_manager.set_data_source(self.data_source)
            
            self.logger.info(f"数据管理器初始化完成，数据源: {self.data_source_name}")
            
        except Exception as e:
            self.logger.error(f"数据管理器初始化失败: {e}")
            raise
    
    def get_stock_data(self, symbols: List[str], start_date: str, end_date: str,
                       frequency: str = "1min", fields: List[str] = None) -> pd.DataFrame:
        """获取股票数据"""
        try:
            # 生成缓存键
            cache_key = f"stock_data_{'-'.join(symbols)}_{start_date}_{end_date}_{frequency}"
            
            # 检查缓存
            if self.cache and self.cache.is_cache_valid(cache_key):
                data = self.cache.get_cached_data(cache_key)
                if data is not None:
                    self.logger.info(f"从缓存获取股票数据: {len(symbols)}只股票")
                    return data
            
            # 从数据源获取数据
            self.logger.info(f"从数据源获取股票数据: {len(symbols)}只股票")
            data = self.data_source.get_stock_data(symbols, start_date, end_date, frequency)
            
            if data is not None and not data.empty:
                # 预处理数据
                data = self.preprocess_data(data, self.config)
                
                # 缓存数据
                if self.cache:
                    self.cache.cache_data(cache_key, data)
                
                self.logger.info(f"获取股票数据完成: {len(data)}条记录")
                return data
            else:
                self.logger.warning("获取到的股票数据为空")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"获取股票数据失败: {e}")
            return pd.DataFrame()
    
    def get_index_data(self, index_code: str, start_date: str, end_date: str,
                       frequency: str = "1min") -> pd.DataFrame:
        """获取指数数据"""
        try:
            # 生成缓存键
            cache_key = f"index_data_{index_code}_{start_date}_{end_date}_{frequency}"
            
            # 检查缓存
            if self.cache and self.cache.is_cache_valid(cache_key):
                data = self.cache.get_cached_data(cache_key)
                if data is not None:
                    self.logger.info(f"从缓存获取指数数据: {index_code}")
                    return data
            
            # 从数据源获取数据
            self.logger.info(f"从数据源获取指数数据: {index_code}")
            data = self.data_source.get_index_data(index_code, start_date, end_date, frequency)
            
            if data is not None and not data.empty:
                # 预处理数据
                data = self.preprocess_data(data, self.config)
                
                # 缓存数据
                if self.cache:
                    self.cache.cache_data(cache_key, data)
                
                self.logger.info(f"获取指数数据完成: {len(data)}条记录")
                return data
            else:
                self.logger.warning("获取到的指数数据为空")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"获取指数数据失败: {e}")
            return pd.DataFrame()
    
    def get_fundamental_data(self, symbols: List[str], start_date: str,
                            end_date: str) -> pd.DataFrame:
        """获取基本面数据"""
        try:
            # 生成缓存键
            cache_key = f"fundamental_data_{'-'.join(symbols)}_{start_date}_{end_date}"
            
            # 检查缓存
            if self.cache and self.cache.is_cache_valid(cache_key, expire_hours=168):  # 7天过期
                data = self.cache.get_cached_data(cache_key)
                if data is not None:
                    self.logger.info(f"从缓存获取基本面数据: {len(symbols)}只股票")
                    return data
            
            # 从数据源获取数据
            self.logger.info(f"从数据源获取基本面数据: {len(symbols)}只股票")
            data = self.data_source.get_fundamental_data(symbols, start_date, end_date)
            
            if data is not None and not data.empty:
                # 缓存数据
                if self.cache:
                    self.cache.cache_data(cache_key, data, expire_hours=168)
                
                self.logger.info(f"获取基本面数据完成: {len(data)}条记录")
                return data
            else:
                self.logger.warning("获取到的基本面数据为空")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"获取基本面数据失败: {e}")
            return pd.DataFrame()
    
    def get_universe(self, universe_name: str, date: str = None) -> List[str]:
        """获取股票池"""
        try:
            if universe_name.upper() == "HS300":
                return self.universe_manager.get_hs300_constituents(date)
            elif universe_name.upper() == "ZZ500":
                return self.universe_manager.get_zz500_constituents(date)
            else:
                # 自定义股票池
                criteria = self.config.get("universe_criteria", {})
                return self.universe_manager.get_custom_universe(criteria)
                
        except Exception as e:
            self.logger.error(f"获取股票池失败: {e}")
            return []
    
    def get_stock_list(self) -> List[str]:
        """获取股票列表"""
        try:
            # 从配置中获取股票列表
            if 'test' in self.config and 'symbols' in self.config['test']:
                symbols = self.config['test']['symbols']
                return symbols
            
            # 如果有AkShare数据源，尝试获取股票基本信息
            if isinstance(self.data_source, AkshareDataSource):
                try:
                    stock_info = self.data_source.get_stock_basic_info()
                    if not stock_info.empty and '代码' in stock_info.columns:
                        symbols = stock_info['代码'].tolist()[:100]  # 返回前100只股票
                        # 转换为标准格式
                        formatted_symbols = []
                        for code in symbols:
                            if code.startswith('6'):
                                formatted_symbols.append(f"{code}.SH")
                            else:
                                formatted_symbols.append(f"{code}.SZ")
                        return formatted_symbols
                except Exception as e:
                    self.logger.warning(f"从AkShare获取股票列表失败: {e}")
            
            # 默认返回一些测试股票
            return ['000001', '000002', '600000', '600036', '000858']
            
        except Exception as e:
            self.logger.error(f"获取股票列表失败: {e}")
            return ['000001', '000002', '600000', '600036', '000858']
    
    def preprocess_data(self, data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """预处理数据"""
        try:
            processed_data = data.copy()
            
            # 清洗价格数据
            processed_data = self.preprocessor.clean_price_data(processed_data)
            
            # 处理缺失数据
            missing_method = config.get("missing_method", "forward_fill")
            processed_data = self.preprocessor.handle_missing_data(processed_data, missing_method)
            
            # 去除异常值
            outlier_method = config.get("outlier_method", "iqr")
            processed_data = self.preprocessor.remove_outliers(processed_data, outlier_method)
            
            # 计算收益率
            return_periods = config.get("return_periods", [1, 5, 10, 20])
            processed_data = self.preprocessor.calculate_returns(processed_data, return_periods)
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"预处理数据失败: {e}")
            return data
    
    def get_multi_frequency_data(self, symbols: List[str], start_date: str, 
                               end_date: str, frequencies: List[str]) -> Dict[str, pd.DataFrame]:
        """获取多频率数据"""
        try:
            result = {}
            
            for frequency in frequencies:
                data = self.get_stock_data(symbols, start_date, end_date, frequency)
                result[frequency] = data
                
            self.logger.info(f"获取多频率数据完成: {frequencies}")
            return result
            
        except Exception as e:
            self.logger.error(f"获取多频率数据失败: {e}")
            return {}
    
    def update_data(self, end_date: str = None):
        """更新数据"""
        try:
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            
            # 清理过期缓存
            if self.cache:
                self.cache.clear_cache()
            
            self.logger.info(f"数据更新完成，更新到: {end_date}")
            
        except Exception as e:
            self.logger.error(f"更新数据失败: {e}")
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, bool]:
        """验证数据质量"""
        try:
            quality_report = {}
            
            # 检查数据是否为空
            quality_report['not_empty'] = not data.empty
            
            # 检查是否有缺失值
            quality_report['no_missing'] = not data.isnull().any().any()
            
            # 检查价格数据的有效性
            if 'close' in data.columns:
                quality_report['positive_prices'] = (data['close'] > 0).all()
            
            # 检查数据的时间连续性
            if isinstance(data.index, pd.DatetimeIndex):
                quality_report['continuous_time'] = True  # 简化处理
            else:
                quality_report['continuous_time'] = False
            
            self.logger.info(f"数据质量验证完成: {quality_report}")
            return quality_report
            
        except Exception as e:
            self.logger.error(f"验证数据质量失败: {e}")
            return {'error': True} 