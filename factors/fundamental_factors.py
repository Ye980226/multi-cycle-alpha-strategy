#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本面因子模块
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional

# 导入日志模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import Logger


class FundamentalFactors:
    """基本面因子计算类"""
    
    def __init__(self, logger: Logger = None):
        """初始化基本面因子计算器"""
        if logger is not None:
            self.logger = logger.get_logger("fundamental_factors")
        else:
            self.logger = Logger().get_logger("fundamental_factors")
    
    def calculate_all_factors(self, price_data: pd.DataFrame, 
                                        fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """计算所有基本面因子"""
        try:
            result = price_data.copy()
            
            if fundamental_data is None or fundamental_data.empty:
                self.logger.warning("基本面数据为空，使用模拟数据")
                fundamental_data = self._create_mock_fundamental_data(price_data)
            
            # 估值因子
            valuation_factors = self.calculate_valuation_factors(price_data, fundamental_data)
            result = pd.concat([result, valuation_factors], axis=1)
            
            # 盈利能力因子
            profitability_factors = self.calculate_profitability_factors(fundamental_data)
            result = pd.concat([result, profitability_factors], axis=1)
            
            # 成长因子
            growth_factors = self.calculate_growth_factors(fundamental_data)
            result = pd.concat([result, growth_factors], axis=1)
            
            # 质量因子
            quality_factors = self.calculate_quality_factors(fundamental_data)
            result = pd.concat([result, quality_factors], axis=1)
            
            self.logger.info(f"计算基本面因子完成，新增{len([col for col in result.columns if col not in price_data.columns])}个因子")
            return result
            
        except Exception as e:
            self.logger.error(f"计算基本面因子失败: {e}")
            return price_data
    
    def _create_mock_fundamental_data(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """创建模拟基本面数据"""
        try:
            symbols = price_data['symbol'].unique() if 'symbol' in price_data.columns else ['000001.SZ']
            
            mock_data = []
            for symbol in symbols:
                # 创建季度数据
                quarters = pd.date_range(start='2020-01-01', end='2024-01-01', freq='Q')
                for quarter in quarters:
                    mock_data.append({
                        'symbol': symbol,
                        'report_date': quarter,
                        'total_revenue': np.random.uniform(1000000, 10000000),  # 营业收入
                        'net_income': np.random.uniform(100000, 1000000),       # 净利润
                        'total_assets': np.random.uniform(5000000, 50000000),   # 总资产
                        'total_equity': np.random.uniform(2000000, 20000000),   # 股东权益
                        'total_debt': np.random.uniform(1000000, 15000000),     # 总负债
                        'operating_cash_flow': np.random.uniform(50000, 1500000), # 经营现金流
                        'book_value_per_share': np.random.uniform(5, 50),       # 每股净资产
                        'earnings_per_share': np.random.uniform(0.1, 5),        # 每股收益
                        'market_cap': np.random.uniform(10000000, 100000000)    # 市值
                    })
            
            return pd.DataFrame(mock_data)
            
        except Exception as e:
            self.logger.error(f"创建模拟基本面数据失败: {e}")
            return pd.DataFrame()
    
    def calculate_valuation_factors(self, price_data: pd.DataFrame, 
                                  fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """计算估值因子"""
        try:
            result = pd.DataFrame(index=price_data.index)
            
            # 获取最新基本面数据
            latest_fundamental = fundamental_data.groupby('symbol').last()
            
            symbols = price_data['symbol'].unique() if 'symbol' in price_data.columns else ['default']
            for symbol in symbols:
                if symbol in latest_fundamental.index:
                    symbol_data = latest_fundamental.loc[symbol]
                    symbol_mask = price_data['symbol'] == symbol if 'symbol' in price_data.columns else slice(None)
                    
                    # PE ratio (市盈率)
                    if 'earnings_per_share' in symbol_data:
                        pe_ratio = price_data.loc[symbol_mask, 'close'] / symbol_data['earnings_per_share']
                        result.loc[symbol_mask, 'pe_ratio'] = pe_ratio
                    
                    # PB ratio (市净率)
                    if 'book_value_per_share' in symbol_data:
                        pb_ratio = price_data.loc[symbol_mask, 'close'] / symbol_data['book_value_per_share']
                        result.loc[symbol_mask, 'pb_ratio'] = pb_ratio
                    
                    # PS ratio (市销率)
                    if 'total_revenue' in symbol_data and 'market_cap' in symbol_data:
                        ps_ratio = symbol_data['market_cap'] / symbol_data['total_revenue']
                        result.loc[symbol_mask, 'ps_ratio'] = ps_ratio
                    
                    # PCF ratio (市现率)
                    if 'operating_cash_flow' in symbol_data and 'market_cap' in symbol_data:
                        pcf_ratio = symbol_data['market_cap'] / symbol_data['operating_cash_flow']
                        result.loc[symbol_mask, 'pcf_ratio'] = pcf_ratio
            
            return result
            
        except Exception as e:
            self.logger.error(f"计算估值因子失败: {e}")
            return pd.DataFrame(index=price_data.index)
    
    # 估值因子
    def calculate_pe_ratio(self, price: pd.Series, earnings: pd.Series) -> pd.Series:
        """市盈率"""
        try:
            pe_ratio = price / earnings
            return pe_ratio.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            self.logger.error(f"计算PE失败: {e}")
            return pd.Series(index=price.index, dtype=float)
    
    def calculate_pb_ratio(self, price: pd.Series, book_value: pd.Series) -> pd.Series:
        """市净率"""
        try:
            pb_ratio = price / book_value
            return pb_ratio.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            self.logger.error(f"计算PB失败: {e}")
            return pd.Series(index=price.index, dtype=float)
    
    def calculate_ps_ratio(self, market_cap: pd.Series, sales: pd.Series) -> pd.Series:
        """市销率"""
        try:
            ps_ratio = market_cap / sales
            return ps_ratio.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            self.logger.error(f"计算PS失败: {e}")
            return pd.Series(index=market_cap.index, dtype=float)
    
    def calculate_pcf_ratio(self, market_cap: pd.Series, cash_flow: pd.Series) -> pd.Series:
        """市现率"""
        try:
            pcf_ratio = market_cap / cash_flow
            return pcf_ratio.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            self.logger.error(f"计算PCF失败: {e}")
            return pd.Series(index=market_cap.index, dtype=float)
    
    def calculate_ev_ebitda(self, enterprise_value: pd.Series, ebitda: pd.Series) -> pd.Series:
        """企业价值倍数"""
        try:
            ev_ebitda = enterprise_value / ebitda
            return ev_ebitda.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            self.logger.error(f"计算EV/EBITDA失败: {e}")
            return pd.Series(index=enterprise_value.index, dtype=float)
    
    # 盈利能力因子
    def calculate_profitability_factors(self, fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """计算盈利能力因子"""
        try:
            result = pd.DataFrame()
            
            if fundamental_data.empty:
                return result
            
            # ROE (净资产收益率)
            if 'net_income' in fundamental_data.columns and 'total_equity' in fundamental_data.columns:
                result['roe'] = fundamental_data['net_income'] / fundamental_data['total_equity']
            
            # ROA (总资产收益率)
            if 'net_income' in fundamental_data.columns and 'total_assets' in fundamental_data.columns:
                result['roa'] = fundamental_data['net_income'] / fundamental_data['total_assets']
            
            # 毛利率 (假设有gross_profit数据)
            if 'gross_profit' in fundamental_data.columns and 'total_revenue' in fundamental_data.columns:
                result['gross_margin'] = fundamental_data['gross_profit'] / fundamental_data['total_revenue']
            
            # 营业利润率 (假设有operating_income数据)
            if 'operating_income' in fundamental_data.columns and 'total_revenue' in fundamental_data.columns:
                result['operating_margin'] = fundamental_data['operating_income'] / fundamental_data['total_revenue']
            
            # 净利润率
            if 'net_income' in fundamental_data.columns and 'total_revenue' in fundamental_data.columns:
                result['net_margin'] = fundamental_data['net_income'] / fundamental_data['total_revenue']
            
            return result
            
        except Exception as e:
            self.logger.error(f"计算盈利能力因子失败: {e}")
            return pd.DataFrame()
    
    def calculate_roe(self, net_income: pd.Series, equity: pd.Series) -> pd.Series:
        """净资产收益率"""
        try:
            roe = net_income / equity
            return roe.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            self.logger.error(f"计算ROE失败: {e}")
            return pd.Series(index=net_income.index, dtype=float)
    
    def calculate_roa(self, net_income: pd.Series, assets: pd.Series) -> pd.Series:
        """总资产收益率"""
        try:
            roa = net_income / assets
            return roa.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            self.logger.error(f"计算ROA失败: {e}")
            return pd.Series(index=net_income.index, dtype=float)
    
    def calculate_roic(self, operating_income: pd.Series, invested_capital: pd.Series) -> pd.Series:
        """投入资本回报率"""
        try:
            roic = operating_income / invested_capital
            return roic.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            self.logger.error(f"计算ROIC失败: {e}")
            return pd.Series(index=operating_income.index, dtype=float)
    
    def calculate_gross_margin(self, gross_profit: pd.Series, revenue: pd.Series) -> pd.Series:
        """毛利率"""
        try:
            gross_margin = gross_profit / revenue
            return gross_margin.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            self.logger.error(f"计算毛利率失败: {e}")
            return pd.Series(index=gross_profit.index, dtype=float)
    
    def calculate_operating_margin(self, operating_income: pd.Series, revenue: pd.Series) -> pd.Series:
        """营业利润率"""
        try:
            operating_margin = operating_income / revenue
            return operating_margin.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            self.logger.error(f"计算营业利润率失败: {e}")
            return pd.Series(index=operating_income.index, dtype=float)
    
    # 成长因子
    def calculate_growth_factors(self, fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """计算成长因子"""
        try:
            result = pd.DataFrame()
            
            if fundamental_data.empty or 'symbol' not in fundamental_data.columns:
                return result
            
            # 按股票分组计算成长率
            for symbol in fundamental_data['symbol'].unique():
                symbol_data = fundamental_data[fundamental_data['symbol'] == symbol].copy()
                symbol_data = symbol_data.sort_values('report_date')
                
                # 营收增长率
                if 'total_revenue' in symbol_data.columns:
                    revenue_growth = symbol_data['total_revenue'].pct_change(periods=4)  # 年同比
                    result = pd.concat([result, pd.DataFrame({
                        'symbol': symbol,
                        'revenue_growth_yoy': revenue_growth
                    })], ignore_index=True)
                
                # 净利润增长率
                if 'net_income' in symbol_data.columns:
                    earnings_growth = symbol_data['net_income'].pct_change(periods=4)
                    result.loc[result['symbol'] == symbol, 'earnings_growth_yoy'] = earnings_growth.values
                
                # 总资产增长率
                if 'total_assets' in symbol_data.columns:
                    asset_growth = symbol_data['total_assets'].pct_change(periods=4)
                    result.loc[result['symbol'] == symbol, 'asset_growth_yoy'] = asset_growth.values
            
            return result
            
        except Exception as e:
            self.logger.error(f"计算成长因子失败: {e}")
            return pd.DataFrame()
    
    def calculate_revenue_growth(self, revenue: pd.Series, periods: int = 4) -> pd.Series:
        """营收增长率"""
        try:
            growth = revenue.pct_change(periods=periods)
            return growth.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            self.logger.error(f"计算营收增长率失败: {e}")
            return pd.Series(index=revenue.index, dtype=float)
    
    def calculate_earnings_growth(self, earnings: pd.Series, periods: int = 4) -> pd.Series:
        """盈利增长率"""
        try:
            growth = earnings.pct_change(periods=periods)
            return growth.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            self.logger.error(f"计算盈利增长率失败: {e}")
            return pd.Series(index=earnings.index, dtype=float)
    
    def calculate_asset_growth(self, assets: pd.Series, periods: int = 4) -> pd.Series:
        """资产增长率"""
        try:
            growth = assets.pct_change(periods=periods)
            return growth.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            self.logger.error(f"计算资产增长率失败: {e}")
            return pd.Series(index=assets.index, dtype=float)
    
    # 质量因子
    def calculate_quality_factors(self, fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """计算质量因子"""
        try:
            result = pd.DataFrame()
            
            if fundamental_data.empty:
                return result
            
            # 资产负债率
            if 'total_debt' in fundamental_data.columns and 'total_assets' in fundamental_data.columns:
                result['debt_ratio'] = fundamental_data['total_debt'] / fundamental_data['total_assets']
            
            # 流动比率 (假设有current_assets和current_liabilities数据)
            if 'current_assets' in fundamental_data.columns and 'current_liabilities' in fundamental_data.columns:
                result['current_ratio'] = fundamental_data['current_assets'] / fundamental_data['current_liabilities']
            
            # 速动比率 (假设有quick_assets数据)
            if 'quick_assets' in fundamental_data.columns and 'current_liabilities' in fundamental_data.columns:
                result['quick_ratio'] = fundamental_data['quick_assets'] / fundamental_data['current_liabilities']
            
            # 资产周转率
            if 'total_revenue' in fundamental_data.columns and 'total_assets' in fundamental_data.columns:
                result['asset_turnover'] = fundamental_data['total_revenue'] / fundamental_data['total_assets']
            
            # 现金流量比率
            if 'operating_cash_flow' in fundamental_data.columns and 'current_liabilities' in fundamental_data.columns:
                result['cash_flow_ratio'] = fundamental_data['operating_cash_flow'] / fundamental_data['current_liabilities']
            
            return result
            
        except Exception as e:
            self.logger.error(f"计算质量因子失败: {e}")
            return pd.DataFrame()
    
    def calculate_debt_ratio(self, debt: pd.Series, assets: pd.Series) -> pd.Series:
        """资产负债率"""
        try:
            debt_ratio = debt / assets
            return debt_ratio.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            self.logger.error(f"计算资产负债率失败: {e}")
            return pd.Series(index=debt.index, dtype=float)
    
    def calculate_current_ratio(self, current_assets: pd.Series, current_liabilities: pd.Series) -> pd.Series:
        """流动比率"""
        try:
            current_ratio = current_assets / current_liabilities
            return current_ratio.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            self.logger.error(f"计算流动比率失败: {e}")
            return pd.Series(index=current_assets.index, dtype=float)
    
    def calculate_interest_coverage(self, ebit: pd.Series, interest_expense: pd.Series) -> pd.Series:
        """利息保障倍数"""
        try:
            interest_coverage = ebit / interest_expense
            return interest_coverage.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            self.logger.error(f"计算利息保障倍数失败: {e}")
            return pd.Series(index=ebit.index, dtype=float)
    
    # 额外的基本面因子
    def calculate_dupont_factors(self, fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """杜邦分析因子"""
        try:
            result = pd.DataFrame()
            
            if fundamental_data.empty:
                return result
            
            # 净利润率
            if 'net_income' in fundamental_data.columns and 'total_revenue' in fundamental_data.columns:
                result['net_margin'] = fundamental_data['net_income'] / fundamental_data['total_revenue']
            
            # 资产周转率
            if 'total_revenue' in fundamental_data.columns and 'total_assets' in fundamental_data.columns:
                result['asset_turnover'] = fundamental_data['total_revenue'] / fundamental_data['total_assets']
            
            # 权益乘数
            if 'total_assets' in fundamental_data.columns and 'total_equity' in fundamental_data.columns:
                result['equity_multiplier'] = fundamental_data['total_assets'] / fundamental_data['total_equity']
            
            return result
            
        except Exception as e:
            self.logger.error(f"计算杜邦分析因子失败: {e}")
            return pd.DataFrame()
    
    def calculate_piotroski_score(self, fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """皮奥特洛斯基F评分"""
        try:
            result = pd.DataFrame()
            
            if fundamental_data.empty or 'symbol' not in fundamental_data.columns:
                return result
            
            for symbol in fundamental_data['symbol'].unique():
                symbol_data = fundamental_data[fundamental_data['symbol'] == symbol].copy()
                symbol_data = symbol_data.sort_values('report_date')
                
                score = pd.Series(0, index=symbol_data.index)
                
                # 盈利能力指标 (4个)
                if 'net_income' in symbol_data.columns:
                    score += (symbol_data['net_income'] > 0).astype(int)  # 正净利润
                
                if 'operating_cash_flow' in symbol_data.columns:
                    score += (symbol_data['operating_cash_flow'] > 0).astype(int)  # 正经营现金流
                
                if 'roa' in symbol_data.columns:
                    score += (symbol_data['roa'].diff() > 0).astype(int)  # ROA改善
                
                if 'operating_cash_flow' in symbol_data.columns and 'net_income' in symbol_data.columns:
                    score += (symbol_data['operating_cash_flow'] > symbol_data['net_income']).astype(int)  # 现金流质量
                
                result = pd.concat([result, pd.DataFrame({
                    'symbol': symbol,
                    'piotroski_score': score
                })], ignore_index=True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"计算皮奥特洛斯基评分失败: {e}")
            return pd.DataFrame()
    
    def calculate_altman_z_score(self, fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """奥特曼Z评分（破产预测模型）"""
        try:
            result = pd.DataFrame()
            
            if fundamental_data.empty:
                return result
            
            # 简化的Z-Score计算
            if all(col in fundamental_data.columns for col in ['total_assets', 'total_equity', 'total_revenue', 'net_income']):
                # 工作资本/总资产
                working_capital_ratio = 0  # 需要流动资产和流动负债数据
                
                # 留存收益/总资产 (用净利润代替)
                retained_earnings_ratio = fundamental_data['net_income'] / fundamental_data['total_assets']
                
                # EBIT/总资产 (用净利润代替EBIT)
                ebit_ratio = fundamental_data['net_income'] / fundamental_data['total_assets']
                
                # 股权市值/总负债
                if 'market_cap' in fundamental_data.columns and 'total_debt' in fundamental_data.columns:
                    market_value_ratio = fundamental_data['market_cap'] / fundamental_data['total_debt']
                else:
                    market_value_ratio = 1  # 默认值
                
                # 营业收入/总资产
                sales_ratio = fundamental_data['total_revenue'] / fundamental_data['total_assets']
                
                # Z-Score = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
                z_score = (1.2 * working_capital_ratio + 
                          1.4 * retained_earnings_ratio + 
                          3.3 * ebit_ratio + 
                          0.6 * market_value_ratio + 
                          1.0 * sales_ratio)
                
                result['altman_z_score'] = z_score
            
            return result
            
        except Exception as e:
            self.logger.error(f"计算奥特曼Z评分失败: {e}")
            return pd.DataFrame()
    
    def calculate_momentum_factors(self, fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """计算基本面动量因子"""
        try:
            result = pd.DataFrame()
            
            if fundamental_data.empty or 'symbol' not in fundamental_data.columns:
                return result
            
            for symbol in fundamental_data['symbol'].unique():
                symbol_data = fundamental_data[fundamental_data['symbol'] == symbol].copy()
                symbol_data = symbol_data.sort_values('report_date')
                
                # 盈利预期修正
                if 'earnings_per_share' in symbol_data.columns:
                    eps_revision = symbol_data['earnings_per_share'].diff()
                    result = pd.concat([result, pd.DataFrame({
                        'symbol': symbol,
                        'eps_revision': eps_revision
                    })], ignore_index=True)
                
                # 营收预期修正
                if 'total_revenue' in symbol_data.columns:
                    revenue_revision = symbol_data['total_revenue'].pct_change()
                    result.loc[result['symbol'] == symbol, 'revenue_revision'] = revenue_revision.values
            
            return result
            
        except Exception as e:
            self.logger.error(f"计算基本面动量因子失败: {e}")
            return pd.DataFrame()
    
    def calculate_efficiency_factors(self, fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """计算效率因子"""
        try:
            result = pd.DataFrame()
            
            if fundamental_data.empty:
                return result
            
            # 存货周转率
            if 'cost_of_goods_sold' in fundamental_data.columns and 'inventory' in fundamental_data.columns:
                result['inventory_turnover'] = fundamental_data['cost_of_goods_sold'] / fundamental_data['inventory']
            
            # 应收账款周转率
            if 'total_revenue' in fundamental_data.columns and 'accounts_receivable' in fundamental_data.columns:
                result['receivables_turnover'] = fundamental_data['total_revenue'] / fundamental_data['accounts_receivable']
            
            # 固定资产周转率
            if 'total_revenue' in fundamental_data.columns and 'fixed_assets' in fundamental_data.columns:
                result['fixed_asset_turnover'] = fundamental_data['total_revenue'] / fundamental_data['fixed_assets']
            
            return result
            
        except Exception as e:
            self.logger.error(f"计算效率因子失败: {e}")
            return pd.DataFrame() 