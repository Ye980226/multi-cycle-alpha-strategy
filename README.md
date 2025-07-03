# 多周期Alpha策略框架

## 项目简介

这是一个完整的量化投资策略框架，专为多周期alpha策略设计。框架基于分钟频数据，提供了从数据获取到策略执行的完整解决方案。

## 核心特性

### 🚀 多周期架构
- 支持日内、日频、周频等多时间周期
- 智能周期权重分配
- 动态周期切换机制

### 📊 数据处理
- 支持AkShare、Tushare等多数据源
- 分钟级高频数据处理
- 数据质量检验和清洗
- 智能缓存管理

### 🔬 因子系统
- **技术因子**: 动量、趋势、波动率、成交量等
- **基本面因子**: 估值、盈利、成长、质量等
- **情绪因子**: 资金流、市场情绪等
- **高频因子**: 日内动量、微观结构等

### 🤖 机器学习
- LightGBM、XGBoost、神经网络等多模型
- 自动特征选择和超参优化
- 模型集成和在线学习
- 模型性能监控和漂移检测

### 📈 信号生成
- 多种信号生成方法（排序、阈值、ML等）
- 信号组合和过滤
- 多周期信号对齐
- 信号质量监控

### 💼 组合优化
- 均值方差优化
- Black-Litterman模型
- 风险平价/层次风险平价
- 多周期权重优化
- 交易成本优化

### ⚠️ 风险管理
- VaR/CVaR风险度量
- 实时风险监控
- 仓位和杠杆控制
- 回撤管理
- 压力测试

### 🔄 交易执行
- TWAP/VWAP算法交易
- 智能订单路由
- 市场冲击建模
- 执行成本分析

### 📋 回测分析
- 向前分析和滚动回测
- 多策略比较
- 归因分析
- 压力测试和蒙特卡洛模拟

### 📱 监控系统
- 实时性能监控
- 风险告警
- 模型监控
- 自动报告生成

## 项目结构

```
multi_cycle_alpha_strategy/
├── main.py                     # 主入口文件
├── config/                     # 配置模块
│   └── strategy_config.py      # 策略配置
├── data/                       # 数据管理模块
│   └── data_manager.py         # 数据管理器
├── factors/                    # 因子模块
│   ├── factor_engine.py        # 因子引擎
│   ├── technical_factors.py    # 技术因子
│   ├── fundamental_factors.py  # 基本面因子
│   └── sentiment_factors.py    # 情绪因子
├── models/                     # 模型模块
│   └── model_manager.py        # 模型管理器
├── signals/                    # 信号模块
│   └── signal_generator.py     # 信号生成器
├── portfolio/                  # 组合优化模块
│   └── portfolio_optimizer.py  # 组合优化器
├── backtest/                   # 回测模块
│   └── backtest_engine.py      # 回测引擎
├── risk/                       # 风险管理模块
│   └── risk_manager.py         # 风险管理器
├── execution/                  # 执行模块
│   └── trade_executor.py       # 交易执行器
├── monitoring/                 # 监控模块
│   └── performance_monitor.py  # 性能监控器
├── utils/                      # 工具模块
│   └── logger.py               # 日志工具
├── notebooks/                  # Jupyter笔记本
├── logs/                       # 日志文件
└── results/                    # 结果输出
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 基本使用

```python
from multi_cycle_alpha_strategy import MultiCycleAlphaStrategy

# 初始化策略
strategy = MultiCycleAlphaStrategy()
strategy.initialize()

# 运行回测
results = strategy.run_backtest("2023-01-01", "2024-01-01")

# 生成报告
strategy.generate_research_report()
```

### 3. 自定义配置

```python
from multi_cycle_alpha_strategy.config import StrategyConfig

# 创建配置
config = StrategyConfig()
config.data_config.frequency = "1min"
config.data_config.universe = ["HS300"]
config.portfolio_config.cycles = ["intraday", "daily", "weekly"]

# 使用配置初始化策略
strategy = MultiCycleAlphaStrategy(config)
```

## 配置说明

### 数据配置
- `data_source`: 数据源选择
- `frequency`: 数据频率（1min, 5min, 15min等）
- `universe`: 股票池选择
- `benchmark`: 基准指数

### 因子配置
- `factor_groups`: 因子组别
- `lookback_periods`: 回望期设置
- `factor_neutralization`: 因子中性化
- `factor_standardization`: 标准化方法

### 模型配置
- `model_types`: 模型类型列表
- `ensemble_method`: 集成方法
- `retrain_frequency`: 重训练频率
- `validation_method`: 验证方法

### 组合配置
- `optimization_method`: 优化方法
- `rebalance_frequency`: 再平衡频率
- `max_position`: 最大仓位限制
- `transaction_cost`: 交易成本

### 风险配置
- `max_drawdown`: 最大回撤限制
- `var_confidence`: VaR置信度
- `sector_exposure_limit`: 行业暴露限制

## 示例策略

查看 `notebooks/` 目录下的示例笔记本：

- `01_基础使用示例.ipynb`: 框架基本使用方法
- `02_因子研究示例.ipynb`: 因子开发和测试
- `03_模型训练示例.ipynb`: 模型训练和验证
- `04_组合优化示例.ipynb`: 组合优化策略
- `05_风险管理示例.ipynb`: 风险控制应用
- `06_回测分析示例.ipynb`: 完整回测流程

## 注意事项

1. **数据质量**: 确保使用高质量的分钟级数据
2. **计算资源**: 分钟级回测需要较大内存和计算资源
3. **模型更新**: 定期重训练模型以适应市场变化
4. **风险控制**: 严格执行风险限制，避免过度集中
5. **交易成本**: 考虑实际交易成本对策略收益的影响

## 扩展开发

框架采用模块化设计，支持灵活扩展：

- 添加新的数据源
- 开发自定义因子
- 集成新的机器学习模型
- 实现新的优化算法
- 扩展风险管理功能

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue或Pull Request。 