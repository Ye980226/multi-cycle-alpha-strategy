#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理模块
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod
import joblib
import pickle
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .ml_models import LightGBMModel, XGBoostModel, LinearModel, NeuralNetworkModel
from .ensemble_models import StackingEnsemble, BlendingEnsemble, VotingEnsemble
from .time_series_models import ARIMAModel, LSTMModel, GRUModel
from utils.logger import Logger


class BaseModel(ABC):
    """模型基类"""
    
    def __init__(self, name: str, params: Dict = None):
        """初始化模型"""
        pass
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series = None):
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        pass
    
    def get_feature_importance(self) -> pd.Series:
        """获取特征重要性"""
        pass
    
    def save_model(self, file_path: str):
        """保存模型"""
        pass
    
    def load_model(self, file_path: str):
        """加载模型"""
        pass


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self):
        """初始化评估器"""
        pass
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """回归模型评估"""
        try:
            # 过滤掉NaN值
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) == 0:
                return {}
            
            metrics = {
                'mse': mean_squared_error(y_true_clean, y_pred_clean),
                'mae': mean_absolute_error(y_true_clean, y_pred_clean),
                'r2': r2_score(y_true_clean, y_pred_clean),
                'ic': self.calculate_ic(y_true_clean, y_pred_clean),
                'rank_ic': self.calculate_rank_ic(y_true_clean, y_pred_clean)
            }
            
            # 计算RMSE
            metrics['rmse'] = np.sqrt(metrics['mse'])
            
            return metrics
            
        except Exception as e:
            print(f"回归评估失败: {e}")
            return {}
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """分类模型评估"""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # 过滤掉NaN值
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) == 0:
                return {}
            
            metrics = {
                'accuracy': accuracy_score(y_true_clean, y_pred_clean),
                'precision': precision_score(y_true_clean, y_pred_clean, average='weighted', zero_division=0),
                'recall': recall_score(y_true_clean, y_pred_clean, average='weighted', zero_division=0),
                'f1': f1_score(y_true_clean, y_pred_clean, average='weighted', zero_division=0)
            }
            
            return metrics
            
        except Exception as e:
            print(f"分类评估失败: {e}")
            return {}
    
    def calculate_ic(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算信息系数"""
        try:
            # 过滤掉NaN值
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) < 2:
                return np.nan
            
            # 计算相关系数
            correlation = np.corrcoef(y_true_clean, y_pred_clean)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            print(f"计算IC失败: {e}")
            return np.nan
    
    def calculate_rank_ic(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算排序信息系数"""
        try:
            from scipy.stats import spearmanr
            
            # 过滤掉NaN值
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) < 2:
                return np.nan
            
            # 计算Spearman相关系数
            correlation, _ = spearmanr(y_true_clean, y_pred_clean)
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            print(f"计算Rank IC失败: {e}")
            return np.nan
    
    def calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """计算夏普比率"""
        try:
            # 过滤掉NaN值
            returns_clean = returns[~np.isnan(returns)]
            
            if len(returns_clean) == 0:
                return np.nan
            
            mean_return = np.mean(returns_clean)
            std_return = np.std(returns_clean)
            
            if std_return == 0:
                return np.nan
            
            # 假设无风险利率为0
            sharpe_ratio = mean_return / std_return * np.sqrt(252)  # 年化
            return sharpe_ratio
            
        except Exception as e:
            print(f"计算夏普比率失败: {e}")
            return np.nan
    
    def calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """计算最大回撤"""
        try:
            # 过滤掉NaN值
            cum_returns_clean = cumulative_returns[~np.isnan(cumulative_returns)]
            
            if len(cum_returns_clean) == 0:
                return np.nan
            
            # 计算累计净值
            cum_value = np.cumprod(1 + cum_returns_clean)
            
            # 计算历史最高点
            running_max = np.maximum.accumulate(cum_value)
            
            # 计算回撤
            drawdown = (cum_value - running_max) / running_max
            
            # 最大回撤
            max_drawdown = np.min(drawdown)
            return max_drawdown
            
        except Exception as e:
            print(f"计算最大回撤失败: {e}")
            return np.nan
    
    def cross_validate(self, model: BaseModel, X: pd.DataFrame, y: pd.Series,
                      cv_method: str = "time_series", n_splits: int = 5) -> Dict[str, float]:
        """交叉验证"""
        try:
            from sklearn.model_selection import cross_val_score, KFold
            from sklearn.linear_model import LinearRegression
            
            # 处理缺失值
            X_clean = X.fillna(X.mean())
            y_clean = y.fillna(y.mean())
            
            # 如果模型没有sklearn接口，使用线性回归作为代替
            if hasattr(model, 'model') and hasattr(model.model, 'fit'):
                estimator = model.model
            else:
                estimator = LinearRegression()
            
            # 选择交叉验证方法
            if cv_method == "time_series":
                cv = TimeSeriesSplit(n_splits=n_splits)
            else:
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            # 执行交叉验证
            scores = cross_val_score(estimator, X_clean, y_clean, cv=cv, 
                                   scoring='neg_mean_squared_error')
            
            return {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'scores': scores.tolist()
            }
            
        except Exception as e:
            print(f"交叉验证失败: {e}")
            return {}


class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(self, method: str = "bayesian"):
        """初始化优化器"""
        pass
    
    def optimize(self, model_class, X: pd.DataFrame, y: pd.Series,
                param_space: Dict, n_trials: int = 100) -> Dict:
        """超参数优化"""
        pass
    
    def grid_search(self, model_class, X: pd.DataFrame, y: pd.Series,
                   param_grid: Dict) -> Dict:
        """网格搜索"""
        pass
    
    def random_search(self, model_class, X: pd.DataFrame, y: pd.Series,
                     param_distributions: Dict, n_iter: int = 100) -> Dict:
        """随机搜索"""
        pass
    
    def bayesian_optimization(self, model_class, X: pd.DataFrame, y: pd.Series,
                             param_space: Dict, n_trials: int = 100) -> Dict:
        """贝叶斯优化"""
        pass


class FeatureSelector:
    """特征选择器"""
    
    def __init__(self):
        """初始化特征选择器"""
        self.logger = Logger().get_logger("feature_selector")
        self.selected_features = []
        self.feature_scores = {}
    
    def select_by_importance(self, model: BaseModel, X: pd.DataFrame,
                           top_k: int = 50) -> List[str]:
        """基于重要性选择特征"""
        try:
            # 获取特征重要性
            feature_importance = model.get_feature_importance()
            
            if feature_importance is None or len(feature_importance) == 0:
                self.logger.warning("模型没有特征重要性信息，返回所有特征")
                return X.columns.tolist()[:top_k]
            
            # 按重要性排序
            importance_series = pd.Series(feature_importance, index=X.columns)
            selected_features = importance_series.nlargest(top_k).index.tolist()
            
            self.feature_scores['importance'] = importance_series.to_dict()
            self.selected_features = selected_features
            
            self.logger.info(f"基于重要性选择了{len(selected_features)}个特征")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"基于重要性选择特征失败: {e}")
            return X.columns.tolist()[:top_k]
    
    def select_by_correlation(self, X: pd.DataFrame, y: pd.Series,
                            threshold: float = 0.05) -> List[str]:
        """基于相关性选择特征"""
        try:
            # 计算与目标变量的相关性
            correlations = X.corrwith(y).abs()
            
            # 过滤掉NaN值
            correlations = correlations.dropna()
            
            # 选择相关性高于阈值的特征
            selected_features = correlations[correlations > threshold].index.tolist()
            
            # 如果没有特征满足条件，选择前50个最相关的
            if len(selected_features) == 0:
                selected_features = correlations.nlargest(50).index.tolist()
                self.logger.warning(f"没有特征相关性高于{threshold}，选择前50个最相关的特征")
            
            self.feature_scores['correlation'] = correlations.to_dict()
            self.selected_features = selected_features
            
            self.logger.info(f"基于相关性选择了{len(selected_features)}个特征")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"基于相关性选择特征失败: {e}")
            return X.columns.tolist()[:50]
    
    def select_by_mutual_info(self, X: pd.DataFrame, y: pd.Series,
                             top_k: int = 50) -> List[str]:
        """基于互信息选择特征"""
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            # 处理缺失值
            X_clean = X.fillna(X.mean())
            y_clean = y.fillna(y.mean())
            
            # 计算互信息
            mi_scores = mutual_info_regression(X_clean, y_clean, random_state=42)
            
            # 创建互信息Series
            mi_series = pd.Series(mi_scores, index=X.columns)
            
            # 选择前top_k个特征
            selected_features = mi_series.nlargest(top_k).index.tolist()
            
            self.feature_scores['mutual_info'] = mi_series.to_dict()
            self.selected_features = selected_features
            
            self.logger.info(f"基于互信息选择了{len(selected_features)}个特征")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"基于互信息选择特征失败: {e}")
            return X.columns.tolist()[:top_k]
    
    def recursive_feature_elimination(self, model: BaseModel, X: pd.DataFrame,
                                    y: pd.Series, n_features: int = 50) -> List[str]:
        """递归特征消除"""
        try:
            from sklearn.feature_selection import RFE
            from sklearn.linear_model import LinearRegression
            
            # 如果模型没有feature_importances_属性，使用线性回归作为代替
            if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
                estimator = model.model
            else:
                estimator = LinearRegression()
            
            # 处理缺失值
            X_clean = X.fillna(X.mean())
            y_clean = y.fillna(y.mean())
            
            # 执行递归特征消除
            rfe = RFE(estimator=estimator, n_features_to_select=n_features)
            rfe.fit(X_clean, y_clean)
            
            # 获取选择的特征
            selected_features = X.columns[rfe.support_].tolist()
            
            # 记录特征排名
            feature_ranking = pd.Series(rfe.ranking_, index=X.columns)
            self.feature_scores['rfe_ranking'] = feature_ranking.to_dict()
            self.selected_features = selected_features
            
            self.logger.info(f"递归特征消除选择了{len(selected_features)}个特征")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"递归特征消除失败: {e}")
            return X.columns.tolist()[:n_features]
    
    def forward_selection(self, model: BaseModel, X: pd.DataFrame,
                         y: pd.Series, max_features: int = 50) -> List[str]:
        """前向选择"""
        try:
            from sklearn.metrics import mean_squared_error
            from sklearn.linear_model import LinearRegression
            
            # 处理缺失值
            X_clean = X.fillna(X.mean())
            y_clean = y.fillna(y.mean())
            
            selected_features = []
            remaining_features = X.columns.tolist()
            best_score = float('inf')
            
            # 使用简单的线性回归进行前向选择
            base_model = LinearRegression()
            
            for i in range(min(max_features, len(remaining_features))):
                best_feature = None
                best_iter_score = float('inf')
                
                # 尝试添加每个剩余特征
                for feature in remaining_features:
                    test_features = selected_features + [feature]
                    
                    try:
                        # 训练模型
                        base_model.fit(X_clean[test_features], y_clean)
                        
                        # 计算预测误差
                        y_pred = base_model.predict(X_clean[test_features])
                        score = mean_squared_error(y_clean, y_pred)
                        
                        if score < best_iter_score:
                            best_iter_score = score
                            best_feature = feature
                            
                    except Exception as e:
                        continue
                
                # 如果找到了改进的特征，添加它
                if best_feature and best_iter_score < best_score:
                    selected_features.append(best_feature)
                    remaining_features.remove(best_feature)
                    best_score = best_iter_score
                    self.logger.debug(f"添加特征 {best_feature}, 当前分数: {best_score}")
                else:
                    # 如果没有找到改进的特征，停止
                    break
            
            self.selected_features = selected_features
            self.feature_scores['forward_selection_score'] = best_score
            
            self.logger.info(f"前向选择选择了{len(selected_features)}个特征")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"前向选择失败: {e}")
            return X.columns.tolist()[:max_features]


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self):
        """初始化训练器"""
        self.logger = Logger().get_logger("model_trainer")
        self.training_history = []
        self.model_cache = {}
    
    def train_single_model(self, model: BaseModel, X: pd.DataFrame,
                          y: pd.Series, validation_data: tuple = None) -> BaseModel:
        """训练单个模型"""
        try:
            self.logger.info(f"开始训练模型: {model.name}")
            
            # 处理缺失值
            X_clean = X.fillna(X.mean())
            y_clean = y.fillna(y.mean())
            
            # 训练模型
            start_time = time.time()
            model.fit(X_clean, y_clean)
            training_time = time.time() - start_time
            
            # 记录训练历史
            training_record = {
                'model_name': model.name,
                'training_time': training_time,
                'training_samples': len(X_clean),
                'features': X_clean.columns.tolist(),
                'timestamp': datetime.now().isoformat()
            }
            
            # 如果有验证数据，计算验证指标
            if validation_data:
                X_val, y_val = validation_data
                X_val_clean = X_val.fillna(X_val.mean())
                y_val_clean = y_val.fillna(y_val.mean())
                
                val_pred = model.predict(X_val_clean)
                val_score = np.corrcoef(y_val_clean, val_pred)[0, 1]
                
                training_record['validation_score'] = val_score
                self.logger.info(f"模型 {model.name} 验证得分: {val_score:.4f}")
            
            self.training_history.append(training_record)
            self.logger.info(f"模型 {model.name} 训练完成，耗时: {training_time:.2f}秒")
            
            return model
            
        except Exception as e:
            self.logger.error(f"训练模型 {model.name} 失败: {e}")
            raise
    
    def train_ensemble_model(self, base_models: List[BaseModel], X: pd.DataFrame,
                           y: pd.Series, ensemble_method: str = "stacking") -> BaseModel:
        """训练集成模型"""
        try:
            self.logger.info(f"开始训练集成模型，方法: {ensemble_method}")
            
            # 处理缺失值
            X_clean = X.fillna(X.mean())
            y_clean = y.fillna(y.mean())
            
            # 训练所有基础模型
            trained_models = []
            for model in base_models:
                trained_model = self.train_single_model(model, X_clean, y_clean)
                trained_models.append(trained_model)
            
            # 创建集成模型
            if ensemble_method == "stacking":
                ensemble_model = StackingEnsemble(
                    base_models=trained_models,
                    cv_folds=5
                )
            elif ensemble_method == "blending":
                ensemble_model = BlendingEnsemble(
                    base_models=trained_models,
                    blend_ratio=0.2
                )
            elif ensemble_method in ["voting", "weighted_average"]:
                ensemble_model = VotingEnsemble(
                    base_models=trained_models,
                    voting="soft"
                )
            else:
                # 默认使用投票集成
                ensemble_model = VotingEnsemble(
                    base_models=trained_models,
                    voting="soft"
                )
            
            # 训练集成模型
            ensemble_model.fit(X_clean, y_clean)
            
            self.logger.info(f"集成模型训练完成，包含{len(trained_models)}个基础模型")
            return ensemble_model
            
        except Exception as e:
            self.logger.error(f"训练集成模型失败: {e}")
            raise
    
    def online_learning(self, model: BaseModel, new_X: pd.DataFrame,
                       new_y: pd.Series) -> BaseModel:
        """在线学习"""
        try:
            self.logger.info(f"开始在线学习: {model.name}")
            
            # 处理缺失值
            new_X_clean = new_X.fillna(new_X.mean())
            new_y_clean = new_y.fillna(new_y.mean())
            
            # 检查模型是否支持在线学习
            if hasattr(model, 'partial_fit'):
                # 使用partial_fit进行增量学习
                model.partial_fit(new_X_clean, new_y_clean)
            else:
                # 如果不支持，则重新训练（需要保存历史数据）
                self.logger.warning(f"模型 {model.name} 不支持在线学习，使用重新训练")
                
                # 这里需要历史数据，实际应用中需要维护训练数据缓存
                if hasattr(model, 'X_train') and hasattr(model, 'y_train'):
                    combined_X = pd.concat([model.X_train, new_X_clean])
                    combined_y = pd.concat([model.y_train, new_y_clean])
                    model.fit(combined_X, combined_y)
                else:
                    # 仅使用新数据训练
                    model.fit(new_X_clean, new_y_clean)
            
            self.logger.info(f"在线学习完成: {model.name}")
            return model
            
        except Exception as e:
            self.logger.error(f"在线学习失败: {e}")
            raise
    
    def incremental_training(self, model: BaseModel, new_data: pd.DataFrame,
                           retrain_threshold: float = 0.1) -> BaseModel:
        """增量训练"""
        try:
            self.logger.info(f"开始增量训练: {model.name}")
            
            # 分离特征和标签
            if 'target' in new_data.columns:
                new_X = new_data.drop('target', axis=1)
                new_y = new_data['target']
            else:
                # 假设最后一列是目标变量
                new_X = new_data.iloc[:, :-1]
                new_y = new_data.iloc[:, -1]
            
            # 计算模型性能退化
            if hasattr(model, 'last_performance'):
                current_pred = model.predict(new_X)
                current_performance = np.corrcoef(new_y, current_pred)[0, 1]
                performance_drop = model.last_performance - current_performance
                
                if performance_drop > retrain_threshold:
                    self.logger.info(f"性能下降 {performance_drop:.4f} 超过阈值 {retrain_threshold}，触发重训练")
                    
                    # 重新训练
                    if hasattr(model, 'X_train') and hasattr(model, 'y_train'):
                        combined_X = pd.concat([model.X_train, new_X])
                        combined_y = pd.concat([model.y_train, new_y])
                        model.fit(combined_X, combined_y)
                    else:
                        model.fit(new_X, new_y)
                    
                    # 更新性能记录
                    model.last_performance = current_performance
                else:
                    self.logger.info(f"性能下降 {performance_drop:.4f} 未超过阈值，继续使用当前模型")
            else:
                # 首次训练
                model.fit(new_X, new_y)
                pred = model.predict(new_X)
                model.last_performance = np.corrcoef(new_y, pred)[0, 1]
            
            self.logger.info(f"增量训练完成: {model.name}")
            return model
            
        except Exception as e:
            self.logger.error(f"增量训练失败: {e}")
            raise


class ModelManager:
    """模型管理主类"""
    
    def __init__(self, config: Dict = None):
        """初始化模型管理器"""
        self.config = config or {}
        self.logger = Logger().get_logger("model_manager")
        self.feature_selector = FeatureSelector()
        self.model_trainer = ModelTrainer()
        self.model_evaluator = ModelEvaluator()
        self.hyperparameter_optimizer = HyperparameterOptimizer()
        
        # 模型存储
        self.models = {}
        self.model_performance = {}
        self.training_history = []
        
    def initialize(self, config: Dict):
        """初始化配置"""
        self.config.update(config)
        self.logger.info("模型管理器初始化完成")
    
    def create_model(self, model_type: str, params: Dict = None) -> BaseModel:
        """创建模型"""
        try:
            params = params or {}
            
            if model_type == "lightgbm":
                return LightGBMModel(params)
            elif model_type == "xgboost":
                return XGBoostModel(params)
            elif model_type == "linear":
                return LinearModel(params.get("model_type", "ridge"), params)
            elif model_type == "neural_network":
                return NeuralNetworkModel(params)
            elif model_type == "arima":
                return ARIMAModel(params)
            elif model_type == "lstm":
                return LSTMModel(params)
            elif model_type == "gru":
                return GRUModel(params)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
                
        except Exception as e:
            self.logger.error(f"创建模型失败: {e}")
            raise
    
    def train_models(self, X: pd.DataFrame, y: pd.Series,
                    model_configs: List[Dict]) -> Dict[str, BaseModel]:
        """训练多个模型"""
        try:
            trained_models = {}
            
            for config in model_configs:
                model_name = config.get("name", config["type"])
                model_type = config["type"]
                model_params = config.get("params", {})
                
                self.logger.info(f"开始训练模型: {model_name}")
                
                # 创建模型
                model = self.create_model(model_type, model_params)
                
                # 特征选择
                if config.get("feature_selection", False):
                    feature_method = config.get("feature_method", "correlation")
                    if feature_method == "importance":
                        # 先训练一个简单模型用于特征选择
                        temp_model = self.create_model("lightgbm", {})
                        temp_model.fit(X, y)
                        selected_features = self.feature_selector.select_by_importance(
                            temp_model, X, config.get("n_features", 50)
                        )
                    elif feature_method == "correlation":
                        selected_features = self.feature_selector.select_by_correlation(
                            X, y, config.get("correlation_threshold", 0.05)
                        )
                    elif feature_method == "mutual_info":
                        selected_features = self.feature_selector.select_by_mutual_info(
                            X, y, config.get("n_features", 50)
                        )
                    else:
                        selected_features = X.columns.tolist()
                    
                    X_selected = X[selected_features]
                else:
                    X_selected = X
                
                # 超参数优化
                if config.get("hyperparameter_optimization", False):
                    param_space = config.get("param_space", {})
                    n_trials = config.get("n_trials", 50)
                    
                    best_params = self.hyperparameter_optimizer.optimize(
                        type(model), X_selected, y, param_space, n_trials
                    )
                    model = self.create_model(model_type, best_params)
                
                # 训练模型
                validation_data = config.get("validation_data")
                trained_model = self.model_trainer.train_single_model(
                    model, X_selected, y, validation_data
                )
                
                trained_models[model_name] = trained_model
                
                # 计算性能
                performance = self.get_model_performance(trained_model, X_selected, y)
                self.model_performance[model_name] = performance
                
                self.logger.info(f"模型 {model_name} 训练完成")
            
            self.models.update(trained_models)
            return trained_models
            
        except Exception as e:
            self.logger.error(f"训练模型失败: {e}")
            raise
    
    def select_best_model(self, models: Dict[str, BaseModel],
                         X_val: pd.DataFrame, y_val: pd.Series,
                         metric: str = "ic") -> str:
        """选择最佳模型"""
        try:
            best_model = None
            best_score = -np.inf
            
            for model_name, model in models.items():
                predictions = model.predict(X_val)
                
                if metric == "ic":
                    score = np.corrcoef(y_val, predictions)[0, 1]
                elif metric == "rank_ic":
                    score = pd.Series(y_val.values).corr(
                        pd.Series(predictions).rank(), method='spearman'
                    )
                elif metric == "mse":
                    score = -mean_squared_error(y_val, predictions)
                elif metric == "mae":
                    score = -mean_absolute_error(y_val, predictions)
                elif metric == "r2":
                    score = r2_score(y_val, predictions)
                else:
                    raise ValueError(f"不支持的评估指标: {metric}")
                
                if score > best_score:
                    best_score = score
                    best_model = model_name
                
                self.logger.info(f"模型 {model_name} {metric} 得分: {score:.4f}")
            
            self.logger.info(f"最佳模型: {best_model}, 得分: {best_score:.4f}")
            return best_model
            
        except Exception as e:
            self.logger.error(f"选择最佳模型失败: {e}")
            raise
    
    def ensemble_models(self, models: Dict[str, BaseModel],
                       method: str = "weighted_average",
                       weights: Dict[str, float] = None) -> BaseModel:
        """模型集成"""
        try:
            base_models = list(models.values())
            
            if method == "stacking":
                ensemble_model = StackingEnsemble(base_models, cv_folds=5)
            elif method == "blending":
                ensemble_model = BlendingEnsemble(base_models, blend_ratio=0.2)
            elif method == "voting":
                ensemble_model = VotingEnsemble(base_models, voting="soft")
            elif method == "weighted_average":
                if weights:
                    model_weights = [weights.get(name, 1.0) for name in models.keys()]
                    ensemble_model = VotingEnsemble(base_models, voting="soft", weights=model_weights)
                else:
                    ensemble_model = VotingEnsemble(base_models, voting="soft")
            else:
                raise ValueError(f"不支持的集成方法: {method}")
            
            self.logger.info(f"创建集成模型完成，方法: {method}")
            return ensemble_model
            
        except Exception as e:
            self.logger.error(f"集成模型失败: {e}")
            raise
    
    def predict_multi_horizon(self, models: Dict[str, BaseModel],
                             X: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
        """多周期预测"""
        try:
            predictions = {}
            
            for horizon in horizons:
                horizon_predictions = {}
                
                for model_name, model in models.items():
                    # 对于多周期预测，可能需要调整输入特征
                    # 这里假设模型已经支持多周期预测
                    pred = model.predict(X)
                    horizon_predictions[f"{model_name}_h{horizon}"] = pred
                
                predictions.update(horizon_predictions)
            
            result_df = pd.DataFrame(predictions, index=X.index)
            self.logger.info(f"多周期预测完成，周期: {horizons}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"多周期预测失败: {e}")
            raise
    
    def retrain_models(self, models: Dict[str, BaseModel],
                      new_data: pd.DataFrame, retrain_schedule: Dict):
        """重训练模型"""
        try:
            for model_name, model in models.items():
                if model_name in retrain_schedule:
                    schedule = retrain_schedule[model_name]
                    
                    if schedule.get("should_retrain", True):
                        self.logger.info(f"重训练模型: {model_name}")
                        
                        # 增量训练
                        retrain_threshold = schedule.get("threshold", 0.1)
                        updated_model = self.model_trainer.incremental_training(
                            model, new_data, retrain_threshold
                        )
                        
                        models[model_name] = updated_model
                        
                        # 更新性能记录
                        if 'target' in new_data.columns:
                            X_new = new_data.drop('target', axis=1)
                            y_new = new_data['target']
                            performance = self.get_model_performance(updated_model, X_new, y_new)
                            self.model_performance[model_name] = performance
                
            self.logger.info("模型重训练完成")
            
        except Exception as e:
            self.logger.error(f"重训练模型失败: {e}")
            raise
    
    def validate_models(self, models: Dict[str, BaseModel],
                       validation_data: pd.DataFrame) -> Dict[str, Dict]:
        """验证模型"""
        try:
            validation_results = {}
            
            X_val = validation_data.drop('target', axis=1)
            y_val = validation_data['target']
            
            for model_name, model in models.items():
                predictions = model.predict(X_val)
                
                # 计算各种评估指标
                results = {
                    'ic': np.corrcoef(y_val, predictions)[0, 1],
                    'rank_ic': pd.Series(y_val.values).corr(
                        pd.Series(predictions).rank(), method='spearman'
                    ),
                    'mse': mean_squared_error(y_val, predictions),
                    'mae': mean_absolute_error(y_val, predictions),
                    'r2': r2_score(y_val, predictions)
                }
                
                validation_results[model_name] = results
                self.logger.info(f"模型 {model_name} 验证完成")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"验证模型失败: {e}")
            raise
    
    def get_model_performance(self, model: BaseModel, X: pd.DataFrame,
                             y: pd.Series) -> Dict[str, float]:
        """获取模型性能"""
        try:
            predictions = model.predict(X)
            
            performance = {
                'ic': np.corrcoef(y, predictions)[0, 1],
                'rank_ic': pd.Series(y.values).corr(
                    pd.Series(predictions).rank(), method='spearman'
                ),
                'mse': mean_squared_error(y, predictions),
                'mae': mean_absolute_error(y, predictions),
                'r2': r2_score(y, predictions)
            }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"获取模型性能失败: {e}")
            return {}
    
    def save_models(self, models: Dict[str, BaseModel], save_dir: str):
        """保存模型"""
        try:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            for model_name, model in models.items():
                model_path = os.path.join(save_dir, f"{model_name}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                self.logger.info(f"模型 {model_name} 保存至: {model_path}")
            
            # 保存性能记录
            performance_path = os.path.join(save_dir, "performance.json")
            import json
            with open(performance_path, 'w') as f:
                json.dump(self.model_performance, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"保存模型失败: {e}")
            raise
    
    def load_models(self, model_dir: str) -> Dict[str, BaseModel]:
        """加载模型"""
        try:
            import os
            import glob
            
            loaded_models = {}
            model_files = glob.glob(os.path.join(model_dir, "*.pkl"))
            
            for model_file in model_files:
                model_name = os.path.basename(model_file).replace(".pkl", "")
                
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                
                loaded_models[model_name] = model
                self.logger.info(f"模型 {model_name} 加载完成")
            
            # 加载性能记录
            performance_path = os.path.join(model_dir, "performance.json")
            if os.path.exists(performance_path):
                import json
                with open(performance_path, 'r') as f:
                    self.model_performance = json.load(f)
            
            self.models.update(loaded_models)
            return loaded_models
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            raise
    
    def monitor_model_drift(self, model: BaseModel, X_train: pd.DataFrame,
                           X_current: pd.DataFrame) -> Dict[str, float]:
        """监控模型漂移"""
        try:
            from scipy import stats
            
            drift_metrics = {}
            
            # 计算特征分布差异
            for col in X_train.columns:
                if col in X_current.columns:
                    # KS检验
                    ks_stat, ks_pvalue = stats.ks_2samp(X_train[col], X_current[col])
                    drift_metrics[f"{col}_ks_stat"] = ks_stat
                    drift_metrics[f"{col}_ks_pvalue"] = ks_pvalue
                    
                    # 均值差异
                    mean_diff = abs(X_train[col].mean() - X_current[col].mean())
                    drift_metrics[f"{col}_mean_diff"] = mean_diff
                    
                    # 方差差异
                    var_diff = abs(X_train[col].var() - X_current[col].var())
                    drift_metrics[f"{col}_var_diff"] = var_diff
            
            # 整体漂移得分
            ks_stats = [v for k, v in drift_metrics.items() if k.endswith('_ks_stat')]
            overall_drift = np.mean(ks_stats) if ks_stats else 0
            drift_metrics['overall_drift'] = overall_drift
            
            self.logger.info(f"模型漂移监控完成，整体漂移得分: {overall_drift:.4f}")
            return drift_metrics
            
        except Exception as e:
            self.logger.error(f"监控模型漂移失败: {e}")
            return {}
    
    def auto_model_selection(self, X: pd.DataFrame, y: pd.Series,
                           model_types: List[str] = None) -> BaseModel:
        """自动模型选择"""
        try:
            if model_types is None:
                model_types = ["lightgbm", "xgboost", "linear"]
            
            models = {}
            
            # 创建和训练多个模型
            for model_type in model_types:
                model = self.create_model(model_type)
                trained_model = self.model_trainer.train_single_model(model, X, y)
                models[model_type] = trained_model
            
            # 交叉验证选择最佳模型
            best_model_name = self.select_best_model(models, X, y, metric="ic")
            best_model = models[best_model_name]
            
            self.logger.info(f"自动模型选择完成，最佳模型: {best_model_name}")
            return best_model
            
        except Exception as e:
            self.logger.error(f"自动模型选择失败: {e}")
            raise
    
    def explain_model(self, model: BaseModel, X: pd.DataFrame,
                     method: str = "shap") -> Dict:
        """模型解释"""
        try:
            explanation = {}
            
            if method == "shap":
                try:
                    import shap
                    
                    # 选择解释器
                    if hasattr(model, 'model') and hasattr(model.model, 'predict'):
                        explainer = shap.Explainer(model.model)
                        shap_values = explainer(X.sample(min(100, len(X))))
                        explanation['shap_values'] = shap_values.values
                        explanation['feature_names'] = X.columns.tolist()
                    else:
                        self.logger.warning("SHAP解释器不支持此模型类型")
                        
                except ImportError:
                    self.logger.warning("SHAP未安装，无法进行模型解释")
                    
            elif method == "feature_importance":
                importance = model.get_feature_importance()
                if importance is not None:
                    explanation['feature_importance'] = importance.to_dict()
                    
            elif method == "permutation_importance":
                from sklearn.inspection import permutation_importance
                
                # 创建一个包装器以适配sklearn的permutation_importance
                def model_predict(X):
                    return model.predict(pd.DataFrame(X, columns=X.columns))
                
                perm_importance = permutation_importance(
                    model_predict, X, y, n_repeats=10, random_state=42
                )
                explanation['permutation_importance'] = {
                    'importances_mean': perm_importance.importances_mean,
                    'importances_std': perm_importance.importances_std,
                    'feature_names': X.columns.tolist()
                }
            
            self.logger.info(f"模型解释完成，方法: {method}")
            return explanation
            
        except Exception as e:
            self.logger.error(f"模型解释失败: {e}")
            return {} 