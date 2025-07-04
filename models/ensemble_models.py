#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成模型模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 导入基础模型类
from .ml_models import BaseMLModel

# 导入日志
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import Logger


class StackingEnsemble(BaseMLModel):
    """堆叠集成模型"""
    
    def __init__(self, base_models: List[BaseMLModel], meta_model: BaseMLModel = None, 
                 cv_folds: int = 5, logger: Logger = None):
        """
        初始化堆叠集成模型
        
        Args:
            base_models: 基学习器列表
            meta_model: 元学习器，默认使用线性回归
            cv_folds: 交叉验证折数
            logger: 日志记录器
        """
        self.base_models = base_models
        self.meta_model = meta_model or LinearRegression()
        self.cv_folds = cv_folds
        
        super().__init__("StackingEnsemble", {}, logger)
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        """训练堆叠集成模型"""
        try:
            n_samples = len(X)
            n_models = len(self.base_models)
            
            # 创建元特征矩阵
            meta_features = np.zeros((n_samples, n_models))
            
            # K折交叉验证生成元特征
            kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            
            for i, (train_idx, val_idx) in enumerate(kfold.split(X)):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                
                # 训练每个基学习器
                for j, base_model in enumerate(self.base_models):
                    # 创建模型副本
                    model_copy = self._copy_model(base_model)
                    
                    # 训练模型
                    model_copy.fit(X_train_fold, y_train_fold, 
                                 sample_weight.iloc[train_idx] if sample_weight is not None else None)
                    
                    # 预测验证集
                    val_pred = model_copy.predict(X_val_fold)
                    meta_features[val_idx, j] = val_pred
            
            # 训练所有基学习器在全量数据上
            self.fitted_base_models = []
            for base_model in self.base_models:
                model_copy = self._copy_model(base_model)
                model_copy.fit(X, y, sample_weight)
                self.fitted_base_models.append(model_copy)
            
            # 训练元学习器
            meta_features_df = pd.DataFrame(meta_features, 
                                          columns=[f"model_{i}" for i in range(n_models)])
            
            if hasattr(self.meta_model, 'fit'):
                self.meta_model.fit(meta_features_df, y, sample_weight)
            else:
                # 如果meta_model是我们自定义的BaseMLModel
                self.meta_model.fit(meta_features_df, y, sample_weight)
            
            self.logger.info(f"Stacking ensemble trained with {n_models} base models")
            
        except Exception as e:
            self.logger.error(f"Error training stacking ensemble: {str(e)}")
            raise
    
    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """堆叠集成预测"""
        try:
            # 获取基学习器预测
            base_predictions = np.zeros((len(X), len(self.fitted_base_models)))
            
            for i, model in enumerate(self.fitted_base_models):
                base_predictions[:, i] = model.predict(X)
            
            # 构造元特征
            meta_features = pd.DataFrame(base_predictions, 
                                       columns=[f"model_{i}" for i in range(len(self.fitted_base_models))])
            
            # 元学习器预测
            if hasattr(self.meta_model, 'predict'):
                return self.meta_model.predict(meta_features)
            else:
                return self.meta_model.predict(meta_features)
                
        except Exception as e:
            self.logger.error(f"Error predicting with stacking ensemble: {str(e)}")
            raise
    
    def _copy_model(self, model: BaseMLModel):
        """复制模型"""
        # 创建相同类型的新模型实例
        model_class = type(model)
        if hasattr(model, 'model_type'):
            # 处理LinearModel等有特殊参数的模型
            return model_class(model.model_type, model.params.copy(), model.logger)
        else:
            return model_class(model.params.copy(), model.logger)
    
    def _get_feature_importance(self) -> Optional[np.ndarray]:
        """获取集成模型特征重要性（取平均）"""
        try:
            importances = []
            for model in self.fitted_base_models:
                imp = model._get_feature_importance()
                if imp is not None:
                    importances.append(imp)
            
            if importances:
                return np.mean(importances, axis=0)
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return None


class BlendingEnsemble(BaseMLModel):
    """混合集成模型"""
    
    def __init__(self, base_models: List[BaseMLModel], blend_ratio: float = 0.2, 
                 logger: Logger = None):
        """
        初始化混合集成模型
        
        Args:
            base_models: 基学习器列表
            blend_ratio: 混合数据比例
            logger: 日志记录器
        """
        self.base_models = base_models
        self.blend_ratio = blend_ratio
        
        super().__init__("BlendingEnsemble", {}, logger)
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        """训练混合集成模型"""
        try:
            n_samples = len(X)
            blend_size = int(n_samples * self.blend_ratio)
            
            # 分割数据
            blend_indices = np.random.choice(n_samples, blend_size, replace=False)
            train_indices = np.setdiff1d(np.arange(n_samples), blend_indices)
            
            X_train = X.iloc[train_indices]
            y_train = y.iloc[train_indices]
            X_blend = X.iloc[blend_indices]
            y_blend = y.iloc[blend_indices]
            
            train_weight = sample_weight.iloc[train_indices] if sample_weight is not None else None
            blend_weight = sample_weight.iloc[blend_indices] if sample_weight is not None else None
            
            # 训练基学习器
            self.fitted_base_models = []
            blend_predictions = np.zeros((len(X_blend), len(self.base_models)))
            
            for i, base_model in enumerate(self.base_models):
                model_copy = self._copy_model(base_model)
                model_copy.fit(X_train, y_train, train_weight)
                self.fitted_base_models.append(model_copy)
                
                # 在混合集上预测
                blend_predictions[:, i] = model_copy.predict(X_blend)
            
            # 计算最优权重
            self.weights = self._optimize_weights(blend_predictions, y_blend.values)
            
            self.logger.info(f"Blending ensemble trained with weights: {self.weights}")
            
        except Exception as e:
            self.logger.error(f"Error training blending ensemble: {str(e)}")
            raise
    
    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """混合集成预测"""
        try:
            predictions = np.zeros((len(X), len(self.fitted_base_models)))
            
            for i, model in enumerate(self.fitted_base_models):
                predictions[:, i] = model.predict(X)
            
            # 加权平均
            return np.dot(predictions, self.weights)
            
        except Exception as e:
            self.logger.error(f"Error predicting with blending ensemble: {str(e)}")
            raise
    
    def _optimize_weights(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """优化混合权重"""
        try:
            from scipy.optimize import minimize
            
            n_models = predictions.shape[1]
            
            # 初始权重（等权重）
            init_weights = np.ones(n_models) / n_models
            
            # 优化函数
            def objective(weights):
                weighted_pred = np.dot(predictions, weights)
                return mean_squared_error(targets, weighted_pred)
            
            # 约束条件：权重和为1，且非负
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(n_models)]
            
            # 优化
            result = minimize(objective, init_weights, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            return result.x if result.success else init_weights
            
        except Exception:
            # 如果优化失败，返回等权重
            return np.ones(predictions.shape[1]) / predictions.shape[1]
    
    def _copy_model(self, model: BaseMLModel):
        """复制模型"""
        model_class = type(model)
        if hasattr(model, 'model_type'):
            return model_class(model.model_type, model.params.copy(), model.logger)
        else:
            return model_class(model.params.copy(), model.logger)
    
    def _get_feature_importance(self) -> Optional[np.ndarray]:
        """获取加权特征重要性"""
        try:
            importances = []
            for model in self.fitted_base_models:
                imp = model._get_feature_importance()
                if imp is not None:
                    importances.append(imp)
            
            if importances and hasattr(self, 'weights'):
                weighted_importance = np.zeros_like(importances[0])
                for i, imp in enumerate(importances):
                    weighted_importance += self.weights[i] * imp
                return weighted_importance
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return None


class VotingEnsemble(BaseMLModel):
    """投票集成模型"""
    
    def __init__(self, base_models: List[BaseMLModel], voting: str = "soft", 
                 weights: Optional[List[float]] = None, logger: Logger = None):
        """
        初始化投票集成模型
        
        Args:
            base_models: 基学习器列表
            voting: 投票方式，'hard'或'soft'
            weights: 模型权重
            logger: 日志记录器
        """
        self.base_models = base_models
        self.voting = voting
        self.weights = weights or [1.0] * len(base_models)
        
        # 标准化权重
        self.weights = np.array(self.weights)
        self.weights = self.weights / np.sum(self.weights)
        
        super().__init__("VotingEnsemble", {}, logger)
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        """训练投票集成模型"""
        try:
            self.fitted_base_models = []
            
            for base_model in self.base_models:
                model_copy = self._copy_model(base_model)
                model_copy.fit(X, y, sample_weight)
                self.fitted_base_models.append(model_copy)
            
            self.logger.info(f"Voting ensemble trained with {len(self.base_models)} models")
            
        except Exception as e:
            self.logger.error(f"Error training voting ensemble: {str(e)}")
            raise
    
    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """投票集成预测"""
        try:
            predictions = np.zeros((len(X), len(self.fitted_base_models)))
            
            for i, model in enumerate(self.fitted_base_models):
                predictions[:, i] = model.predict(X)
            
            if self.voting == "soft":
                # 软投票：加权平均
                return np.dot(predictions, self.weights)
            else:
                # 硬投票：多数投票（对于回归任务使用加权平均）
                return np.dot(predictions, self.weights)
                
        except Exception as e:
            self.logger.error(f"Error predicting with voting ensemble: {str(e)}")
            raise
    
    def _copy_model(self, model: BaseMLModel):
        """复制模型"""
        model_class = type(model)
        if hasattr(model, 'model_type'):
            return model_class(model.model_type, model.params.copy(), model.logger)
        else:
            return model_class(model.params.copy(), model.logger)
    
    def _get_feature_importance(self) -> Optional[np.ndarray]:
        """获取加权特征重要性"""
        try:
            importances = []
            for model in self.fitted_base_models:
                imp = model._get_feature_importance()
                if imp is not None:
                    importances.append(imp)
            
            if importances:
                weighted_importance = np.zeros_like(importances[0])
                for i, imp in enumerate(importances):
                    weighted_importance += self.weights[i] * imp
                return weighted_importance
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return None


class AdaptiveEnsemble(BaseMLModel):
    """自适应集成模型"""
    
    def __init__(self, base_models: List[BaseMLModel], adaptation_window: int = 50,
                 learning_rate: float = 0.01, logger: Logger = None):
        """
        初始化自适应集成模型
        
        Args:
            base_models: 基学习器列表
            adaptation_window: 自适应窗口大小
            learning_rate: 学习率
            logger: 日志记录器
        """
        self.base_models = base_models
        self.adaptation_window = adaptation_window
        self.learning_rate = learning_rate
        
        super().__init__("AdaptiveEnsemble", {}, logger)
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        """训练自适应集成模型"""
        try:
            # 训练基学习器
            self.fitted_base_models = []
            
            for base_model in self.base_models:
                model_copy = self._copy_model(base_model)
                model_copy.fit(X, y, sample_weight)
                self.fitted_base_models.append(model_copy)
            
            # 初始化权重
            self.weights = np.ones(len(self.base_models)) / len(self.base_models)
            self.performance_history = []
            
            self.logger.info(f"Adaptive ensemble trained with {len(self.base_models)} models")
            
        except Exception as e:
            self.logger.error(f"Error training adaptive ensemble: {str(e)}")
            raise
    
    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """自适应集成预测"""
        try:
            predictions = np.zeros((len(X), len(self.fitted_base_models)))
            
            for i, model in enumerate(self.fitted_base_models):
                predictions[:, i] = model.predict(X)
            
            # 加权预测
            return np.dot(predictions, self.weights)
            
        except Exception as e:
            self.logger.error(f"Error predicting with adaptive ensemble: {str(e)}")
            raise
    
    def update_weights(self, X: pd.DataFrame, y_true: pd.Series):
        """根据新数据更新权重"""
        try:
            # 获取各模型预测
            predictions = []
            for model in self.fitted_base_models:
                pred = model.predict(X)
                predictions.append(pred)
            
            predictions = np.array(predictions).T
            
            # 计算各模型误差
            errors = []
            for i in range(len(self.fitted_base_models)):
                error = np.mean((predictions[:, i] - y_true.values) ** 2)
                errors.append(error)
            
            # 更新权重（误差越小权重越大）
            errors = np.array(errors)
            # 避免除零
            errors = np.maximum(errors, 1e-8)
            
            # 计算新权重（误差的倒数）
            new_weights = 1.0 / errors
            new_weights = new_weights / np.sum(new_weights)
            
            # 指数移动平均更新权重
            self.weights = (1 - self.learning_rate) * self.weights + self.learning_rate * new_weights
            
            # 记录性能历史
            ensemble_pred = np.dot(predictions, self.weights)
            ensemble_error = np.mean((ensemble_pred - y_true.values) ** 2)
            self.performance_history.append(ensemble_error)
            
            # 保持历史记录窗口
            if len(self.performance_history) > self.adaptation_window:
                self.performance_history.pop(0)
            
            self.logger.debug(f"Updated weights: {self.weights}")
            
        except Exception as e:
            self.logger.error(f"Error updating adaptive weights: {str(e)}")
    
    def _copy_model(self, model: BaseMLModel):
        """复制模型"""
        model_class = type(model)
        if hasattr(model, 'model_type'):
            return model_class(model.model_type, model.params.copy(), model.logger)
        else:
            return model_class(model.params.copy(), model.logger)
    
    def _get_feature_importance(self) -> Optional[np.ndarray]:
        """获取加权特征重要性"""
        try:
            importances = []
            for model in self.fitted_base_models:
                imp = model._get_feature_importance()
                if imp is not None:
                    importances.append(imp)
            
            if importances:
                weighted_importance = np.zeros_like(importances[0])
                for i, imp in enumerate(importances):
                    weighted_importance += self.weights[i] * imp
                return weighted_importance
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return None