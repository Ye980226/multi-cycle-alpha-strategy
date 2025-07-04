#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习模型模块
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 导入必要的库
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 导入日志和基类
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import Logger


class BaseMLModel:
    """机器学习模型基类"""
    
    def __init__(self, name: str, params: Dict = None, logger: Logger = None):
        """初始化模型"""
        self.name = name
        self.params = params or {}
        self.logger = logger or Logger()
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        """训练模型"""
        try:
            self.feature_names = list(X.columns)
            
            # 子类实现具体的训练逻辑
            self._fit_model(X, y, sample_weight)
            self.is_fitted = True
            
            self.logger.info(f"Model {self.name} fitted successfully with {len(X)} samples and {len(X.columns)} features")
            
        except Exception as e:
            self.logger.error(f"Error fitting model {self.name}: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
            
            # 确保特征顺序一致
            if self.feature_names:
                X = X[self.feature_names]
            
            predictions = self._predict_model(X)
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting with model {self.name}: {str(e)}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率（对于回归任务，返回预测值）"""
        return self.predict(X)
    
    def get_feature_importance(self) -> pd.Series:
        """获取特征重要性"""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before getting feature importance")
            
            importance = self._get_feature_importance()
            if importance is not None and self.feature_names:
                return pd.Series(importance, index=self.feature_names)
            else:
                return pd.Series()
                
        except Exception as e:
            self.logger.error(f"Error getting feature importance for model {self.name}: {str(e)}")
            return pd.Series()
    
    def save_model(self, file_path: str):
        """保存模型"""
        try:
            model_data = {
                'name': self.name,
                'params': self.params,
                'model': self.model,
                'is_fitted': self.is_fitted,
                'feature_names': self.feature_names
            }
            joblib.dump(model_data, file_path)
            self.logger.info(f"Model {self.name} saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model {self.name}: {str(e)}")
            raise
    
    def load_model(self, file_path: str):
        """加载模型"""
        try:
            model_data = joblib.load(file_path)
            self.name = model_data['name']
            self.params = model_data['params']
            self.model = model_data['model']
            self.is_fitted = model_data['is_fitted']
            self.feature_names = model_data['feature_names']
            
            self.logger.info(f"Model {self.name} loaded from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model from {file_path}: {str(e)}")
            raise
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        """子类需要实现的训练逻辑"""
        raise NotImplementedError("Subclasses must implement _fit_model")
    
    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """子类需要实现的预测逻辑"""
        raise NotImplementedError("Subclasses must implement _predict_model")
    
    def _get_feature_importance(self) -> Optional[np.ndarray]:
        """子类需要实现的特征重要性获取逻辑"""
        return None


class LightGBMModel(BaseMLModel):
    """LightGBM模型"""
    
    def __init__(self, params: Dict = None, logger: Logger = None):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available. Please install it using: pip install lightgbm")
        
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        if params:
            default_params.update(params)
        
        super().__init__("LightGBM", default_params, logger)
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        """训练LightGBM模型"""
        train_data = lgb.Dataset(X, label=y, weight=sample_weight)
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.params.get('num_boost_round', 100),
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
    
    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """LightGBM预测"""
        return self.model.predict(X)
    
    def _get_feature_importance(self) -> Optional[np.ndarray]:
        """获取LightGBM特征重要性"""
        if hasattr(self.model, 'feature_importance'):
            return self.model.feature_importance()
        return None


class XGBoostModel(BaseMLModel):
    """XGBoost模型"""
    
    def __init__(self, params: Dict = None, logger: Logger = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Please install it using: pip install xgboost")
        
        default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        if params:
            default_params.update(params)
        
        super().__init__("XGBoost", default_params, logger)
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        """训练XGBoost模型"""
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y, sample_weight=sample_weight)
    
    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """XGBoost预测"""
        return self.model.predict(X)
    
    def _get_feature_importance(self) -> Optional[np.ndarray]:
        """获取XGBoost特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None


class LinearModel(BaseMLModel):
    """线性模型"""
    
    def __init__(self, model_type: str = "linear", params: Dict = None, logger: Logger = None):
        default_params = {
            'alpha': 1.0,
            'l1_ratio': 0.5,
            'random_state': 42
        }
        
        if params:
            default_params.update(params)
        
        self.model_type = model_type
        super().__init__(f"Linear_{model_type}", default_params, logger)
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        """训练线性模型"""
        if self.model_type == "linear":
            self.model = LinearRegression()
        elif self.model_type == "ridge":
            self.model = Ridge(alpha=self.params['alpha'], random_state=self.params['random_state'])
        elif self.model_type == "lasso":
            self.model = Lasso(alpha=self.params['alpha'], random_state=self.params['random_state'])
        elif self.model_type == "elastic":
            self.model = ElasticNet(
                alpha=self.params['alpha'],
                l1_ratio=self.params['l1_ratio'],
                random_state=self.params['random_state']
            )
        else:
            raise ValueError(f"Unknown linear model type: {self.model_type}")
        
        self.model.fit(X, y, sample_weight=sample_weight)
    
    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """线性模型预测"""
        return self.model.predict(X)
    
    def _get_feature_importance(self) -> Optional[np.ndarray]:
        """获取线性模型系数"""
        if hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_)
        return None


class RandomForestModel(BaseMLModel):
    """随机森林模型"""
    
    def __init__(self, params: Dict = None, logger: Logger = None):
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        if params:
            default_params.update(params)
        
        super().__init__("RandomForest", default_params, logger)
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        """训练随机森林模型"""
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X, y, sample_weight=sample_weight)
    
    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """随机森林预测"""
        return self.model.predict(X)
    
    def _get_feature_importance(self) -> Optional[np.ndarray]:
        """获取随机森林特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None


class NeuralNetworkModel(BaseMLModel):
    """神经网络模型"""
    
    def __init__(self, params: Dict = None, logger: Logger = None):
        default_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'max_iter': 200,
            'random_state': 42
        }
        
        if params:
            default_params.update(params)
        
        super().__init__("NeuralNetwork", default_params, logger)
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        """训练神经网络模型"""
        # 标准化特征
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = MLPRegressor(**self.params)
        self.model.fit(X_scaled, y, sample_weight=sample_weight)
    
    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """神经网络预测"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def _get_feature_importance(self) -> Optional[np.ndarray]:
        """神经网络没有传统的特征重要性"""
        return None


class GradientBoostingModel(BaseMLModel):
    """梯度提升模型"""
    
    def __init__(self, params: Dict = None, logger: Logger = None):
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'subsample': 1.0,
            'random_state': 42
        }
        
        if params:
            default_params.update(params)
        
        super().__init__("GradientBoosting", default_params, logger)
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        """训练梯度提升模型"""
        self.model = GradientBoostingRegressor(**self.params)
        self.model.fit(X, y, sample_weight=sample_weight)
    
    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """梯度提升预测"""
        return self.model.predict(X)
    
    def _get_feature_importance(self) -> Optional[np.ndarray]:
        """获取梯度提升特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None


# 模型工厂函数
def create_model(model_type: str, params: Dict = None, logger: Logger = None) -> BaseMLModel:
    """创建模型工厂函数"""
    model_classes = {
        'lightgbm': LightGBMModel,
        'xgboost': XGBoostModel,
        'linear': lambda p, l: LinearModel("linear", p, l),
        'ridge': lambda p, l: LinearModel("ridge", p, l),
        'lasso': lambda p, l: LinearModel("lasso", p, l),
        'elastic': lambda p, l: LinearModel("elastic", p, l),
        'random_forest': RandomForestModel,
        'neural_network': NeuralNetworkModel,
        'gradient_boosting': GradientBoostingModel
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Available types: {list(model_classes.keys())}")
    
    return model_classes[model_type](params, logger)