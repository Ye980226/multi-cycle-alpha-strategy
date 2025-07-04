#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间序列模型模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


class BaseTimeSeriesModel:
    """时间序列模型基类"""
    
    def __init__(self, name: str, params: Dict = None, logger=None):
        """初始化模型"""
        self.name = name
        self.params = params or {}
        self.logger = logger
        self.model = None
        self.scaler = None
        self.is_fitted = False
        
    def fit(self, data: pd.Series, **kwargs):
        """训练模型"""
        raise NotImplementedError
        
    def predict(self, steps: int = 1, **kwargs) -> np.ndarray:
        """预测"""
        raise NotImplementedError
        
    def forecast(self, steps: int = 1, **kwargs) -> pd.DataFrame:
        """预测并返回置信区间"""
        raise NotImplementedError
        
    def get_params(self) -> Dict:
        """获取模型参数"""
        return self.params
        
    def set_params(self, **params):
        """设置模型参数"""
        self.params.update(params)
        
    def save_model(self, file_path: str):
        """保存模型"""
        import joblib
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'params': self.params}, file_path)
        
    def load_model(self, file_path: str):
        """加载模型"""
        import joblib
        data = joblib.load(file_path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.params = data['params']
        self.is_fitted = True


class ARIMAModel(BaseTimeSeriesModel):
    """ARIMA模型"""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), 
                 seasonal_order: Tuple[int, int, int, int] = None,
                 logger=None):
        """初始化ARIMA模型"""
        super().__init__("ARIMA", logger=logger)
        self.order = order
        self.seasonal_order = seasonal_order
        self.params = {
            'order': order,
            'seasonal_order': seasonal_order
        }
        
    def _check_stationarity(self, data: pd.Series) -> bool:
        """检查时间序列平稳性"""
        if not HAS_STATSMODELS:
            if self.logger:
                self.logger.warning("statsmodels not available, skipping stationarity test")
            return True
            
        try:
            result = adfuller(data.dropna())
            p_value = result[1]
            return p_value <= 0.05
        except Exception as e:
            if self.logger:
                self.logger.error(f"Stationarity test failed: {e}")
            return True
            
    def fit(self, data: pd.Series, **kwargs):
        """训练ARIMA模型"""
        if not HAS_STATSMODELS:
            if self.logger:
                self.logger.error("statsmodels not available, cannot fit ARIMA model")
            return self
            
        try:
            # 检查平稳性
            if not self._check_stationarity(data):
                if self.logger:
                    self.logger.warning("Data is not stationary, consider differencing")
                    
            # 拟合模型
            if self.seasonal_order:
                self.model = SARIMAX(data, order=self.order, seasonal_order=self.seasonal_order)
            else:
                self.model = ARIMA(data, order=self.order)
                
            self.model = self.model.fit()
            self.is_fitted = True
            
            if self.logger:
                self.logger.info(f"ARIMA model fitted successfully, AIC: {self.model.aic:.4f}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to fit ARIMA model: {e}")
            raise
            
        return self
        
    def predict(self, steps: int = 1, **kwargs) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        try:
            forecast = self.model.forecast(steps=steps)
            return forecast.values
        except Exception as e:
            if self.logger:
                self.logger.error(f"Prediction failed: {e}")
            raise
            
    def forecast(self, steps: int = 1, **kwargs) -> pd.DataFrame:
        """预测并返回置信区间"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        try:
            forecast = self.model.get_forecast(steps=steps)
            forecast_df = pd.DataFrame({
                'forecast': forecast.predicted_mean,
                'lower_ci': forecast.conf_int().iloc[:, 0],
                'upper_ci': forecast.conf_int().iloc[:, 1]
            })
            return forecast_df
        except Exception as e:
            if self.logger:
                self.logger.error(f"Forecast failed: {e}")
            raise


class LSTMModel(BaseTimeSeriesModel):
    """LSTM模型"""
    
    def __init__(self, sequence_length: int = 60, 
                 lstm_units: int = 50, 
                 dropout_rate: float = 0.2,
                 epochs: int = 50,
                 batch_size: int = 32,
                 logger=None):
        """初始化LSTM模型"""
        super().__init__("LSTM", logger=logger)
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        
        self.params = {
            'sequence_length': sequence_length,
            'lstm_units': lstm_units,
            'dropout_rate': dropout_rate,
            'epochs': epochs,
            'batch_size': batch_size
        }
        
    def _prepare_data(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        # 数据标准化
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        # 创建序列数据
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
            
        return np.array(X), np.array(y)
        
    def _create_model(self, input_shape: Tuple[int, int]):
        """创建LSTM模型"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not available")
            
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
        
    def fit(self, data: pd.Series, validation_split: float = 0.2, **kwargs):
        """训练LSTM模型"""
        if not HAS_TENSORFLOW:
            if self.logger:
                self.logger.error("TensorFlow not available, cannot fit LSTM model")
            return self
            
        try:
            # 准备数据
            X, y = self._prepare_data(data)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # 创建模型
            self.model = self._create_model((X.shape[1], 1))
            
            # 设置回调
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0001)
            ]
            
            # 训练模型
            history = self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=0
            )
            
            self.is_fitted = True
            
            if self.logger:
                final_loss = history.history['loss'][-1]
                self.logger.info(f"LSTM model fitted successfully, final loss: {final_loss:.6f}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to fit LSTM model: {e}")
            raise
            
        return self
        
    def predict(self, data: pd.Series, steps: int = 1, **kwargs) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        try:
            # 准备输入数据
            scaled_data = self.scaler.transform(data.values.reshape(-1, 1))
            last_sequence = scaled_data[-self.sequence_length:]
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(steps):
                # 预测下一个值
                next_pred = self.model.predict(current_sequence.reshape(1, self.sequence_length, 1), verbose=0)
                predictions.append(next_pred[0, 0])
                
                # 更新序列
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = next_pred[0, 0]
                
            # 反标准化
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions)
            
            return predictions.flatten()
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Prediction failed: {e}")
            raise


class GRUModel(BaseTimeSeriesModel):
    """GRU模型"""
    
    def __init__(self, sequence_length: int = 60, 
                 gru_units: int = 50, 
                 dropout_rate: float = 0.2,
                 epochs: int = 50,
                 batch_size: int = 32,
                 logger=None):
        """初始化GRU模型"""
        super().__init__("GRU", logger=logger)
        self.sequence_length = sequence_length
        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        
        self.params = {
            'sequence_length': sequence_length,
            'gru_units': gru_units,
            'dropout_rate': dropout_rate,
            'epochs': epochs,
            'batch_size': batch_size
        }
        
    def _prepare_data(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        # 数据标准化
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        # 创建序列数据
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
            
        return np.array(X), np.array(y)
        
    def _create_model(self, input_shape: Tuple[int, int]):
        """创建GRU模型"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not available")
            
        model = Sequential([
            GRU(self.gru_units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout_rate),
            GRU(self.gru_units, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
        
    def fit(self, data: pd.Series, validation_split: float = 0.2, **kwargs):
        """训练GRU模型"""
        if not HAS_TENSORFLOW:
            if self.logger:
                self.logger.error("TensorFlow not available, cannot fit GRU model")
            return self
            
        try:
            # 准备数据
            X, y = self._prepare_data(data)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # 创建模型
            self.model = self._create_model((X.shape[1], 1))
            
            # 设置回调
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0001)
            ]
            
            # 训练模型
            history = self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=0
            )
            
            self.is_fitted = True
            
            if self.logger:
                final_loss = history.history['loss'][-1]
                self.logger.info(f"GRU model fitted successfully, final loss: {final_loss:.6f}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to fit GRU model: {e}")
            raise
            
        return self
        
    def predict(self, data: pd.Series, steps: int = 1, **kwargs) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        try:
            # 准备输入数据
            scaled_data = self.scaler.transform(data.values.reshape(-1, 1))
            last_sequence = scaled_data[-self.sequence_length:]
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(steps):
                # 预测下一个值
                next_pred = self.model.predict(current_sequence.reshape(1, self.sequence_length, 1), verbose=0)
                predictions.append(next_pred[0, 0])
                
                # 更新序列
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = next_pred[0, 0]
                
            # 反标准化
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions)
            
            return predictions.flatten()
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Prediction failed: {e}")
            raise


class VARModel(BaseTimeSeriesModel):
    """向量自回归模型"""
    
    def __init__(self, max_lags: int = 12, logger=None):
        """初始化VAR模型"""
        super().__init__("VAR", logger=logger)
        self.max_lags = max_lags
        self.params = {'max_lags': max_lags}
        
    def fit(self, data: pd.DataFrame, **kwargs):
        """训练VAR模型"""
        if not HAS_STATSMODELS:
            if self.logger:
                self.logger.error("statsmodels not available, cannot fit VAR model")
            return self
            
        try:
            from statsmodels.tsa.vector_ar.var_model import VAR
            
            # 创建VAR模型
            var_model = VAR(data)
            
            # 选择最优滞后阶数
            lag_order = var_model.select_order(maxlags=self.max_lags)
            optimal_lag = lag_order.aic
            
            # 拟合模型
            self.model = var_model.fit(optimal_lag)
            self.is_fitted = True
            
            if self.logger:
                self.logger.info(f"VAR model fitted successfully, optimal lag: {optimal_lag}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to fit VAR model: {e}")
            raise
            
        return self
        
    def predict(self, steps: int = 1, **kwargs) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        try:
            forecast = self.model.forecast(y=self.model.y, steps=steps)
            return forecast
        except Exception as e:
            if self.logger:
                self.logger.error(f"Prediction failed: {e}")
            raise


class ProphetModel(BaseTimeSeriesModel):
    """Prophet模型"""
    
    def __init__(self, seasonality_mode: str = 'additive',
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = False,
                 logger=None):
        """初始化Prophet模型"""
        super().__init__("Prophet", logger=logger)
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        
        self.params = {
            'seasonality_mode': seasonality_mode,
            'yearly_seasonality': yearly_seasonality,
            'weekly_seasonality': weekly_seasonality,
            'daily_seasonality': daily_seasonality
        }
        
    def fit(self, data: pd.DataFrame, **kwargs):
        """训练Prophet模型"""
        try:
            from prophet import Prophet
            
            # 创建Prophet模型
            self.model = Prophet(
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality
            )
            
            # 拟合模型
            self.model.fit(data)
            self.is_fitted = True
            
            if self.logger:
                self.logger.info("Prophet model fitted successfully")
                
        except ImportError:
            if self.logger:
                self.logger.error("Prophet not available, install with: pip install prophet")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to fit Prophet model: {e}")
            raise
            
        return self
        
    def predict(self, future_periods: int = 1, **kwargs) -> pd.DataFrame:
        """预测"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        try:
            # 创建未来时间点
            future = self.model.make_future_dataframe(periods=future_periods)
            
            # 预测
            forecast = self.model.predict(future)
            
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(future_periods)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Prediction failed: {e}")
            raise


def create_time_series_model(model_type: str, **kwargs) -> BaseTimeSeriesModel:
    """创建时间序列模型工厂函数"""
    models = {
        'arima': ARIMAModel,
        'lstm': LSTMModel,
        'gru': GRUModel,
        'var': VARModel,
        'prophet': ProphetModel
    }
    
    if model_type.lower() not in models:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return models[model_type.lower()](**kwargs)
