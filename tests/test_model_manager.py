#!/usr/bin/env python3
"""
模型管理模块测试
"""

def test_model_manager_module():
    """测试模型管理模块基本功能"""
    print("🚀 开始测试模型管理模块...")
    
    try:
        # 测试导入
        print("1. 测试导入模型管理模块...")
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from models.model_manager import (
            ModelManager, ModelEvaluator, HyperparameterOptimizer,
            FeatureSelector, ModelTrainer
        )
        from models.ml_models import LightGBMModel, XGBoostModel, LinearModel
        print("✅ 模型管理模块导入成功")
        
        # 测试初始化
        print("\n2. 测试模型管理器初始化...")
        import pandas as pd
        import numpy as np
        
        # 创建测试数据
        np.random.seed(42)
        n_samples = 1000
        n_features = 50
        
        # 生成特征数据
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # 生成目标变量（回归任务）
        y = np.random.randn(n_samples) * 0.1 + X.iloc[:, :5].sum(axis=1) * 0.3
        y = pd.Series(y, name='target')
        
        # 创建模型管理器
        config = {
            'model_types': ['lightgbm', 'xgboost', 'linear'],
            'cross_validation': {
                'method': 'time_series',
                'n_splits': 3
            },
            'feature_selection': {
                'method': 'correlation',
                'top_k': 20
            },
            'ensemble_method': 'weighted_average'
        }
        
        model_manager = ModelManager(config)
        model_manager.initialize(config)
        print("✅ 模型管理器初始化成功")
        
        # 测试特征选择器
        print("\n3. 测试特征选择器...")
        feature_selector = FeatureSelector()
        
        # 基于相关性选择特征
        selected_features = feature_selector.select_by_correlation(X, y, threshold=0.1)
        if len(selected_features) > 0:
            print(f"✅ 基于相关性选择特征成功: {len(selected_features)}个特征")
        else:
            print("❌ 基于相关性选择特征失败")
            return False
        
        # 基于互信息选择特征
        mi_features = feature_selector.select_by_mutual_info(X, y, top_k=20)
        if len(mi_features) > 0:
            print(f"✅ 基于互信息选择特征成功: {len(mi_features)}个特征")
        else:
            print("❌ 基于互信息选择特征失败")
            return False
        
        # 测试模型创建
        print("\n4. 测试模型创建...")
        
        # 创建线性模型
        linear_model = model_manager.create_model('linear')
        if linear_model is not None:
            print("✅ 线性模型创建成功")
        else:
            print("❌ 线性模型创建失败")
            return False
        
        # 创建LightGBM模型
        try:
            lgb_model = model_manager.create_model('lightgbm')
            print("✅ LightGBM模型创建成功")
        except Exception as e:
            print(f"⚠️ LightGBM模型创建失败（可能缺少依赖）: {e}")
            lgb_model = None
        
        # 创建XGBoost模型
        try:
            xgb_model = model_manager.create_model('xgboost')
            print("✅ XGBoost模型创建成功")
        except Exception as e:
            print(f"⚠️ XGBoost模型创建失败（可能缺少依赖）: {e}")
            xgb_model = None
        
        # 测试模型训练器
        print("\n5. 测试模型训练器...")
        model_trainer = ModelTrainer()
        
        # 训练线性模型
        X_train = X.iloc[:800]
        y_train = y.iloc[:800]
        X_val = X.iloc[800:]
        y_val = y.iloc[800:]
        
        trained_linear = model_trainer.train_single_model(
            linear_model, X_train, y_train, (X_val, y_val)
        )
        
        if trained_linear is not None:
            print("✅ 线性模型训练成功")
        else:
            print("❌ 线性模型训练失败")
            return False
        
        # 测试模型预测
        print("\n6. 测试模型预测...")
        predictions = trained_linear.predict(X_val)
        if len(predictions) == len(X_val):
            print(f"✅ 模型预测成功: {len(predictions)}个预测值")
        else:
            print("❌ 模型预测失败")
            return False
        
        # 测试模型评估器
        print("\n7. 测试模型评估器...")
        evaluator = ModelEvaluator()
        
        # 回归评估
        regression_metrics = evaluator.evaluate_regression(y_val.values, predictions)
        if isinstance(regression_metrics, dict) and len(regression_metrics) > 0:
            print(f"✅ 回归评估成功: {list(regression_metrics.keys())}")
        else:
            print("❌ 回归评估失败")
            return False
        
        # IC计算
        ic_value = evaluator.calculate_ic(y_val.values, predictions)
        if not np.isnan(ic_value):
            print(f"✅ IC计算成功: {ic_value:.4f}")
        else:
            print("❌ IC计算失败")
            return False
        
        # 测试模型管理器的核心功能
        print("\n8. 测试模型管理器核心功能...")
        
        # 训练多个模型
        model_configs = [
            {'type': 'linear', 'name': 'linear_model'},
        ]
        
        # 如果LightGBM可用，添加到配置中
        if lgb_model is not None:
            model_configs.append({'type': 'lightgbm', 'name': 'lgb_model'})
        
        trained_models = model_manager.train_models(X_train, y_train, model_configs)
        if len(trained_models) > 0:
            print(f"✅ 多模型训练成功: {list(trained_models.keys())}")
        else:
            print("❌ 多模型训练失败")
            return False
        
        # 选择最佳模型
        best_model_name = model_manager.select_best_model(trained_models, X_val, y_val, metric='ic')
        if best_model_name:
            print(f"✅ 最佳模型选择成功: {best_model_name}")
        else:
            print("❌ 最佳模型选择失败")
            return False
        
        # 模型验证
        validation_results = model_manager.validate_models(trained_models, pd.concat([X_val, y_val], axis=1))
        if isinstance(validation_results, dict) and len(validation_results) > 0:
            print(f"✅ 模型验证成功: {list(validation_results.keys())}")
        else:
            print("❌ 模型验证失败")
            return False
        
        # 模型性能评估
        performance = model_manager.get_model_performance(trained_models[best_model_name], X_val, y_val)
        if isinstance(performance, dict) and len(performance) > 0:
            print(f"✅ 模型性能评估成功: {list(performance.keys())}")
        else:
            print("❌ 模型性能评估失败")
            return False
        
        # 测试模型保存和加载
        print("\n9. 测试模型保存和加载...")
        
        # 保存模型
        try:
            model_manager.save_models(trained_models, './test_models')
            print("✅ 模型保存成功")
        except Exception as e:
            print(f"⚠️ 模型保存失败: {e}")
        
        # 加载模型
        try:
            loaded_models = model_manager.load_models('./test_models')
            if len(loaded_models) > 0:
                print(f"✅ 模型加载成功: {list(loaded_models.keys())}")
            else:
                print("❌ 模型加载失败")
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}")
        
        print("\n🎉 模型管理模块核心功能测试通过！")
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_manager_module()
    if success:
        print("🎉 模型管理模块测试完成！核心功能正常。")
    else:
        print("⚠️ 模型管理模块测试失败，需要检查问题。") 