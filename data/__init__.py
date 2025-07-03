#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据管理模块

提供数据获取、存储、预处理和管理功能。
支持多种数据源和数据格式。
"""

from .data_manager import (
    DataManager,
    DataSource,
    AkshareDataSource,
    TushareDataSource,
    DataPreprocessor,
    DataCache,
    UniverseManager
)

__all__ = [
    "DataManager",
    "DataSource",
    "AkshareDataSource", 
    "TushareDataSource",
    "DataPreprocessor",
    "DataCache",
    "UniverseManager"
] 