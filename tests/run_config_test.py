#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行配置测试的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 直接导入并运行测试函数
from test_config_validation import test_config_functionality

if __name__ == "__main__":
    print("🚀 开始运行配置模块测试...")
    success = test_config_functionality()
    
    if success:
        print("\n✅ 配置模块测试全部通过！")
    else:
        print("\n❌ 配置模块测试存在问题，需要修复。")
    
    print("\n📋 测试完成，继续后续模块的测试...") 