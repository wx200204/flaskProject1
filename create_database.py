#!/usr/bin/env python3
"""
脊柱侧弯分析系统 - 数据库初始化脚本
此脚本用于创建数据库和表
"""

import pymysql
import os
from pathlib import Path
import sys

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '200204'  # 尝试空密码
}

def create_database():
    """创建数据库"""
    try:
        # 连接MySQL服务器
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 创建数据库
        cursor.execute("CREATE DATABASE IF NOT EXISTS spine_analysis CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        
        print("数据库创建成功: spine_analysis")
        
        # 关闭连接
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        import traceback
        print(f"创建数据库失败: {str(e)}")
        print(f"详细错误信息:")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("脊柱侧弯分析系统 - 数据库初始化")
    
    # 创建数据库
    if not create_database():
        print("数据库初始化失败")
        return
    
    print("\n数据库初始化成功")
    print("\n现在可以启动应用程序，系统将自动创建表结构")
    print("请确保在app/config.py中配置了正确的数据库连接信息")

if __name__ == "__main__":
    main() 