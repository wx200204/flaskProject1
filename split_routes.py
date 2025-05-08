#!/usr/bin/env python3
"""
脊柱侧弯分析系统 - 路由拆分脚本
此脚本将大型routes.py文件拆分为多个小模块文件
"""

import os
import re
import sys

def read_file(filename):
    """读取文件内容"""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.readlines()

def write_file(filename, content):
    """写入文件内容"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(content)

def extract_imports(lines):
    """提取import语句"""
    imports = []
    for line in lines:
        if line.strip().startswith(('import ', 'from ')):
            imports.append(line)
    return imports

def extract_route_sections(lines):
    """根据路由装饰器提取路由函数段落"""
    sections = {}
    current_section = None
    current_content = []
    
    # 合并所有行为一个字符串，便于使用正则表达式
    content = ''.join(lines)
    
    # 使用正则表达式匹配所有路由函数
    route_pattern = r'(@bp\.route\([^\)]+\)(?:\s*@[^\n]+\n)*\s*def\s+([a-zA-Z0-9_]+)\([^\)]*\):(?:.|\n)*?)(?=@bp\.route|$)'
    matches = re.finditer(route_pattern, content, re.MULTILINE)
    
    for match in matches:
        route_func = match.group(0)
        func_name = match.group(2)
        
        # 根据路由URL确定分类
        if '/api/v1/analyze' in route_func:
            section_type = 'analysis'
        elif '/api/v1/models' in route_func or '/models/' in route_func:
            section_type = 'models'
        elif '/rehab' in route_func:
            section_type = 'rehab'
        elif '/api/v1/system' in route_func:
            section_type = 'system'
        elif func_name in ['index', 'status']:
            section_type = 'main'
        else:
            section_type = 'other'
        
        if section_type not in sections:
            sections[section_type] = []
        
        sections[section_type].append(route_func)
    
    return sections

def extract_helper_functions(lines):
    """提取辅助函数"""
    helpers = []
    current_helper = None
    collecting = False
    
    for line in lines:
        # 辅助函数定义（非路由处理函数）
        if re.match(r'^def\s+([a-zA-Z0-9_]+)\(', line) and not collecting:
            # 确保这不是路由处理函数
            if not any(prev_line.strip().startswith('@bp.route') for prev_line in lines[max(0, lines.index(line)-3):lines.index(line)]):
                collecting = True
                current_helper = [line]
        elif collecting:
            current_helper.append(line)
            # 如果遇到新的函数定义或类，结束当前函数收集
            if (line.strip() == '' and 
                (lines.index(line)+1 < len(lines)) and 
                (lines[lines.index(line)+1].startswith('def ') or 
                 lines[lines.index(line)+1].startswith('class '))):
                collecting = False
                helpers.append(''.join(current_helper))
                current_helper = None
    
    # 处理最后一个辅助函数
    if current_helper:
        helpers.append(''.join(current_helper))
    
    return helpers

def categorize_helpers(helpers, routes_sections):
    """将辅助函数分类到各个模块"""
    helper_sections = {
        'analysis': [],
        'models': [],
        'system': [],
        'rehab': [],
        'main': []
    }
    
    # 使用已知模式匹配辅助函数
    for helper in helpers:
        helper_name = re.search(r'def\s+([a-zA-Z0-9_]+)\(', helper).group(1)
        
        if helper_name in ['get_analyzer']:
            helper_sections['analysis'].append(helper)
        elif helper_name in ['get_model_manager']:
            helper_sections['models'].append(helper)
        elif helper_name in ['get_rehab_manager', 'get_pose_detector', 'get_camera_frame']:
            helper_sections['rehab'].append(helper)
        elif helper_name in ['get_system_info', 'check_health']:
            helper_sections['system'].append(helper)
        else:
            # 通过函数体内容推测分类
            found = False
            for section_name, section_routes in routes_sections.items():
                for route in section_routes:
                    if helper_name in route:
                        helper_sections[section_name].append(helper)
                        found = True
                        break
                if found:
                    break
            
            # 如果无法匹配，放入main模块
            if not found:
                helper_sections['main'].append(helper)
    
    return helper_sections

def generate_module_files(imports, route_sections, helper_sections):
    """生成各个模块文件"""
    module_templates = {
        'main': "# app/routes/main.py\n{imports}\nfrom . import bp\n\n{helpers}\n{routes}\n",
        'analysis': "# app/routes/analysis.py\n{imports}\nfrom . import bp\n\n{helpers}\n{routes}\n",
        'models': "# app/routes/models.py\n{imports}\nfrom . import bp\n\n{helpers}\n{routes}\n",
        'system': "# app/routes/system.py\n{imports}\nfrom . import bp\n\n{helpers}\n{routes}\n",
        'rehab': "# app/routes/rehab.py\n{imports}\nfrom . import bp\n\n{helpers}\n{routes}\n"
    }
    
    # 确保目录存在
    os.makedirs('app/routes', exist_ok=True)
    
    # 生成各个模块文件
    for module_name, template in module_templates.items():
        helpers_content = ''.join(helper_sections.get(module_name, []))
        routes_content = ''.join(route_sections.get(module_name, []))
        
        if not routes_content and not helpers_content:
            continue
        
        content = template.format(
            imports=''.join(imports),
            helpers=helpers_content,
            routes=routes_content
        )
        
        # 移除重复的空行
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # 写入文件
        filename = f'app/routes/{module_name}.py'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"已创建模块文件: {filename}")

def create_init_file():
    """创建或更新__init__.py文件"""
    init_content = """# app/routes/__init__.py
from flask import Blueprint

bp = Blueprint('api', __name__)

# 导入所有路由模块
from . import main
from . import analysis
from . import models
from . import system
from . import rehab
"""
    
    init_file = 'app/routes/__init__.py'
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write(init_content)
    
    print(f"已创建/更新初始化文件: {init_file}")

def process_others_section(route_sections, helper_sections):
    """处理其他未分类的路由"""
    if 'other' in route_sections and route_sections['other']:
        print("\n发现未分类的路由函数:")
        for route in route_sections['other']:
            func_name = re.search(r'def\s+([a-zA-Z0-9_]+)\(', route).group(1)
            print(f" - {func_name}")
        
        print("\n请手动检查这些路由并将它们添加到适当的模块中。")

def parse_args():
    """解析命令行参数"""
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("用法: python split_routes.py [routes_file]")
        print("默认路由文件: app/routes.py")
        sys.exit(0)
    
    routes_file = sys.argv[1] if len(sys.argv) > 1 else 'app/routes.py'
    return routes_file

def main():
    """主函数"""
    print("=" * 60)
    print("脊柱侧弯分析系统 - 路由拆分脚本")
    print("=" * 60)
    
    routes_file = parse_args()
    
    if not os.path.exists(routes_file):
        print(f"错误: 路由文件 {routes_file} 不存在")
        return 1
    
    print(f"正在处理路由文件: {routes_file}")
    
    try:
        # 读取路由文件
        lines = read_file(routes_file)
        
        # 提取import语句
        imports = extract_imports(lines)
        
        # 提取路由函数
        route_sections = extract_route_sections(lines)
        
        # 提取辅助函数
        helpers = extract_helper_functions(lines)
        
        # 将辅助函数分类
        helper_sections = categorize_helpers(helpers, route_sections)
        
        # 生成模块文件
        generate_module_files(imports, route_sections, helper_sections)
        
        # 创建__init__.py
        create_init_file()
        
        # 处理未分类的路由
        process_others_section(route_sections, helper_sections)
        
        print("\n=" * 60)
        print("路由拆分完成！")
        print("=" * 60)
        print("\n注意：")
        print("1. 请检查生成的模块文件，确保所有路由和辅助函数都正确分类")
        print("2. 确保import语句正确且完整")
        print("3. 更新app/__init__.py中的蓝图导入")
        
    except Exception as e:
        print(f"\n错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 