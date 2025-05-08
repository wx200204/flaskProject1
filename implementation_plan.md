# 项目优化实施计划

根据对项目的分析，我们将按照以下步骤优化项目结构和代码：

## 第一阶段：目录结构重组

### 步骤1：创建新的目录结构

```bash
# 创建新的目录结构
mkdir -p data/models data/uploads data/debug 
mkdir -p tests
```

### 步骤2：清理冗余文件

1. 删除空文件：
   ```bash
   rm geometry.py
   rm app/models/spine_model.pth
   ```

2. 整合配置文件：
   - 将根目录的config.py内容合并到app/config.py中
   - 删除根目录的config.py

### 步骤3：整合重复目录

1. 移动文件：
   ```bash
   # 将根目录的models文件移动到data/models目录
   mv models/spine_model.pth data/models/
   
   # 移动debug文件到data/debug
   mkdir -p data/debug/images
   mv debug/*.jpg debug/*.png data/debug/images/
   
   # 整合uploads目录
   mv uploads/* data/uploads/
   ```

2. 删除空目录：
   ```bash
   rm -rf models
   rm -rf debug
   rm -rf examples
   rm -rf uploads
   ```

3. 更新文件路径引用：
   - 修改app.py中模型路径的引用
   - 修改app/__init__.py中的目录配置

## 第二阶段：代码优化

### 步骤1：拆分routes.py

根据功能将app/routes.py拆分为以下文件：

1. **app/routes/__init__.py**：初始化蓝图
2. **app/routes/main.py**：主页和基础路由
3. **app/routes/analysis.py**：图像分析相关API
4. **app/routes/models.py**：模型管理相关API
5. **app/routes/system.py**：系统监控相关API
6. **app/routes/rehab.py**：康复训练相关API

### 步骤2：重构routes/__init__.py文件

```python
# app/routes/__init__.py
from flask import Blueprint

bp = Blueprint('api', __name__)

from . import main
from . import analysis  
from . import models
from . import system
from . import rehab
```

### 步骤3：实现各个路由模块

1. **main.py**：移动主页和基础路由
2. **analysis.py**：移动图像分析相关API
3. **models.py**：移动模型管理相关API
4. **system.py**：移动系统监控相关API
5. **rehab.py**：移动康复训练相关API

### 步骤4：更新应用入口

更新app/__init__.py中蓝图注册方式：

```python
# 原代码
from .routes import bp
app.register_blueprint(bp)

# 修改为
from .routes import bp
app.register_blueprint(bp)
```

## 第三阶段：优化模型管理

### 步骤1：更新模型路径

修改app.py中的模型路径引用：

```python
# 原代码
model_path = models_dir / 'spine_model.pth'

# 修改为
model_path = Path(__file__).parent / 'data' / 'models' / 'spine_model.pth'
```

### 步骤2：统一配置

更新app/config.py，整合所有配置项并使用环境变量：

```python
import os
from pathlib import Path

class Config:
    """应用配置类"""
    # 基础目录
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-please-change-in-production'
    DEBUG = os.environ.get('DEBUG_MODE', 'True').lower() == 'true'
    
    # 目录配置 - 使用新的目录结构
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    UPLOAD_DIR = os.path.join(DATA_DIR, 'uploads')
    MODEL_DIR = os.path.join(DATA_DIR, 'models')
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    DEBUG_DIR = os.path.join(DATA_DIR, 'debug')
    
    # 康复训练相关目录
    REHAB_IMAGES_DIR = os.path.join(UPLOAD_DIR, 'rehab_images')
    REHAB_VIDEOS_DIR = os.path.join(UPLOAD_DIR, 'rehab_videos')
    
    # 上传文件配置
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'} 
```

## 第四阶段：添加文档和清理机制

### 步骤1：创建README.md

```markdown
# 脊柱侧弯分析系统

## 概述
脊柱侧弯分析系统是一个基于Flask的Web应用，提供脊柱X光片分析和康复训练功能。

## 功能特点
- 脊柱X光片分析
- Cobb角计算
- 康复训练指导
- 康复效果评估

## 目录结构
```

### 步骤2：添加debug文件清理机制

在app/utils/目录下添加cleanup.py:

```python
# app/utils/cleanup.py
import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta

def cleanup_debug_files(debug_dir, max_age_days=7):
    """清理超过指定天数的debug文件"""
    if not os.path.exists(debug_dir):
        return
        
    now = datetime.now()
    cutoff_date = now - timedelta(days=max_age_days)
    
    for root, _, files in os.walk(debug_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
            if file_modified < cutoff_date:
                os.remove(file_path)
                
def cleanup_all_debug_files(debug_dir):
    """清空debug目录的所有文件"""
    if os.path.exists(debug_dir):
        for item in os.listdir(debug_dir):
            item_path = os.path.join(debug_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
```

### 步骤3：添加.gitignore文件

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# 虚拟环境
venv/
ENV/

# IDE
.idea/
.vscode/

# 数据和日志文件
data/debug/
data/uploads/
logs/
instance/
```

## 执行顺序

1. **备份项目**：在开始重构前，先备份整个项目
2. **执行第一阶段**：重组目录结构
3. **执行第二阶段**：拆分和优化代码
4. **执行第三阶段**：更新模型管理
5. **执行第四阶段**：添加文档和清理机制
6. **测试应用**：确保所有功能正常工作 