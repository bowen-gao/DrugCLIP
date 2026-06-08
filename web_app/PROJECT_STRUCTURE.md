# DrugCLIP Web Application 项目结构

## 📁 目录结构

```
web_app/
├── backend/                 # Flask 后端应用
│   ├── app.py              # 主应用文件
│   ├── config.py           # 配置管理
│   ├── model_service.py    # 模型推理服务
│   └── requirements.txt    # Python 依赖
│
├── frontend/               # 前端应用
│   ├── templates/
│   │   └── index.html      # 主页面（HTML + Vue）
│   └── static/
│       ├── style.css       # 样式表
│       └── main.js         # 主逻辑脚本
│
├── Dockerfile             # Docker 镜像配置
├── docker-compose.yml     # Docker Compose 编排
├── setup.py              # 初始化脚本
├── quickstart.sh         # Linux/Mac 快速启动
├── quickstart.bat        # Windows 快速启动
├── README.md             # 使用文档
├── PROJECT_STRUCTURE.md  # 本文件
└── .gitignore           # Git 忽略配置
```

## 🚀 启动方式

### 方式 1: 快速启动脚本（推荐）

#### Linux/Mac:
```bash
chmod +x web_app/quickstart.sh
./web_app/quickstart.sh
```

#### Windows:
```cmd
web_app\quickstart.bat
```

### 方式 2: 手动启动

```bash
# 进入后端目录
cd web_app/backend

# 安装依赖
pip install -r requirements.txt

# 启动应用
python app.py
```

### 方式 3: Docker 启动

```bash
cd web_app
docker-compose up --build
```

## 📝 核心文件说明

### backend/app.py
**主应用文件** - Flask 应用主体

主要功能：
- `/` - 主页面
- `/api/health` - 健康检查
- `/api/predict` - 单分子预测
- `/api/batch_predict` - 批量预测
- `/api/upload_molecules` - 分子文件上传
- `/api/examples` - 获取示例数据

### backend/config.py
**配置管理** - 应用配置

配置项：
- Flask 设置（HOST、PORT、DEBUG）
- 模型设置（MODEL_PATH、DEVICE）
- 数据路径（LMDB、缓存目录）
- 文件上传限制
- 批处理参数

### backend/model_service.py
**模型服务** - 推理逻辑

核心类：
- `ModelService` - 模型加载和推理
- `MockDrugCLIPModel` - 测试用模型

主要方法：
- `predict()` - 单分子预测
- `batch_predict()` - 批量预测
- `_mol_to_features()` - 分子特征提取
- `_pocket_to_features()` - 口袋特征提取

### frontend/templates/index.html
**主页面** - Web 界面

包含内容：
- 四个功能标签页
- 表单输入组件
- 结果展示区域
- 帮助文档

### frontend/static/style.css
**样式表** - 美化界面

主要样式：
- 响应式布局
- 暗色/亮色主题
- 图表容器样式
- 表格和表单样式
- 移动设备适配

### frontend/static/main.js
**主逻辑脚本** - 交互逻辑

核心功能：
- API 调用处理
- 标签页切换
- 单分子预测流程
- 批量预测流程
- 结果展示和图表
- 数据导出（CSV/JSON）

## 🔌 API 端点一览

| 方法 | 端点 | 功能 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| POST | `/api/predict` | 单分子预测 |
| POST | `/api/batch_predict` | 批量预测 |
| POST | `/api/upload_molecules` | 上传分子文件 |
| GET | `/api/examples` | 获取示例 |

详细文档见 [README.md](README.md)

## 📋 数据流

### 单分子预测流程
```
用户输入 (SMILES, Pocket ID)
    ↓
前端验证
    ↓
POST /api/predict
    ↓
后端加载特征
    ↓
模型推理
    ↓
返回结果 (score, probability, confidence)
    ↓
前端展示结果和图表
```

### 批量预测流程
```
用户上传 SMILES 文件
    ↓
解析文件
    ↓
显示分子列表预览
    ↓
用户选择 Pocket ID
    ↓
POST /api/batch_predict
    ↓
后端逐个预测
    ↓
返回排序结果列表
    ↓
前端生成表格和图表
    ↓
用户导出 CSV/JSON
```

## 🛠️ 自定义和扩展

### 修改配置
编辑 `.env` 文件：
```bash
FLASK_PORT=5001              # 改变端口
DEVICE=cpu                   # 使用 CPU
BATCH_SIZE=64               # 改变批处理大小
```

### 集成真实模型
在 `model_service.py` 中修改 `_load_model()`:
```python
# 替换 MockDrugCLIPModel 为实际模型
from unimol.models import build_model
checkpoint = torch.load(self.model_path)
self.model = build_model(checkpoint['args'])
self.model.load_state_dict(checkpoint['model'])
```

### 添加新的 API 端点
在 `app.py` 中添加路由：
```python
@app.route('/api/my_endpoint', methods=['POST'])
def my_endpoint():
    # 实现逻辑
    return jsonify({'result': 'success'})
```

### 修改前端样式
编辑 `frontend/static/style.css` 中的 CSS 变量：
```css
:root {
    --primary-color: #3b82f6;  # 改变主色
    --text-color: #1f2937;      # 改变文字颜色
}
```

## 🐛 调试

### 查看后端日志
```bash
# 启用调试模式
FLASK_ENV=development python app.py
```

### 浏览器控制台
打开浏览器的开发者工具（F12），查看：
- 控制台错误和日志
- 网络请求和响应
- 性能分析

### 测试 API
使用 curl 或 Postman：
```bash
# 测试单个预测
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"mol_smi":"CCO","pocket_id":"1a30"}'
```

## 📊 性能优化

### 后端优化
- 使用 GPU 推理（设置 DEVICE=cuda）
- 增加 BATCH_SIZE 参数
- 启用模型缓存

### 前端优化
- 图表库使用 ECharts（高效）
- 异步加载数据
- 支持大文件上传（16MB）

## 🔐 安全建议

1. **生产环境**：
   - 关闭 DEBUG 模式
   - 使用环境变量存储敏感信息
   - 添加身份验证（如果需要）
   - 使用 HTTPS

2. **输入验证**：
   - 验证 SMILES 格式
   - 限制文件大小
   - 防止 SQL 注入（如使用数据库）

3. **资源限制**：
   - 限制批量预测大小
   - 设置请求超时
   - 防止 DDoS 攻击

## 📚 相关资源

- **官方论文**: https://arxiv.org/pdf/2310.06367.pdf
- **GitHub**: https://github.com/bowen-gao/DrugCLIP
- **数据下载**: https://drive.google.com/drive/folders/1zW1MGpgunynFxTKXC2Q4RgWxZmg6CInV
- **PDB 数据库**: https://www.rcsb.org/
- **RDKit 文档**: https://www.rdkit.org/
- **Flask 文档**: https://flask.palletsprojects.com/

## 📞 技术支持

如有问题，请：
1. 查看 [README.md](README.md) 中的常见问题
2. 检查后端日志
3. 查看浏览器控制台错误
4. 联系开发者：billgao0111@gmail.com

## 📝 版本历史

- **v1.0** (2024-06-08) - 初始版本
  - 单分子预测
  - 批量预测
  - 结果分析
  - Web 界面
  - Docker 支持
