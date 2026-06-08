# DrugCLIP Web Application 使用指南

## 📋 目录

- [项目介绍](#项目介绍)
- [功能说明](#功能说明)
- [安装部署](#安装部署)
- [API 文档](#api-文档)
- [使用示例](#使用示例)
- [常见问题](#常见问题)

## 项目介绍

**DrugCLIP** 是一个基于深度学习的虚拟药物筛选 Web 平台。它使用对比学习方法同时学习蛋白质和分子的 3D 结构表示，用于快速预测分子与蛋白质的结合亲和力。

该项目发表于 **NeurIPS 2023**，论文地址：[arXiv:2310.06367](https://arxiv.org/pdf/2310.06367.pdf)

## 功能说明

### 1. 单分子预测
- **输入**：分子 SMILES 字符串 + 蛋白质口袋 ID
- **输出**：结合分数、结合概率、置信度
- **用途**：快速预测单个分子与特定蛋白质的结合亲和力

### 2. 批量预测
- **输入**：SMILES 文件（支持 .smi、.txt、.smiles 格式）+ 蛋白质口袋 ID
- **输出**：分子排序列表、结合分数排行
- **用途**：从化合物库中快速筛选有前景的候选药物

### 3. 结果分析
- **功能**：
  - 统计信息（总数、最高分、最低分、平均分）
  - 评分排行图表
  - 结合概率分布
  - 评分分布直方图
- **用途**：深入分析批量预测结果

### 4. 数据导出
- **支持格式**：CSV、JSON
- **用途**：将结果导出用于后续分析

## 安装部署

### 前置要求
- Python 3.8+
- PyTorch 2.0+（推荐 GPU 版本）
- CUDA 11.7+（如使用 GPU）

### 本地开发

#### 1. 初始化项目
```bash
cd /workspaces/DrugCLIP
python web_app/setup.py
```

#### 2. 安装依赖
```bash
cd web_app/backend
pip install -r requirements.txt
```

#### 3. 准备数据和模型
从 [Google Drive](https://drive.google.com/drive/folders/1zW1MGpgunynFxTKXC2Q4RgWxZmg6CInV?usp=sharing) 下载：
- `checkpoint_best.pt` → `checkpoints/`
- `train_no_test_af.zip` 中的数据文件 → `data/`

#### 4. 启动服务
```bash
python app.py
```

打开浏览器访问：`http://localhost:5000`

### Docker 部署

#### 1. 构建镜像
```bash
cd /workspaces/DrugCLIP/web_app
docker-compose up --build
```

#### 2. 访问应用
```
http://localhost:5000
```

#### 3. 停止服务
```bash
docker-compose down
```

## API 文档

### 1. 健康检查

**端点**：`GET /api/health`

**响应**：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-06-08T10:30:00"
}
```

### 2. 单分子预测

**端点**：`POST /api/predict`

**请求体**：
```json
{
  "mol_smi": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
  "pocket_id": "1a30"
}
```

**响应**：
```json
{
  "success": true,
  "data": {
    "mol_smi": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "pocket_id": "1a30",
    "score": -3.45,
    "binding_probability": 0.92,
    "confidence": 0.85
  }
}
```

**参数说明**：
- `mol_smi`：分子 SMILES 字符串
- `pocket_id`：PDB 中蛋白质口袋 ID
- `score`：结合分数（负值表示更强的结合）
- `binding_probability`：0-1，结合概率
- `confidence`：0-1，模型置信度

### 3. 批量预测

**端点**：`POST /api/batch_predict`

**请求体**：
```json
{
  "molecules": [
    {"smi": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "name": "Ibuprofen"},
    {"smi": "CC(=O)Oc1ccccc1C(=O)O", "name": "Aspirin"}
  ],
  "pocket_id": "1a30"
}
```

**响应**：
```json
{
  "success": true,
  "results": [
    {
      "rank": 1,
      "name": "Ibuprofen",
      "smi": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
      "score": -3.45,
      "binding_probability": 0.92,
      "confidence": 0.85
    },
    {
      "rank": 2,
      "name": "Aspirin",
      "smi": "CC(=O)Oc1ccccc1C(=O)O",
      "score": -2.10,
      "binding_probability": 0.78,
      "confidence": 0.72
    }
  ],
  "timestamp": "2024-06-08T10:30:00"
}
```

### 4. 上传分子文件

**端点**：`POST /api/upload_molecules`

**请求**：
```
Content-Type: multipart/form-data
file: <SMILES file>
```

**文件格式** (.smi/.txt/.smiles)：
```
CC(C)Cc1ccc(cc1)C(C)C(=O)O Ibuprofen
CC(=O)Oc1ccccc1C(=O)O Aspirin
c1ccccc1 Benzene
```

**响应**：
```json
{
  "success": true,
  "molecules": [
    {"smi": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "name": "Ibuprofen"},
    {"smi": "CC(=O)Oc1ccccc1C(=O)O", "name": "Aspirin"}
  ],
  "count": 2
}
```

### 5. 获取示例数据

**端点**：`GET /api/examples`

**响应**：
```json
{
  "molecules": [
    {"smi": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "name": "Ibuprofen"},
    {"smi": "CC(=O)Oc1ccccc1C(=O)O", "name": "Aspirin"},
    {"smi": "c1ccccc1", "name": "Benzene"}
  ],
  "pockets": [
    {"id": "1a30", "name": "HIV Protease"},
    {"id": "1bma", "name": "Estrogen Receptor"},
    {"id": "3pbl", "name": "Kinase"}
  ]
}
```

## 使用示例

### 示例 1：预测 Ibuprofen 与 HIV 蛋白酶的结合

1. 进入"单分子预测"标签
2. 输入 SMILES：`CC(C)Cc1ccc(cc1)C(C)C(=O)O`
3. 输入口袋 ID：`1a30`
4. 点击"开始预测"
5. 查看结果：结合分数、概率、置信度

### 示例 2：批量筛选 1000 个分子

1. 创建 SMILES 文件 `candidates.smi`：
   ```
   CC(C)Cc1ccc(cc1)C(C)C(=O)O Ibuprofen
   CC(=O)Oc1ccccc1C(=O)O Aspirin
   ...（更多 SMILES）
   ```

2. 进入"批量预测"标签
3. 上传文件
4. 输入口袋 ID：`1a30`
5. 点击"开始批量预测"
6. 等待处理完成
7. 查看"结果分析"中的图表
8. 下载 CSV 或 JSON 结果

## 常见问题

### Q: 怎样获取蛋白质口袋 ID？
**A**: 可以从 [PDB 数据库](https://www.rcsb.org/) 查询。例如：
- HIV 蛋白酶：1a30
- 雌激素受体：1bma
- 激酶：3pbl

### Q: 结合分数的范围是多少？
**A**: 通常为 -10 到 +10。负值表示更强的结合亲和力，即分子与蛋白质结合能力更强。

### Q: 如何生成分子的 SMILES？
**A**: 可以使用 RDKit、ChemDraw 或在线工具如 [SMILES translator](https://smiles.surya.me/) 转换分子结构。

### Q: 支持哪些文件格式？
**A**: 支持以下分子文件格式：
- `.smi` - SMILES 文件
- `.txt` - 文本文件（每行一个 SMILES）
- `.smiles` - SMILES 文件

### Q: 预测需要多长时间？
**A**: 
- 单分子预测：< 1 秒
- 1000 个分子批量预测：约 2-5 分钟（取决于硬件）

### Q: 可以使用 CPU 进行预测吗？
**A**: 可以，但速度较慢。编辑 `.env` 文件，将 `DEVICE` 改为 `cpu`。

### Q: 如何提高预测速度？
**A**: 
1. 使用 GPU（比 CPU 快 10-50 倍）
2. 增加 BATCH_SIZE 参数（取决于显存）
3. 使用 FP16 精度

### Q: 预测结果的准确性如何？
**A**: DrugCLIP 模型在 DUD-E 数据集上的 BEDROC 分数为 0.65+，是当前最先进水平之一。但建议结果用于初步筛选，最终需要实验验证。

## 联系方式

- 论文：[arXiv:2310.06367](https://arxiv.org/pdf/2310.06367.pdf)
- GitHub：[bowen-gao/DrugCLIP](https://github.com/bowen-gao/DrugCLIP)
- Email：billgao0111@gmail.com

## 许可证

MIT License - 详见 LICENSE 文件
