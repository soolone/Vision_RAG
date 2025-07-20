# Vision RAG System

一个基于多模态RAG（检索增强生成）技术的智能文档问答系统，支持PDF、图片、Office文档等多种格式的文档解析、向量化存储和智能问答。

## 🌟 主要特性

### 📄 多格式文档支持
- **PDF文档**: 支持复杂PDF的解析，包括文本、图片、表格、公式等
- **图片文件**: 支持JPG、PNG、BMP、TIFF、WebP等格式
- **Office文档**: 支持Word (.docx, .doc)、PowerPoint (.pptx, .ppt)、Excel (.xlsx, .xls)
- **自动格式识别**: 智能识别文件类型并选择最佳解析策略

### 🤖 智能解析与处理
- **MinerU 2.0集成**: 使用先进的文档解析引擎，精确提取文档结构
- **多模态内容提取**: 同时处理文本、图片、表格、公式等多种内容类型
- **智能切片**: 基于语义边界的智能文档切片，保持内容完整性
- **自动图片描述**: 使用视觉语言模型自动生成图片描述

### 🔍 高效检索系统
- **多模态嵌入**: 使用通义千问多模态嵌入模型，支持文本和图像的统一向量化
- **ChromaDB存储**: 高性能向量数据库，支持大规模文档存储和快速检索
- **查询扩写**: 智能查询扩写功能，提高检索准确性
- **混合检索**: 支持文本检索、图像检索和多查询融合检索

### 💬 智能问答
- **多模态理解**: 基于通义千问视觉语言模型，理解文本和图像内容
- **上下文感知**: 结合检索到的相关内容，提供准确的答案
- **引用标注**: 自动标注答案来源，提高可信度
- **流式输出**: 支持实时流式回答，提升用户体验

## 🏗️ 系统架构

### 核心组件

```
Vision RAG System
├── 📁 agent/                    # 智能代理模块
│   ├── data_agent.py           # 文档解析代理
│   ├── chunk_agent.py          # 文档切片代理
│   ├── chromadb_agent.py       # 向量数据库代理
│   ├── qwen_embedding.py       # 通义千问嵌入代理
│   ├── query_agent.py          # 查询扩写代理
│   ├── answer_agent.py         # 问答代理
│   └── caption_agent.py        # 图片描述代理
├── 📁 utils/                    # 工具模块
│   ├── apis.py                 # API配置
│   └── mineru_parser.py        # MinerU解析器
├── 📁 data/                     # 数据目录
│   ├── doc_data/               # 原始文档
│   ├── image_data/             # 图片数据
│   └── output/                 # 解析输出
├── vision_rag_qwen.py          # 主应用程序（通义千问版本）
├── vision_rag.py               # 主应用程序（Cohere版本）
└── requirements.txt            # 依赖配置
```

### 数据流程

1. **文档上传** → 用户上传各种格式的文档
2. **智能解析** → DataAgent使用MinerU解析文档结构和内容
3. **内容切片** → ChunkAgent将解析结果切分为语义完整的片段
4. **向量化** → QwenEmbedding将文本和图像转换为向量表示
5. **存储索引** → ChromaDBAgent将向量存储到数据库中
6. **查询处理** → QueryAgent对用户查询进行扩写和优化
7. **相似检索** → 在向量数据库中检索相关内容
8. **智能问答** → AnswerAgent基于检索结果生成答案

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 8GB+ RAM（推荐16GB+）
- GPU支持（可选，用于加速推理）

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd Vision_RAG
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **安装MinerU**

建议参考 [MinerU官方文档](https://github.com/opendatalab/MinerU) 进行安装，以获取最新的安装指南和依赖要求。

```bash
# 安装MinerU核心组件
pip install -U 'mineru[core]'

# 如果需要GPU加速
pip install -U 'mineru[core,gpu]'
```

4. **配置API密钥**

⚠️ **重要**: 在正常使用前，您必须将 `utils/apis.py` 文件中的所有API密钥替换为您自己的有效密钥。

编辑 `utils/apis.py` 文件，配置相应的API密钥：
- **通义千问API密钥**: 替换 `"your modelscope api"` 为您的ModelScope API密钥（用于嵌入和问答）
- **Cohere API密钥**: 替换 `"your cohere api"` 为您的Cohere API密钥（可选，用于Cohere版本）
- **Google Gemini API密钥**: 替换 `"your google ai studio api"` 为您的Google AI Studio API密钥（可选，用于Gemini版本）
- **其他API密钥**: 根据需要配置其他服务的API密钥

示例配置：
```python
class Qwen25VL72BInstruct:
    def __init__(self):
        self.model = "Qwen/Qwen2.5-VL-72B-Instruct"
        self.api_key = "your_actual_modelscope_api_key_here"  # 替换为实际密钥
        self.api_base = "https://api-inference.modelscope.cn/v1/"
```

### 启动应用

```bash
# 启动通义千问版本（推荐）
streamlit run vision_rag_qwen.py

# 或启动Cohere版本
streamlit run vision_rag.py
```

应用将在 `http://localhost:8501` 启动。

## 📖 使用指南

### 1. 文档上传与解析

1. 在侧边栏选择"📄 文档解析"选项卡
2. 上传支持的文档格式（PDF、图片、Office文档）
3. 配置解析参数：
   - 语言设置（中文/英文）
   - 启用公式解析
   - 启用表格解析
   - 自动图片描述
4. 点击"开始解析"按钮
5. 查看解析结果和统计信息

### 2. 知识库管理

1. 在"📚 知识库管理"选项卡中查看已解析的文档
2. 浏览文档切片，包括文本、图片、表格等
3. 管理文档：删除不需要的文档或切片
4. 查看数据库统计信息

### 3. 智能问答

1. 在主界面输入问题
2. 系统自动检索相关内容
3. 查看智能生成的答案和引用来源
4. 浏览检索到的相关图片和文档片段

### 4. 高级功能

- **查询扩写**: 系统自动扩写短查询，提高检索准确性
- **多模态检索**: 同时检索文本和图像内容
- **相似度调节**: 调整检索结果的相似度阈值
- **批量处理**: 支持批量上传和处理多个文档

## ⚙️ 配置说明

### 解析配置

```python
@dataclass
class VisionRAGConfig:
    # 数据解析配置
    output_dir: str = "data/output"  # 解析输出目录
    cls_dir: str = "default"         # 知识库分类目录
    lang: str = "ch"                 # 解析语言
    enable_formula: bool = True      # 启用公式解析
    enable_table: bool = True        # 启用表格解析
    auto_caption: bool = True        # 自动生成图片描述
    
    # 切片配置
    max_chunk_size: int = 1000       # 最大切片大小
    min_chunk_size: int = 100        # 最小切片大小
    overlap_size: int = 100          # 重叠大小
    preserve_sentences: bool = True  # 保持句子完整性
    preserve_paragraphs: bool = True # 保持段落完整性
```

### API配置

在 `utils/apis.py` 中配置各种API密钥和端点：

- **Qwen25VL72BInstruct**: 通义千问视觉语言模型
- **Qwen3_235B_A22B**: 通义千问大语言模型
- **AliBailian_API**: 阿里百炼API
- **Cohere_API**: Cohere API
- **Google_API**: Google Gemini API

## 🔧 技术栈

### 核心技术

- **[MinerU 2.0](https://github.com/opendatalab/MinerU)**: 先进的文档解析引擎
- **[ChromaDB](https://www.trychroma.com/)**: 高性能向量数据库
- **[LangChain](https://langchain.com/)**: LLM应用开发框架
- **[LangGraph](https://langchain-ai.github.io/langgraph/)**: 工作流编排框架
- **[Streamlit](https://streamlit.io/)**: Web应用框架

### AI模型

- **通义千问多模态嵌入**: 文本和图像的统一向量化
- **通义千问视觉语言模型**: 多模态理解和问答
- **Cohere Embed-4**: 企业级多模态嵌入模型（可选）
- **Google Gemini**: 多模态大语言模型（可选）

## 📁 重要文件说明

### `/utils/mineru_parser.py`

本文件来源于 [RAG-Anything项目](https://github.com/HKUDS/RAG-Anything/blob/main/raganything/mineru_parser.py)，是MinerU文档解析器的封装工具，提供了统一的文档解析接口。

**主要功能**:
- PDF文档解析
- 图像文档解析  
- 结构化数据提取
- Markdown和JSON格式输出

**使用许可**: 请遵循原项目的开源许可证

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出改进建议！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [MinerU](https://github.com/opendatalab/MinerU) - 优秀的文档解析引擎
- [RAG-Anything](https://github.com/HKUDS/RAG-Anything) - MinerU解析器封装
- [ChromaDB](https://www.trychroma.com/) - 高性能向量数据库
- [LangChain](https://langchain.com/) - LLM应用开发框架
- [Streamlit](https://streamlit.io/) - 简洁的Web应用框架

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [Issue](../../issues)
- 发起 [Discussion](../../discussions)

---

**Vision RAG System** - 让文档智能化，让知识触手可及！ 🚀