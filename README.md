# 律所智能问答 RAG 系统

法律垂直领域多 Agent 智能问答系统。基于 RAG（检索增强生成）技术，结合 LangGraph 多 Agent 协作架构，为民商事诉讼、刑事辩护、行政诉讼三类案件提供专业法律咨询服务。支持文档智能分块、混合向量检索、重排序、会话记忆、检索参数动态调整。

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI |
| 前端框架 | Vue 3 + Vite |
| 向量数据库 | Milvus 2.6|
| 多 Agent 框架 | LangGraph |
| 嵌入模型 | BGE-M3 |
| 重排序模型 | BGE Reranker v2-M3 |
| 大模型 | DeepSeek API | 
| LLM 框架 | LangChain | 

---

## 项目结构

```
RAG2.0/
├── test/                              # 后端代码（Python）
│   ├── agent/                          # 多 Agent 模块
│   │   ├── legal_graph.py              # LangGraph 工作流编排（核心调度）
│   │   ├── reception_agent.py          # 客户接待 Agent（信息收集/分类/追问）
│   │   ├── civil_lawyer_agent.py       # 民商事诉讼律师 Agent
│   │   ├── criminal_lawyer_agent.py    # 刑事辩护律师 Agent
│   │   ├── admin_lawyer_agent.py       # 行政诉讼律师 Agent
│   │   ├── FengKuaiAgent.py            # 语义分块 Agent（BGE-M3 + 百分位 + LLM）
│   │   ├── prompt_loader.py            # Prompt 文件加载工具
│   │   └── prompts/                    # Prompt 模板（.txt）
│   │       ├── legal_reception.txt     #   接待 Agent 提示词
│   │       ├── legal_civil_lawyer.txt  #   民商事律师提示词
│   │       ├── legal_criminal_lawyer.txt # 刑事律师提示词
│   │       ├── legal_admin_lawyer.txt  #   行政律师提示词
│   │       ├── legal_search_strategy.txt # 检索策略生成提示词
│   │       ├── gray_zone_chunker.txt   #   灰色地带判断提示词
│   │       └── smart_chunker.txt       #   智能分块提示词
│   ├── api/                            # FastAPI 路由层
│   │   ├── __init__.py                 #   FastAPI 应用工厂、CORS、启动事件
│   │   ├── chat.py                     #   法律咨询接口（/api/legal/chat）
│   │   ├── collections.py              #   集合管理接口（CRUD）
│   │   └── documents.py                #   文档上传/删除接口
│   ├── database/
│   │   └── milvus.py                   # Milvus 全部操作（建集合/插入/检索/删除/file_registry）
│   ├── jiansuo/
│   │   └── qurey.py                    # 查询入口（编码 → 混合检索）
│   ├── qianru/
│   │   └── BGE.py                      # BGE-M3 稠密+稀疏向量编码
│   ├── chongpai/
│   │   └── reranker.py                 # BGE Reranker 精排
│   ├── load/
│   │   ├── load_word.py                # Word 文档语义分块（核心分块逻辑）
│   │   └── load_xlsx.py                # Excel 加载（备用）
│   ├── config.py                       # 环境变量加载与配置中心
│   ├── llm.py                          # LLM 初始化（命令行入口）
│   ├── main.py                         # 启动入口（uvicorn）
│   ├── file_registry.json              # 文件注册表（文件→集合→source_id 映射）
│   ├── .env.example                    # 环境变量模板
│   └── requirements.txt                # Python 依赖
│
├── qianduan/                           # 前端代码（Vue 3）
│   ├── src/
│   │   ├── App.vue                     # 根组件
│   │   ├── main.js                     # Vue 应用入口
│   │   ├── router/index.js             # 路由配置
│   │   ├── api/entries.js              # API 请求封装
│   │   └── views/
│   │       ├── ChatPage.vue            # 法律咨询页面
│   │       ├── CollectionsPage.vue     # Milvus 集合管理页面
│   │       ├── DocumentsPage.vue       # 文档上传与管理页面
│   │       └── ParamsPage.vue          # 检索参数调整页面
│   ├── index.html
│   ├── vite.config.js                  # Vite 配置（含 API 代理）
│   └── package.json
│
└── .gitignore
```

---

## 核心架构

### 1. LangGraph 多 Agent 工作流

```
                        ┌────────────┐
                        │  用户输入   │
                        └─────┬──────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │      客户接待 Agent            │
              │     
              │                               │
              │  · 分析对话历史 + 最新问题      │
              │  · 提取/更新三要素：            │
              │    案件类型 | 用户诉求 | 基本信息 │
              │  · 判断三要素是否全部收集完毕     │
              └───────────────┬───────────────┘
                              │
                    _route_after_reception()
                              │
              ┌───────┬───────┼───────┬───────┐
              │       │       │       │       │
              ▼       ▼       ▼       ▼       ▼
          incomplete 民商事   刑事    行政   其他案件
              │       │       │       │       │
              ▼       ▼       ▼       ▼       ▼
         ┌────────┐┌──────┐┌──────┐┌──────┐┌──────────┐
         │追问用户││民事  ││刑事  ││行政  ││"本律所暂 │
         │ END   ││律师  ││律师  ││律师  ││无法受理" │
         └────────┘└──┬───┘└──┬───┘└──┬───┘└────┬─────┘
                      │       │       │         │
                      ▼       ▼       ▼         ▼
                    ┌──────────────────────┐  ┌───┐
                    │   生成法律分析意见     │  │END│
                    │        END           │  └───┘
                    └──────────────────────┘
```

**关键设计**：
- 接待 Agent 必须收齐「案件类型 + 用户诉求 + 案件基本信息」三点后才交接；不全则持续追问
- 传递是单向的：接待 → 律师，律师不能反向回传
- 每个 Agent 独立维护对话历史，通过 LangGraph State 传递

### 2. RAG 混合检索流程

```
律师Agent接收: 案件类型 + 用户诉求 + 基本信息
    │
    ├─→ LLM 生成检索策略（JSON）:
    │       · main_query: 1个完整案情摘要检索式
    │       · element_queries: 3-4个法律要素检索式
    │       · final_top_k: 动态精排数量
    │
    ├─→ 对每个检索式并行执行:
    │       ┌──────────────┐   ┌──────────────┐
    │       │ BGE-M3 稠密   │   │ BGE-M3 稀疏   │
    │       │ 语义相似度     │   │ 词汇级匹配     │
    │       └──────┬───────┘   └──────┬───────┘
    │              └────────┬─────────┘
    │         
    │
    ├─→ 多路结果合并 + 按 ID 去重
    │
    ├─→ BGE Reranker 精排 → Top final_top_k
    │
    └─→ 检索结果注入 LLM → 生成法律分析意见
```



### 3. 文档语义分块流程

```
Word 文档
    │
    ├─→ 提取所有段落 [P1, P2, P3, ..., Pn]
    │
    ├─→ BGE-M3 编码每个段落 → 稠密向量
    │
    ├─→ 计算相邻段落余弦相似度 (P1↔P2, P2↔P3, ...)
    │
    ├─→ 动态百分位阈值判定:
    │       · < P25（下界）→ 明确分割点
    │       · > P75（上界）→ 明确合并点
    │       · P25 ~ P75     → 灰色地带
    │
    ├─→ 灰色地带交由 LLM 二次判断（是否需要分割）
    │
    └─→ 根据决策结果构建最终 chunks
```

**优势**：
- 阈值由当前文档相似度分布动态计算，不依赖固定值
- 灰色地带引入 LLM 语义判断，提升分块准确性
- 分块边界基于段落间语义关系，保留法律条文上下文


## 快速启动

### 环境要求

| 组件 | 版本要求 |
|------|----------|
| Python | ≥ 3.10 |
| Node.js | ≥ 18 |
| Milvus | ≥ 2.6（需支持 SPARSE_FLOAT_VECTOR） |
| BGE-M3 模型 | 本地路径 |
| BGE Reranker v2-M3 模型 | 本地路径 |
| DeepSeek API Key | sk-xxx |

### 1. 安装 Python 依赖

```bash
cd test
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入实际配置
```

`.env` 配置项说明：

| 变量 | 说明 |
|------|------|
| `BGE_MODEL_PATH` | BGE-M3 模型本地路径 |
| `RERANKER_MODEL_PATH` | BGE Reranker 模型本地路径 |
| `MILVUS_HOST` | Milvus 服务地址 |
| `MILVUS_PORT` | Milvus 服务端口 |
| `COLLECTION_NAME` | 默认集合名称 |
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 |
| `DEEPSEEK_BASE_URL` | DeepSeek API 地址 |
| `HF_HOME` | HuggingFace 缓存目录 |
| `HF_ENDPOINT` | HuggingFace 镜像地址 |

### 3. 启动后端

```bash
cd test
python main.py
# 或：uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

后端启动成功后访问 `http://localhost:8000/docs` 查看 Swagger API 文档。

### 4. 启动前端

```bash
cd qianduan
npm install
npm run dev
```

前端启动后访问 `http://localhost:5173`。Vite 自动代理 `/api` 请求到 `localhost:8000`。

### 5. 创建 Milvus 集合

通过前端「集合管理」页面或 API 创建三个集合：

| 集合名称 | 描述 | 用途 |
|----------|------|------|
| `minshi` | 民商事诉讼案例库 | 婚姻家庭/合同/侵权/劳动等 |
| `xingshi` | 刑事辩护案例库 | 杀人/抢劫/拐卖等刑事案件 |
| `xingzheng` | 行政诉讼案例库 | 行政处罚/许可/强制案件 |

### 6. 上传法律文档

通过前端「文档管理」页面上传 `.docx` 格式法律文档，选择目标集合，系统自动完成：
1. 语义分块
2. BGE-M3 向量编码
3. 写入 Milvus
4. 注册文件记录









