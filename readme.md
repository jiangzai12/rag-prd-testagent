# 🤖 Auto_prd_test_expert

![Status](https://img.shields.io/badge/Status-Beta-blue)
![Python](https://img.shields.io/badge/Python-3.10-green)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![LLM](https://img.shields.io/badge/LLM-Google%20Gemini-orange)

> **"从精准检索到智能共创，重塑测试用例生成体验。"**
本项目来源于 https://github.com/zxLeva/ByteDance--Auto_prd_test_agent  的项目，感谢大佬开源，基于项目进行二次开发，希望对尝试测开方向小伙伴们有一定的学习与帮忙
Auto_prd_test_expert 是一款基于 **“轻量化本地部署 + 企业级大模型 API”** 混合架构的智能测试助手。它采用 **Advanced RAG** 和 **Human-in-the-Loop** 闭环设计，解决了传统工具“各种幻觉”、“无法微调”、“流程割裂”的痛点，是一个具备记忆与质检能力的可进化测试管家。

---

## 🎯 一、项目背景与定位

### 1.1 行业痛点
在软件测试领域，测试人员长期面临以下挑战：
*   **🕒 文档理解耗时**：PRD（需求文档）冗长，人工提取测试点效率低。
*   **🕳️ 用例覆盖不全**：容易遗漏边界值、异常场景或安全隐患。
*   **♻️ 经验难以复用**：优质历史用例沉淀在文档坟墓中，无法自动联想。
*   **😵‍💫 AI 生成幻觉**：通用大模型不懂内部规范（如密码策略），生成内容“假大空”。

### 1.2 产品定位
构建 **“精准检索 -> 智能共创 -> 对抗评估”** 的完整闭环体系。
*   **精准检索**：漏斗式 RAG，拒绝噪音。
*   **智能共创**：多模态输入 + 双屏交互，支持持续 Fine-tune。
*   **对抗评估**：引入 AI Critic 角色，像 QA 专家一样对用例进行质检。

---

## 🚀 二、核心功能亮点

### 🛡️ 功能一：漏斗式 RAG 精准检索体系
> **解决痛点**：解决通用大模型不懂业务规范、引用内容包含噪音的问题。

*   **📥 输入**：非结构化资产（PDF/图片/历史用例） + 当前 PRD。
*   **⚙️ 处理 (The Funnel)**：
    1.  **智能切片 (Smart Chunking)**：基于滑窗机制将文档切分为细粒度语义片段。
    2.  **向量粗筛 (Vector Retrieval)**：利用 Embedding 快速召回 Top-K 相关片段。
    3.  **LLM 智能细筛 (Contextual Filtering)**：引入轻量级模型作为“过滤器”，剔除无关噪音（如背景介绍），只保留核心干货。
*   **📤 输出**：100% 纯净的核心知识上下文，并向用户**透明展示**引用的具体来源文件名。

### 🧠 功能二：多模态智能共创体系
> **解决痛点**：交互体验差、无法微调、视觉逻辑丢失。

*   **📥 输入**：多模态材料（PDF 文档、**UI 设计图**、补充文本）。
*   **⚙️ 处理**：
    1.  **视觉逻辑推导**：利用 Gemini Vision 识别 UI 元素（按钮/输入框）并推导交互逻辑。
    2.  **双屏共创交互**：
        *   *左侧（思维链）*：展示 AI 分析思路，支持多轮对话微调（Fine-tune）。
        *   *右侧（实时交付）*：动态渲染结构化表格，所见即所得。
    3.  **意图路由 (Intent Routing)**：自动识别用户是“闲聊”还是“修改指令”，动态切换 Prompt 策略。
*   **📤 结果**：
    *   支持 **CSV/Excel/YAML/JSON/Markdown** 五种格式一键导出。
    *   **资产回流**：确认后的用例可“归档”回知识库，成为下一次生成的“历史参考”。

### ⚖️ 功能三：智能对抗评估体系
> **解决痛点**：人工验收慢、缺乏标准度量。

*   **⚙️ 处理**：
    *   **独立评审 Agent (AI Critic)**：构建“QA 验收专家”角色，与生成者形成对抗。
    *   **多维质检**：覆盖率分析、逻辑自洽性检查、去重与规范性检测。
    *   **差异化比对**：若上传了标准用例 (Golden Sample)，自动计算偏差。
*   **📤 结果**：输出可视化的**质量评估报告**（包含评分、漏测点、优化建议）。

---

## 🛠️ 三、技术栈与架构

### 3.1 核心技术栈
| 模块 | 技术选型 | 理由 |
| :--- | :--- | :--- |
| **Lang** | **Python 3.10** | AI 领域标准语言，生态最丰富。 |
| **Frontend** | **Streamlit** | 专为数据科学设计，支持 `Session State` 状态管理与快速迭代。 |
| **Backend** | **Google Generative AI SDK** | 原生支持 Gemini Pro/Flash，具备 1M+ Token 上下文与多模态能力。 |
| **RAG DB** | **ChromaDB** | 轻量级本地向量数据库，无需服务器，保障**数据隐私**。 |
| **Data** | **Pandas** | 强大的数据清洗与格式转换能力 (JSON <-> DataFrame)。 |

### 3.2 核心代码模块说明

本项目采用模块化分层架构：

#### 📂 配置层 (`config/`)
*   **`prompts.py`**: 统一管理所有 Prompt。内置**智能路由逻辑**，根据用户意图动态组装 System Prompt，并强制执行“解释与数据分离”策略。

#### 📂 核心逻辑层 (`core/`)
*   **`rag_engine.py` (知识引擎)**: 封装 ChromaDB 操作。实现了 `TextSplitter` 递归切片算法，以及多模态解析接口 `parse_file_content`（支持 UI 图转文字）。
*   **`llm_client.py` (AI 网关)**: 封装 Gemini API，实现带历史记忆的对话接口 `get_gemini_chat_response`。
*   **`evaluator.py` (评估引擎)**: 独立的质检模块，负责调用 LLM 输出结构化的质量报告。

#### 📂 前端交互层 (`ui/`)
*   **`main.py` (主控台)**: 业务编排中心。
    *   **RAG 智能双重筛选**: 串联 `rag_engine` (粗筛) 与 `llm_client` (细筛/去噪)。
    *   **思维链共创**: 利用 `split_text_and_json` 实现左侧聊天流与右侧数据流的视觉分离。
    *   **资产归档**: 闭环逻辑，将最终用例回流至 ChromaDB。
*   **`sidebar.py`**: 全局配置入口，包含动态模型加载与 API Key 管理。
*   **`components.py`**: 专注于数据表格渲染与多格式导出。

---

## 🔮 四、未来展望

1.  **输入端拓展**：
    *   集成 **FastAPI**，支持飞书/钉钉机器人接入，实现 IM 群组中的“对话即测试”。
    *   增加 URL 解析能力（直接读取在线 PRD/Jira）。
2.  **输出端 Agent 化**：
    *   自动化脚本生成：利用生成的结构化数据，调用大模型生成 **Pytest / Midscene.js** 执行脚本。
    *   集成 CI/CD 流水线，实现“从文档到代码”的自动化闭环。
3.  **上游优化**：
    *   增加 **“PRD 质检”** 模块，在生成用例前先对需求文档本身进行逻辑漏洞分析。

---

## 📥 安装与运行

## 🛠️ 一、 环境部署与配置

本项目提供 **Docker（推荐）** 和 **本地源码** 两种部署方式。由于项目依赖 Google Gemini API，请确保你的运行环境能够访问 Google 服务。

### 🐳 方式一：使用 Docker 部署（推荐）

这是最简单、最稳定的部署方式，无需配置复杂的 Python 环境，且已配置好数据持久化。

#### 1. 前置准备
*   安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac/Linux)。
*   克隆本项目代码：
    ```bash
    git clone https://github.com/YourUsername/Auto_prd_test_expert.git
    cd Auto_prd_test_expert
    ```

#### 2. 配置 API Key
为了安全起见，API Key 不包含在代码库中。请按照以下步骤配置：
1.  进入 `data/` 目录。
2.  将 `user_config.example.json` 重命名为 `user_config.json`。
3.  编辑该文件，填入你的 Gemini API Key：
    ```json
    {
        "api_key": "YOUR_GEMINI_API_KEY_HERE"
    }
    ```

#### 3. 配置网络代理（国内用户必读）
如果你在中国大陆地区使用，必须配置代理才能连接 Gemini API。
打开项目根目录下的 `docker-compose.yml`，找到 `environment` 部分，根据你的实际代理端口修改：

```yaml
    environment:
      # host.docker.internal 代表宿主机 IP
      # 请将 7897 修改为你本地代理软件（如 v2ray/clash）的端口号
      - HTTP_PROXY=http://host.docker.internal:7897
      - HTTPS_PROXY=http://host.docker.internal:7897
```
![alt text](image.png)
#### 4. 一键启动
在项目根目录下打开终端，运行：

```bash
docker-compose up -d --build
```

*   构建过程可能需要几分钟（已配置国内镜像源加速）。
*   启动成功后，浏览器访问：[http://127.0.0.1:8501](http://127.0.0.1:8501) 即可使用。

#### 5. 停止服务
```bash
docker-compose down
```
> **关于数据持久化**：Docker 已配置挂载卷，你的知识库（向量数据）和上传的原始文件会保存在本地 `data/` 目录下，重启容器数据**不会丢失**。

---

### 🐍 方式二：本地源码部署（开发调试）

如果你需要修改代码或进行二次开发，建议使用 Anaconda 环境。

#### 1. 创建虚拟环境
```bash
conda create -n gemini_test python=3.10
conda activate gemini_test
```

#### 2. 安装依赖
```bash
pip install -r requirements.txt
```

#### 3. 配置代理 (可选)
项目会自动读取系统环境变量。如果在代码中未检测到代理，默认会尝试使用 `http://127.0.0.1:7897`。你可以直接修改 `config/settings.py` 中的默认端口，或者在终端设置环境变量：
```bash
# Windows PowerShell
$env:HTTP_PROXY="http://127.0.0.1:7890"
$env:HTTPS_PROXY="http://127.0.0.1:7890"
```

#### 4. 启动应用
```bash
streamlit run ui/main.py
```

---

## 🧪 回归套件与指标看板

为固化质量改造收益，项目新增了最小回归集与可复测指标看板。

### 回归样本

回归样本位于 `test_prd/regression_suite.json`，覆盖以下典型问题：
1. 标准结构化输出。
2. 字段缺失与非法优先级。
3. 硬重复 + 边界值相似场景。
4. 非结构化回复（需修复）。

### 运行方式

1. UI 运行（推荐）
    - 启动应用后，进入 `📈 回归看板` Tab。
    - 点击 `▶️ 运行最小回归套件`。

2. 命令行运行
```bash
python scripts/run_regression_suite.py
```

### 产物文件

产物会写入 `reports/regression/`：
1. `latest_regression_report.json`（机器可读）
2. `latest_regression_report.md`（人工看板）

### 指标说明

1. Schema通过率
2. 平均规则分
3. 平均覆盖率
4. 硬去重移除总数
5. 语义近似告警总数（仅告警不自动删除）

## 🧪 RAG 离线评测看板

为评估 RAG 检索与问答质量，项目新增离线评测体系（评测集驱动）。

### 评测集

- 评测集文件：`test_prd/rag_eval_suite.json`
- 内容结构：`corpus`（评测语料）+ `qa_samples`（问答样本与期望来源）

### 指标

1. Recall@K：Top-K 召回命中期望来源的平均比例。
2. 误召回率：Top-K 召回中非期望来源的平均占比。
3. 幻觉率：命中禁用关键词、无依据强答、或不可回答却未拒答的比例。

### 运行方式

1. UI 运行（推荐）
    - 启动应用后，进入 `🧪 评测看板` Tab。
    - 点击 `▶️ 运行 RAG 离线评测`。

2. 命令行运行
```bash
python scripts/run_rag_eval_suite.py
```

3. 纯离线运行（断网可用）
```bash
python scripts/run_rag_eval_suite.py --offline
```

离线模式说明：
1. 不调用 Gemini API，使用本地词法召回评估检索指标。
2. 若评测样本包含 `offline_answer` 字段，可继续计算幻觉率。
3. 若未提供 `offline_answer`，则仅评估 Recall@K 与误召回率，幻觉率标记为不可评估。

### 产物文件

产物会写入 `reports/rag_eval/`：
1. `latest_rag_eval_report.json`（机器可读）
2. `latest_rag_eval_report.md`（人工看板）
3. `history/*.json`（每次运行历史快照）

