# 最小 Agent（多文件版）

从零手写的教学 Agent：不依赖任何 Agent 框架，只用 `openai` SDK，
把 Agent 的每个零件都摊开来看。当前约 700 行，功能见下方清单。

## 它是什么

Agent 没有任何魔法，只有三样东西：

```
        ┌────────────────────────────────────────┐
        │                                        │
        ▼                                        │
   ① 发给 LLM：完整对话历史 + 工具说明书          │
        │                                        │
        ▼                                        │
   ② LLM 回复：要么是最终答案，要么说             │
      "我想调用工具 xxx，参数是 yyy"               │
        │                                        │
        ├── 最终答案 → 流式显示给用户，结束        │
        │                                        │
        └── 要调用工具 → ③ 你的代码真正执行函数 ──┘
                          结果放回对话历史
```

| 概念 | 通俗解释 | 在代码里的位置 |
|------|----------|----------------|
| 工具 Tool | 普通 Python 函数 + 一份给 LLM 看的说明书 | `tools.py` |
| 记忆 Memory | 一个列表 + 自动压缩，每次请求完整重发 | `memory.py` |
| 循环 Loop | 问 LLM → 调工具 → 结果塞回 → 再问，直到给出最终回答 | `agent.py` 的 `agent_loop()` |

**两条最重要的认知**：

1. LLM 从来没有"执行"过任何东西，它只会说"我想调用 calculate，参数是 3+5"。
   真正动手的是我们的代码——**LLM 出主意，代码动手**。
2. 凡是能写成规则的活（数数、切分、流程控制）写进代码；
   需要"理解意思"的活（摘要、判断信息重要性）才叫 LLM。

## 项目结构

```
agent/
├── config.py    ← 最底层：client、模型名、所有可调常量（谁都能依赖它，它不依赖谁）
├── tools.py     ← 工具层：工具函数 + TOOLBOX + TOOL_SCHEMAS
├── memory.py    ← 记忆层：滑动窗口、摘要压缩、防雪球
├── rag.py       ← 知识层：私有文档检索（RAG，待建）
├── agent.py     ← 对话循环 + 入口（agent_loop + main）
├── docs/        ← 练习任务书、学习笔记
├── knowledge/   ← RAG 检索的文档库（待建）
├── README.md / requirements.txt / .gitignore
```

依赖只允许自下而上指向 `config`，严禁反向（循环导入）。
加一个新工具的固定流程（三处注册）：写函数 → `TOOLBOX` 登记 → `TOOL_SCHEMAS` 写说明书。

## 功能清单

- **流式输出**：回答逐字蹦出（`stream=True` + 碎块拼装）
- **工具循环 + 保险丝**：单次提问最多连续 `MAX_TOOL_ROUNDS` 轮工具调用，防失控
- **双份记忆**：`[0]` system 人设 + 滚动摘要 + 最近原文
- **自动压缩**：记忆超限自动"摘要压缩"，工具配对永不被切断
- **防雪球**：摘要自身超长时降分辨率再压
- **异常兜底**：单轮失败（网络/接口/参数）只作废本轮，程序不崩
- **token 监控**：每次调用打印 `[Token] 输入+输出=总计`
- **调试命令**：`/dump` 查看长期记忆、`/forget` 一键清空、`q` 退出

## 怎么运行

```powershell
pip install openai        # 本项目用 conda 环境 agent（PyCharm 里选它）
python agent.py
```

模型接口：任何 OpenAI 兼容服务商都行。**API Key 只放在根目录的 `.env`
文件里**（已被 `.gitignore` 挡住，永不进版本库）：

```
LLM_API_KEY=sk-你的key
```

也可以用系统环境变量设置，优先级高于 `.env`：

```powershell
$env:LLM_API_KEY  = "sk-你的key"
$env:LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
$env:LLM_MODEL    = "glm-5.3-flash"
```

## 可调常量（都在 config.py）

| 常量 | 含义 | 当前值 |
|------|------|--------|
| `MEMORY_ON` | 总开关：False = 每次提问都失忆（金鱼模式） | `True` |
| `MAX_KEPT_MESSAGES` | 触发线：记忆超过几条就压缩 | `4`（日常建议 20） |
| `SUMMARY_KEEP_RECENT` | 压缩后保留最近几条原文 | `2`（日常建议 10） |
| `MAX_SUMMARY_CHARS` | 摘要字数上限，超过就"摘要的摘要" | `200` |
| `MAX_TOOL_ROUNDS` | 保险丝：单次提问最多几轮工具调用 | `2`（日常建议 8） |

## 学习实验

1. **失忆实验**：`MEMORY_ON = False` 重启，或运行中输入 `/forget`，
   然后问"我叫什么名字？"
2. **保险丝实验**：`MAX_TOOL_ROUNDS = 1`，问个需要工具的问题，
   看熔断但程序不死
3. **加工具实验**：照 `get_weather` 的样子三处注册一个新工具
4. **上下文长度管理**：完整任务书在 `docs/练习-上下文长度管理.md`（已完成 ✅）

## 版本控制

```powershell
git add .
git commit -m "这次改了什么的一句话"
git log --oneline     # 看历史；改崩了用 git diff / git checkout -- 文件 回滚
```

**铁律**：API Key 只住 `.env`（本项目已落实）。以后换 Key、加其他密钥，
一律写进 `.env`，永远别写进任何会被提交的文件。

## 路线图

- [x] 工具调用 / 记忆 / 循环（三件套）
- [x] 上下文长度管理（滑动窗口、自动摘要、防雪球）
- [x] 流式输出、保险丝、异常兜底
- [x] 多文件拆分 + git 版本控制
- [ ] **RAG：私有文档检索（下一站）**
- [ ] 结构化输出 / MCP / 多 Agent 协作
