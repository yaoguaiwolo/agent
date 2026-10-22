"""
config.py —— 全项目最底层的公共配置。

所有模块都允许依赖它，但它不依赖任何本地模块——
这是避免"循环导入"的关键：依赖必须只朝一个方向流。

换服务商只需要改 .env 或环境变量，代码一行不动：
    OpenAI   : https://api.openai.com/v1              模型如 gpt-4o-mini
    DeepSeek : https://api.deepseek.com/v1            模型如 deepseek-chat
    智谱 GLM : https://open.bigmodel.cn/api/paas/v4   模型如 glm-4-flash（有免费额度）
    Kimi     : https://api.moonshot.cn/v1             模型如 moonshot-v1-8k
"""

import os
from pathlib import Path

from openai import OpenAI


def _load_dotenv() -> None:
    """
    从项目根目录的 .env 读取配置（每行一条 KEY=VALUE，# 开头是注释），
    写进环境变量。已存在的环境变量优先，.env 只是兜底。
    .env 已被 .gitignore 挡住，永远不会进版本库 —— 密钥只住在这里。
    """
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

if not os.environ.get("LLM_API_KEY"):
    raise RuntimeError(
        "找不到 API Key。请在项目根目录创建 .env 文件（或设置环境变量），内容：\n"
        "LLM_API_KEY=你的key"
    )

client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ.get("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
)
MODEL = os.environ.get("LLM_MODEL", "glm-5.3-flash")

# ★ 失忆实验总开关：True = 正常记忆；False = 每次提问都是全新对话（金鱼模式）
MEMORY_ON = True

# —— 上下文长度管理（滑动窗口 + 自动摘要）——
# 触发线：长期记忆超过这个条数（不含第 0 条 system）就触发"压缩"。
# 你之前改的 4 先留着，正好用来快速验证；日常使用建议改回 20
MAX_KEPT_MESSAGES = 4
# 压缩后保留的最近原文条数；更早的内容被压成一条摘要
SUMMARY_KEEP_RECENT = 2
# 摘要字数上限：超过就触发"摘要的摘要"，防止滚动摘要越滚越肥。
# 用字数当 token 的廉价替身（中文 1 字 ≈ 1 个 token 左右，够用了）
MAX_SUMMARY_CHARS = 200
# 保险丝：一次提问里最多连续调用几轮工具，防止 Agent 失控刷工具（README 实验 3）
MAX_TOOL_ROUNDS = 2

# —— RAG 私有文档检索（rag.py 用）——
EMBEDDING_MODEL = "embedding-3"   # 智谱的嵌入模型（把文字变成向量）
CHUNK_SIZE = 300                  # 切块目标字数
TOP_K = 3                         # 检索时捞最相关的几段
RAG_MIN_SCORE = 0.3               # 相似度低于它视为"没找到"（经验值：实测无关问题也能得 0.25 左右）
