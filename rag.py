"""
rag.py —— 知识层：私有文档检索（RAG，Retrieval-Augmented Generation）。

数据流：
    【启动后首次使用】knowledge/*.md → 读文件 → 切块 → 每段调嵌入模型 → 内存向量库
    【每次检索】      问题 → 嵌入成向量 → 和所有段落算余弦相似度 → 捞最相关的几段原文

两个关键认知：
    1. 向量是"意思坐标"：嵌入模型学出来的，意思近的文字坐标就近。
       关键词搜索匹配字面，向量搜索匹配意思。
    2. 切块和向量都是"派生数据"，可以从原文档重新算出来；
       放在内存里只为省事，knowledge/ 里的文件才是唯一的真身。
"""

import math
from pathlib import Path

from config import CHUNK_SIZE, EMBEDDING_MODEL, RAG_MIN_SCORE, TOP_K, client

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"


# ---------------------------------------------------------------------------
# 第 1 步：读文档 + 切块
# ---------------------------------------------------------------------------

def chunk_text(text: str, max_chars: int = CHUNK_SIZE) -> list[str]:
    """
    把长文本切成小块：按空行分出自然段，攒到 max_chars 字就封一块。
    每块是一个独立的检索单元 —— 太长语义被稀释，太短丢上下文。
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if current and len(current) + len(p) > max_chars:
            chunks.append(current)  # 装不下了：当前块封存，开新块
            current = p
        else:
            current = f"{current}\n{p}".strip()  # 还装得下：继续攒
    if current:
        chunks.append(current)
    return chunks


def load_chunks() -> list[dict]:
    """读 knowledge/ 下所有 .md / .txt，切块，返回 [{text, source}]"""
    chunks = []
    paths = sorted(KNOWLEDGE_DIR.glob("*.md")) + sorted(KNOWLEDGE_DIR.glob("*.txt"))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for chunk in chunk_text(text):
            chunks.append({"text": chunk, "source": path.name})
    return chunks


# ---------------------------------------------------------------------------
# 第 2 步：嵌入 —— 把文字变成"意思坐标"
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    调嵌入模型，把一批文本变成一批向量（每条是一个几百到几千长的数字列表）。
    注意：嵌入模型和聊天模型是两种模型，这个调用不带 tools、不产生对话。
    """
    vectors = []
    for i in range(0, len(texts), 10):  # 分批发送，每批最多 10 条
        batch = texts[i:i + 10]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    余弦相似度：把两串向量看成两支箭头，算夹角。
    1 = 方向完全一致（意思很近），0 = 垂直（无关），负数 = 指向相反。
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# 第 3 步：向量库（内存）+ 检索
# ---------------------------------------------------------------------------

_VECTOR_STORE: list[dict] = []  # [{text, source, vector}] —— 派生数据，启动后重建
_INDEX_BUILT = False


def _build_index() -> None:
    """读文档 → 切块 → 全部嵌入 → 装进内存向量库"""
    chunks = load_chunks()
    if not chunks:
        print("[RAG] knowledge/ 里没有文档，search_documents 将始终报告无结果")
        return
    print(f"[RAG] 构建索引：{len(chunks)} 块，正在调用嵌入模型……")
    vectors = embed_texts([c["text"] for c in chunks])
    for chunk, vector in zip(chunks, vectors):
        _VECTOR_STORE.append({**chunk, "vector": vector})
    print(f"[RAG] 索引就绪（{EMBEDDING_MODEL}，共 {len(vectors)} 个向量）")


def _get_index() -> list[dict]:
    """懒加载：第一次检索时才建索引，之前不花一分钱 API 调用"""
    global _INDEX_BUILT
    if not _INDEX_BUILT:
        _build_index()
        _INDEX_BUILT = True
    return _VECTOR_STORE


def search_documents(query: str) -> str:
    """
    工具函数：在用户的私有文档里检索与 query 最相关的 TOP_K 段原文。
    返回带相关度和来源标注的原文；低于阈值视为"没找到"——
    宁可承认没找到，也不能拿不相干的段落误导 LLM 瞎编。
    """
    store = _get_index()
    if not store:
        return "（知识库为空：knowledge/ 文件夹里没有文档）"

    query_vector = embed_texts([query])[0]  # 问题也要嵌入成同一种"坐标"
    scored = [(cosine_similarity(query_vector, item["vector"]), item) for item in store]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = scored[:TOP_K]

    if top[0][0] < RAG_MIN_SCORE:
        return f"知识库里没有找到与「{query}」相关的内容（最高相关度 {top[0][0]:.2f}）"

    parts = [
        f"【相关度 {score:.2f}｜来源 {item['source']}】\n{item['text']}"
        for score, item in top
    ]
    return "\n\n".join(parts)
