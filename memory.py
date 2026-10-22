"""
memory.py —— 记忆层：长期记忆的瘦身与压缩。

记忆就是一个列表，每次请求都原样重发给 LLM。本层的职责：
历史太长时把它变短 —— 优先"摘要压缩"（compact_messages），
失败则退回"滑动窗口"（trim_messages），保证 Agent 永不因此崩溃。
"""

from config import MAX_SUMMARY_CHARS, MODEL, client


def get_role(msg) -> str:
    """
    统一取出一条消息的 role。
    列表里混着两种消息：我们 append 的字典（system/user/tool）有 ["role"]，
    SDK 返回的助手消息对象用 .role —— 这个小函数把它们抹平
    """
    return msg["role"] if isinstance(msg, dict) else msg.role


def get_content(msg) -> str:
    """同 get_role：把字典消息和 SDK 对象消息的正文统一取出来"""
    if isinstance(msg, dict):
        return str(msg.get("content") or "")
    return str(msg.content or "")


def trim_messages(messages: list, max_kept: int) -> list:
    """
    滑动窗口瘦身：历史超过 max_kept 条时，只保留 system + 最近 max_kept 条。

    为什么不直接切片完事？因为 API 要求 tool 消息必须紧跟在
    "assistant 的调用请求"后面。从中间一刀切，很可能把一对
    （assistant(tool_calls) + tool 结果）拦腰切断，留下一条"孤儿 tool 消息"，
    下一轮请求直接报 400。所以切完还要检查边界、把孤儿丢干净。
    """
    # 没超限（+1 是把 system 那条算进去），原样返回，一个字不动
    if len(messages) <= max_kept + 1:
        return messages

    # 第 0 条永远是 system（人设不能丢），后面接最近 max_kept 条
    kept = [messages[0]] + messages[-max_kept:]

    # 配对安全检查：如果保留部分的开头是 tool 消息，
    # 说明它的"调用请求"搭档被切掉了，继续丢，直到开头不是 tool
    while len(kept) > 1 and get_role(kept[1]) == "tool":
        kept.pop(1)

    return kept


def summarize_messages(old_messages: list) -> str:
    """
    自动摘要核心：把一批旧消息压缩成一段摘要。
    注意这是【独立的 API 调用】——不带 tools，用的是另一个"摘要员"人设。
    摘要员只需要会说话，不需要会动手。
    """
    # 把旧消息渲染成纯文本（role: 内容），作为一次普通提问发给摘要员
    transcript = "\n".join(f"{get_role(m)}: {get_content(m)}" for m in old_messages)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是对话摘要员。把用户发来的历史对话压缩成简明要点，"
                           "必须保留其中的关键事实（人名、数字、决定、结论）。"
                           f"摘要全文不得超过 {MAX_SUMMARY_CHARS} 字。"
                           "直接输出摘要本身，不要客套话。",
            },
            {"role": "user", "content": f"请总结以下对话：\n\n{transcript}"},
        ],
    )
    return response.choices[0].message.content


def shrink_summary(summary: str) -> str:
    """
    防雪球补丁：摘要的摘要 —— 摘要本身超长时，降低"分辨率"再压。
    像地图不断缩小比例尺：省 → 市 → 国，主干永在，细节逐层舍弃。
    最多尝试 3 遍，防止死循环；实在压不下来也照常使用（总比崩了好）。
    """
    for attempt in range(1, 4):
        before = len(summary)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是对话摘要员。把这份摘要压缩得更短："
                               "只保留最重要的主干事实（人名、决定、结论），"
                               f"全文不得超过 {MAX_SUMMARY_CHARS} 字。直接输出结果。",
                },
                {"role": "user", "content": summary},
            ],
        )
        summary = response.choices[0].message.content
        print(f"[摘要瘦身] 第 {attempt} 遍：{before} 字 → {len(summary)} 字")
        if len(summary) <= MAX_SUMMARY_CHARS:
            break
    return summary


def compact_messages(messages: list, trigger: int, keep_recent: int) -> list:
    """
    自动摘要版瘦身（比单纯的滑动窗口聪明）。

    超过 trigger 条时：
      1. 在最近的 user 消息处下刀 —— 一刀同时解决两个配对问题：
         旧部分结尾不会是没回应的调用请求，新部分开头不会是孤儿 tool 消息
      2. 旧部分发给"摘要员"压缩成一段摘要
      3. 新记忆 = [system 人设, 摘要, ...最近原文]

    压缩后长度（摘要 + keep_recent）远低于触发线 trigger，
    所以不会每轮都压缩，天然留出了缓冲间隔。
    下次再压缩时，旧摘要也会被一起送去合并（滚动摘要），信息不丢主线。
    """
    # 没超限就不动（+1 是把 system 那条算进去）
    if len(messages) <= trigger + 1:
        return messages

    # ---- 选切割点：从目标位置向前退，退到最近的 user 消息处 ----
    cut = len(messages) - keep_recent
    while cut > 1 and get_role(messages[cut]) != "user":
        cut -= 1
    if cut <= 1:  # 一路退到头都不是 user，找不到安全的下刀处
        return trim_messages(messages, keep_recent)  # 退回普通滑动窗口

    old_part = messages[1:cut]  # 要被压缩的部分（不含 system 人设）
    if not old_part:
        return messages

    # ---- 调摘要员；万一网络/接口出错，退回滑动窗口，保证 Agent 不崩 ----
    try:
        summary = summarize_messages(old_part)
        # 防雪球：摘要本身超长时，降分辨率再压一遍
        if len(summary) > MAX_SUMMARY_CHARS:
            summary = shrink_summary(summary)
    except Exception as e:
        print(f"\n[记忆压缩失败，退回滑动窗口] {e}")
        return trim_messages(messages, keep_recent)

    print(f"\n[记忆压缩] {len(old_part)} 条旧消息 → 1 条摘要：{summary[:60]}...")

    # 摘要用 system 角色、放在第 1 位：排在人设之后、正文之前，
    # LLM 会把它当作"背景资料"来读
    return (
            [messages[0]]
            + [{
        "role": "system",
        "content": f"（以下是更早对话的摘要，供你参考）\n{summary}",
    }]
            + messages[cut:]
    )
