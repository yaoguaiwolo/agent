"""
一个最小的 Agent —— 用来理解 Agent 的本质。（多文件版）

Agent 没有任何魔法，只有三样东西：
    1. 工具 (Tool)  : 普通的 Python 函数 + 给 LLM 看的"说明书"        → tools.py
    2. 记忆 (Memory): 一个列表 + 自动压缩机制，每次请求原样重发        → memory.py
    3. 循环 (Loop)  : 反复"思考 → 调工具 → 结果塞回 → 再思考"          → 本文件 agent_loop

最容易误解的一点：
    LLM 从来没有"执行"过任何东西。它只会输出一段结构化文字，说
    "我想调用 calculate 函数，参数是 3+5"。真正动手的是我们的代码。
    LLM 出主意，代码动手 —— 凡是能写成规则的活写进代码，
    需要"理解意思"的活才叫 LLM。

项目结构（依赖只允许自下而上指向 config，严禁反向，否则循环导入）：
    config.py : client、模型名、所有可调常量 —— 最底层
    tools.py  : 工具函数 + 工具说明书
    memory.py : 长期记忆的瘦身与压缩
    rag.py    : 私有知识检索（RAG，下一步）
    agent.py  : 对话循环 + 入口（本文件）

运行前：pip install openai，然后 python agent.py
"""

import json

from config import (
    MAX_KEPT_MESSAGES,
    MAX_TOOL_ROUNDS,
    MEMORY_ON,
    MODEL,
    SUMMARY_KEEP_RECENT,
    client,
)
from memory import compact_messages
from tools import TOOLBOX, TOOL_SCHEMAS


def agent_loop(turn: list) -> None:
    """
    处理"一轮对话"。turn = 长期记忆 + 你刚说的这句话，是一份临时草稿：
        问 LLM -> LLM 要调用工具？ -> 执行 -> 结果追加进草稿 -> 再问 LLM
        直到 LLM 不再要求调用工具（说明它想给出最终回答了）才退出。

    注意：这里对 turn 的 append 只是让"本轮"能运转——工具结果必须紧跟在
    调用请求后面，这是 API 的硬性要求。它们不是长期记忆，
    真正的长期记忆保存只在 main() 里的那一行（带 MEMORY_ON 开关）。
    """
    for _round in range(MAX_TOOL_ROUNDS):  # 保险丝：最多连续 N 轮工具调用
        try:
            # stream=True：回答不再一次性返回，而是碎成小块源源传来
            stream = client.chat.completions.create(
                model=MODEL,
                messages=turn,  # 发送本轮草稿（含完整历史）
                tools=TOOL_SCHEMAS,  # 每次都把工具说明书带上
                stream=True,
                # 让最后一个块顺路捎上 token 统计（个别服务商不支持这个参数，
                # 如果报错就把这一行删掉，只是看不到 [Token] 统计，其他不受影响）
                stream_options={"include_usage": True},
            )

            # ---- 把碎块重新拼成一条完整消息 ----
            content_parts = []  # 文字碎片都堆这里，最后拼成完整回答
            tool_calls = {}  # 工具调用的碎片：编号 -> {id, 函数名, 参数}
            usage = None  # token 统计（最后一个块才会带来）

            for chunk in stream:
                if chunk.usage:  # 最后一块捎带的 token 统计
                    usage = chunk.usage
                if not chunk.choices:  # 有的块只有统计、没有内容
                    continue
                delta = chunk.choices[0].delta
                if not delta:
                    continue
                if delta.content:  # 文字碎片：边收边打印
                    if not content_parts:  # 第一个碎片：先打出前缀
                        print("\nAgent: ", end="", flush=True)
                    content_parts.append(delta.content)
                    print(delta.content, end="", flush=True)
                for tc in (delta.tool_calls or []):  # 工具调用碎片：按编号拼接
                    slot = tool_calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["args"] += tc.function.arguments

            # 拼装：有工具调用就还原成结构化请求，否则就是一条普通回答。
            # 用字典代替 SDK 对象，效果完全等价（get_role/get_content 都兼容）
            if tool_calls:
                reply = {
                    "role": "assistant",
                    "content": "".join(content_parts) or None,
                    "tool_calls": [
                        {
                            "id": slot["id"],
                            "type": "function",
                            "function": {"name": slot["name"], "arguments": slot["args"]},
                        }
                        for _, slot in sorted(tool_calls.items())
                    ],
                }
            else:
                reply = {"role": "assistant", "content": "".join(content_parts)}

            turn.append(reply)  # 放进草稿，成为本轮对话记录的一部分
            print()  # 流式打印没有自带换行，补一个
            if usage:
                print(f"[Token] 输入 {usage.prompt_tokens} + 输出 {usage.completion_tokens} = {usage.total_tokens}")

            # 情况 A：没有工具调用 -> 最终回答已经流式打完，本轮结束
            if not tool_calls:
                return

            # 情况 B：LLM 要求调用工具（可能一次要求好几个），逐个执行
            for call in reply["tool_calls"]:
                name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"])  # 参数是 JSON 字符串
                print(f"\n[Agent 想调用工具] {name}，参数：{args}")

                func = TOOLBOX.get(name)
                if func is None:
                    result = f"错误：不存在名为 {name} 的工具"
                else:
                    result = func(**args)  # 真正执行的是我们的代码，不是 LLM

                print(f"[工具结果] {result}")

                # 工具结果必须紧跟在"调用请求"后面塞回草稿，下一轮 LLM 才能读到
                turn.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": result,
                    }
                )
        except Exception as e:
            print(f"\n[工具调用出错] {e}")
            turn.append({"role": "assistant", "content": f"错误：{e}"})
            return

        # 回到循环开头，带着工具结果的草稿再发给 LLM，让它继续思考

    # 保险丝熔断：跑满 MAX_TOOL_ROUNDS 轮还没结论，强制收尾。
    # 必须补一条 assistant 消息，否则草稿以 tool 消息结尾，下一轮 API 会报错
    turn.append({
        "role": "assistant",
        "content": f"（连续调用 {MAX_TOOL_ROUNDS} 轮工具仍未得出结论，先停在这里）",
    })
    print(f"\n[保险丝] 连续 {MAX_TOOL_ROUNDS} 轮工具调用，已强制停止本轮")


def main() -> None:
    print("最小 Agent 已启动！试试问我：现在几点？/ 3+5*2 等于几？/ D:\\workcode 里有什么文件？")
    print("输入 /forget 清空记忆、/dump 查看记忆（做实验用），输入 q 退出。\n")

    # 记忆的起点：一条 system 消息，告诉 LLM 它是谁
    messages = [
        {
            "role": "system",
            "content": "你是一个乐于助人的中文助手，可以使用提供的工具来回答问题。"
                       "需要使用工具时直接调用，拿到结果后用简洁的中文回答。",
        }
    ]

    # 外层循环：一问一答；内层循环（agent_loop）：一次提问内部可能要调用好几次工具
    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in ("q", "quit", "exit"):
            print("再见！")
            break
        if not user_input:
            continue

        # /forget：一键清空长期记忆，只保留 system 提示词
        if user_input == "/forget":
            messages[:] = [messages[0]]  # 原地清空，只留第一条 system 消息
            print("（记忆已清空，列表里只剩下 system 提示词）\n")
            continue

        # /dump：打印长期记忆，亲眼看它到底"记得"什么
        if user_input == "/dump":
            print("\n===== 长期记忆（messages 列表）=====")
            for i, m in enumerate(messages):
                if isinstance(m, dict):  # 字典形式的消息（system/user/tool）
                    print(f"[{i}] {m['role']}: {str(m.get('content'))[:100]}")
                else:  # SDK 返回的助手消息对象
                    extra = f"（要求调用 {len(m.tool_calls)} 个工具）" if m.tool_calls else ""
                    print(f"[{i}] {m.role}: {str(m.content or '')[:100]}{extra}")
            print(f"===== 共 {len(messages)} 条 =====\n")
            continue

        # 本轮草稿 = 长期记忆 + 你刚说的这句话。
        # 即使处于失忆模式，这句话也必须发出去，否则它连问题都听不见
        turn = messages + [{"role": "user", "content": user_input}]

        agent_loop(turn)  # Agent 在草稿上完成本轮所有思考和工具调用

        if MEMORY_ON:
            # ★ 全项目唯一的长期记忆保存点！存档前先做"摘要压缩"
            messages = compact_messages(turn, MAX_KEPT_MESSAGES, SUMMARY_KEEP_RECENT)

        # MEMORY_ON = False 时，turn 用完即扔：下一轮对它来说又是初次见面


if __name__ == "__main__":
    main()
