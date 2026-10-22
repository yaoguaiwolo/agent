"""
tools.py —— 工具层：普通的 Python 函数 + 给 LLM 看的"说明书"。

加一个新工具的标准流程（三处注册，缺一不可）：
    1. 写一个普通函数（接收参数、返回字符串结果）
    2. 在 TOOLBOX 里登记：函数名 -> 函数（LLM 点名时靠它找到人）
    3. 在 TOOL_SCHEMAS 里写说明书（LLM 全靠它决定什么时候用、怎么传参）
"""

import os
from datetime import datetime

from rag import search_documents


def get_current_time() -> str:
    """获取当前的日期和时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


def calculate(expression: str) -> str:
    """计算一个四则运算表达式，例如 3+5*2"""
    # eval 有安全风险，这里只放行数字和运算符号，学习用途足够
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "错误：表达式包含不允许的字符"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算出错：{e}"


def list_files(path: str = ".") -> str:
    """列出某个文件夹里的文件和子文件夹"""
    try:
        entries = os.listdir(path)
        if not entries:
            return "这个文件夹是空的"
        return "\n".join(entries)
    except Exception as e:
        return f"出错：{e}"


def read_file(path: str, limit: int = 50) -> str:
    """读取一个文本文件的前 N 行（默认 50 行）"""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [next(f, "") for _ in range(limit)]
        return f"文件 {path} 的前 {limit} 行：\n" + "".join(lines)
    except Exception as e:
        return f"出错：{e}"


def get_weather(city: str) -> str:
    """获取某个城市的天气情况"""
    return f"{city} 的天气是晴天，温度 25 度。"


# 工具名 -> 函数。LLM 说"我想调用 xxx"，我们就从这个表里找到对应的函数执行
TOOLBOX = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "list_files": list_files,
    "read_file": read_file,
    "get_weather": get_weather,
    "search_documents": search_documents,
}

# 工具说明书：用 JSON Schema 描述每个函数的名字、用途、参数。
# 这份说明书会原样发给 LLM，它全靠这些描述来决定"什么时候用哪个工具"，
# 所以说明书写得越清楚，Agent 用得越准。
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期、时间和星期几。当用户问到和时间有关的问题时使用。",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算四则运算表达式的值，例如 3+5*2 或 (10-4)/3",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，例如 1+2*3"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出指定文件夹里有哪些文件和子文件夹",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件夹路径，例如 D:\\workcode"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取一个文本文件的开头部分",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "limit": {"type": "integer", "description": "读取前几行，默认 50"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取某个城市的天气情况",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，例如北京"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "在用户的私有文档/笔记（knowledge 文件夹）里检索相关内容。"
                           "当用户问到自己的笔记、文档、计划里记了什么，"
                           "或问题可能记录在私有文档中时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要检索的主题或问题关键词"}
                },
                "required": ["query"],
            },
        },
    }
]
