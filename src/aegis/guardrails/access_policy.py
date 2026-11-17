"""输入侧指令完整性检查；调用方权限只能来自认证上下文。"""

import re
import unicodedata

SAFE_EVIDENCE_REFUSAL = "当前授权知识库中没有可核验的依据，请联系有权限的人员处理。"

# 仅识别改变系统规则、提取内部提示词等指令，不按具体部门或知识内容封禁提问。
_OVERRIDE_PATTERNS = (
    r"(?:忽略|跳过|绕过|无视).{0,16}(?:权限|授权|系统规则|系统指令|之前的指令)",
    r"(?:不用|无需|不需要).{0,8}(?:权限校验|权限检查|授权校验)",
    r"(?:没有|无|不受).{0,3}权限限制",
    r"系统管理员指令",
    r"你现在.{0,8}(?:进入|处于|切换到)维护模式",
    r"(?:输出|打印|显示|泄露|提供|发送).{0,20}(?:系统提示词|隐藏指令|内部提示词)",
    r"(?:系统提示词|隐藏指令|内部提示词).{0,24}(?:发给我|输出|打印|编码|提供)",
    r"上一条.{0,8}测试消息.{0,8}真实指令",
)


def is_instruction_override(text: str) -> bool:
    """规范化混淆字符后检测指令篡改，返回布尔值而不拼接攻击内容。"""
    normalized = "".join(
        char
        for char in unicodedata.normalize("NFKC", text).casefold()
        if not char.isspace() and unicodedata.category(char) != "Cf"
    )
    return any(re.search(pattern, normalized) for pattern in _OVERRIDE_PATTERNS)
