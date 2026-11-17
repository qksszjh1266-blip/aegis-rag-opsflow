"""前置高危词规则只进行本地确定性匹配，不访问检索器或模型。"""

import unicodedata
from collections.abc import Sequence

MANDATORY_HIGH_RISK_KEYWORDS = ("漏电", "冒烟", "起火", "批量资损", "监管投诉")


class HighRiskPrecheck:
    """可扩展词库，但必须保留基础安全红线；否定或引用同样转人工复核。"""

    def __init__(self, keywords: Sequence[str] | None = None) -> None:
        if isinstance(keywords, str):
            raise ValueError("高危词库必须为词条序列")
        configured = tuple(keywords or ())
        if any(not isinstance(word, str) or not word.strip() for word in configured):
            raise ValueError("高危词条必须为非空字符串")
        self.keywords = tuple(dict.fromkeys((*MANDATORY_HIGH_RISK_KEYWORDS, *configured)))

    def check(self, text: str) -> tuple[str, ...]:
        """规范化全角字符与格式控制符，避免不可见字符绕过红线匹配。"""
        normalized = self._normalize(text)
        return tuple(word for word in self.keywords if self._normalize(word) in normalized)

    @staticmethod
    def _normalize(text: str) -> str:
        return "".join(
            char
            for char in unicodedata.normalize("NFKC", text).casefold()
            if not char.isspace() and unicodedata.category(char) != "Cf"
        )
