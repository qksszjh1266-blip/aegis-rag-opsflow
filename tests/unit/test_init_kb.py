"""种子知识清单必须完整且逐份声明权限，校验失败前不连接基础设施。"""

import asyncio
import importlib.util
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("init_kb", _ROOT / "scripts" / "init_kb.py")
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.parametrize("unregistered", [False, True])
async def test_seed_manifest_fails_closed_before_storage_access(
    tmp_path, monkeypatch, unregistered
):
    def prepare():
        shutil.copytree(_ROOT / "data" / "seed_docs", tmp_path, dirs_exist_ok=True)
        if unregistered:
            (tmp_path / "undeclared_finance.md").write_text(
                "# 尚未登记权限的财务文档", encoding="utf-8"
            )
        else:
            (tmp_path / "ec_finance_internal.md").unlink()

    await asyncio.to_thread(prepare)
    engine = Mock()
    monkeypatch.setattr(_MODULE, "create_mysql_engine", engine)
    with pytest.raises(ValueError, match="种子清单不完整"):
        await _MODULE.initialize(tmp_path)
    engine.assert_not_called()
