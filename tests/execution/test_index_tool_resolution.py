"""Index-tool lookup only; no dataset export or DALI process is executed."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from nexuml.data.export import webdataset


@pytest.mark.parametrize("on_path", [True, False])
def test_index_tool_preserves_path_then_uses_running_interpreter_scripts(monkeypatch, on_path):
    tool = "/explicit/wds2idx" if on_path else "/private/env/bin/wds2idx"
    lookup = Mock(side_effect=[tool] if on_path else [None, tool])
    run = Mock()
    monkeypatch.setattr(webdataset.shutil, "which", lookup)
    monkeypatch.setattr(webdataset.sysconfig, "get_path", lambda name: "/private/env/bin")
    monkeypatch.setattr(webdataset.subprocess, "run", run)
    result = webdataset.WebDatasetBackend()._generate_index(Path("shard.tar"))
    assert result == Path("shard.idx")
    assert lookup.call_args_list[0].args == ("wds2idx",)
    assert lookup.call_count == (1 if on_path else 2)
    if not on_path:
        assert lookup.call_args.kwargs == {"path": "/private/env/bin"}
    assert run.call_args.args[0] == [tool, "shard.tar", "shard.idx"]
    assert run.call_args.kwargs["check"] is True
