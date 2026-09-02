"""_meta.request_strategies — the locked decoding regime, recorded on disk.

Two runs of the same model can lock different rungs (one provider held the
schema, another was walked down to json_object). Without this key the two
reports are indistinguishable. The client records the lock on GLOBAL_STATS —
the same blackboard the token counts travel on — so report builders read it
without importing the client.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from contextchecker.llmclient import LLMClient, RETRY_MATRIX
from contextchecker.stats import GLOBAL_STATS


@pytest.fixture(autouse=True)
def _clean():
    GLOBAL_STATS._strategies.clear()
    LLMClient._STRATEGY_CACHE.clear()
    yield
    GLOBAL_STATS._strategies.clear()
    LLMClient._STRATEGY_CACHE.clear()


class TestBlackboard:

    def test_records_and_reads_back(self):
        GLOBAL_STATS.record_strategy("m1", "http://x/v1", "Schema Only")
        GLOBAL_STATS.record_strategy("m2", "http://x/v1", "Reasoning + JSON")
        assert GLOBAL_STATS.strategies() == {"m1": "Schema Only", "m2": "Reasoning + JSON"}

    def test_empty_gives_empty_dict(self):
        assert GLOBAL_STATS.strategies() == {}

    def test_same_model_on_two_endpoints_keeps_both(self):
        GLOBAL_STATS.record_strategy("m", "http://a/v1", "Reasoning + Schema")
        GLOBAL_STATS.record_strategy("m", "http://b/v1", "Unguided Decoding")
        out = GLOBAL_STATS.strategies()
        assert out["m"] == "Reasoning + Schema"
        assert out["m @ http://b/v1"] == "Unguided Decoding"

    def test_relock_on_same_endpoint_overwrites(self):
        GLOBAL_STATS.record_strategy("m", "http://a/v1", "Reasoning + Schema")
        GLOBAL_STATS.record_strategy("m", "http://a/v1", "Schema Only")
        assert GLOBAL_STATS.strategies() == {"m": "Schema Only"}


def _ok():
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = '{"triplets": []}'
    r.choices[0].finish_reason = "stop"
    r.usage.model_dump.return_value = {"prompt_tokens": 1, "completion_tokens": 1}
    r._hidden_params = {}
    return r


class TestClientRecordsTheLock:
    """The client writes at the moment it locks — no other code path does."""

    def test_direct_endpoint_records_the_rung_name(self):
        from contextchecker.workers.extractor import ExtractionResult
        with patch("contextchecker.llmclient.AsyncOpenAI"):
            c = LLMClient(api_key="k", model="m", base_url="http://x/v1")
        c._connection_verified = True
        c._strategy_index = 1            # discovery starts on 'Schema Only'
        c.client.chat.completions.create = AsyncMock(return_value=_ok())

        asyncio.run(c.generate([{"role": "user", "content": "x"}],
                               schema=ExtractionResult, task="extract"))
        assert GLOBAL_STATS.strategies() == {"m": "Schema Only"}

    def test_every_rung_name_is_the_matrix_name(self):
        from contextchecker.workers.extractor import ExtractionResult
        for idx, rung in enumerate(RETRY_MATRIX[:2]):     # the schema rungs lock cleanly
            GLOBAL_STATS._strategies.clear()
            LLMClient._STRATEGY_CACHE.clear()
            with patch("contextchecker.llmclient.AsyncOpenAI"):
                c = LLMClient(api_key="k", model="m", base_url="http://x/v1")
            c._connection_verified = True
            c._strategy_index = idx
            c.client.chat.completions.create = AsyncMock(return_value=_ok())
            asyncio.run(c.generate([{"role": "user", "content": "x"}],
                                   schema=ExtractionResult, task="extract"))
            assert GLOBAL_STATS.strategies()["m"] == rung.name

    def test_litellm_records_the_routing_mode_not_a_rung(self):
        """LiteLLM locks an index without ever consulting the matrix."""
        c = LLMClient(api_key="k", model="openrouter/foo", base_url=None)
        c._connection_verified = True
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=_ok()):
            asyncio.run(c.generate([{"role": "user", "content": "x"}], task="extract"))
        assert GLOBAL_STATS.strategies() == {"openrouter/foo": "LiteLLM passthrough"}

    def test_adopting_a_cached_lock_does_not_rerecord(self):
        """A sibling adopting the process cache never locked — it must not write."""
        LLMClient._STRATEGY_CACHE[("http://x/v1", "m")] = 2
        with patch("contextchecker.llmclient.AsyncOpenAI"):
            c = LLMClient(api_key="k", model="m", base_url="http://x/v1")
        assert c._strategy_discovered is True
        assert GLOBAL_STATS.strategies() == {}


class TestWiring:
    """The key must reach a real report's _meta, without the pipeline touching the client."""

    def test_refcheck_meta_carries_the_locked_strategy(self):
        from contextchecker.pipelines.refchecker import RefCheckerPipeline

        GLOBAL_STATS.record_strategy("ext-model", "http://x/v1", "Reasoning + JSON")

        with patch("contextchecker.pipelines.refchecker.ExtractionService"), \
             patch("contextchecker.pipelines.refchecker.CheckingService"):
            p = RefCheckerPipeline(extractor_model="ext-model",
                                   checker_model="chk-model", verbosity="silent")

        async def passthrough(data):
            return data
        p._extraction.run = passthrough
        p._checking.run = passthrough

        p.run_sync([{"response": "r", "reference": ["ref"]}])
        assert p.last_report["_meta"]["request_strategies"] == {"ext-model": "Reasoning + JSON"}

    def test_only_workers_import_the_client(self):
        """The layering rule this feature must not break."""
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[2] / "src" / "contextchecker"
        offenders = sorted(
            str(p.relative_to(root)) for p in root.rglob("*.py")
            if "from contextchecker.llmclient import" in p.read_text(encoding="utf-8")
            and p.parent.name != "workers"
        )
        assert offenders == []

    def test_meta_core_keys_still_come_first(self):
        from contextchecker.utils import build_meta
        m = build_meta("refcheck", timestamp="t", duration_seconds=1.0,
                       total_items=1, evaluated_items=1, dropped_items=0,
                       request_strategies={"m": "Schema Only"})
        assert list(m)[:8] == [
            "schema_version", "report_type", "contextchecker_version",
            "timestamp", "duration_seconds", "total_items",
            "evaluated_items", "dropped_items",
        ]
        assert m["request_strategies"] == {"m": "Schema Only"}
