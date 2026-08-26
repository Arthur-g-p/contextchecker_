"""Real sockets, real deadline. No mocked SDK anywhere in this file.

httpx has no total-request timeout — its `read` budget measures the gap
between two chunks, so a server that trickles bytes keeps a request alive
indefinitely. These tests hold that behaviour to the wall clock instead.
"""

import asyncio
import time
from unittest.mock import patch

import pytest

from contextchecker.exceptions import LLMTimeoutError
from contextchecker.llmclient import LLMClient, TIMEOUT_RETRIES

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

DEADLINE = 2.0


async def _silent(reader, writer):
    """Accepts the request and never answers."""
    try:
        await reader.read(65536)
        await asyncio.sleep(3600)
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    finally:
        writer.close()


async def _trickle(reader, writer):
    """Headers at once, then one byte per 0.2s — keeps httpx's read timer alive."""
    try:
        await reader.read(65536)
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                     b"Transfer-Encoding: chunked\r\n\r\n")
        await writer.drain()
        for _ in range(300):
            await asyncio.sleep(0.2)
            writer.write(b"1\r\n \r\n")
            await writer.drain()
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


@pytest.fixture
async def servers():
    started = []
    for handler in (_silent, _trickle):
        s = await asyncio.start_server(handler, "127.0.0.1", 0)
        started.append((s, s.sockets[0].getsockname()[1]))
    yield [port for _, port in started]
    for s, _ in started:
        s.close()


def _client(port: int) -> LLMClient:
    with patch("contextchecker.llmclient.default_config") as cfg:
        cfg.LLM_TIMEOUT = DEADLINE
        cfg.LLM_MAX_TOKENS = None
        cfg.PROMPTS = {}
        c = LLMClient(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
    c._connection_verified = True
    c._strategy_discovered = True
    c._discovery_succeeded = True
    return c


@pytest.mark.parametrize("which,label", [(0, "server stays silent"), (1, "server trickles")])
async def test_deadline_fires_and_item_is_skipped(servers, which, label):
    c = _client(servers[which])
    t0 = time.monotonic()
    with pytest.raises(LLMTimeoutError):
        await c.generate([{"role": "user", "content": "hi"}])
    elapsed = time.monotonic() - t0

    attempts = TIMEOUT_RETRIES + 1
    assert elapsed < DEADLINE * attempts * 1.8, f"{label}: ran {elapsed:.1f}s"
    assert elapsed >= DEADLINE * attempts * 0.8, f"{label}: returned too early"
    assert c._fatal_error_occurred is False


async def test_trickling_server_no_longer_runs_forever(servers):
    """The regression: httpx alone would keep this request open indefinitely."""
    c = _client(servers[1])
    t0 = time.monotonic()
    with pytest.raises(LLMTimeoutError):
        await c.generate([{"role": "user", "content": "hi"}])
    assert time.monotonic() - t0 < 10.0


async def test_batch_survives_a_hanging_endpoint(servers):
    c = _client(servers[0])
    results = await c.generate_batch(
        [{"messages": [{"role": "user", "content": f"i{i}"}]} for i in range(3)]
    )
    assert len(results) == 3
    assert all(isinstance(r, LLMTimeoutError) for r in results)
    assert c._fatal_error_occurred is False
