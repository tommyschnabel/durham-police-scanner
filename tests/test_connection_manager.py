"""Tests for the WebSocket fan-out."""

import json

import pytest

from connection_manager import ConnectionManager


class FakeWebSocket:
    def __init__(self, fail=False):
        self.accepted = False
        self.sent = []
        self.fail = fail

    async def accept(self):
        self.accepted = True

    async def send_text(self, message):
        if self.fail:
            raise ConnectionResetError("client went away")
        self.sent.append(message)


@pytest.fixture
def manager():
    return ConnectionManager()


@pytest.mark.asyncio
async def test_connect_accepts_and_registers(manager):
    ws = FakeWebSocket()
    await manager.connect(ws)
    assert ws.accepted is True
    assert manager.active_connections == [ws]


@pytest.mark.asyncio
async def test_disconnect_removes_the_client(manager):
    ws = FakeWebSocket()
    await manager.connect(ws)
    manager.disconnect(ws)
    assert manager.active_connections == []


def test_disconnecting_an_unknown_client_is_a_no_op(manager):
    manager.disconnect(FakeWebSocket())  # must not raise
    assert manager.active_connections == []


@pytest.mark.asyncio
async def test_broadcast_reaches_every_client(manager):
    a, b = FakeWebSocket(), FakeWebSocket()
    await manager.connect(a)
    await manager.connect(b)

    await manager.broadcast("hello")
    assert a.sent == ["hello"] and b.sent == ["hello"]


@pytest.mark.asyncio
async def test_a_dead_client_is_dropped_and_does_not_block_the_others(manager):
    dead, alive = FakeWebSocket(fail=True), FakeWebSocket()
    await manager.connect(dead)
    await manager.connect(alive)

    await manager.broadcast("hello")

    assert alive.sent == ["hello"]
    assert manager.active_connections == [alive]


@pytest.mark.asyncio
async def test_broadcast_json_serialises_the_payload(manager):
    ws = FakeWebSocket()
    await manager.connect(ws)

    await manager.broadcast_json({"type": "transcript", "text": "unit 12"})
    assert json.loads(ws.sent[0]) == {"type": "transcript", "text": "unit 12"}


@pytest.mark.asyncio
async def test_send_message_drops_a_client_that_errors(manager):
    ws = FakeWebSocket(fail=True)
    await manager.connect(ws)

    await manager.send_message("hello", ws)
    assert manager.active_connections == []
