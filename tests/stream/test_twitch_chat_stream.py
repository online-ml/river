from __future__ import annotations

import datetime as dt
import socket
from collections.abc import Generator, Sequence
from unittest.mock import MagicMock, call

import pytest

from river import stream

MESSAGE = ":alice!alice@alice.tmi.twitch.tv PRIVMSG #channel :hello café 🌊\r\n".encode()
PING = b"PING :tmi.twitch.tv\r\n"


@pytest.fixture
def sock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    connection = MagicMock(spec=socket.socket)
    connection.__enter__.return_value = connection
    monkeypatch.setattr(socket, "socket", lambda: connection)
    return connection


def make_stream() -> stream.TwitchChatStream:
    return stream.TwitchChatStream("nickname", "token", ["channel"], buffer_size=73, timeout=12)


def receive(sock: MagicMock, chunks: Sequence[bytes | Exception]) -> None:
    sock.recv.side_effect = [*chunks, b"", AssertionError("Received again after disconnect")]


def test_immediate_disconnect(sock: MagicMock) -> None:
    receive(sock, [])
    with pytest.raises(ConnectionError):
        next(iter(make_stream()))
    sock.recv.assert_called_once_with(73)
    sock.__exit__.assert_called_once()


@pytest.mark.parametrize("split", range(1, len(MESSAGE)))
def test_message_split(sock: MagicMock, split: int) -> None:
    receive(sock, [MESSAGE[:split], MESSAGE[split:]])
    items = iter(make_stream())
    item = next(items)
    assert item["username"] == "alice"
    assert item["channel"] == "channel"
    assert item["msg"] == "hello café 🌊"
    assert isinstance(item["dt"], dt.datetime)
    with pytest.raises(ConnectionError):
        next(items)
    assert sock.recv.call_args_list == [call(73)] * 3
    sock.__exit__.assert_called_once()


def test_multiple_messages_and_partial_line(sock: MagicMock) -> None:
    second = MESSAGE.replace(b"alice", b"bob").replace(b"hello", b"goodbye")
    receive(sock, [MESSAGE + second + MESSAGE[:20], MESSAGE[20:]])
    items = iter(make_stream())
    assert [next(items)["username"] for _ in range(3)] == ["alice", "bob", "alice"]
    with pytest.raises(ConnectionError):
        next(items)
    sock.__exit__.assert_called_once()


def test_complete_message_then_disconnect(sock: MagicMock) -> None:
    receive(sock, [MESSAGE])
    items = iter(make_stream())
    assert next(items)["msg"] == "hello café 🌊"
    with pytest.raises(ConnectionError):
        next(items)
    assert sock.recv.call_args_list == [call(73), call(73)]
    sock.__exit__.assert_called_once()


def test_split_ping(sock: MagicMock) -> None:
    receive(sock, [bytes([byte]) for byte in PING] + [MESSAGE + PING])
    items = iter(make_stream())
    assert next(items)["msg"] == "hello café 🌊"
    with pytest.raises(ConnectionError):
        next(items)
    assert sock.send.call_args_list == [
        call(b"PASS token\n"),
        call(b"NICK nickname\n"),
        call(b"JOIN #channel\n"),
        call(b"PONG :tmi.twitch.tv\n"),
        call(b"PONG :tmi.twitch.tv\n"),
    ]
    sock.__exit__.assert_called_once()


@pytest.mark.parametrize("partial", [MESSAGE[:-2], MESSAGE[:-1], b"\xc3"])
def test_unterminated_line_at_disconnect(sock: MagicMock, partial: bytes) -> None:
    receive(sock, [partial])
    with pytest.raises(ConnectionError, match="incomplete IRC message"):
        next(iter(make_stream()))
    assert sock.recv.call_args_list == [call(73), call(73)]
    sock.__exit__.assert_called_once()


def test_invalid_utf8(sock: MagicMock) -> None:
    receive(sock, [b"\xff\r\n"])
    with pytest.raises(UnicodeDecodeError):
        next(iter(make_stream()))
    sock.__exit__.assert_called_once()


def test_timeout(sock: MagicMock) -> None:
    error = TimeoutError("socket timeout")
    receive(sock, [MESSAGE[:20], error])
    with pytest.raises(TimeoutError, match="Twitch did not respond in 12 seconds") as exc:
        next(iter(make_stream()))
    assert exc.value.__cause__ is error
    sock.settimeout.assert_called_once_with(12)
    sock.__exit__.assert_called_once()


def test_generator_close(sock: MagicMock) -> None:
    receive(sock, [MESSAGE + MESSAGE])
    items = iter(make_stream())
    assert next(items)["msg"] == "hello café 🌊"
    assert isinstance(items, Generator)
    items.close()
    sock.recv.assert_called_once_with(73)
    sock.__exit__.assert_called_once()
