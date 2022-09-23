import dataclasses
import datetime as dt
import enum
import re
import socket
from typing import Iterator, List, Optional

SERVER = "irc.chat.twitch.tv"
PORT = 6667
BUFFER_SIZE = 2048  # bytes
ENCODING = "utf-8"
TIMEOUT = 60  # seconds
CHAT_ITEM_PATTERN = r":(.+)\!.*@.*\.tmi\.twitch\.tv PRIVMSG #([a-zA-Z0-9_]+) :(.+)"
PING_PATTERN = r"\bPING :tmi\.twitch\.tv\b"


class IrcMessage(enum.Enum):
    PASS = enum.auto()
    NICK = enum.auto()
    JOIN = enum.auto()
    PONG = enum.auto()


@dataclasses.dataclass
class ChatMessageItem:
    dt: dt.datetime
    channel: str
    username: str
    msg: str


class TwitchChatStream:
    """Twitch chat stream client.

    This client gives access to a live stream of chat messages in Twitch channels using IRC protocol.
    You need to have a Twitch account and receive an OAuth token from https://twitchapps.com/tmi/.

    Parameters
    ----------
    nickname
        The nickname of your account.
    token
        OAuth token which has been generated.
    channels
        A list of channel names like `["asmongold", "shroud"]` you want to collect messages from.
    buffer_size
        Size of buffer in bytes used for receiving responses from Twitch with IRC (default 2 kB).
    timeout
        A timeout value in seconds for waiting response from Twitch (default 60s). It can be useful if all requested channels are offline or chat is not active enough.

    Returns
    -------
    Stream items are python dictionaries with keys:
    dt (datetime)
        Datetime indicating when response with this message has been received from Twitch.
    channel (str)
        A channel where message has been posted.
    username (str)
        User who posted the message.
    msg (str)
        The message itself.

    Examples
    --------
    The live stream is instantiated by passing your Twitch account nickname, OAuth token and list of channels. Other parameters are optional.

    >>> from river import stream

    >>> twitch_chat = stream.TwitchChatStream(
    ...     nickname="twitch_user1",
    ...     token="oauth:okrip6j6fjio8n5xpy2oum1lph4fbve",
    ...     channels=["asmongold", "shroud"]
    ... )

    The stream can be iterated over like this:

    ```py
    for item in twitch_chat:
        print(item)
    ```

    Here's a single stream item example:
    ```py
    {
        'dt': datetime.datetime(2022, 9, 14, 10, 33, 37, 989560),
        'channel': 'asmongold',
        'username': 'moojiejaa',
        'msg': 'damn this chat mod are wild'
    }
    ```

    References
    ----------
    [^1]: [Twitch IRC doc](https://dev.twitch.tv/docs/irc)

    """

    def __init__(
        self,
        nickname: str,
        token: str,
        channels: List[str],
        buffer_size: int = BUFFER_SIZE,
        timeout: int = TIMEOUT,
    ):
        self.nickname = nickname
        self.token = token
        self.channels = channels
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.chat_item_pattern = re.compile(CHAT_ITEM_PATTERN)
        self.ping_pattern = re.compile(PING_PATTERN)

    def _send(self, s: socket.socket, msg: IrcMessage, payload: Optional[str] = None) -> None:
        text = msg.name
        if payload:
            text += f" {payload}"
        s.send(f"{text}\n".encode(ENCODING))

    def _setup_connection(self, s: socket.socket) -> None:
        s.connect((SERVER, PORT))
        self._send(s, IrcMessage.PASS, self.token)
        self._send(s, IrcMessage.NICK, self.nickname)
        for chan in self.channels:
            self._send(s, IrcMessage.JOIN, f"#{chan}")
        s.settimeout(self.timeout)

    def _is_ping(self, resp: str) -> bool:
        return bool(self.ping_pattern.search(resp))

    def _extract_chat_messages(self, resp: str, dt: dt.datetime) -> Iterator[ChatMessageItem]:
        for m in self.chat_item_pattern.finditer(resp):
            if not m or not m.groups():
                continue
            user, chan, msg = m.groups()
            yield ChatMessageItem(dt=dt, channel=chan, username=user, msg=msg.strip())

    def _gen_items(self, sock: socket.socket) -> Iterator[ChatMessageItem]:
        while True:
            try:
                data = sock.recv(self.buffer_size)
                if not data:
                    continue
                resp = data.decode(ENCODING)
                now = dt.datetime.now()
            except socket.timeout as e:
                raise TimeoutError(f"Twitch did not respond in {self.timeout:,d} seconds") from e
            except UnicodeDecodeError:
                continue

            if self._is_ping(resp):
                self._send(sock, IrcMessage.PONG, ":tmi.twitch.tv")

            yield from self._extract_chat_messages(resp, now)

    def __iter__(self) -> Iterator[dict]:
        with socket.socket() as sock:
            self._setup_connection(sock)
            yield from map(dataclasses.asdict, self._gen_items(sock))
