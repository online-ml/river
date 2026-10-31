# TwitchChatStream

Twitch chat stream client.

This client gives access to a live stream of chat messages in Twitch channels using IRC protocol. You need to have a Twitch account and receive an OAuth token from https://twitchapps.com/tmi/.

## Parameters

- **nickname**

  *Type* → `str`

  The nickname of your account.
- **token**

  *Type* → `str`

  OAuth token which has been generated.
- **channels**

  *Type* → `list[str]`

  A list of channel names like `\["asmongold", "shroud"\]` you want to collect messages from.
- **buffer\_size**

  *Type* → `int`

  *Default* → `2048`

  Size of buffer in bytes used for receiving responses from Twitch with IRC (default 2 KiB).
- **timeout**

  *Type* → `int`

  *Default* → `60`

  A timeout value in seconds for waiting response from Twitch (default 60s). It can be useful if all requested channels are offline or chat is not active enough.

## Examples

The live stream is instantiated by passing your Twitch account nickname, OAuth token and list of channels. Other parameters are optional.

```python
from river import stream

twitch_chat = stream.TwitchChatStream(
    nickname="twitch_user1",
    token="oauth:okrip6j6fjio8n5xpy2oum1lph4fbve",
    channels=["asmongold", "shroud"]
)
```

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

## References

- [Twitch IRC doc](<https://dev.twitch.tv/docs/irc>)
