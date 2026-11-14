# Unreleased

## stream

- stream.TwitchChatStream now reconstructs messages split across socket reads and raises ConnectionError when the server disconnects.
