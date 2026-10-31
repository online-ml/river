# TwitterLiveStream

Twitter API v2 live stream client.

This client gives access to a live stream of Tweets. That is, Tweets that have just been published. This is different to `stream.TwitterRecentStream`, which also covers Tweets that have been published over recent days, and not necessarily in real-time.

A list of filtering rules has to be provided. For instance, this allows focusing on a subset of topics and/or users.

> [!NOTE]
>
> Using this requires having the [`requests`](<https://requests.readthedocs.io/en/latest/>) package installed.

## Parameters

- **rules**

  *Type* → `list[str]`

  See the [documentation](<https://developer.twitter.com/en/docs/twitter-api/tweets/filtered-stream/integrate/build-a-rule>) for a comprehensive overview of filtering rules.
- **bearer\_token**

  *Type* → `str`

  A bearer token that is available in each account's developer portal.

## Examples

The live stream is instantiated by passing a list of filtering rules, as well as a bearer token. For instance, we can listen to all the breaking news Tweets from the BBC and CNN.

```python
from river import stream

tweets = stream.TwitterLiveStream(
    rules=["from:BBCBreaking", "from:cnnbrk"],
    bearer_token="<insert_bearer_token>"
)
```

````text
The stream can then be iterated over, possibly in an infinite loop. This will listen to the
live feed of Tweets and produce a Tweet right after it's been published.

```py
import logging

while True:
    try:
        for tweet in tweets:
            print(tweet)
    except requests.exceptions.RequestException as e:
        logging.warning(str(e))
        time.sleep(10)
```

Here's a Tweet example:

```py
{
    'data': {
        'author_id': '428333',
        'created_at': '2022-08-26T12:59:48.000Z',
        'id': '1563149212774445058',
        'text': "Ukraine's Zaporizhzhia nuclear power plant, which is currently held by
````

Russian forces, has been reconnected to Ukraine's electricity grid, according to the country's nuclear operator https://t.co/xfylkBs4JR" }, 'includes': { 'users': \[ { 'created\_at': '2007-01-02T01:48:14.000Z', 'id': '428333', 'name': 'CNN Breaking News', 'username': 'cnnbrk' } \] }, 'matching\_rules': \[{'id': '1563148866333151233', 'tag': 'from:cnnbrk'}\] } \`\`\`

## References

- [Filtered stream introduction](<https://developer.twitter.com/en/docs/twitter-api/tweets/filtered-stream/introduction>)
- [Building rules for filtered stream](<https://developer.twitter.com/en/docs/twitter-api/tweets/filtered-stream/integrate/build-a-rule>)
- [Stream Tweets in real-time](<https://developer.twitter.com/en/docs/tutorials/stream-tweets-in-real-time>)
