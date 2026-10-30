from __future__ import annotations

import json
import typing
from urllib.parse import urljoin

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    import requests

    class _RequestOptions(typing.TypedDict, total=False):
        json: dict[str, typing.Any]
        params: dict[str, str]
        stream: bool

    class _Rule(typing.TypedDict):
        id: str
        value: str
        tag: typing.NotRequired[str]

    class _Rules(typing.TypedDict, total=False):
        data: list[_Rule]
        meta: dict[str, typing.Any]

    class _RuleSpec(typing.TypedDict):
        value: str
        tag: str


class TwitterLiveStream:
    """Twitter API v2 live stream client.

        This client gives access to a live stream of Tweets. That is, Tweets that have just been
        published. This is different to `stream.TwitterRecentStream`, which also covers Tweets that
        have been published over recent days, and not necessarily in real-time.

        A list of filtering rules has to be provided. For instance, this allows focusing on a subset of
        topics and/or users.

        !!! note

            Using this requires having the [`requests`](https://requests.readthedocs.io/en/latest/)
            package installed.

        Parameters
        ----------
        rules
            See the [documentation](https://developer.twitter.com/en/docs/twitter-api/tweets/filtered-stream/integrate/build-a-rule) for a comprehensive overview of filtering rules.
        bearer_token
            A bearer token that is available in each account's developer portal.

        Examples
        --------

        The live stream is instantiated by passing a list of filtering rules, as well as a bearer
        token. For instance, we can listen to all the breaking news Tweets from the BBC and CNN.

        >>> from river import stream

        >>> tweets = stream.TwitterLiveStream(
        ...     rules=["from:BBCBreaking", "from:cnnbrk"],
        ...     bearer_token="<insert_bearer_token>"
        ... )

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
    Russian forces, has been reconnected to Ukraine's electricity grid, according to the
    country's nuclear operator https://t.co/xfylkBs4JR"
            },
            'includes': {
                'users': [
                    {
                        'created_at': '2007-01-02T01:48:14.000Z',
                        'id': '428333',
                        'name': 'CNN Breaking News',
                        'username': 'cnnbrk'
                    }
                ]
            },
            'matching_rules': [{'id': '1563148866333151233', 'tag': 'from:cnnbrk'}]
        }
        ```

        References
        ----------
        [^1]: [Filtered stream introduction](https://developer.twitter.com/en/docs/twitter-api/tweets/filtered-stream/introduction)
        [^2]: [Building rules for filtered stream](https://developer.twitter.com/en/docs/twitter-api/tweets/filtered-stream/integrate/build-a-rule)
        [^3]: [Stream Tweets in real-time](https://developer.twitter.com/en/docs/tutorials/stream-tweets-in-real-time)

    """

    def __init__(self, rules: list[str], bearer_token: str) -> None:
        self.rules = rules
        self.bearer_token = bearer_token

    def _request(
        self, method: str, endpoint: str, **kwargs: typing.Unpack[_RequestOptions]
    ) -> requests.Response:
        import requests

        url = urljoin("https://api.twitter.com/2/", endpoint)
        r = requests.request(
            method,
            url,
            headers={
                "Authorization": f"Bearer {self.bearer_token}",
                "User-Agent": "v2FilteredStreamPython",
            },
            **kwargs,
        )
        r.raise_for_status()
        return r

    def _get_rules(self) -> _Rules:
        return typing.cast("_Rules", self._request("GET", "tweets/search/stream/rules").json())

    def _delete_rules(self, rules: _Rules) -> dict[str, typing.Any] | None:
        if rule_ids := [rule["id"] for rule in rules.get("data", [])]:
            payload = {"delete": {"ids": rule_ids}}
            response = self._request("POST", "tweets/search/stream/rules", json=payload)
            return typing.cast("dict[str, typing.Any]", response.json())
        return None

    def _set_rules(self, rules: list[_RuleSpec]) -> dict[str, typing.Any]:
        payload = {"add": rules}
        response = self._request("POST", "tweets/search/stream/rules", json=payload)
        return typing.cast("dict[str, typing.Any]", response.json())

    def __iter__(self) -> Iterator[dict[str, typing.Any]]:
        existing_rules = self._get_rules()
        self._delete_rules(existing_rules)
        self._set_rules([{"value": rule, "tag": rule} for rule in existing_rules])
        params = {
            "tweet.fields": "created_at",
            "expansions": "author_id",
            "user.fields": "created_at",
        }
        r = self._request("GET", "tweets/search/stream", stream=True, params=params)
        for response_line in r.iter_lines():
            if not response_line:
                continue
            tweet = json.loads(response_line)
            yield tweet
