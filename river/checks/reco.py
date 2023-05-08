from __future__ import annotations

import random


def check_reco_routine(ranker):
    users = ["Tom", "Anna"]
    items = {"politics", "sports", "music", "food", "finance", "health", "camping"}

    def get_reward(user, item) -> bool:
        if user == "Tom":
            return item in {"music", "politics"}
        return item in {"politics", "sports"}

    for i in range(100):
        user = random.choice(users)
        item = ranker.rank(user, items)[0]

        clicked = get_reward(user, item)

        ranker.learn_one(user, item, clicked)
