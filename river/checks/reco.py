import random


def check_reco_routine(recommender):

    users = ["Tom", "Anna"]
    items = {"politics", "sports", "music", "food", "finance", "health", "camping"}

    def get_reward(user, item) -> bool:
        if user == "Tom":
            return item in {"music", "politics"}
        if user == "Anna":
            return item in {"politics", "sports"}

    for i in range(100):

        user = random.choice(users)
        item = recommender.recommend(user, k=1, items=items, strategy="best")[0]

        clicked = get_reward(user, item)

        recommender.learn_one({"user": user, "item": item}, clicked)
