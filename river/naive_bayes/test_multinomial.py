import math

from river import compose, feature_extraction, naive_bayes


def test_predict_class_given_unseen_features():
    model = compose.Pipeline(
        ("tokenize", feature_extraction.BagOfWords()),
        ("nb", naive_bayes.MultinomialNB(alpha=1)),
    )

    docs = [("cloudy cold", 0), ("sunny warm", 1)]

    for sentence, label in docs:
        model = model.learn_one(sentence, label)

    # Assert model parameters needed to calculate the likelihoods
    assert model["nb"].n_terms == 4
    assert model["nb"].class_totals[0] == 2
    assert model["nb"].class_totals[1] == 2

    # Given new, unseen text, predict the label
    text = "new word"
    tokens = model["tokenize"].transform_one(text)
    cp = model["nb"].p_feature_given_class

    # P(new|0)
    #   = (N_new_0 + 1) / N_0 + N_terms)
    #   = (0 + 1) / (model['nb'].class_totals[0] + model['nb'].n_terms)
    assert cp("new", 0) == (0 + 1) / (2 + 4)

    # Since class_totals[0] == class_totals[1], and both words in text are new/unseen,
    # expect the class-conditional probabilities to be the same
    assert cp("new", 0) == cp("word", 0)
    assert cp("new", 0) == cp("new", 1)
    assert cp("new", 0) == cp("word", 1)

    jll = model["nb"].joint_log_likelihood(tokens)

    # Expect JLLs to be equal
    assert jll[0] == jll[1]

    # P(0|new word)
    #   = P(new|0) * P(word|0) * P(0)
    assert jll[0] == math.log(cp("new", 0) * cp("word", 0) * (1 / 2))

    # JLLs for both labels are the same, but 0 was the first label to be added to model['nb'].class_counts
    assert model.predict_one(text) == 0
