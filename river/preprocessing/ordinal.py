from __future__ import annotations

import collections

import numpy as np

from river import base


class OrdinalEncoder(base.MiniBatchTransformer):
    # INFO: makes sense to add None initially? so it'll be easier to spot
    def __init__(
        self, handle_unknown="use_reserved_value", unknown_value=1, encoded_missing_value=0
    ):
        # self.categories = categories
        # self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value
        self.categories_ = collections.defaultdict(dict)
        # self.categories_ = collections.defaultdict(collections.defaultdict)

    def learn_one(self, x):
        for i, xi in x.items():
            if self.handle_unknown == "use_reserved_value":
                # if self.categories_[i].get(xi, None) is None:
                if xi not in self.categories_[i]:
                    if xi is None:
                        self.categories_[i][xi] = self.encoded_missing_value
                    else:
                        self.categories_[i][xi] = 2 + len(self.categories_[i])
        # self.abs_max[i].update(xi)

        return self

    def transform_one(self, x):
        # return {i: self.categories_[i][xi] for i, xi in x.items()}
        return {i: self.categories_[i].get(xi, self.unknown_value) for i, xi in x.items()}

    def learn_many(self, X):
        for col in X.columns:
            # self.values[col].update(X[col].unique())
            known_uniqs = np.array(
                list(self.categories_[col].keys())
            )  # INFO: convert to numpy array for np.isin to work properly, otherwise unexpected behavior
            current_uniqs = X[col].unique()

            # if None in known_uniqs:
            #     current_uniqs.dropna(inplace=True)
            # elif
            upd_encoding = dict()

            # current_uniqs = pd.DataFrame(X)['country'].dropna().unique()

            # new_unique_mask = np.isin(current_uniqs, known_uniqs, assume_unique=True, invert=True)
            new_unique_mask = np.isin(current_uniqs, known_uniqs, assume_unique=True, invert=True)

            new_uniqs = current_uniqs[new_unique_mask]
            print(f"{known_uniqs=}")
            print(f"{current_uniqs=}")
            print(f"{new_uniqs=}")

            # INFO: record and mask the first encounters of None or equivalents in the batch
            if None in new_uniqs:
                print("None detected")

                upd_encoding.update({None: self.encoded_missing_value})
                none_mask = np.isin(new_uniqs, [None], assume_unique=True, invert=True)
                new_uniqs = new_uniqs[none_mask]

            # INFO: process first encounters of other unique values except None or equivalents
            upd_encoding.update(
                {
                    k: v + 2 + len(known_uniqs)
                    # for k, v in zip(current_uniqs, range(0, current_uniqs.shape[0]))
                    for v, k in enumerate(new_uniqs)
                    # if (k not in known_uniqs) and (k is not None)
                }
            )

            print(upd_encoding)

            self.categories_[col].update(upd_encoding)

        return self

    def transform_many(self, X):
        X_encoded = X.copy()
        for col in X_encoded.columns:
            X_encoded[col] = X_encoded[col].transform(
                lambda x: self.categories_[col].get(x, self.unknown_value)
            )

        return X_encoded
