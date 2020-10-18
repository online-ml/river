from river import linear_model


class LinearRegression:

    params = [2, 10, 100]

    def setup(self, p):
        self.x = {i: i for i in range(p)}
        self.model = linear_model.LinearRegression()
        self.fitted_model = linear_model.LinearRegression().learn_one(self.x, 1)

    def time_learn_one(self, p):
        self.model.learn_one(self.x, 1)

    def time_predict_one(self, p):
        self.fitted_model.predict_one(self.x)

    def track_memory_usage(self, p):
        return self.fitted_model._memory_usage_raw
