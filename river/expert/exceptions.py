class NotEnoughModels(ValueError):
    def __init__(self, n_expected, n_obtained):
        message = (
            f"At least {n_expected} models are expected, only {n_obtained} were passed"
        )
        super().__init__(message)
