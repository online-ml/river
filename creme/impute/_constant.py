class Constant:
    '''
    Class designed to return a constant.

    Args:
        constant_value (float): A constant_value
    '''

    def __init__(self, constant_value):
        self.constant_value = constant_value

    def update(self, x):
        return self

    def get(self):
        return self.constant_value
