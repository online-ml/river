from river import base


class Uncertainty(): 

    def fixed_uncertainty(self,maximum_posteriori, theta):
        selected = False
        if maximum_posteriori < theta:
            selected = True
        return selected

    def variable_uncertainty(self, maximum_posteriori, s, theta):
        selected = False
        if maximum_posteriori < theta:
            theta = theta*(1-s)
            selected = True
        else:
            theta = theta*(1+s)
            selected = False
        return selected
        