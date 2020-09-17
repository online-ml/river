from .htc_nodes import ActiveLearningNodeMC, ActiveLearningNodeNB, ActiveLearningNodeNBA, \
    InactiveLearningNodeMC


def transform_label(y):
    y = ''.join(str(e) for e in y)
    return int(y, 2)


class LCActiveLearningNodeMC(ActiveLearningNodeMC):
    """ Active Learning node for the Label Combination Hoeffding Tree.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations

    """
    def __init__(self, initial_stats=None):
        super().__init__(initial_stats)

    def learn_one(self, X, y, *, weight=1.0, tree=None):
        y = transform_label(y)
        super().learn_one(X, y, weight=weight, tree=tree)


class LCInactiveLearningNodeMC(InactiveLearningNodeMC):
    """ Inactive Learning node for the Label Combination Hoeffding Tree.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations

    """
    def __init__(self, initial_stats=None):
        """ LCInactiveLearningNode class constructor. """
        super().__init__(initial_stats)

    def learn_one(self, X, y, *, weight=1.0, tree=None):
        y = transform_label(y)
        super().learn_one(X, y, weight=weight, tree=tree)


class LCActiveLearningNodeNB(ActiveLearningNodeNB):
    """ Learning node for the Label Combination Hoeffding Tree that uses Naive
    Bayes models.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations

    """
    def __init__(self, initial_stats=None):
        """ LCLearningNodeNB class constructor. """
        super().__init__(initial_stats)

    def learn_one(self, X, y, *, weight=1.0, tree=None):
        y = transform_label(y)
        super().learn_one(X, y, weight=weight, tree=tree)


class LCActiveLearningNodeNBA(ActiveLearningNodeNBA):
    """ Learning node for the Label Combination Hoeffding Tree that uses
    Adaptive Naive Bayes models.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations

    """
    def __init__(self, initial_stats=None):
        """LCLearningNodeNBA class constructor. """
        super().__init__(initial_stats)

    def learn_one(self, X, y, *, weight=1.0, tree=None):
        y = transform_label(y)
        super().learn_one(X, y, weight=weight, tree=tree)
