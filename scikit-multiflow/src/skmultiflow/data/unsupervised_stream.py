from skmultiflow.data.base_stream import Stream


class UnsupervisedStream(Stream):

    def __init__(self):
        super().__init__()
        self.target = None
        self.n_targets = 1
        self.target_values = set()
        self.target_names = ""
        self.timestamp = -1
        self.weight = 1.0
        self.X = []
        self.y = None

    def prepare_for_use(self):
        pass

    def next_sample(self, batch_size=1, get_weight=False, get_timestamp=False):
        if get_weight:
            return self.X, self.y, self.weight
        if get_timestamp:
            return self.X, self.y, self.timestamp
        return self.X, self.y

    @property
    def target(self):
        """ Retrieve the current label.

        Returns
        -------
        str
            The name of the label.
        """
        return self.target

    @target.setter
    def target(self, target):
        """ Set the current label.

        """
        self.target = target

    @property
    def n_targets(self):
        """ Retrieve the number of labels.

        Returns
        -------
        int
            The number of labels.
        """
        return self.n_targets

    @n_targets.setter
    def n_targets(self, n_targets):
        """ Set the number of labels.

        """
        self.n_targets = n_targets

    @property
    def target_values(self):
        """ Retrieve the set of all the labels.

        Returns
        -------
        set()
            The set of all the labels.
        """
        return self.target_values

    @target_values.setter
    def target_values(self, target_values):
        """ Set the set of labels.

        """
        self.target_values = target_values

    @property
    def target_names(self):
        """ Retrieve all the label names.

        Returns
        -------
        str
            A string of all the label names.
        """
        return self.target_names

    @target_names.setter
    def target_names(self, target_names):
        """ Set the names of all the labels.

        """
        self.target_names = target_names

    @property
    def timestamp(self):
        """ Retrieve the timestamp for the current label.

        Returns
        -------
        int
            An int to denote the timestamp.
        """
        return self.timestamp

    @timestamp.setter
    def timestamp(self, timestamp):
        """ Set the timestamp for the current label.

        """
        self.timestamp = timestamp

    @property
    def weight(self):
        """ Retrieve the weight.

        Returns
        -------
        float
            A float to denote the weight
        """
        return self.weight

    @weight.setter
    def weight(self, weight):
        """ Set the current weight.

        """
        self.weight = weight
