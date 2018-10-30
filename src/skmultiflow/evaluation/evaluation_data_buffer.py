class EvaluationDataBuffer(object):
    """ Stores evaluation data at a given time. It is used to track and distribute
    data across evaluators and visualizers.
    """
    def __init__(self, data_dict):
        """

        Parameters
        ----------
        data_dict: dict
            Dictionary containing metrics as keys and data identifiers as values.

        """
        self.data_dict = None
        self.data = {}
        self.sample_id = -1
        self._validate(data_dict)
        self._clear_data()

    def _validate(self, data_dict):
        if isinstance(data_dict, dict):
            if len(data_dict.keys()) > 0:
                self.data_dict = data_dict
            else:
                raise ValueError('data_dict is empty')
        else:
            raise TypeError('data_dict must be a dictionary, received: {}'.format(type(data_dict)))

    def _clear_data(self):
        for metric_id, data_ids in self.data_dict.items():
            self.data[metric_id] = {}
            for data_id in data_ids:
                self.data[metric_id][data_id] = None

    def get_data(self, metric_id, data_id):
        return self.data[metric_id][data_id]

    def update_data(self, sample_id, metric_id, data_id, value):
        if self.sample_id != sample_id:
            # New sample id, clear values
            self._clear_data()
            self.sample_id = sample_id
        self.data[metric_id][data_id] = value
