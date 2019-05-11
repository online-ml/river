from timeit import default_timer as timer
import logging


class EvaluateStreamGenerationSpeed(object):
    """ EvaluateStreamGeneration
    
    Measures the stream's sample generation time.
    
    Parameters
    ----------
    n_samples: int (Default: 100000)
        The number of samples to generate.
    
    max_time: float (Default: float("inf"))
        The maximum simulation time.
    
    output_file: string, optional (Default: None)
        The name of the output file (Not yet implemented).
        
    batch_size: int (Default: 1)
        The number of samples to generate at a time.
    
    Examples
    --------
    >>> from skmultiflow.data.random_rbf_generator_drift import RandomRBFGeneratorDrift
    >>> from skmultiflow.evaluation.evaluate_stream_gen_speed import EvaluateStreamGenerationSpeed
    >>> stream = RandomRBFGeneratorDrift(change_speed=0.2)
    >>> stream.prepare_for_use()
    >>> evaluator = EvaluateStreamGenerationSpeed(100000, float("inf"), None, 5)
    >>> stream = evaluator.evaluate(stream)
    Evaluation time: 110.064
    Generated 100000 samples
    Samples/second = 908.56
    
    """
    def __init__(self, n_samples=100000, max_time=float("inf"), output_file=None, batch_size=1):
        super().__init__()
        self.num_samples = n_samples
        self.max_time = max_time
        self.output_file = output_file
        self.batch_size = batch_size

    def evaluate(self, stream):
        """ evaluate
        
        This function will evaluate the stream passed as parameter.
        
        Parameters
        ----------
        stream: A stream (an extension from BaseInstanceStream) 
            The stream from which to draw the samples. 
        
        Returns
        -------
        BaseInstanceStream
            The used stream.
        
        """
        self._measure_stream_speed(stream)
        return stream

    def _measure_stream_speed(self, stream):
        logging.basicConfig(format='%(message)s', level=logging.INFO)
        sample_count = 0
        init_time = timer()
        true_percentage_index = 0
        logging.info('Measure generation speed of %s samples', str(self.num_samples))
        logging.info('Evaluating...')
        stream_local_max = float("inf") if (stream.n_remaining_samples() == -1) \
            else stream.n_remaining_samples()
        while ((timer() - init_time <= self.max_time) & (sample_count+self.batch_size <= self.num_samples)
               & (sample_count+self.batch_size <= stream_local_max)):
            stream.next_sample(self.batch_size)
            sample_count += self.batch_size
            while float(sample_count) + self.batch_size >= (((true_percentage_index + 1) * self.num_samples) / 20):
                true_percentage_index += 1
                logging.info('%s%%', str(true_percentage_index*5))
        end_time = timer()
        logging.info('Evaluation time: %s', str(round(end_time - init_time, 3)))
        logging.info('Generated %s samples', str(sample_count))
        logging.info('Samples/second = %s', str(round(sample_count/(end_time-init_time), 3)))

    def set_params(self, parameter_dict):
        """ set_params

        This function allows the users to change some of the evaluator's parameters, 
        by passing a dictionary where keys are the parameters names, and values are 
        the new parameters' values.
        
        Parameters
        ----------
        parameter_dict: Dictionary
            A dictionary where the keys are the names of attributes the user 
            wants to change, and the values are the new values of those attributes.

        """
        for name, value in parameter_dict.items():
            if name == 'n_samples':
                self.num_samples = value
            elif name == 'max_time':
                self.max_time = value
            elif name == 'output_file':
                self.output_file = value
            elif name == 'batch_size':
                self.batch_size = value

    def get_info(self):
        return 'EvaluateStreamGenerationSpeed: n_samples: ' + str(self.num_samples) + \
               ' - max_time: ' + (str(self.max_time)) + \
               ' - output_file: ' + (self.output_file if self.output_file is not None else 'None') + \
               ' - batch_size: ' + str(self.batch_size)
