from skmultiflow.data import WaveformGenerator
from skmultiflow.trees.hoeffding_tree import HoeffdingTreeClassifier
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential

# Create a stream
stream = WaveformGenerator()
stream.prepare_for_use()   # Not required for v0.5.0+

# Instantiate the HoeffdingTreeClassifier
ht = HoeffdingTreeClassifier()

# Setup the evaluator
evaluator = EvaluatePrequential(show_plot=False,
                                pretrain_size=200,
                                max_samples=20000)

# Run evaluation
evaluator.evaluate(stream=stream, model=ht)