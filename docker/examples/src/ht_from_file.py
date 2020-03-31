from skmultiflow.data.file_stream import FileStream
from skmultiflow.trees.hoeffding_tree import HoeffdingTreeClassifier
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential

# Create a stream
stream = FileStream("elec.csv")
stream.prepare_for_use()   # Not required for v0.5.0+

# Instantiate the HoeffdingTreeClassifier
ht = HoeffdingTreeClassifier()

# Setup the evaluator
evaluator = EvaluatePrequential(pretrain_size=1000,
                                max_samples=10000,
                                output_file='results.csv')

# Run evaluation
evaluator.evaluate(stream=stream, model=ht)

