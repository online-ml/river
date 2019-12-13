from skmultiflow.data import WaveformGenerator
from skmultiflow.trees.hoeffding_tree import HoeffdingTreeClassifier
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential

# 1. Create a stream
stream = WaveformGenerator()
stream.prepare_for_use()

# 2. Instantiate the HoeffdingTreeClassifier classifier
ht = HoeffdingTreeClassifier()

# 3. Setup the evaluator
evaluator = EvaluatePrequential(show_plot=False,
                                pretrain_size=200,
                                max_samples=20000)

# 4. Run evaluation
evaluator.evaluate(stream=stream, model=ht)