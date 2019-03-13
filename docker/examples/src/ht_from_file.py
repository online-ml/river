from skmultiflow.data.file_stream import FileStream
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential

# 1. Create a stream
stream = FileStream("elec.csv")
stream.prepare_for_use()

# 2. Instantiate the HoeffdingTree classifier
ht = HoeffdingTree()

# 3. Setup the evaluator
evaluator = EvaluatePrequential(pretrain_size=1000,
                                max_samples=10000,
                                output_file='results.csv')

# 4. Run evaluation
evaluator.evaluate(stream=stream, model=ht)

