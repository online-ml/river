from skmultiflow.data import SEAGenerator
from skmultiflow.evaluation.evaluate_stream_gen_speed import EvaluateStreamGenerationSpeed


def test_evaluate_stream_gen_speed():
    stream = SEAGenerator(random_state=1)
    stream.prepare_for_use()
    stream_name = stream.name

    evaluator = EvaluateStreamGenerationSpeed(n_samples=100000, max_time=float("inf"), output_file=None, batch_size=5)
    stream = evaluator.evaluate(stream)
    assert stream_name == stream.name

    expected_info = 'EvaluateStreamGenerationSpeed: ' \
                    'n_samples: 100000 - max_time: inf - output_file: None - batch_size: 5'
    assert evaluator.get_info() == expected_info

    evaluator.set_params({'n_samples': 500000,
                          'max_time': 0.05,
                          'output_file': None,
                          'batch_size': 1})
    expected_info = 'EvaluateStreamGenerationSpeed: ' \
                    'n_samples: 500000 - max_time: 0.05 - output_file: None - batch_size: 1'
    assert evaluator.get_info() == expected_info

    # Stop evaluation by reaching max_time
    stream = evaluator.evaluate(stream)
    assert stream_name == stream.name
