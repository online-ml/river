"""Compare same training code in different execution contexts.

Forces fails to dump captured stdout from CI logs.
"""

from __future__ import annotations

import doctest
import io
import sys
import textwrap


def _run_amrules_training() -> tuple:
    from river import drift, rules, tree
    from river.datasets import synth

    dataset = synth.Friedman(seed=42).take(1001)
    model = rules.AMRules(
        n_min=50,
        delta=0.1,
        drift_detector=drift.ADWIN(),
        splitter=tree.splitter.QOSplitter(),
    )

    last_x = None
    for i, (x, y) in enumerate(dataset):
        if i == 1000:
            last_x = x
            break
        model.learn_one(x, y)

    return model.anomaly_score(last_x)


def test_function_context():
    """Plain function call."""
    score = _run_amrules_training()
    print(f"\n[function] score: {score!r}", flush=True)
    assert score == (0, 0, 0), f"intentional fail, got {score!r}"


def test_manual_doctest():
    """Run identical code via the doctest module directly."""
    source = textwrap.dedent("""
        >>> from river import drift, rules, tree
        >>> from river.datasets import synth
        >>> dataset = synth.Friedman(seed=42).take(1001)
        >>> model = rules.AMRules(
        ...     n_min=50,
        ...     delta=0.1,
        ...     drift_detector=drift.ADWIN(),
        ...     splitter=tree.splitter.QOSplitter()
        ... )
        >>> last_x = None
        >>> for i, (x, y) in enumerate(dataset):
        ...     if i == 1000:
        ...         last_x = x
        ...         break
        ...     model.learn_one(x, y)
        >>> result = model.anomaly_score(last_x)
        >>> print(repr(result))
    """).strip()

    parser = doctest.DocTestParser()
    examples = parser.get_examples(source)
    runner = doctest.DocTestRunner(verbose=False)
    test = doctest.DocTest(examples, {}, "manual", "<manual>", 0, source)

    buf = io.StringIO()
    runner.run(test, out=buf.write)
    output = buf.getvalue()
    print(f"\n[manual-doctest] captured: {output!r}", flush=True)
    # Force fail to dump
    raise AssertionError(f"intentional fail, output={output!r}")


def test_with_module_globals_simulation():
    """Run inside a globs dict pre-populated like a doctest's, to test that angle."""
    from river.rules import amrules as amrules_mod

    # Doctest globals = shallow copy of module globals
    globs = dict(amrules_mod.__dict__)

    source = textwrap.dedent("""
        from river import drift, rules, tree
        from river.datasets import synth
        dataset = synth.Friedman(seed=42).take(1001)
        model = rules.AMRules(
            n_min=50,
            delta=0.1,
            drift_detector=drift.ADWIN(),
            splitter=tree.splitter.QOSplitter()
        )
        last_x = None
        for i, (x, y) in enumerate(dataset):
            if i == 1000:
                last_x = x
                break
            model.learn_one(x, y)
        result = model.anomaly_score(last_x)
    """)

    exec(source, globs)
    score = globs["result"]
    print(f"\n[module-globals-exec] score: {score!r}", flush=True)
    print(f"[module-globals-exec] python: {sys.version}", flush=True)
    assert score == (0, 0, 0), f"intentional fail, got {score!r}"
