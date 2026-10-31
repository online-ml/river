# NoDrift

Dummy class used to turn off concept drift detection capabilities of adaptive models. It always signals that no concept drift was detected. Examples --------

> > > from river import drift \>\>\> from river import evaluate \>\>\> from river import forest \>\>\> from river import metrics \>\>\> from river.datasets import synth
> > >
> > > dataset = [datasets.synth.ConceptDriftStream](<https://riverml.xyz/dev/api/datasets/synth/ConceptDriftStream>)( ... seed=8, ... position=500, ... width=40, ... ).take(700)

We can turn off the warning detection capabilities of Adaptive Random Forest (ARF) or other similar models. Thus, the base models will reset immediately after identifying a drift, bypassing the background model building phase:

> > > adaptive\_model = [forest.ARFClassifier](<https://riverml.xyz/dev/api/forest/ARFClassifier>)( ... leaf\_prediction="mc", ... warning\_detector=[drift.NoDrift](<https://riverml.xyz/dev/api/drift/NoDrift>)(), ... seed=8 ... )

We can also turn off the concept drift handling capabilities completely:

> > > stationary\_model = [forest.ARFClassifier](<https://riverml.xyz/dev/api/forest/ARFClassifier>)( ... leaf\_prediction="mc", ... warning\_detector=[drift.NoDrift](<https://riverml.xyz/dev/api/drift/NoDrift>)(), ... drift\_detector=[drift.NoDrift](<https://riverml.xyz/dev/api/drift/NoDrift>)(), ... seed=8 ... )

Let's put that to test:

> > > for x, y in dataset: ... adaptive\_model.learn\_one(x, y) ... stationary\_model.learn\_one(x, y)

The adaptive model:

> > > adaptive\_model.n\_drifts\_detected() 2
> > >
> > > adaptive\_model.n\_warnings\_detected() 0

The stationary one:

> > > stationary\_model.n\_drifts\_detected() 0
> > >
> > > stationary\_model.n\_warnings\_detected() 0

## Attributes

- **drift\_detected**

  Whether or not a drift is detected following the last update.

## Methods

> [!NOTE]
>
> **update**
>
> Update the detector with a single data point.
>
> **Parameters**
>
> - **x** — `int | float`
