Optimization
------------

.. currentmodule:: creme.optim

Optimizers
++++++++++

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: class.rst

   Adam
   AdaBound
   AdaDelta
   AdaGrad
   AdaMax
   FTRLProximal
   MiniBatcher
   Momentum
   NesterovMomentum
   Optimizer
   RMSProp
   SGD


Learning rate schedulers
++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: class.rst

   schedulers.Constant
   schedulers.InverseScaling
   schedulers.Optimal


Loss functions
++++++++++++++

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: class.rst

   losses.Absolute
   losses.BinaryClassificationLoss
   losses.Cauchy
   losses.CrossEntropy
   losses.EpsilonInsensitiveHinge
   losses.Hinge
   losses.Log
   losses.MultiClassificationLoss
   losses.Quantile
   losses.RegressionLoss
   losses.Squared


Weight initializers
+++++++++++++++++++

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: class.rst

   initializers.Constant
   initializers.Normal
   initializers.Zeros
