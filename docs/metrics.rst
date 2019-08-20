Streaming metrics
-----------------

Most of the metrics have a rolling version which only applies to a window of most recent values.

.. currentmodule:: creme.metrics

Binary classification
+++++++++++++++++++++

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: class.rst

   LogLoss
   F1
   FBeta
   MCC
   Precision
   Recall
   ROCAUC
   RollingLogLoss
   RollingF1
   RollingFBeta
   RollingMCC
   RollingPrecision
   RollingRecall

Multi-class classification
++++++++++++++++++++++++++

Note that every multi-class classification metric also works for binary classification. For example you may use the ``Accuracy`` metric in both binary and multi-class classification.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: class.rst

   Accuracy
   ConfusionMatrix
   CrossEntropy
   MacroF1
   MacroFBeta
   MacroPrecision
   MacroRecall
   MicroF1
   MicroFBeta
   MicroPrecision
   MicroRecall
   MultiFBeta
   RollingAccuracy
   RollingConfusionMatrix
   RollingCrossEntropy
   RollingMacroF1
   RollingMacroFBeta
   RollingMacroPrecision
   RollingMacroRecall
   RollingMicroF1
   RollingMicroFBeta
   RollingMicroPrecision
   RollingMicroRecall
   RollingMultiFBeta

Regression
++++++++++

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: class.rst

   MAE
   MSE
   RMSE
   RMSLE
   SMAPE
   RollingMAE
   RollingMSE
   RollingRMSE
   RollingRMSLE
   RollingSMAPE

Multi-output
++++++++++++

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: class.rst

   Jaccard
   RegressionMultiOutput
