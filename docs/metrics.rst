Streaming metrics
-----------------

.. automodule:: creme.metrics
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme.metrics

Binary classification
+++++++++++++++++++++

.. autosummary::
   :toctree: generated/
   :nosignatures:

   LogLoss
   F1Score
   Precision
   Recall

Multi-class classification
++++++++++++++++++++++++++

Note that every multi-class classification metric also works for binary classification. For example you may use the ``Accuracy`` metric in both cases.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Accuracy
   ConfusionMatrix
   CrossEntropy
   MacroF1Score
   MacroPrecision
   MacroRecall
   MicroF1Score
   MicroPrecision
   MicroRecall

Regression
++++++++++

.. autosummary::
   :toctree: generated/
   :nosignatures:

   MAE
   MSE
   RMSE
   RMSLE
   SMAPE

Rolling metrics
+++++++++++++++

Rolling metrics only apply to a window of most recent values.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   RollingAccuracy
   RollingCrossEntropy
   RollingF1Score
   RollingLogLoss
   RollingMacroF1Score
   RollingMacroPrecision
   RollingMacroRecall
   RollingMAE
   RollingMicroF1Score
   RollingMicroPrecision
   RollingMicroRecall
   RollingMSE
   RollingPrecision
   RollingRecall
   RollingRMSE
   RollingRMSLE
   RollingSMAPE
