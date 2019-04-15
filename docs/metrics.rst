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
