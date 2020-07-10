DATA_POINTS = 'data_points'
ACCURACY = 'accuracy'
KAPPA = 'kappa'
KAPPA_T = 'kappa_t'
KAPPA_M = 'kappa_m'
HAMMING_SCORE = 'hamming_score'
HAMMING_LOSS = 'hamming_loss'
EXACT_MATCH = 'exact_match'
J_INDEX = 'j_index'
MSE = 'mean_square_error'
MAE = 'mean_absolute_error'
AMSE = 'average_mean_square_error'
AMAE = 'average_mean_absolute_error'
ARMSE = 'average_root_mean_square_error'
TRUE_VS_PREDICTED = 'true_vs_predicted'
RUNNING_TIME = 'running_time'
MODEL_SIZE = 'model_size'
F1_SCORE = 'f1'
GMEAN = 'gmean'
PRECISION = 'precision'
RECALL = 'recall'

PLOT_TYPES = [ACCURACY,
              KAPPA,
              KAPPA_T,
              KAPPA_M,
              HAMMING_SCORE,
              HAMMING_LOSS,
              EXACT_MATCH,
              J_INDEX,
              MSE,
              MAE,
              AMSE,
              AMAE,
              ARMSE,
              TRUE_VS_PREDICTED,
              PRECISION,
              RECALL,
              F1_SCORE,
              GMEAN,

              DATA_POINTS,
              RUNNING_TIME,
              MODEL_SIZE]
CLASSIFICATION_METRICS = [ACCURACY,
                          KAPPA,
                          KAPPA_T,
                          KAPPA_M,
                          TRUE_VS_PREDICTED,
                          RECALL,
                          PRECISION,
                          GMEAN,
                          F1_SCORE,

                          DATA_POINTS,
                          RUNNING_TIME,
                          MODEL_SIZE]
REGRESSION_METRICS = [MSE,
                      MAE,
                      TRUE_VS_PREDICTED,
                      RUNNING_TIME,
                      MODEL_SIZE]
MULTI_TARGET_CLASSIFICATION_METRICS = [HAMMING_SCORE,
                                       HAMMING_LOSS,
                                       EXACT_MATCH,
                                       J_INDEX,
                                       RUNNING_TIME,
                                       MODEL_SIZE]
MULTI_TARGET_REGRESSION_METRICS = [AMSE,
                                   AMAE,
                                   ARMSE,
                                   RUNNING_TIME,
                                   MODEL_SIZE]
CLASSIFICATION = 'classification'
REGRESSION = 'regression'
MULTI_OUTPUT = 'multi_output'
SINGLE_OUTPUT = 'single-output'
MULTI_TARGET_CLASSIFICATION = 'multi_target_classification'
MULTI_TARGET_REGRESSION = 'multi_target_regression'
UNDEFINED = 'undefined'
TASK_TYPES = [CLASSIFICATION,
              REGRESSION,
              SINGLE_OUTPUT,
              MULTI_OUTPUT,
              MULTI_TARGET_CLASSIFICATION,
              MULTI_TARGET_REGRESSION,
              UNDEFINED]
MEAN = 'mean'
CURRENT = 'current'
Y_TRUE = 'y_true'
Y_PRED = 'y_pred'
