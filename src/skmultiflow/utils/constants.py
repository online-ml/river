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

              DATA_POINTS]
CLASSIFICATION_METRICS = [ACCURACY,
                          KAPPA,
                          KAPPA_T,
                          KAPPA_M,
                          TRUE_VS_PREDICTED,

                          DATA_POINTS]
REGRESSION_METRICS = [MSE,
                      MAE,
                      TRUE_VS_PREDICTED
                      ]
MULTI_TARGET_CLASSIFICATION_METRICS = [HAMMING_SCORE,
                                       HAMMING_LOSS,
                                       EXACT_MATCH,
                                       J_INDEX]
MULTI_TARGET_REGRESSION_METRICS = [AMSE,
                                   AMAE,
                                   ARMSE]
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
