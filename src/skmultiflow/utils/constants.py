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
MULTI_OUTPUT_METRICS = [HAMMING_SCORE,
                        HAMMING_LOSS,
                        EXACT_MATCH,
                        J_INDEX]
CLASSIFICATION = 'classification'
REGRESSION = 'regression'
MULTI_OUTPUT = 'multi_output'
SINGLE_OUTPUT = 'single-output'
UNDEFINED = 'undefined'
TASK_TYPES = [CLASSIFICATION,
              REGRESSION,
              MULTI_OUTPUT,
              SINGLE_OUTPUT,
              UNDEFINED]
