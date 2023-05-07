# Multiclass classification



=== "Table"

    | Model                    | Dataset       |   Accuracy |   MicroF1 |   MacroF1 |   Memory in Mb |   Time in s |
    |:-------------------------|:--------------|-----------:|----------:|----------:|---------------:|------------:|
    | ADWIN Bagging            | ImageSegments |   0.777729 |  0.777729 |  0.764912 |     4.14768    |   482.736   |
    | ADWIN Bagging            | Insects       |   0.579424 |  0.579424 |  0.570136 |    15.4446     | 12525.9     |
    | ADWIN Bagging            | Keystroke     |   0.805824 |  0.805824 |  0.80625  |    32.1812     |  8923.61    |
    | AdaBoost                 | ImageSegments |   0.805133 |  0.805133 |  0.798078 |     4.12853    |   438.219   |
    | AdaBoost                 | Insects       |   0.554082 |  0.554082 |  0.543927 |    28.2902     | 12481.3     |
    | AdaBoost                 | Keystroke     |   0.842492 |  0.842492 |  0.843635 |   177.385      | 12366.9     |
    | Adaptive Random Forest   | ImageSegments |   0.819052 |  0.819052 |  0.814425 |     4.66081    |   227.541   |
    | Adaptive Random Forest   | Insects       |   0.744257 |  0.744257 |  0.741932 |     0.369647   |  4404.71    |
    | Adaptive Random Forest   | Keystroke     |   0.969851 |  0.969851 |  0.969867 |     2.33717    |   937.846   |
    | Bagging                  | ImageSegments |   0.77686  |  0.77686  |  0.764461 |     4.18729    |   482.036   |
    | Bagging                  | Insects       |   0.606053 |  0.606053 |  0.598222 |     3.75006    | 14067.2     |
    | Bagging                  | Keystroke     |   0.667974 |  0.667974 |  0.668853 |    50.4872     | 13509.1     |
    | Hoeffding Adaptive Tree  | ImageSegments |   0.774685 |  0.774685 |  0.763496 |     0.425819   |    53.9974  |
    | Hoeffding Adaptive Tree  | Insects       |   0.611962 |  0.611962 |  0.602993 |     0.147679   |  1507.07    |
    | Hoeffding Adaptive Tree  | Keystroke     |   0.723712 |  0.723712 |  0.722393 |     0.727901   |  1274.73    |
    | Hoeffding Tree           | ImageSegments |   0.77599  |  0.77599  |  0.763027 |     0.419177   |    39.4879  |
    | Hoeffding Tree           | Insects       |   0.537018 |  0.537018 |  0.527071 |     2.5392     |   921.351   |
    | Hoeffding Tree           | Keystroke     |   0.648218 |  0.648218 |  0.647249 |     5.09806    |   914.037   |
    | Leveraging Bagging       | ImageSegments |   0.778164 |  0.778164 |  0.765914 |     4.13275    |  1135.16    |
    | Leveraging Bagging       | Insects       |   0.691547 |  0.691547 |  0.686411 |    18.1413     | 32334.1     |
    | Leveraging Bagging       | Keystroke     |   0.95039  |  0.95039  |  0.950468 |    10.4201     |  7265.02    |
    | Naive Bayes              | ImageSegments |   0.731622 |  0.731622 |  0.730042 |     0.390004   |    38.4724  |
    | Naive Bayes              | Insects       |   0.506847 |  0.506847 |  0.493003 |     0.611693   |   557.606   |
    | Naive Bayes              | Keystroke     |   0.652532 |  0.652532 |  0.651577 |     4.86901    |   473.747   |
    | Stacking                 | ImageSegments |   0.849065 |  0.849065 |  0.847922 |     5.29567    |   399.289   |
    | Stacking                 | Insects       |   0.752154 |  0.752154 |  0.750251 |    11.339      |  9741.14    |
    | Stacking                 | Keystroke     |   0.976518 |  0.976518 |  0.976517 |    12.2203     |  4556.33    |
    | Streaming Random Patches | ImageSegments |   0.754676 |  0.754676 |  0.752727 |    10.4257     |   832.07    |
    | Streaming Random Patches | Insects       |   0.739578 |  0.739578 |  0.737512 |     8.34194    | 26942.3     |
    | Streaming Random Patches | Keystroke     |   0.953233 |  0.953233 |  0.953239 |    74.5521     |  5886.48    |
    | Voting                   | ImageSegments |   0.803393 |  0.803393 |  0.794975 |     0.951658   |   146.236   |
    | Voting                   | Insects       |   0.647929 |  0.647929 |  0.635943 |     3.38862    |  3141.99    |
    | Voting                   | Keystroke     |   0.793274 |  0.793274 |  0.798424 |    10.3088     |  2173.75    |
    | [baseline] Last Class    | ImageSegments |   0.14789  |  0.14789  |  0.147887 |     0.00136757 |     2.67732 |
    | [baseline] Last Class    | Insects       |   0.289115 |  0.289115 |  0.289295 |     0.00138664 |    59.1503  |
    | [baseline] Last Class    | Keystroke     |   0.997549 |  0.997549 |  0.997549 |     0.00504208 |    24.227   |
    | k-Nearest Neighbors      | ImageSegments |   0.819922 |  0.819922 |  0.815895 |     0.12676    |    38.8794  |
    | k-Nearest Neighbors      | Insects       |   0.686547 |  0.686547 |  0.683661 |     0.216656   |  1254.78    |
    | k-Nearest Neighbors      | Keystroke     |   0.984509 |  0.984509 |  0.984508 |     0.214242   |   515.415   |

=== "Chart"

    *Try reloading the page if something is buggy*

    ```vegalite
    {
      "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
      "data": {
        "values": [
          {
            "step": 46,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.4666666666666667,
            "MicroF1": 0.4666666666666667,
            "MacroF1": 0.4009102009102009,
            "Memory in Mb": 0.3899507522583008,
            "Time in s": 0.163216
          },
          {
            "step": 92,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.5604395604395604,
            "MicroF1": 0.5604395604395604,
            "MacroF1": 0.5279334700387331,
            "Memory in Mb": 0.3899507522583008,
            "Time in s": 0.349738
          },
          {
            "step": 138,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.5474452554744526,
            "MicroF1": 0.5474452554744526,
            "MacroF1": 0.5191892873237388,
            "Memory in Mb": 0.3899774551391601,
            "Time in s": 0.5584899999999999
          },
          {
            "step": 184,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.5573770491803278,
            "MicroF1": 0.5573770491803278,
            "MacroF1": 0.5225713529323662,
            "Memory in Mb": 0.3899507522583008,
            "Time in s": 0.789485
          },
          {
            "step": 230,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.5545851528384279,
            "MicroF1": 0.5545851528384279,
            "MacroF1": 0.5217226223148511,
            "Memory in Mb": 0.3899774551391601,
            "Time in s": 1.042858
          },
          {
            "step": 276,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.56,
            "MicroF1": 0.56,
            "MacroF1": 0.5450388711329708,
            "Memory in Mb": 0.3899774551391601,
            "Time in s": 1.324703
          },
          {
            "step": 322,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.5825545171339563,
            "MicroF1": 0.5825545171339563,
            "MacroF1": 0.5566705826058684,
            "Memory in Mb": 0.3900041580200195,
            "Time in s": 1.637036
          },
          {
            "step": 368,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.5940054495912807,
            "MicroF1": 0.5940054495912807,
            "MacroF1": 0.5613773296963412,
            "Memory in Mb": 0.3900041580200195,
            "Time in s": 1.979491
          },
          {
            "step": 414,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.5980629539951574,
            "MicroF1": 0.5980629539951574,
            "MacroF1": 0.5624927052752284,
            "Memory in Mb": 0.3900041580200195,
            "Time in s": 2.352111
          },
          {
            "step": 460,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.599128540305011,
            "MicroF1": 0.599128540305011,
            "MacroF1": 0.5669821167583783,
            "Memory in Mb": 0.3899774551391601,
            "Time in s": 2.754918
          },
          {
            "step": 506,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.6099009900990099,
            "MicroF1": 0.6099009900990099,
            "MacroF1": 0.5922286190986811,
            "Memory in Mb": 0.3900041580200195,
            "Time in s": 3.188186
          },
          {
            "step": 552,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.6116152450090744,
            "MicroF1": 0.6116152450090744,
            "MacroF1": 0.5983340184133136,
            "Memory in Mb": 0.3899507522583008,
            "Time in s": 3.651555
          },
          {
            "step": 598,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.6180904522613065,
            "MicroF1": 0.6180904522613065,
            "MacroF1": 0.611527101723203,
            "Memory in Mb": 0.3899774551391601,
            "Time in s": 4.145135
          },
          {
            "step": 644,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.6158631415241057,
            "MicroF1": 0.6158631415241057,
            "MacroF1": 0.6113311881078581,
            "Memory in Mb": 0.3899774551391601,
            "Time in s": 4.668896
          },
          {
            "step": 690,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.6182873730043541,
            "MicroF1": 0.6182873730043541,
            "MacroF1": 0.615018998714676,
            "Memory in Mb": 0.3899774551391601,
            "Time in s": 5.223075
          },
          {
            "step": 736,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.617687074829932,
            "MicroF1": 0.617687074829932,
            "MacroF1": 0.6157912419016742,
            "Memory in Mb": 0.3899774551391601,
            "Time in s": 5.807397
          },
          {
            "step": 782,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.6274007682458387,
            "MicroF1": 0.6274007682458387,
            "MacroF1": 0.6216325704223051,
            "Memory in Mb": 0.3899774551391601,
            "Time in s": 6.422078
          },
          {
            "step": 828,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.6324062877871826,
            "MicroF1": 0.6324062877871826,
            "MacroF1": 0.6280704917469789,
            "Memory in Mb": 0.3899774551391601,
            "Time in s": 7.066915
          },
          {
            "step": 874,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.6426116838487973,
            "MicroF1": 0.6426116838487973,
            "MacroF1": 0.6349558095046656,
            "Memory in Mb": 0.3899774551391601,
            "Time in s": 7.742184
          },
          {
            "step": 920,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.6485310119695321,
            "MicroF1": 0.6485310119695321,
            "MacroF1": 0.6384515982514894,
            "Memory in Mb": 0.3899774551391601,
            "Time in s": 8.447577
          },
          {
            "step": 966,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.6507772020725389,
            "MicroF1": 0.6507772020725389,
            "MacroF1": 0.6399118827528387,
            "Memory in Mb": 0.3899774551391601,
            "Time in s": 9.183146
          },
          {
            "step": 1012,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.6508407517309595,
            "MicroF1": 0.6508407517309595,
            "MacroF1": 0.6387857120889422,
            "Memory in Mb": 0.3899774551391601,
            "Time in s": 9.95137
          },
          {
            "step": 1058,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.6537369914853358,
            "MicroF1": 0.6537369914853358,
            "MacroF1": 0.6398811322847952,
            "Memory in Mb": 0.3899774551391601,
            "Time in s": 10.747402
          },
          {
            "step": 1104,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.658204895738894,
            "MicroF1": 0.658204895738894,
            "MacroF1": 0.6463297068165035,
            "Memory in Mb": 0.3899774551391601,
            "Time in s": 11.559914
          },
          {
            "step": 1150,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.6640557006092254,
            "MicroF1": 0.6640557006092254,
            "MacroF1": 0.6508930463144657,
            "Memory in Mb": 0.3900041580200195,
            "Time in s": 12.388643000000002
          },
          {
            "step": 1196,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.6702928870292887,
            "MicroF1": 0.6702928870292887,
            "MacroF1": 0.6599370641329335,
            "Memory in Mb": 0.3900041580200195,
            "Time in s": 13.233598000000002
          },
          {
            "step": 1242,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.6736502820306205,
            "MicroF1": 0.6736502820306205,
            "MacroF1": 0.669511465798708,
            "Memory in Mb": 0.3900041580200195,
            "Time in s": 14.094776000000005
          },
          {
            "step": 1288,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.6822066822066822,
            "MicroF1": 0.6822066822066822,
            "MacroF1": 0.6790074545382362,
            "Memory in Mb": 0.3900041580200195,
            "Time in s": 14.972203000000004
          },
          {
            "step": 1334,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.6841710427606902,
            "MicroF1": 0.6841710427606902,
            "MacroF1": 0.6834974476087325,
            "Memory in Mb": 0.3900041580200195,
            "Time in s": 15.866030000000004
          },
          {
            "step": 1380,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.6874546773023931,
            "MicroF1": 0.6874546773023931,
            "MacroF1": 0.6876766922721351,
            "Memory in Mb": 0.3900041580200195,
            "Time in s": 16.775981000000005
          },
          {
            "step": 1426,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.6919298245614035,
            "MicroF1": 0.6919298245614035,
            "MacroF1": 0.6930786661709784,
            "Memory in Mb": 0.3900041580200195,
            "Time in s": 17.702176000000005
          },
          {
            "step": 1472,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.698844323589395,
            "MicroF1": 0.698844323589395,
            "MacroF1": 0.6985606658027722,
            "Memory in Mb": 0.3899774551391601,
            "Time in s": 18.644575000000003
          },
          {
            "step": 1518,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.7027027027027027,
            "MicroF1": 0.7027027027027027,
            "MacroF1": 0.7017787722939461,
            "Memory in Mb": 0.3900041580200195,
            "Time in s": 19.603248000000004
          },
          {
            "step": 1564,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.7056941778630839,
            "MicroF1": 0.7056941778630839,
            "MacroF1": 0.7062915374924865,
            "Memory in Mb": 0.3899774551391601,
            "Time in s": 20.578282000000005
          },
          {
            "step": 1610,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.7078931013051585,
            "MicroF1": 0.7078931013051585,
            "MacroF1": 0.7081385387673029,
            "Memory in Mb": 0.3899774551391601,
            "Time in s": 21.573844000000005
          },
          {
            "step": 1656,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.7093655589123867,
            "MicroF1": 0.7093655589123867,
            "MacroF1": 0.7109488618373111,
            "Memory in Mb": 0.3899507522583008,
            "Time in s": 22.586424000000004
          },
          {
            "step": 1702,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.7101704879482658,
            "MicroF1": 0.7101704879482658,
            "MacroF1": 0.7132092257742534,
            "Memory in Mb": 0.3899774551391601,
            "Time in s": 23.615335000000005
          },
          {
            "step": 1748,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.7143674871207785,
            "MicroF1": 0.7143674871207784,
            "MacroF1": 0.7178399485500211,
            "Memory in Mb": 0.3899507522583008,
            "Time in s": 24.660526000000004
          },
          {
            "step": 1794,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.7172336865588399,
            "MicroF1": 0.7172336865588399,
            "MacroF1": 0.7191260584555578,
            "Memory in Mb": 0.3899774551391601,
            "Time in s": 25.721983000000005
          },
          {
            "step": 1840,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.7199564980967917,
            "MicroF1": 0.7199564980967917,
            "MacroF1": 0.7217017555070445,
            "Memory in Mb": 0.3900041580200195,
            "Time in s": 26.79968000000001
          },
          {
            "step": 1886,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.7204244031830239,
            "MicroF1": 0.7204244031830238,
            "MacroF1": 0.7234495525792994,
            "Memory in Mb": 0.3900041580200195,
            "Time in s": 27.893629000000004
          },
          {
            "step": 1932,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.7219057483169342,
            "MicroF1": 0.7219057483169342,
            "MacroF1": 0.723848351214801,
            "Memory in Mb": 0.3900041580200195,
            "Time in s": 29.003837000000004
          },
          {
            "step": 1978,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.723823975720789,
            "MicroF1": 0.723823975720789,
            "MacroF1": 0.725139923863974,
            "Memory in Mb": 0.3900041580200195,
            "Time in s": 30.130512000000003
          },
          {
            "step": 2024,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.726643598615917,
            "MicroF1": 0.726643598615917,
            "MacroF1": 0.7268553573885639,
            "Memory in Mb": 0.3900041580200195,
            "Time in s": 31.273399000000005
          },
          {
            "step": 2070,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.7269212179797003,
            "MicroF1": 0.7269212179797003,
            "MacroF1": 0.7276782991451582,
            "Memory in Mb": 0.3900041580200195,
            "Time in s": 32.432577
          },
          {
            "step": 2116,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.7286052009456265,
            "MicroF1": 0.7286052009456266,
            "MacroF1": 0.7283656039279266,
            "Memory in Mb": 0.3900041580200195,
            "Time in s": 33.608017000000004
          },
          {
            "step": 2162,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.7306802406293382,
            "MicroF1": 0.7306802406293383,
            "MacroF1": 0.7303992643507475,
            "Memory in Mb": 0.3900041580200195,
            "Time in s": 34.7997
          },
          {
            "step": 2208,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.733574988672406,
            "MicroF1": 0.733574988672406,
            "MacroF1": 0.7322842940126589,
            "Memory in Mb": 0.3900041580200195,
            "Time in s": 36.007612
          },
          {
            "step": 2254,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.7314691522414558,
            "MicroF1": 0.7314691522414558,
            "MacroF1": 0.7300322879925133,
            "Memory in Mb": 0.3900041580200195,
            "Time in s": 37.231763
          },
          {
            "step": 2300,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "ImageSegments",
            "Accuracy": 0.7316224445411048,
            "MicroF1": 0.7316224445411048,
            "MacroF1": 0.7300416811383057,
            "Memory in Mb": 0.3900041580200195,
            "Time in s": 38.472431
          },
          {
            "step": 1056,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.623696682464455,
            "MicroF1": 0.623696682464455,
            "MacroF1": 0.5870724729616661,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 0.909568
          },
          {
            "step": 2112,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.6148744670772146,
            "MicroF1": 0.6148744670772146,
            "MacroF1": 0.5800776869595597,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 2.67356
          },
          {
            "step": 3168,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.6065677297126618,
            "MicroF1": 0.6065677297126618,
            "MacroF1": 0.5714781230184183,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 5.143102000000001
          },
          {
            "step": 4224,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.6043097324177126,
            "MicroF1": 0.6043097324177126,
            "MacroF1": 0.5697541737710122,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 7.993857
          },
          {
            "step": 5280,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.6088274294373934,
            "MicroF1": 0.6088274294373934,
            "MacroF1": 0.5727560614138387,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 11.225513
          },
          {
            "step": 6336,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.6023677979479084,
            "MicroF1": 0.6023677979479084,
            "MacroF1": 0.5679597008529512,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 14.839337
          },
          {
            "step": 7392,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5995129211202814,
            "MicroF1": 0.5995129211202814,
            "MacroF1": 0.5652603100832261,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 18.839998
          },
          {
            "step": 8448,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.6019888717888008,
            "MicroF1": 0.6019888717888008,
            "MacroF1": 0.5673514925692325,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 23.223853
          },
          {
            "step": 9504,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5993896664211301,
            "MicroF1": 0.5993896664211301,
            "MacroF1": 0.5644951651039589,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 27.990643
          },
          {
            "step": 10560,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5994885879344635,
            "MicroF1": 0.5994885879344635,
            "MacroF1": 0.5645655385998631,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 33.140509
          },
          {
            "step": 11616,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5972449418854929,
            "MicroF1": 0.5972449418854929,
            "MacroF1": 0.5631227877868952,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 38.672833
          },
          {
            "step": 12672,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.6001894088864336,
            "MicroF1": 0.6001894088864336,
            "MacroF1": 0.5684733590606373,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 44.58831000000001
          },
          {
            "step": 13728,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.6120783856632913,
            "MicroF1": 0.6120783856632913,
            "MacroF1": 0.5935173038317552,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 50.89270200000001
          },
          {
            "step": 14784,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.6024487587093282,
            "MicroF1": 0.6024487587093282,
            "MacroF1": 0.5841270876002981,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 57.581734
          },
          {
            "step": 15840,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5676494728202538,
            "MicroF1": 0.5676494728202538,
            "MacroF1": 0.5507155080701159,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 64.65553700000001
          },
          {
            "step": 16896,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5418762947617638,
            "MicroF1": 0.5418762947617638,
            "MacroF1": 0.5256197352354143,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 72.114698
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5232020500250683,
            "MicroF1": 0.5232020500250683,
            "MacroF1": 0.5066898143269706,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 79.958388
          },
          {
            "step": 19008,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5118640500868101,
            "MicroF1": 0.5118640500868101,
            "MacroF1": 0.4926543583964285,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 88.190503
          },
          {
            "step": 20064,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5103922643672432,
            "MicroF1": 0.5103922643672432,
            "MacroF1": 0.4900586962359796,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 96.808684
          },
          {
            "step": 21120,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5115772527108291,
            "MicroF1": 0.5115772527108291,
            "MacroF1": 0.4910837640903744,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 105.81178
          },
          {
            "step": 22176,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5140022547914318,
            "MicroF1": 0.5140022547914318,
            "MacroF1": 0.4932541888231956,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 115.205863
          },
          {
            "step": 23232,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5154319659076234,
            "MicroF1": 0.5154319659076234,
            "MacroF1": 0.4943013417599926,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 124.990845
          },
          {
            "step": 24288,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5184254951208466,
            "MicroF1": 0.5184254951208466,
            "MacroF1": 0.4965832238311332,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 135.166218
          },
          {
            "step": 25344,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5225111470623052,
            "MicroF1": 0.5225111470623052,
            "MacroF1": 0.499893079239698,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 145.739141
          },
          {
            "step": 26400,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5257396113489148,
            "MicroF1": 0.5257396113489148,
            "MacroF1": 0.5022487669255871,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 156.702601
          },
          {
            "step": 27456,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5301402294663996,
            "MicroF1": 0.5301402294663996,
            "MacroF1": 0.5051550433324518,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 168.057909
          },
          {
            "step": 28512,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5277261407877661,
            "MicroF1": 0.5277261407877661,
            "MacroF1": 0.5036945145235057,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 179.80420999999998
          },
          {
            "step": 29568,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5204450908107011,
            "MicroF1": 0.5204450908107011,
            "MacroF1": 0.4989008712312767,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 191.944501
          },
          {
            "step": 30624,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5147111648107632,
            "MicroF1": 0.5147111648107632,
            "MacroF1": 0.495826840073632,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 204.478499
          },
          {
            "step": 31680,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5105590454244137,
            "MicroF1": 0.5105590454244137,
            "MacroF1": 0.4941101813344875,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 217.402092
          },
          {
            "step": 32736,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5075607148312204,
            "MicroF1": 0.5075607148312204,
            "MacroF1": 0.4931947798921405,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 230.716201
          },
          {
            "step": 33792,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5044538486579266,
            "MicroF1": 0.5044538486579266,
            "MacroF1": 0.4905626123916189,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 244.420884
          },
          {
            "step": 34848,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5020231296811777,
            "MicroF1": 0.5020231296811777,
            "MacroF1": 0.487879842488124,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 258.51509
          },
          {
            "step": 35904,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.4998746622844887,
            "MicroF1": 0.4998746622844887,
            "MacroF1": 0.4853435061152475,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 273.003699
          },
          {
            "step": 36960,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.4967937444194918,
            "MicroF1": 0.4967937444194918,
            "MacroF1": 0.4819418474093529,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 287.883522
          },
          {
            "step": 38016,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.4955938445350519,
            "MicroF1": 0.4955938445350519,
            "MacroF1": 0.4801892436835747,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 303.152298
          },
          {
            "step": 39072,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.4940237004427836,
            "MicroF1": 0.4940237004427836,
            "MacroF1": 0.478380783820526,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 318.807697
          },
          {
            "step": 40128,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.493508111745209,
            "MicroF1": 0.493508111745209,
            "MacroF1": 0.4785213801670671,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 334.85223
          },
          {
            "step": 41184,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.4936988563242114,
            "MicroF1": 0.4936988563242114,
            "MacroF1": 0.4794201499427274,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 351.286644
          },
          {
            "step": 42240,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.4938800634484718,
            "MicroF1": 0.4938800634484718,
            "MacroF1": 0.4802377497532936,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 368.105611
          },
          {
            "step": 43296,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.4943757939715902,
            "MicroF1": 0.4943757939715902,
            "MacroF1": 0.4812132921167227,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 385.310693
          },
          {
            "step": 44352,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.494036211133909,
            "MicroF1": 0.494036211133909,
            "MacroF1": 0.4812388919618418,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 402.906414
          },
          {
            "step": 45408,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.4944832294580131,
            "MicroF1": 0.4944832294580131,
            "MacroF1": 0.4818441874360224,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 420.888505
          },
          {
            "step": 46464,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.4945225232981082,
            "MicroF1": 0.4945225232981082,
            "MacroF1": 0.4820791268335544,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 439.259743
          },
          {
            "step": 47520,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.4956333256171216,
            "MicroF1": 0.4956333256171216,
            "MacroF1": 0.4833168636021498,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 458.017368
          },
          {
            "step": 48576,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.4970869788986104,
            "MicroF1": 0.4970869788986104,
            "MacroF1": 0.4846703771634363,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 477.16088800000006
          },
          {
            "step": 49632,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.4987608551107171,
            "MicroF1": 0.4987608551107171,
            "MacroF1": 0.4862426724473749,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 496.692936
          },
          {
            "step": 50688,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5009568528419516,
            "MicroF1": 0.5009568528419516,
            "MacroF1": 0.4881725476999718,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 516.6094800000001
          },
          {
            "step": 51744,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5034497419940862,
            "MicroF1": 0.5034497419940862,
            "MacroF1": 0.4903712806540024,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 536.9146260000001
          },
          {
            "step": 52800,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Insects",
            "Accuracy": 0.5068467205818292,
            "MicroF1": 0.5068467205818292,
            "MacroF1": 0.4930025316136313,
            "Memory in Mb": 0.6116933822631836,
            "Time in s": 557.6057650000001
          },
          {
            "step": 408,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.9852579852579852,
            "MicroF1": 0.9852579852579852,
            "MacroF1": 0.6962686567164179,
            "Memory in Mb": 0.1935644149780273,
            "Time in s": 0.122414
          },
          {
            "step": 816,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.947239263803681,
            "MicroF1": 0.947239263803681,
            "MacroF1": 0.7418606503288051,
            "Memory in Mb": 0.2889022827148437,
            "Time in s": 0.375804
          },
          {
            "step": 1224,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.884709730171709,
            "MicroF1": 0.884709730171709,
            "MacroF1": 0.8705899666065842,
            "Memory in Mb": 0.3842401504516601,
            "Time in s": 0.764348
          },
          {
            "step": 1632,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.8933169834457388,
            "MicroF1": 0.8933169834457388,
            "MacroF1": 0.8791291775937072,
            "Memory in Mb": 0.4795780181884765,
            "Time in s": 1.303316
          },
          {
            "step": 2040,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.8921039725355566,
            "MicroF1": 0.8921039725355566,
            "MacroF1": 0.8831785360852743,
            "Memory in Mb": 0.575160026550293,
            "Time in s": 2.01214
          },
          {
            "step": 2448,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.851655087862689,
            "MicroF1": 0.851655087862689,
            "MacroF1": 0.858198428951664,
            "Memory in Mb": 0.6704978942871094,
            "Time in s": 2.906585
          },
          {
            "step": 2856,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.8598949211908932,
            "MicroF1": 0.8598949211908932,
            "MacroF1": 0.8469962214365345,
            "Memory in Mb": 0.7658357620239258,
            "Time in s": 3.994802
          },
          {
            "step": 3264,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.8513637756665645,
            "MicroF1": 0.8513637756665645,
            "MacroF1": 0.8281280134770848,
            "Memory in Mb": 0.8611736297607422,
            "Time in s": 5.296884
          },
          {
            "step": 3672,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.8422773086352493,
            "MicroF1": 0.8422773086352493,
            "MacroF1": 0.8409307955747314,
            "Memory in Mb": 0.9565114974975586,
            "Time in s": 6.831079000000001
          },
          {
            "step": 4080,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.8367246874233881,
            "MicroF1": 0.8367246874233881,
            "MacroF1": 0.8249418657104467,
            "Memory in Mb": 1.0523834228515625,
            "Time in s": 8.617788000000001
          },
          {
            "step": 4488,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.8203699576554491,
            "MicroF1": 0.8203699576554491,
            "MacroF1": 0.8300896799820437,
            "Memory in Mb": 1.147721290588379,
            "Time in s": 10.679552
          },
          {
            "step": 4896,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.8192032686414709,
            "MicroF1": 0.8192032686414709,
            "MacroF1": 0.8269731591910484,
            "Memory in Mb": 1.243059158325195,
            "Time in s": 13.032163
          },
          {
            "step": 5304,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.8172732415613804,
            "MicroF1": 0.8172732415613804,
            "MacroF1": 0.8027823390848743,
            "Memory in Mb": 1.3383970260620115,
            "Time in s": 15.695238
          },
          {
            "step": 5712,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.7961828051129399,
            "MicroF1": 0.7961828051129399,
            "MacroF1": 0.8002006091139847,
            "Memory in Mb": 1.433734893798828,
            "Time in s": 18.689224
          },
          {
            "step": 6120,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.793920575257395,
            "MicroF1": 0.793920575257395,
            "MacroF1": 0.7746960355921346,
            "Memory in Mb": 1.5290727615356443,
            "Time in s": 22.034543
          },
          {
            "step": 6528,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.7688064960931515,
            "MicroF1": 0.7688064960931515,
            "MacroF1": 0.7622487598340326,
            "Memory in Mb": 1.624410629272461,
            "Time in s": 25.755146
          },
          {
            "step": 6936,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.7568853640951694,
            "MicroF1": 0.7568853640951694,
            "MacroF1": 0.757813781660983,
            "Memory in Mb": 1.7197484970092771,
            "Time in s": 29.876127
          },
          {
            "step": 7344,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.7669889690862045,
            "MicroF1": 0.7669889690862046,
            "MacroF1": 0.7643943615019535,
            "Memory in Mb": 1.8150863647460935,
            "Time in s": 34.413227
          },
          {
            "step": 7752,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.7676428847890595,
            "MicroF1": 0.7676428847890595,
            "MacroF1": 0.7655695901071293,
            "Memory in Mb": 1.9104242324829104,
            "Time in s": 39.374485
          },
          {
            "step": 8160,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.7714180659394534,
            "MicroF1": 0.7714180659394533,
            "MacroF1": 0.7672011803374248,
            "Memory in Mb": 2.0057621002197266,
            "Time in s": 44.773425
          },
          {
            "step": 8568,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.7702813120112058,
            "MicroF1": 0.7702813120112058,
            "MacroF1": 0.7699263138193526,
            "Memory in Mb": 2.1021223068237305,
            "Time in s": 50.625164000000005
          },
          {
            "step": 8976,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.7680222841225627,
            "MicroF1": 0.7680222841225627,
            "MacroF1": 0.7682287234686137,
            "Memory in Mb": 2.197460174560547,
            "Time in s": 56.940867
          },
          {
            "step": 9384,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.7659597143770649,
            "MicroF1": 0.7659597143770649,
            "MacroF1": 0.7643546547243015,
            "Memory in Mb": 2.2927980422973637,
            "Time in s": 63.725868000000006
          },
          {
            "step": 9792,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.7586559084873864,
            "MicroF1": 0.7586559084873864,
            "MacroF1": 0.7552148692020618,
            "Memory in Mb": 2.38813591003418,
            "Time in s": 70.991963
          },
          {
            "step": 10200,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.7505637807628199,
            "MicroF1": 0.7505637807628199,
            "MacroF1": 0.7430512224080149,
            "Memory in Mb": 2.483473777770996,
            "Time in s": 78.748505
          },
          {
            "step": 10608,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.7290468558499105,
            "MicroF1": 0.7290468558499106,
            "MacroF1": 0.715756093271779,
            "Memory in Mb": 2.5788116455078125,
            "Time in s": 87.01168299999999
          },
          {
            "step": 11016,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.7217430776214253,
            "MicroF1": 0.7217430776214253,
            "MacroF1": 0.7173640789896896,
            "Memory in Mb": 2.674149513244629,
            "Time in s": 95.787317
          },
          {
            "step": 11424,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.7151361288628206,
            "MicroF1": 0.7151361288628206,
            "MacroF1": 0.7011862635194492,
            "Memory in Mb": 2.7694873809814453,
            "Time in s": 105.08400199999998
          },
          {
            "step": 11832,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.705603921900093,
            "MicroF1": 0.705603921900093,
            "MacroF1": 0.6976881379682605,
            "Memory in Mb": 2.8648252487182617,
            "Time in s": 114.917299
          },
          {
            "step": 12240,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.7094533867146009,
            "MicroF1": 0.7094533867146009,
            "MacroF1": 0.705840538940343,
            "Memory in Mb": 2.960163116455078,
            "Time in s": 125.31674099999998
          },
          {
            "step": 12648,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.7053846762077963,
            "MicroF1": 0.7053846762077963,
            "MacroF1": 0.6965736948063981,
            "Memory in Mb": 3.0555009841918945,
            "Time in s": 136.28361299999997
          },
          {
            "step": 13056,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.6927613941018766,
            "MicroF1": 0.6927613941018766,
            "MacroF1": 0.6842255816736497,
            "Memory in Mb": 3.150838851928711,
            "Time in s": 147.832836
          },
          {
            "step": 13464,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.6890737577063062,
            "MicroF1": 0.6890737577063062,
            "MacroF1": 0.6845669389392289,
            "Memory in Mb": 3.246176719665528,
            "Time in s": 159.980064
          },
          {
            "step": 13872,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.6873332852714296,
            "MicroF1": 0.6873332852714296,
            "MacroF1": 0.6839054551822702,
            "Memory in Mb": 3.341514587402344,
            "Time in s": 172.74228
          },
          {
            "step": 14280,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.682960991666083,
            "MicroF1": 0.682960991666083,
            "MacroF1": 0.6781566371919946,
            "Memory in Mb": 3.43685245513916,
            "Time in s": 186.135321
          },
          {
            "step": 14688,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.686185061619119,
            "MicroF1": 0.686185061619119,
            "MacroF1": 0.6843713776162116,
            "Memory in Mb": 3.532190322875977,
            "Time in s": 200.177651
          },
          {
            "step": 15096,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.6928784365684001,
            "MicroF1": 0.6928784365684001,
            "MacroF1": 0.6911392400672977,
            "Memory in Mb": 3.627528190612793,
            "Time in s": 214.888654
          },
          {
            "step": 15504,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.6913500612784622,
            "MicroF1": 0.6913500612784622,
            "MacroF1": 0.687359772989117,
            "Memory in Mb": 3.72286605834961,
            "Time in s": 230.279565
          },
          {
            "step": 15912,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.6819810194205267,
            "MicroF1": 0.6819810194205267,
            "MacroF1": 0.6749159449359359,
            "Memory in Mb": 3.818203926086426,
            "Time in s": 246.365674
          },
          {
            "step": 16320,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.6726515105092223,
            "MicroF1": 0.6726515105092223,
            "MacroF1": 0.6670192172011686,
            "Memory in Mb": 3.913541793823242,
            "Time in s": 263.163212
          },
          {
            "step": 16728,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.6695163508100676,
            "MicroF1": 0.6695163508100676,
            "MacroF1": 0.6664051037977978,
            "Memory in Mb": 4.008879661560059,
            "Time in s": 280.687557
          },
          {
            "step": 17136,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.6650131310183834,
            "MicroF1": 0.6650131310183834,
            "MacroF1": 0.6608988619616459,
            "Memory in Mb": 4.1063079833984375,
            "Time in s": 298.952273
          },
          {
            "step": 17544,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.6568431853160804,
            "MicroF1": 0.6568431853160804,
            "MacroF1": 0.653138289771919,
            "Memory in Mb": 4.201645851135254,
            "Time in s": 317.97399
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.6556180714166342,
            "MicroF1": 0.6556180714166342,
            "MacroF1": 0.6538448358590967,
            "Memory in Mb": 4.29698371887207,
            "Time in s": 337.769402
          },
          {
            "step": 18360,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.6614194672912468,
            "MicroF1": 0.6614194672912468,
            "MacroF1": 0.6603186829199905,
            "Memory in Mb": 4.392321586608887,
            "Time in s": 358.361854
          },
          {
            "step": 18768,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.6669686151222891,
            "MicroF1": 0.6669686151222891,
            "MacroF1": 0.666229361655457,
            "Memory in Mb": 4.487659454345703,
            "Time in s": 379.763277
          },
          {
            "step": 19176,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.6579921773142112,
            "MicroF1": 0.6579921773142112,
            "MacroF1": 0.6554177118629491,
            "Memory in Mb": 4.58299732208252,
            "Time in s": 401.986094
          },
          {
            "step": 19584,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.6622580809886126,
            "MicroF1": 0.6622580809886126,
            "MacroF1": 0.6609360990360077,
            "Memory in Mb": 4.678335189819336,
            "Time in s": 425.04707400000007
          },
          {
            "step": 19992,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.6562453103896754,
            "MicroF1": 0.6562453103896754,
            "MacroF1": 0.6545704957554573,
            "Memory in Mb": 4.773673057556152,
            "Time in s": 448.962927
          },
          {
            "step": 20400,
            "track": "Multiclass classification",
            "model": "Naive Bayes",
            "dataset": "Keystroke",
            "Accuracy": 0.6525319868621011,
            "MicroF1": 0.6525319868621011,
            "MacroF1": 0.6515767870317881,
            "Memory in Mb": 4.869010925292969,
            "Time in s": 473.747426
          },
          {
            "step": 46,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.3555555555555555,
            "MicroF1": 0.3555555555555555,
            "MacroF1": 0.2537942449707155,
            "Memory in Mb": 0.4191083908081054,
            "Time in s": 0.164966
          },
          {
            "step": 92,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.4945054945054945,
            "MicroF1": 0.4945054945054945,
            "MacroF1": 0.5043329927491419,
            "Memory in Mb": 0.4191045761108398,
            "Time in s": 0.355643
          },
          {
            "step": 138,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.5328467153284672,
            "MicroF1": 0.5328467153284672,
            "MacroF1": 0.5564033878668025,
            "Memory in Mb": 0.4191999435424804,
            "Time in s": 0.571134
          },
          {
            "step": 184,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.6010928961748634,
            "MicroF1": 0.6010928961748634,
            "MacroF1": 0.6227664965396451,
            "Memory in Mb": 0.4191999435424804,
            "Time in s": 0.8113900000000001
          },
          {
            "step": 230,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.6375545851528385,
            "MicroF1": 0.6375545851528385,
            "MacroF1": 0.6539827168809461,
            "Memory in Mb": 0.4192228317260742,
            "Time in s": 1.079154
          },
          {
            "step": 276,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.6509090909090909,
            "MicroF1": 0.6509090909090909,
            "MacroF1": 0.6671561759164943,
            "Memory in Mb": 0.4192724227905273,
            "Time in s": 1.371943
          },
          {
            "step": 322,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.67601246105919,
            "MicroF1": 0.67601246105919,
            "MacroF1": 0.6756614325426025,
            "Memory in Mb": 0.4192724227905273,
            "Time in s": 1.689575
          },
          {
            "step": 368,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7029972752043597,
            "MicroF1": 0.7029972752043597,
            "MacroF1": 0.6993447851636565,
            "Memory in Mb": 0.4192457199096679,
            "Time in s": 2.032058
          },
          {
            "step": 414,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7142857142857143,
            "MicroF1": 0.7142857142857143,
            "MacroF1": 0.7108606838045498,
            "Memory in Mb": 0.4191656112670898,
            "Time in s": 2.4019660000000003
          },
          {
            "step": 460,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7145969498910676,
            "MicroF1": 0.7145969498910676,
            "MacroF1": 0.7090365931960759,
            "Memory in Mb": 0.4192419052124023,
            "Time in s": 2.796914
          },
          {
            "step": 506,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7207920792079208,
            "MicroF1": 0.7207920792079208,
            "MacroF1": 0.7126631585949763,
            "Memory in Mb": 0.4192419052124023,
            "Time in s": 3.216844
          },
          {
            "step": 552,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7223230490018149,
            "MicroF1": 0.7223230490018149,
            "MacroF1": 0.7157730164623107,
            "Memory in Mb": 0.4191350936889648,
            "Time in s": 3.6616
          },
          {
            "step": 598,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7286432160804021,
            "MicroF1": 0.7286432160804021,
            "MacroF1": 0.7216745323124732,
            "Memory in Mb": 0.4191579818725586,
            "Time in s": 4.131175
          },
          {
            "step": 644,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7278382581648523,
            "MicroF1": 0.7278382581648523,
            "MacroF1": 0.72291051830875,
            "Memory in Mb": 0.4191312789916992,
            "Time in s": 4.628008
          },
          {
            "step": 690,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7314949201741655,
            "MicroF1": 0.7314949201741654,
            "MacroF1": 0.7263583447448078,
            "Memory in Mb": 0.4191312789916992,
            "Time in s": 5.149870999999999
          },
          {
            "step": 736,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7333333333333333,
            "MicroF1": 0.7333333333333333,
            "MacroF1": 0.729431071218305,
            "Memory in Mb": 0.4191579818725586,
            "Time in s": 5.696603
          },
          {
            "step": 782,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7387964148527529,
            "MicroF1": 0.7387964148527529,
            "MacroF1": 0.7349287389986899,
            "Memory in Mb": 0.4191579818725586,
            "Time in s": 6.268242
          },
          {
            "step": 828,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7376058041112454,
            "MicroF1": 0.7376058041112454,
            "MacroF1": 0.7356226390109742,
            "Memory in Mb": 0.4191579818725586,
            "Time in s": 6.867156
          },
          {
            "step": 874,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7445589919816724,
            "MicroF1": 0.7445589919816724,
            "MacroF1": 0.7409366047432264,
            "Memory in Mb": 0.4191579818725586,
            "Time in s": 7.49107
          },
          {
            "step": 920,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7453754080522307,
            "MicroF1": 0.7453754080522307,
            "MacroF1": 0.7408438328939173,
            "Memory in Mb": 0.4191312789916992,
            "Time in s": 8.139827
          },
          {
            "step": 966,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7471502590673575,
            "MicroF1": 0.7471502590673575,
            "MacroF1": 0.7416651838589269,
            "Memory in Mb": 0.4191312789916992,
            "Time in s": 8.813418
          },
          {
            "step": 1012,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7467853610286844,
            "MicroF1": 0.7467853610286844,
            "MacroF1": 0.7416356251822,
            "Memory in Mb": 0.4191312789916992,
            "Time in s": 9.514287
          },
          {
            "step": 1058,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7492904446546831,
            "MicroF1": 0.7492904446546831,
            "MacroF1": 0.7430778844390782,
            "Memory in Mb": 0.4191312789916992,
            "Time in s": 10.240045
          },
          {
            "step": 1104,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7515865820489573,
            "MicroF1": 0.7515865820489573,
            "MacroF1": 0.7451256886686588,
            "Memory in Mb": 0.4191808700561523,
            "Time in s": 10.990683
          },
          {
            "step": 1150,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7536988685813751,
            "MicroF1": 0.7536988685813751,
            "MacroF1": 0.7468312166689606,
            "Memory in Mb": 0.4191808700561523,
            "Time in s": 11.766057
          },
          {
            "step": 1196,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7564853556485356,
            "MicroF1": 0.7564853556485356,
            "MacroF1": 0.7503479321738039,
            "Memory in Mb": 0.4191808700561523,
            "Time in s": 12.566171
          },
          {
            "step": 1242,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7566478646253022,
            "MicroF1": 0.7566478646253022,
            "MacroF1": 0.7509717522131719,
            "Memory in Mb": 0.4191808700561523,
            "Time in s": 13.393734
          },
          {
            "step": 1288,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7614607614607615,
            "MicroF1": 0.7614607614607615,
            "MacroF1": 0.7547643483779538,
            "Memory in Mb": 0.4191808700561523,
            "Time in s": 14.246394
          },
          {
            "step": 1334,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7614403600900225,
            "MicroF1": 0.7614403600900225,
            "MacroF1": 0.7551060921605869,
            "Memory in Mb": 0.4191808700561523,
            "Time in s": 15.123846
          },
          {
            "step": 1380,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7621464829586657,
            "MicroF1": 0.7621464829586658,
            "MacroF1": 0.7562209880685911,
            "Memory in Mb": 0.4191808700561523,
            "Time in s": 16.026049
          },
          {
            "step": 1426,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7642105263157895,
            "MicroF1": 0.7642105263157895,
            "MacroF1": 0.7575332274919562,
            "Memory in Mb": 0.4191808700561523,
            "Time in s": 16.955566
          },
          {
            "step": 1472,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7688647178789939,
            "MicroF1": 0.768864717878994,
            "MacroF1": 0.760438686053582,
            "Memory in Mb": 0.4191808700561523,
            "Time in s": 17.910383
          },
          {
            "step": 1518,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7705998681608438,
            "MicroF1": 0.7705998681608438,
            "MacroF1": 0.7612069012840875,
            "Memory in Mb": 0.4191808700561523,
            "Time in s": 18.890183
          },
          {
            "step": 1564,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7709532949456174,
            "MicroF1": 0.7709532949456174,
            "MacroF1": 0.7622701654854867,
            "Memory in Mb": 0.4191808700561523,
            "Time in s": 19.895086
          },
          {
            "step": 1610,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7712865133623369,
            "MicroF1": 0.771286513362337,
            "MacroF1": 0.7617247271717752,
            "Memory in Mb": 0.4192037582397461,
            "Time in s": 20.927569
          },
          {
            "step": 1656,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7709969788519637,
            "MicroF1": 0.7709969788519637,
            "MacroF1": 0.7615629120572474,
            "Memory in Mb": 0.4192037582397461,
            "Time in s": 21.985292
          },
          {
            "step": 1702,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.770135214579659,
            "MicroF1": 0.770135214579659,
            "MacroF1": 0.7627316365695141,
            "Memory in Mb": 0.4192037582397461,
            "Time in s": 23.068121
          },
          {
            "step": 1748,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7727532913566113,
            "MicroF1": 0.7727532913566113,
            "MacroF1": 0.7649467707214076,
            "Memory in Mb": 0.4192037582397461,
            "Time in s": 24.176005
          },
          {
            "step": 1794,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7741215839375348,
            "MicroF1": 0.7741215839375348,
            "MacroF1": 0.7649332326562147,
            "Memory in Mb": 0.4191770553588867,
            "Time in s": 25.309107999999995
          },
          {
            "step": 1840,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7754214246873301,
            "MicroF1": 0.7754214246873301,
            "MacroF1": 0.7664700790631906,
            "Memory in Mb": 0.4191770553588867,
            "Time in s": 26.470049
          },
          {
            "step": 1886,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7740053050397878,
            "MicroF1": 0.7740053050397878,
            "MacroF1": 0.7655121135276625,
            "Memory in Mb": 0.4191770553588867,
            "Time in s": 27.656614999999995
          },
          {
            "step": 1932,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7742102537545313,
            "MicroF1": 0.7742102537545313,
            "MacroF1": 0.7648034036287765,
            "Memory in Mb": 0.4191770553588867,
            "Time in s": 28.868293999999995
          },
          {
            "step": 1978,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7754172989377845,
            "MicroF1": 0.7754172989377845,
            "MacroF1": 0.7656013068970458,
            "Memory in Mb": 0.4191770553588867,
            "Time in s": 30.105038
          },
          {
            "step": 2024,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7770637666831438,
            "MicroF1": 0.7770637666831438,
            "MacroF1": 0.7660878232247856,
            "Memory in Mb": 0.4191770553588867,
            "Time in s": 31.36953
          },
          {
            "step": 2070,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7762203963267279,
            "MicroF1": 0.7762203963267279,
            "MacroF1": 0.7654829214385931,
            "Memory in Mb": 0.4191770553588867,
            "Time in s": 32.658967
          },
          {
            "step": 2116,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7768321513002364,
            "MicroF1": 0.7768321513002364,
            "MacroF1": 0.7653071619305024,
            "Memory in Mb": 0.4191770553588867,
            "Time in s": 33.973288999999994
          },
          {
            "step": 2162,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7778806108283203,
            "MicroF1": 0.7778806108283203,
            "MacroF1": 0.7659351904174981,
            "Memory in Mb": 0.4191770553588867,
            "Time in s": 35.312507
          },
          {
            "step": 2208,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7797915722700498,
            "MicroF1": 0.7797915722700498,
            "MacroF1": 0.7668192864082087,
            "Memory in Mb": 0.4191770553588867,
            "Time in s": 36.679284
          },
          {
            "step": 2254,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7767421216156236,
            "MicroF1": 0.7767421216156236,
            "MacroF1": 0.7637794374955548,
            "Memory in Mb": 0.4191770553588867,
            "Time in s": 38.070969
          },
          {
            "step": 2300,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7759895606785558,
            "MicroF1": 0.7759895606785558,
            "MacroF1": 0.763026662835187,
            "Memory in Mb": 0.4191770553588867,
            "Time in s": 39.487872
          },
          {
            "step": 1056,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.6218009478672986,
            "MicroF1": 0.6218009478672986,
            "MacroF1": 0.585266310719421,
            "Memory in Mb": 0.6617898941040039,
            "Time in s": 1.016292
          },
          {
            "step": 2112,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.6153481762198011,
            "MicroF1": 0.6153481762198011,
            "MacroF1": 0.5806436317780949,
            "Memory in Mb": 0.6617898941040039,
            "Time in s": 2.820671
          },
          {
            "step": 3168,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.6071992421850332,
            "MicroF1": 0.6071992421850332,
            "MacroF1": 0.572248584718361,
            "Memory in Mb": 0.6617898941040039,
            "Time in s": 5.468586999999999
          },
          {
            "step": 4224,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.6043097324177126,
            "MicroF1": 0.6043097324177126,
            "MacroF1": 0.5697573109597247,
            "Memory in Mb": 0.6617898941040039,
            "Time in s": 8.970813999999999
          },
          {
            "step": 5280,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.6088274294373934,
            "MicroF1": 0.6088274294373934,
            "MacroF1": 0.5727379077413696,
            "Memory in Mb": 0.6617898941040039,
            "Time in s": 13.304299
          },
          {
            "step": 6336,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.6026835043409629,
            "MicroF1": 0.6026835043409629,
            "MacroF1": 0.568251333238805,
            "Memory in Mb": 0.6617898941040039,
            "Time in s": 18.451533
          },
          {
            "step": 7392,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.600189419564335,
            "MicroF1": 0.600189419564335,
            "MacroF1": 0.5659762112716077,
            "Memory in Mb": 0.6617898941040039,
            "Time in s": 24.373815
          },
          {
            "step": 8448,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.60258079791642,
            "MicroF1": 0.60258079791642,
            "MacroF1": 0.5679781484640409,
            "Memory in Mb": 0.6617898941040039,
            "Time in s": 31.061276
          },
          {
            "step": 9504,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5998105861306956,
            "MicroF1": 0.5998105861306956,
            "MacroF1": 0.5649597336877693,
            "Memory in Mb": 0.6617898941040039,
            "Time in s": 38.490335
          },
          {
            "step": 10560,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5998674116867128,
            "MicroF1": 0.5998674116867128,
            "MacroF1": 0.5650173260529011,
            "Memory in Mb": 0.6617898941040039,
            "Time in s": 46.63726
          },
          {
            "step": 11616,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5974171330176495,
            "MicroF1": 0.5974171330176495,
            "MacroF1": 0.5633067089377386,
            "Memory in Mb": 0.6617898941040039,
            "Time in s": 55.514266
          },
          {
            "step": 12672,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.6001894088864336,
            "MicroF1": 0.6001894088864336,
            "MacroF1": 0.5684760329567131,
            "Memory in Mb": 0.6617898941040039,
            "Time in s": 65.102691
          },
          {
            "step": 13728,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.6120783856632913,
            "MicroF1": 0.6120783856632913,
            "MacroF1": 0.5935956771555828,
            "Memory in Mb": 0.6617898941040039,
            "Time in s": 75.408233
          },
          {
            "step": 14784,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.6024487587093282,
            "MicroF1": 0.6024487587093282,
            "MacroF1": 0.5842148300149193,
            "Memory in Mb": 0.6617898941040039,
            "Time in s": 86.426133
          },
          {
            "step": 15840,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5677757434181451,
            "MicroF1": 0.5677757434181451,
            "MacroF1": 0.5509250187877572,
            "Memory in Mb": 0.6617898941040039,
            "Time in s": 98.158455
          },
          {
            "step": 16896,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5419354838709678,
            "MicroF1": 0.5419354838709678,
            "MacroF1": 0.5257359157219257,
            "Memory in Mb": 0.6617898941040039,
            "Time in s": 110.605121
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5233691716338923,
            "MicroF1": 0.5233691716338923,
            "MacroF1": 0.506858183835206,
            "Memory in Mb": 0.6617898941040039,
            "Time in s": 123.763106
          },
          {
            "step": 19008,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5121271110643447,
            "MicroF1": 0.5121271110643447,
            "MacroF1": 0.4929289906509415,
            "Memory in Mb": 0.6617898941040039,
            "Time in s": 137.636213
          },
          {
            "step": 20064,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5120370831879579,
            "MicroF1": 0.5120370831879579,
            "MacroF1": 0.4920970323041603,
            "Memory in Mb": 1.317840576171875,
            "Time in s": 152.19804
          },
          {
            "step": 21120,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5173066906577016,
            "MicroF1": 0.5173066906577016,
            "MacroF1": 0.497344716983625,
            "Memory in Mb": 1.3185958862304688,
            "Time in s": 167.375493
          },
          {
            "step": 22176,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5229312288613304,
            "MicroF1": 0.5229312288613304,
            "MacroF1": 0.5026343687424488,
            "Memory in Mb": 1.3185958862304688,
            "Time in s": 183.138263
          },
          {
            "step": 23232,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5301536739701261,
            "MicroF1": 0.5301536739701261,
            "MacroF1": 0.5095132087733324,
            "Memory in Mb": 1.3185958862304688,
            "Time in s": 199.493958
          },
          {
            "step": 24288,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5351422571746202,
            "MicroF1": 0.5351422571746202,
            "MacroF1": 0.5135975374357353,
            "Memory in Mb": 1.3185958862304688,
            "Time in s": 216.435818
          },
          {
            "step": 25344,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5403069881229531,
            "MicroF1": 0.5403069881229531,
            "MacroF1": 0.5180803411538233,
            "Memory in Mb": 1.3185958862304688,
            "Time in s": 233.973059
          },
          {
            "step": 26400,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5441493995984696,
            "MicroF1": 0.5441493995984696,
            "MacroF1": 0.5209012984387186,
            "Memory in Mb": 1.3185958862304688,
            "Time in s": 252.0993
          },
          {
            "step": 27456,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5475869604807867,
            "MicroF1": 0.5475869604807867,
            "MacroF1": 0.5230407124785976,
            "Memory in Mb": 1.3185958862304688,
            "Time in s": 270.826115
          },
          {
            "step": 28512,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5442460804601733,
            "MicroF1": 0.5442460804601733,
            "MacroF1": 0.5199893698637053,
            "Memory in Mb": 1.3185958862304688,
            "Time in s": 290.125735
          },
          {
            "step": 29568,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5439848479724017,
            "MicroF1": 0.5439848479724017,
            "MacroF1": 0.5225387960194382,
            "Memory in Mb": 1.3185958862304688,
            "Time in s": 310.131151
          },
          {
            "step": 30624,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5449825294713124,
            "MicroF1": 0.5449825294713124,
            "MacroF1": 0.5260472440529832,
            "Memory in Mb": 1.3185958862304688,
            "Time in s": 330.869455
          },
          {
            "step": 31680,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5469238296663405,
            "MicroF1": 0.5469238296663405,
            "MacroF1": 0.5300194392617626,
            "Memory in Mb": 1.3185958862304688,
            "Time in s": 352.339648
          },
          {
            "step": 32736,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5492286543455017,
            "MicroF1": 0.5492286543455017,
            "MacroF1": 0.5337692045397759,
            "Memory in Mb": 1.3185958862304688,
            "Time in s": 374.544388
          },
          {
            "step": 33792,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5448196265277737,
            "MicroF1": 0.5448196265277737,
            "MacroF1": 0.5298516474077152,
            "Memory in Mb": 1.3185958862304688,
            "Time in s": 397.480297
          },
          {
            "step": 34848,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.539357763939507,
            "MicroF1": 0.539357763939507,
            "MacroF1": 0.5246413689313029,
            "Memory in Mb": 1.3185958862304688,
            "Time in s": 421.148709
          },
          {
            "step": 35904,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5352756037099964,
            "MicroF1": 0.5352756037099964,
            "MacroF1": 0.5204658240271912,
            "Memory in Mb": 1.3185958862304688,
            "Time in s": 445.552724
          },
          {
            "step": 36960,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5307232338537298,
            "MicroF1": 0.5307232338537298,
            "MacroF1": 0.5158458403074863,
            "Memory in Mb": 1.3185958862304688,
            "Time in s": 470.685377
          },
          {
            "step": 38016,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5287912666052874,
            "MicroF1": 0.5287912666052874,
            "MacroF1": 0.5138605376143625,
            "Memory in Mb": 1.8598642349243164,
            "Time in s": 496.544653
          },
          {
            "step": 39072,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5245322617798367,
            "MicroF1": 0.5245322617798367,
            "MacroF1": 0.5100329616180462,
            "Memory in Mb": 1.9744834899902344,
            "Time in s": 523.1337460000001
          },
          {
            "step": 40128,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5244847608841927,
            "MicroF1": 0.5244847608841927,
            "MacroF1": 0.5114466799524962,
            "Memory in Mb": 1.9744834899902344,
            "Time in s": 550.3794180000001
          },
          {
            "step": 41184,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5269650098341548,
            "MicroF1": 0.5269650098341548,
            "MacroF1": 0.5145630920489553,
            "Memory in Mb": 1.9744834899902344,
            "Time in s": 578.1701970000001
          },
          {
            "step": 42240,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5290608205686688,
            "MicroF1": 0.5290608205686688,
            "MacroF1": 0.5171452370879218,
            "Memory in Mb": 1.9744834899902344,
            "Time in s": 606.4941780000001
          },
          {
            "step": 43296,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5316318281556762,
            "MicroF1": 0.5316318281556762,
            "MacroF1": 0.5200714653059242,
            "Memory in Mb": 1.9744834899902344,
            "Time in s": 635.3594070000001
          },
          {
            "step": 44352,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5332912448422809,
            "MicroF1": 0.5332912448422809,
            "MacroF1": 0.521951703681177,
            "Memory in Mb": 1.975238800048828,
            "Time in s": 664.7773900000002
          },
          {
            "step": 45408,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5350937080185875,
            "MicroF1": 0.5350937080185875,
            "MacroF1": 0.5236272112757866,
            "Memory in Mb": 1.975238800048828,
            "Time in s": 694.7425150000001
          },
          {
            "step": 46464,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5374168693368917,
            "MicroF1": 0.5374168693368917,
            "MacroF1": 0.5257977177437826,
            "Memory in Mb": 1.975238800048828,
            "Time in s": 725.2648820000002
          },
          {
            "step": 47520,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5359540394368568,
            "MicroF1": 0.5359540394368568,
            "MacroF1": 0.5247049329892776,
            "Memory in Mb": 1.975238800048828,
            "Time in s": 756.3925470000001
          },
          {
            "step": 48576,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5333196088522902,
            "MicroF1": 0.5333196088522902,
            "MacroF1": 0.5224640186909638,
            "Memory in Mb": 1.975238800048828,
            "Time in s": 788.1537450000002
          },
          {
            "step": 49632,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5314017448771937,
            "MicroF1": 0.5314017448771937,
            "MacroF1": 0.5209076603734538,
            "Memory in Mb": 1.975238800048828,
            "Time in s": 820.5431960000002
          },
          {
            "step": 50688,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5321877404462683,
            "MicroF1": 0.5321877404462683,
            "MacroF1": 0.5219332135179457,
            "Memory in Mb": 2.097897529602051,
            "Time in s": 853.5752100000002
          },
          {
            "step": 51744,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5376959202210927,
            "MicroF1": 0.5376959202210927,
            "MacroF1": 0.5274519689249669,
            "Memory in Mb": 2.335637092590332,
            "Time in s": 887.2128290000002
          },
          {
            "step": 52800,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Insects",
            "Accuracy": 0.5370177465482301,
            "MicroF1": 0.5370177465482301,
            "MacroF1": 0.5270712327692165,
            "Memory in Mb": 2.5391950607299805,
            "Time in s": 921.3507020000002
          },
          {
            "step": 408,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.9803439803439804,
            "MicroF1": 0.9803439803439804,
            "MacroF1": 0.4950372208436724,
            "Memory in Mb": 0.2276544570922851,
            "Time in s": 0.136786
          },
          {
            "step": 816,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.9423312883435584,
            "MicroF1": 0.9423312883435584,
            "MacroF1": 0.7661667470992702,
            "Memory in Mb": 0.3232784271240234,
            "Time in s": 0.4359419999999999
          },
          {
            "step": 1224,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.8830744071954211,
            "MicroF1": 0.883074407195421,
            "MacroF1": 0.8761191747044462,
            "Memory in Mb": 0.4189023971557617,
            "Time in s": 0.926938
          },
          {
            "step": 1632,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.8902513795217658,
            "MicroF1": 0.8902513795217658,
            "MacroF1": 0.8767853151263398,
            "Memory in Mb": 0.5150146484375,
            "Time in s": 1.637883
          },
          {
            "step": 2040,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.8891613536047082,
            "MicroF1": 0.8891613536047082,
            "MacroF1": 0.8807858055314012,
            "Memory in Mb": 0.6221132278442383,
            "Time in s": 2.570345
          },
          {
            "step": 2448,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.848385778504291,
            "MicroF1": 0.848385778504291,
            "MacroF1": 0.8522513926518692,
            "Memory in Mb": 0.7177371978759766,
            "Time in s": 3.758757
          },
          {
            "step": 2856,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.8563922942206655,
            "MicroF1": 0.8563922942206655,
            "MacroF1": 0.8440193478447515,
            "Memory in Mb": 0.8133611679077148,
            "Time in s": 5.245901
          },
          {
            "step": 3264,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.8482991112473184,
            "MicroF1": 0.8482991112473184,
            "MacroF1": 0.8269786301577753,
            "Memory in Mb": 0.9089851379394532,
            "Time in s": 7.065474
          },
          {
            "step": 3672,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.8392808499046581,
            "MicroF1": 0.8392808499046581,
            "MacroF1": 0.8374924160046074,
            "Memory in Mb": 1.0046091079711914,
            "Time in s": 9.269679
          },
          {
            "step": 4080,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.8323118411375338,
            "MicroF1": 0.8323118411375338,
            "MacroF1": 0.8182261307945194,
            "Memory in Mb": 1.1253337860107422,
            "Time in s": 11.89727
          },
          {
            "step": 4488,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.8159126365054602,
            "MicroF1": 0.8159126365054602,
            "MacroF1": 0.8260965842218733,
            "Memory in Mb": 1.2209577560424805,
            "Time in s": 14.983422
          },
          {
            "step": 4896,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.8149131767109296,
            "MicroF1": 0.8149131767109296,
            "MacroF1": 0.8221314665977922,
            "Memory in Mb": 1.3165817260742188,
            "Time in s": 18.566921
          },
          {
            "step": 5304,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.8125589289081652,
            "MicroF1": 0.8125589289081652,
            "MacroF1": 0.797613058026624,
            "Memory in Mb": 1.412205696105957,
            "Time in s": 22.693048
          },
          {
            "step": 5712,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7907546839432674,
            "MicroF1": 0.7907546839432674,
            "MacroF1": 0.7936708037520237,
            "Memory in Mb": 1.507829666137695,
            "Time in s": 27.396131
          },
          {
            "step": 6120,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7886909625755842,
            "MicroF1": 0.7886909625755842,
            "MacroF1": 0.7694478218498494,
            "Memory in Mb": 1.6034536361694336,
            "Time in s": 32.715078
          },
          {
            "step": 6528,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7635973647924008,
            "MicroF1": 0.7635973647924008,
            "MacroF1": 0.75687960152136,
            "Memory in Mb": 1.699077606201172,
            "Time in s": 38.687416
          },
          {
            "step": 6936,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.75155010814708,
            "MicroF1": 0.7515501081470799,
            "MacroF1": 0.7521509466338958,
            "Memory in Mb": 1.7947015762329102,
            "Time in s": 45.356366
          },
          {
            "step": 7344,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7611330518861501,
            "MicroF1": 0.7611330518861501,
            "MacroF1": 0.7576671162861804,
            "Memory in Mb": 1.8917903900146484,
            "Time in s": 52.757111
          },
          {
            "step": 7752,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7617081666881693,
            "MicroF1": 0.7617081666881692,
            "MacroF1": 0.7593340838982118,
            "Memory in Mb": 1.9874143600463867,
            "Time in s": 60.92847
          },
          {
            "step": 8160,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7655349920333374,
            "MicroF1": 0.7655349920333374,
            "MacroF1": 0.7610505848438686,
            "Memory in Mb": 2.083038330078125,
            "Time in s": 69.910689
          },
          {
            "step": 8568,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7644449632310026,
            "MicroF1": 0.7644449632310025,
            "MacroF1": 0.7639417799779614,
            "Memory in Mb": 2.226712226867676,
            "Time in s": 79.742469
          },
          {
            "step": 8976,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7624512534818941,
            "MicroF1": 0.7624512534818941,
            "MacroF1": 0.7625605608371231,
            "Memory in Mb": 2.322336196899414,
            "Time in s": 90.464241
          },
          {
            "step": 9384,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7605243525524885,
            "MicroF1": 0.7605243525524885,
            "MacroF1": 0.7588384348689571,
            "Memory in Mb": 2.4179601669311523,
            "Time in s": 102.115634
          },
          {
            "step": 9792,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.753344908589521,
            "MicroF1": 0.753344908589521,
            "MacroF1": 0.7499438215834663,
            "Memory in Mb": 2.51358413696289,
            "Time in s": 114.735409
          },
          {
            "step": 10200,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7450730463770958,
            "MicroF1": 0.7450730463770959,
            "MacroF1": 0.7369660419615973,
            "Memory in Mb": 2.609208106994629,
            "Time in s": 128.375943
          },
          {
            "step": 10608,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7240501555576506,
            "MicroF1": 0.7240501555576506,
            "MacroF1": 0.7111305646829175,
            "Memory in Mb": 2.704832077026367,
            "Time in s": 143.06648900000002
          },
          {
            "step": 11016,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7166591012256015,
            "MicroF1": 0.7166591012256015,
            "MacroF1": 0.7122511515574345,
            "Memory in Mb": 2.800456047058105,
            "Time in s": 158.84720500000003
          },
          {
            "step": 11424,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.710146196270682,
            "MicroF1": 0.710146196270682,
            "MacroF1": 0.6963016796632095,
            "Memory in Mb": 2.896080017089844,
            "Time in s": 175.75710000000004
          },
          {
            "step": 11832,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7005324993660722,
            "MicroF1": 0.7005324993660722,
            "MacroF1": 0.6925666211338902,
            "Memory in Mb": 2.991703987121582,
            "Time in s": 193.83910100000003
          },
          {
            "step": 12240,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7043876133671052,
            "MicroF1": 0.7043876133671052,
            "MacroF1": 0.7007845610449206,
            "Memory in Mb": 3.0873279571533203,
            "Time in s": 213.15240600000004
          },
          {
            "step": 12648,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7004032576895707,
            "MicroF1": 0.7004032576895707,
            "MacroF1": 0.6915775762792659,
            "Memory in Mb": 3.1829519271850586,
            "Time in s": 233.71252100000004
          },
          {
            "step": 13056,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6877058598238223,
            "MicroF1": 0.6877058598238223,
            "MacroF1": 0.6789768292873962,
            "Memory in Mb": 3.278575897216797,
            "Time in s": 255.55614900000003
          },
          {
            "step": 13464,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6838743222164451,
            "MicroF1": 0.6838743222164451,
            "MacroF1": 0.6791243465680946,
            "Memory in Mb": 3.374199867248535,
            "Time in s": 278.72697300000004
          },
          {
            "step": 13872,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6822146925239708,
            "MicroF1": 0.6822146925239708,
            "MacroF1": 0.6786558938530484,
            "Memory in Mb": 3.469823837280273,
            "Time in s": 303.265051
          },
          {
            "step": 14280,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6777085230058127,
            "MicroF1": 0.6777085230058127,
            "MacroF1": 0.6725285130045525,
            "Memory in Mb": 3.565447807312012,
            "Time in s": 329.21132600000004
          },
          {
            "step": 14688,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6807380676788997,
            "MicroF1": 0.6807380676788997,
            "MacroF1": 0.6786761142186741,
            "Memory in Mb": 3.66107177734375,
            "Time in s": 356.60277
          },
          {
            "step": 15096,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6873799271281882,
            "MicroF1": 0.6873799271281882,
            "MacroF1": 0.68548393064844,
            "Memory in Mb": 3.756695747375488,
            "Time in s": 385.483239
          },
          {
            "step": 15504,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6858027478552539,
            "MicroF1": 0.6858027478552539,
            "MacroF1": 0.6816808496509055,
            "Memory in Mb": 3.8523197174072266,
            "Time in s": 415.890234
          },
          {
            "step": 15912,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6765759537426937,
            "MicroF1": 0.6765759537426937,
            "MacroF1": 0.6694713281964944,
            "Memory in Mb": 3.947943687438965,
            "Time in s": 447.863862
          },
          {
            "step": 16320,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6673815797536614,
            "MicroF1": 0.6673815797536614,
            "MacroF1": 0.6617321933140904,
            "Memory in Mb": 4.043567657470703,
            "Time in s": 481.440864
          },
          {
            "step": 16728,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6643151790518323,
            "MicroF1": 0.6643151790518323,
            "MacroF1": 0.6611780293584051,
            "Memory in Mb": 4.139191627502441,
            "Time in s": 516.667277
          },
          {
            "step": 17136,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6598774438284214,
            "MicroF1": 0.6598774438284214,
            "MacroF1": 0.655734247886306,
            "Memory in Mb": 4.333066940307617,
            "Time in s": 553.571784
          },
          {
            "step": 17544,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6518269395200365,
            "MicroF1": 0.6518269395200365,
            "MacroF1": 0.6481085155228207,
            "Memory in Mb": 4.428690910339356,
            "Time in s": 592.193317
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6507158375577963,
            "MicroF1": 0.6507158375577963,
            "MacroF1": 0.648936899585426,
            "Memory in Mb": 4.524314880371094,
            "Time in s": 632.57823
          },
          {
            "step": 18360,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6566806470940683,
            "MicroF1": 0.6566806470940683,
            "MacroF1": 0.6555764711123697,
            "Memory in Mb": 4.619938850402832,
            "Time in s": 674.762883
          },
          {
            "step": 18768,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.662279533223211,
            "MicroF1": 0.662279533223211,
            "MacroF1": 0.6615432060687811,
            "Memory in Mb": 4.71556282043457,
            "Time in s": 718.781471
          },
          {
            "step": 19176,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6534028683181226,
            "MicroF1": 0.6534028683181226,
            "MacroF1": 0.6508089832432515,
            "Memory in Mb": 4.811186790466309,
            "Time in s": 764.6790530000001
          },
          {
            "step": 19584,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6577643874789358,
            "MicroF1": 0.6577643874789358,
            "MacroF1": 0.6564201177589184,
            "Memory in Mb": 4.906810760498047,
            "Time in s": 812.4977690000001
          },
          {
            "step": 19992,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6518433294982742,
            "MicroF1": 0.6518433294982742,
            "MacroF1": 0.6501496360982538,
            "Memory in Mb": 5.002434730529785,
            "Time in s": 862.2665020000001
          },
          {
            "step": 20400,
            "track": "Multiclass classification",
            "model": "Hoeffding Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6482180499044071,
            "MicroF1": 0.6482180499044071,
            "MacroF1": 0.6472493759146579,
            "Memory in Mb": 5.098058700561523,
            "Time in s": 914.036687
          },
          {
            "step": 46,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.4,
            "MicroF1": 0.4000000000000001,
            "MacroF1": 0.2926704014939309,
            "Memory in Mb": 0.4254798889160156,
            "Time in s": 0.179349
          },
          {
            "step": 92,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.5274725274725275,
            "MicroF1": 0.5274725274725275,
            "MacroF1": 0.5399541634835753,
            "Memory in Mb": 0.425537109375,
            "Time in s": 0.395098
          },
          {
            "step": 138,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.5547445255474452,
            "MicroF1": 0.5547445255474452,
            "MacroF1": 0.5795767508697842,
            "Memory in Mb": 0.4256591796875,
            "Time in s": 0.646414
          },
          {
            "step": 184,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.6174863387978142,
            "MicroF1": 0.6174863387978142,
            "MacroF1": 0.6398140932417979,
            "Memory in Mb": 0.4257431030273437,
            "Time in s": 0.936097
          },
          {
            "step": 230,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.6419213973799127,
            "MicroF1": 0.6419213973799127,
            "MacroF1": 0.6592174177506214,
            "Memory in Mb": 0.4257431030273437,
            "Time in s": 1.2615820000000002
          },
          {
            "step": 276,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.6545454545454545,
            "MicroF1": 0.6545454545454545,
            "MacroF1": 0.6716869228432982,
            "Memory in Mb": 0.4257926940917969,
            "Time in s": 1.6228100000000003
          },
          {
            "step": 322,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.6791277258566978,
            "MicroF1": 0.6791277258566978,
            "MacroF1": 0.6806263486692059,
            "Memory in Mb": 0.4258537292480469,
            "Time in s": 2.022508
          },
          {
            "step": 368,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7029972752043597,
            "MicroF1": 0.7029972752043597,
            "MacroF1": 0.7008299817149242,
            "Memory in Mb": 0.4258270263671875,
            "Time in s": 2.458102
          },
          {
            "step": 414,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7142857142857143,
            "MicroF1": 0.7142857142857143,
            "MacroF1": 0.7121569327354127,
            "Memory in Mb": 0.4257469177246094,
            "Time in s": 2.92926
          },
          {
            "step": 460,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7145969498910676,
            "MicroF1": 0.7145969498910676,
            "MacroF1": 0.7103106155638,
            "Memory in Mb": 0.4258232116699219,
            "Time in s": 3.4385440000000003
          },
          {
            "step": 506,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7227722772277227,
            "MicroF1": 0.7227722772277227,
            "MacroF1": 0.715881182832702,
            "Memory in Mb": 0.4258232116699219,
            "Time in s": 3.983535
          },
          {
            "step": 552,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7241379310344828,
            "MicroF1": 0.7241379310344829,
            "MacroF1": 0.7187949260386588,
            "Memory in Mb": 0.4257164001464844,
            "Time in s": 4.564144000000001
          },
          {
            "step": 598,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7286432160804021,
            "MicroF1": 0.7286432160804021,
            "MacroF1": 0.7227601649788371,
            "Memory in Mb": 0.4257392883300781,
            "Time in s": 5.1830560000000006
          },
          {
            "step": 644,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7278382581648523,
            "MicroF1": 0.7278382581648523,
            "MacroF1": 0.7240595992457829,
            "Memory in Mb": 0.4257736206054687,
            "Time in s": 5.837887
          },
          {
            "step": 690,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7314949201741655,
            "MicroF1": 0.7314949201741654,
            "MacroF1": 0.727547508877315,
            "Memory in Mb": 0.4257736206054687,
            "Time in s": 6.528431
          },
          {
            "step": 736,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7333333333333333,
            "MicroF1": 0.7333333333333333,
            "MacroF1": 0.730585229165138,
            "Memory in Mb": 0.4258003234863281,
            "Time in s": 7.25733
          },
          {
            "step": 782,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7387964148527529,
            "MicroF1": 0.7387964148527529,
            "MacroF1": 0.7359626710287273,
            "Memory in Mb": 0.4258003234863281,
            "Time in s": 8.022590000000001
          },
          {
            "step": 828,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7376058041112454,
            "MicroF1": 0.7376058041112454,
            "MacroF1": 0.7367699509780541,
            "Memory in Mb": 0.4258003234863281,
            "Time in s": 8.823569
          },
          {
            "step": 874,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7434135166093929,
            "MicroF1": 0.7434135166093929,
            "MacroF1": 0.7406779161411566,
            "Memory in Mb": 0.4258003234863281,
            "Time in s": 9.663167
          },
          {
            "step": 920,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7431991294885746,
            "MicroF1": 0.7431991294885745,
            "MacroF1": 0.7396284921253597,
            "Memory in Mb": 0.4257736206054687,
            "Time in s": 10.538696000000002
          },
          {
            "step": 966,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7430051813471502,
            "MicroF1": 0.7430051813471502,
            "MacroF1": 0.7386475429248082,
            "Memory in Mb": 0.4257736206054687,
            "Time in s": 11.449986
          },
          {
            "step": 1012,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7428288822947576,
            "MicroF1": 0.7428288822947575,
            "MacroF1": 0.7387392151852316,
            "Memory in Mb": 0.4257736206054687,
            "Time in s": 12.399906
          },
          {
            "step": 1058,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7445600756859035,
            "MicroF1": 0.7445600756859035,
            "MacroF1": 0.7397141356071754,
            "Memory in Mb": 0.4257736206054687,
            "Time in s": 13.385988
          },
          {
            "step": 1104,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7470534904805077,
            "MicroF1": 0.7470534904805077,
            "MacroF1": 0.7419829508197956,
            "Memory in Mb": 0.4258232116699219,
            "Time in s": 14.408007
          },
          {
            "step": 1150,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7484769364664926,
            "MicroF1": 0.7484769364664926,
            "MacroF1": 0.7430153502407321,
            "Memory in Mb": 0.4258232116699219,
            "Time in s": 15.46854
          },
          {
            "step": 1196,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7514644351464436,
            "MicroF1": 0.7514644351464436,
            "MacroF1": 0.7466450927602833,
            "Memory in Mb": 0.4254570007324219,
            "Time in s": 16.565517
          },
          {
            "step": 1242,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7518130539887188,
            "MicroF1": 0.7518130539887188,
            "MacroF1": 0.7475811251410989,
            "Memory in Mb": 0.4255790710449219,
            "Time in s": 17.698596
          },
          {
            "step": 1288,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7567987567987567,
            "MicroF1": 0.7567987567987567,
            "MacroF1": 0.7515585748403605,
            "Memory in Mb": 0.4256401062011719,
            "Time in s": 18.868108
          },
          {
            "step": 1334,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7576894223555889,
            "MicroF1": 0.7576894223555888,
            "MacroF1": 0.7527145732365901,
            "Memory in Mb": 0.4256401062011719,
            "Time in s": 20.076794
          },
          {
            "step": 1380,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7592458303118201,
            "MicroF1": 0.7592458303118201,
            "MacroF1": 0.754880899709855,
            "Memory in Mb": 0.4257011413574219,
            "Time in s": 21.321463
          },
          {
            "step": 1426,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7621052631578947,
            "MicroF1": 0.7621052631578947,
            "MacroF1": 0.7572480123106181,
            "Memory in Mb": 0.4257011413574219,
            "Time in s": 22.601949
          },
          {
            "step": 1472,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7661454792658056,
            "MicroF1": 0.7661454792658056,
            "MacroF1": 0.7596240117389202,
            "Memory in Mb": 0.4257011413574219,
            "Time in s": 23.921025
          },
          {
            "step": 1518,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7679630850362558,
            "MicroF1": 0.7679630850362558,
            "MacroF1": 0.7604664202984912,
            "Memory in Mb": 0.4257621765136719,
            "Time in s": 25.275945
          },
          {
            "step": 1564,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7683941138835573,
            "MicroF1": 0.7683941138835573,
            "MacroF1": 0.7616623934037686,
            "Memory in Mb": 0.4257621765136719,
            "Time in s": 26.66678
          },
          {
            "step": 1610,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7681789931634556,
            "MicroF1": 0.7681789931634556,
            "MacroF1": 0.7606779105029744,
            "Memory in Mb": 0.4257850646972656,
            "Time in s": 28.096857
          },
          {
            "step": 1656,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7685800604229607,
            "MicroF1": 0.7685800604229607,
            "MacroF1": 0.7611818346958917,
            "Memory in Mb": 0.4257850646972656,
            "Time in s": 29.563118
          },
          {
            "step": 1702,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7683715461493239,
            "MicroF1": 0.768371546149324,
            "MacroF1": 0.7630805397579306,
            "Memory in Mb": 0.4257850646972656,
            "Time in s": 31.065673
          },
          {
            "step": 1748,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7716084716657127,
            "MicroF1": 0.7716084716657126,
            "MacroF1": 0.7661058855209445,
            "Memory in Mb": 0.4257850646972656,
            "Time in s": 32.607308
          },
          {
            "step": 1794,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7730061349693251,
            "MicroF1": 0.7730061349693251,
            "MacroF1": 0.76613283717613,
            "Memory in Mb": 0.4257583618164062,
            "Time in s": 34.185135
          },
          {
            "step": 1840,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7743338771071234,
            "MicroF1": 0.7743338771071234,
            "MacroF1": 0.7676486165305356,
            "Memory in Mb": 0.4258193969726562,
            "Time in s": 35.798944000000006
          },
          {
            "step": 1886,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7729442970822281,
            "MicroF1": 0.7729442970822282,
            "MacroF1": 0.7669643117326908,
            "Memory in Mb": 0.4258193969726562,
            "Time in s": 37.451807
          },
          {
            "step": 1932,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7736923873640601,
            "MicroF1": 0.7736923873640601,
            "MacroF1": 0.7669808567090198,
            "Memory in Mb": 0.4258193969726562,
            "Time in s": 39.140782
          },
          {
            "step": 1978,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7744056651492159,
            "MicroF1": 0.7744056651492159,
            "MacroF1": 0.7669005381948409,
            "Memory in Mb": 0.4258193969726562,
            "Time in s": 40.865953000000005
          },
          {
            "step": 2024,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7755808205635195,
            "MicroF1": 0.7755808205635196,
            "MacroF1": 0.7665616644775576,
            "Memory in Mb": 0.4258193969726562,
            "Time in s": 42.627552
          },
          {
            "step": 2070,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7752537457709038,
            "MicroF1": 0.7752537457709039,
            "MacroF1": 0.7663566554091733,
            "Memory in Mb": 0.4258193969726562,
            "Time in s": 44.428542
          },
          {
            "step": 2116,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.775886524822695,
            "MicroF1": 0.775886524822695,
            "MacroF1": 0.7661827507972012,
            "Memory in Mb": 0.4258193969726562,
            "Time in s": 46.266451
          },
          {
            "step": 2162,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7764923646459972,
            "MicroF1": 0.7764923646459972,
            "MacroF1": 0.7663510353808046,
            "Memory in Mb": 0.4258193969726562,
            "Time in s": 48.14124
          },
          {
            "step": 2208,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7784322609877662,
            "MicroF1": 0.7784322609877662,
            "MacroF1": 0.767276937076619,
            "Memory in Mb": 0.4258193969726562,
            "Time in s": 50.054953000000005
          },
          {
            "step": 2254,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.775410563692854,
            "MicroF1": 0.775410563692854,
            "MacroF1": 0.7642399015136985,
            "Memory in Mb": 0.4258193969726562,
            "Time in s": 52.004875000000006
          },
          {
            "step": 2300,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "ImageSegments",
            "Accuracy": 0.7746846454980426,
            "MicroF1": 0.7746846454980426,
            "MacroF1": 0.7634961218545901,
            "Memory in Mb": 0.4258193969726562,
            "Time in s": 53.99744400000001
          },
          {
            "step": 1056,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.6161137440758294,
            "MicroF1": 0.6161137440758294,
            "MacroF1": 0.5813841513331479,
            "Memory in Mb": 0.6684322357177734,
            "Time in s": 1.380603
          },
          {
            "step": 2112,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.6120322122216959,
            "MicroF1": 0.6120322122216959,
            "MacroF1": 0.5792161554760864,
            "Memory in Mb": 0.6684932708740234,
            "Time in s": 3.972211
          },
          {
            "step": 3168,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.6049889485317335,
            "MicroF1": 0.6049889485317335,
            "MacroF1": 0.5721633809277146,
            "Memory in Mb": 0.6685543060302734,
            "Time in s": 7.807885
          },
          {
            "step": 4224,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.603125739995264,
            "MicroF1": 0.603125739995264,
            "MacroF1": 0.5703574432462962,
            "Memory in Mb": 0.6685543060302734,
            "Time in s": 12.856232
          },
          {
            "step": 5280,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.6061754120098504,
            "MicroF1": 0.6061754120098504,
            "MacroF1": 0.5722430970062696,
            "Memory in Mb": 0.6686153411865234,
            "Time in s": 19.046562
          },
          {
            "step": 6336,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5995264404104184,
            "MicroF1": 0.5995264404104184,
            "MacroF1": 0.5671511237518186,
            "Memory in Mb": 0.6686153411865234,
            "Time in s": 26.345703
          },
          {
            "step": 7392,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5972128264104992,
            "MicroF1": 0.5972128264104992,
            "MacroF1": 0.5650210504998666,
            "Memory in Mb": 0.6686153411865234,
            "Time in s": 34.702495
          },
          {
            "step": 8448,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5989108559251806,
            "MicroF1": 0.5989108559251806,
            "MacroF1": 0.566418690076869,
            "Memory in Mb": 0.6686153411865234,
            "Time in s": 44.110114
          },
          {
            "step": 9504,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5962327685993897,
            "MicroF1": 0.5962327685993897,
            "MacroF1": 0.5633780031885508,
            "Memory in Mb": 0.6686153411865234,
            "Time in s": 54.569631
          },
          {
            "step": 10560,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5964579979164694,
            "MicroF1": 0.5964579979164694,
            "MacroF1": 0.5634236596216465,
            "Memory in Mb": 0.6686763763427734,
            "Time in s": 66.07704199999999
          },
          {
            "step": 11616,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.594317692638829,
            "MicroF1": 0.594317692638829,
            "MacroF1": 0.5620068495149612,
            "Memory in Mb": 0.6686763763427734,
            "Time in s": 78.631408
          },
          {
            "step": 12672,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5975061163286244,
            "MicroF1": 0.5975061163286244,
            "MacroF1": 0.567518061449456,
            "Memory in Mb": 0.6686763763427734,
            "Time in s": 92.232933
          },
          {
            "step": 13728,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.6097472135207984,
            "MicroF1": 0.6097472135207984,
            "MacroF1": 0.5927729676671933,
            "Memory in Mb": 0.6686763763427734,
            "Time in s": 106.889718
          },
          {
            "step": 14784,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.6001488195900697,
            "MicroF1": 0.6001488195900697,
            "MacroF1": 0.5832911478837771,
            "Memory in Mb": 0.6683712005615234,
            "Time in s": 122.594966
          },
          {
            "step": 15840,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5673969316244712,
            "MicroF1": 0.5673969316244712,
            "MacroF1": 0.5522471754341497,
            "Memory in Mb": 0.8954944610595703,
            "Time in s": 139.36423
          },
          {
            "step": 16896,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5712340929269014,
            "MicroF1": 0.5712340929269014,
            "MacroF1": 0.559038323684958,
            "Memory in Mb": 1.4438505172729492,
            "Time in s": 157.284858
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5741184335134533,
            "MicroF1": 0.5741184335134533,
            "MacroF1": 0.5632919959429029,
            "Memory in Mb": 1.874833106994629,
            "Time in s": 176.64490099999998
          },
          {
            "step": 19008,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5867312042931552,
            "MicroF1": 0.5867312042931552,
            "MacroF1": 0.5723846445183199,
            "Memory in Mb": 0.4898128509521484,
            "Time in s": 197.018382
          },
          {
            "step": 20064,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5939789662562927,
            "MicroF1": 0.5939789662562927,
            "MacroF1": 0.5773993022741072,
            "Memory in Mb": 0.6687717437744141,
            "Time in s": 218.485499
          },
          {
            "step": 21120,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.595908897201572,
            "MicroF1": 0.595908897201572,
            "MacroF1": 0.5788762098776178,
            "Memory in Mb": 0.6688938140869141,
            "Time in s": 241.01085
          },
          {
            "step": 22176,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5977452085682075,
            "MicroF1": 0.5977452085682075,
            "MacroF1": 0.5801804614049403,
            "Memory in Mb": 1.2152299880981443,
            "Time in s": 264.603623
          },
          {
            "step": 23232,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5997158968619517,
            "MicroF1": 0.5997158968619517,
            "MacroF1": 0.5818597835760811,
            "Memory in Mb": 1.3294572830200195,
            "Time in s": 289.451742
          },
          {
            "step": 24288,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.6025033968789888,
            "MicroF1": 0.6025033968789888,
            "MacroF1": 0.5841484049015139,
            "Memory in Mb": 1.3295183181762695,
            "Time in s": 315.702926
          },
          {
            "step": 25344,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.6047823856686264,
            "MicroF1": 0.6047823856686264,
            "MacroF1": 0.5859943093850892,
            "Memory in Mb": 1.3296403884887695,
            "Time in s": 343.366363
          },
          {
            "step": 26400,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.6074472517898405,
            "MicroF1": 0.6074472517898405,
            "MacroF1": 0.5878557237787366,
            "Memory in Mb": 1.3296403884887695,
            "Time in s": 372.430483
          },
          {
            "step": 27456,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.6086323074121289,
            "MicroF1": 0.6086323074121289,
            "MacroF1": 0.5880340775890752,
            "Memory in Mb": 1.3298234939575195,
            "Time in s": 402.902886
          },
          {
            "step": 28512,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.6087124267826453,
            "MicroF1": 0.6087124267826453,
            "MacroF1": 0.5895354690395743,
            "Memory in Mb": 1.3298234939575195,
            "Time in s": 434.780231
          },
          {
            "step": 29568,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.6080765718537559,
            "MicroF1": 0.6080765718537559,
            "MacroF1": 0.5920130278134075,
            "Memory in Mb": 1.3298234939575195,
            "Time in s": 468.066846
          },
          {
            "step": 30624,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.6071253632890311,
            "MicroF1": 0.6071253632890311,
            "MacroF1": 0.5937369304389161,
            "Memory in Mb": 1.3293352127075195,
            "Time in s": 502.755803
          },
          {
            "step": 31680,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.6071845702200196,
            "MicroF1": 0.6071845702200196,
            "MacroF1": 0.5960066132315273,
            "Memory in Mb": 1.3295793533325195,
            "Time in s": 538.8589619999999
          },
          {
            "step": 32736,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.6079425691156255,
            "MicroF1": 0.6079425691156255,
            "MacroF1": 0.59836863034629,
            "Memory in Mb": 1.3296403884887695,
            "Time in s": 576.3667869999999
          },
          {
            "step": 33792,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.6027936432777958,
            "MicroF1": 0.6027936432777958,
            "MacroF1": 0.5936321389881086,
            "Memory in Mb": 0.6688251495361328,
            "Time in s": 615.5159199999999
          },
          {
            "step": 34848,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.6018882543690992,
            "MicroF1": 0.6018882543690992,
            "MacroF1": 0.5927698243358274,
            "Memory in Mb": 0.6689472198486328,
            "Time in s": 655.7217059999999
          },
          {
            "step": 35904,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.601398211848592,
            "MicroF1": 0.601398211848592,
            "MacroF1": 0.592182344393812,
            "Memory in Mb": 0.6690082550048828,
            "Time in s": 696.988077
          },
          {
            "step": 36960,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5999080061689981,
            "MicroF1": 0.5999080061689981,
            "MacroF1": 0.5906275041314122,
            "Memory in Mb": 0.6690082550048828,
            "Time in s": 739.3140189999999
          },
          {
            "step": 38016,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5996054189135868,
            "MicroF1": 0.5996054189135868,
            "MacroF1": 0.5899615119365567,
            "Memory in Mb": 0.6690082550048828,
            "Time in s": 782.70332
          },
          {
            "step": 39072,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5989608661155332,
            "MicroF1": 0.5989608661155332,
            "MacroF1": 0.5889868403975307,
            "Memory in Mb": 0.6687030792236328,
            "Time in s": 827.150323
          },
          {
            "step": 40128,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5947865526951928,
            "MicroF1": 0.5947865526951928,
            "MacroF1": 0.5855600636799734,
            "Memory in Mb": 0.6687030792236328,
            "Time in s": 872.66104
          },
          {
            "step": 41184,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5926717334822621,
            "MicroF1": 0.5926717334822621,
            "MacroF1": 0.5840930914391779,
            "Memory in Mb": 0.6688861846923828,
            "Time in s": 919.237173
          },
          {
            "step": 42240,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5913018774118706,
            "MicroF1": 0.5913018774118706,
            "MacroF1": 0.5832685369240246,
            "Memory in Mb": 0.6689472198486328,
            "Time in s": 966.874788
          },
          {
            "step": 43296,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5898833583554683,
            "MicroF1": 0.5898833583554683,
            "MacroF1": 0.5823904732646675,
            "Memory in Mb": 0.6690082550048828,
            "Time in s": 1015.577362
          },
          {
            "step": 44352,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5883745575071588,
            "MicroF1": 0.5883745575071588,
            "MacroF1": 0.5813207633940128,
            "Memory in Mb": 1.112539291381836,
            "Time in s": 1065.314826
          },
          {
            "step": 45408,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5863853590856035,
            "MicroF1": 0.5863853590856035,
            "MacroF1": 0.5797569747943008,
            "Memory in Mb": 1.3286066055297852,
            "Time in s": 1116.269805
          },
          {
            "step": 46464,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5850461657663086,
            "MicroF1": 0.5850461657663086,
            "MacroF1": 0.5780695197887614,
            "Memory in Mb": 1.3287897109985352,
            "Time in s": 1168.116792
          },
          {
            "step": 47520,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5867968602032871,
            "MicroF1": 0.5867968602032871,
            "MacroF1": 0.5799343284152632,
            "Memory in Mb": 1.328934669494629,
            "Time in s": 1221.156775
          },
          {
            "step": 48576,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5917035512094699,
            "MicroF1": 0.5917035512094699,
            "MacroF1": 0.5847625919047718,
            "Memory in Mb": 1.329483985900879,
            "Time in s": 1275.6038119999998
          },
          {
            "step": 49632,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.5968447139892405,
            "MicroF1": 0.5968447139892405,
            "MacroF1": 0.5895877351185161,
            "Memory in Mb": 1.329422950744629,
            "Time in s": 1331.456239
          },
          {
            "step": 50688,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.601673012804072,
            "MicroF1": 0.601673012804072,
            "MacroF1": 0.5939045014873635,
            "Memory in Mb": 1.329606056213379,
            "Time in s": 1388.7152379999998
          },
          {
            "step": 51744,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.6067487389598593,
            "MicroF1": 0.6067487389598593,
            "MacroF1": 0.5983547975185618,
            "Memory in Mb": 1.329606056213379,
            "Time in s": 1447.387746
          },
          {
            "step": 52800,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Insects",
            "Accuracy": 0.6119623477717381,
            "MicroF1": 0.6119623477717381,
            "MacroF1": 0.6029934068442723,
            "Memory in Mb": 0.147679328918457,
            "Time in s": 1507.071307
          },
          {
            "step": 408,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.9803439803439804,
            "MicroF1": 0.9803439803439804,
            "MacroF1": 0.4950372208436724,
            "Memory in Mb": 0.2342357635498047,
            "Time in s": 0.174096
          },
          {
            "step": 816,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.943558282208589,
            "MicroF1": 0.943558282208589,
            "MacroF1": 0.7669956277713079,
            "Memory in Mb": 0.3298597335815429,
            "Time in s": 0.617741
          },
          {
            "step": 1224,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.8863450531479967,
            "MicroF1": 0.8863450531479967,
            "MacroF1": 0.8786592421362931,
            "Memory in Mb": 0.4254837036132812,
            "Time in s": 1.425386
          },
          {
            "step": 1632,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.891477621091355,
            "MicroF1": 0.891477621091355,
            "MacroF1": 0.8818548670971932,
            "Memory in Mb": 0.5215349197387695,
            "Time in s": 2.789107
          },
          {
            "step": 2040,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.889651790093183,
            "MicroF1": 0.889651790093183,
            "MacroF1": 0.8812768038030504,
            "Memory in Mb": 0.6287555694580078,
            "Time in s": 4.864698000000001
          },
          {
            "step": 2448,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.8414384961176952,
            "MicroF1": 0.8414384961176952,
            "MacroF1": 0.8420581397672002,
            "Memory in Mb": 0.7242574691772461,
            "Time in s": 7.621751000000001
          },
          {
            "step": 2856,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.8500875656742557,
            "MicroF1": 0.8500875656742557,
            "MacroF1": 0.834558203718852,
            "Memory in Mb": 0.8199424743652344,
            "Time in s": 10.917147
          },
          {
            "step": 3264,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.8406374501992032,
            "MicroF1": 0.8406374501992032,
            "MacroF1": 0.8151418555553325,
            "Memory in Mb": 0.9155054092407228,
            "Time in s": 14.806837
          },
          {
            "step": 3672,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.8321983110868973,
            "MicroF1": 0.8321983110868973,
            "MacroF1": 0.8307198315203921,
            "Memory in Mb": 1.011190414428711,
            "Time in s": 19.353183
          },
          {
            "step": 4080,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.826182887962736,
            "MicroF1": 0.826182887962736,
            "MacroF1": 0.812376785603362,
            "Memory in Mb": 1.1319761276245115,
            "Time in s": 24.603589
          },
          {
            "step": 4488,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.809226654780477,
            "MicroF1": 0.809226654780477,
            "MacroF1": 0.8196273526663149,
            "Memory in Mb": 1.2275390625,
            "Time in s": 30.588099
          },
          {
            "step": 4896,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.8081716036772216,
            "MicroF1": 0.8081716036772216,
            "MacroF1": 0.815232111826365,
            "Memory in Mb": 1.3230409622192385,
            "Time in s": 37.350443
          },
          {
            "step": 5304,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.8057703186875353,
            "MicroF1": 0.8057703186875353,
            "MacroF1": 0.7903391475861199,
            "Memory in Mb": 1.4186649322509766,
            "Time in s": 44.935392
          },
          {
            "step": 5712,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7860269655051655,
            "MicroF1": 0.7860269655051656,
            "MacroF1": 0.7895763142947655,
            "Memory in Mb": 1.5144109725952148,
            "Time in s": 53.372574
          },
          {
            "step": 6120,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.784441902271613,
            "MicroF1": 0.784441902271613,
            "MacroF1": 0.7657785418705475,
            "Memory in Mb": 1.6098518371582031,
            "Time in s": 62.716148
          },
          {
            "step": 6528,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7585414432357898,
            "MicroF1": 0.7585414432357898,
            "MacroF1": 0.751418836389106,
            "Memory in Mb": 1.7056589126586914,
            "Time in s": 73.022548
          },
          {
            "step": 6936,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7473684210526316,
            "MicroF1": 0.7473684210526316,
            "MacroF1": 0.7484284412750403,
            "Memory in Mb": 1.8010997772216797,
            "Time in s": 84.351871
          },
          {
            "step": 7344,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7565027917744791,
            "MicroF1": 0.7565027917744791,
            "MacroF1": 0.7526701844923946,
            "Memory in Mb": 1.898371696472168,
            "Time in s": 96.7592
          },
          {
            "step": 7752,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7577086827506129,
            "MicroF1": 0.7577086827506129,
            "MacroF1": 0.7557350658705178,
            "Memory in Mb": 1.9939956665039065,
            "Time in s": 110.303168
          },
          {
            "step": 8160,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7617355068023042,
            "MicroF1": 0.7617355068023042,
            "MacroF1": 0.7576049653668415,
            "Memory in Mb": 2.0895586013793945,
            "Time in s": 125.047569
          },
          {
            "step": 8568,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7604762460604646,
            "MicroF1": 0.7604762460604646,
            "MacroF1": 0.7596175662696861,
            "Memory in Mb": 2.2332935333251958,
            "Time in s": 141.03972299999998
          },
          {
            "step": 8976,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.756991643454039,
            "MicroF1": 0.7569916434540391,
            "MacroF1": 0.7575313939177277,
            "Memory in Mb": 2.328978538513184,
            "Time in s": 158.34463899999997
          },
          {
            "step": 9384,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7558350207822658,
            "MicroF1": 0.7558350207822658,
            "MacroF1": 0.7548436696787698,
            "Memory in Mb": 2.424480438232422,
            "Time in s": 177.02417899999998
          },
          {
            "step": 9792,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.748340312531917,
            "MicroF1": 0.7483403125319169,
            "MacroF1": 0.7443908596260193,
            "Memory in Mb": 2.52004337310791,
            "Time in s": 197.139779
          },
          {
            "step": 10200,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7393862143347387,
            "MicroF1": 0.7393862143347387,
            "MacroF1": 0.7315892779928432,
            "Memory in Mb": 2.6156673431396484,
            "Time in s": 218.763511
          },
          {
            "step": 10608,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7196191194494201,
            "MicroF1": 0.7196191194494201,
            "MacroF1": 0.7089541376321257,
            "Memory in Mb": 2.7114133834838867,
            "Time in s": 241.930258
          },
          {
            "step": 11016,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7123921924648207,
            "MicroF1": 0.7123921924648208,
            "MacroF1": 0.7092068316988943,
            "Memory in Mb": 2.806976318359375,
            "Time in s": 266.699543
          },
          {
            "step": 11424,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7062943184802591,
            "MicroF1": 0.7062943184802591,
            "MacroF1": 0.694671323095531,
            "Memory in Mb": 2.9026002883911133,
            "Time in s": 293.16281
          },
          {
            "step": 11832,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6967289324655566,
            "MicroF1": 0.6967289324655566,
            "MacroF1": 0.6902328307983061,
            "Memory in Mb": 2.9981632232666016,
            "Time in s": 321.350926
          },
          {
            "step": 12240,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7007108423890841,
            "MicroF1": 0.7007108423890841,
            "MacroF1": 0.6983689907908355,
            "Memory in Mb": 3.09378719329834,
            "Time in s": 351.321335
          },
          {
            "step": 12648,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6969241717403337,
            "MicroF1": 0.6969241717403337,
            "MacroF1": 0.6892508246262707,
            "Memory in Mb": 3.189472198486328,
            "Time in s": 383.138799
          },
          {
            "step": 13056,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6836461126005362,
            "MicroF1": 0.6836461126005362,
            "MacroF1": 0.6755391962059191,
            "Memory in Mb": 3.2851572036743164,
            "Time in s": 416.860837
          },
          {
            "step": 13464,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6793433855752804,
            "MicroF1": 0.6793433855752804,
            "MacroF1": 0.6754035266161622,
            "Memory in Mb": 3.3807201385498047,
            "Time in s": 452.545925
          },
          {
            "step": 13872,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6769519140653161,
            "MicroF1": 0.6769519140653161,
            "MacroF1": 0.6742482232309566,
            "Memory in Mb": 3.476466178894043,
            "Time in s": 490.25119899999993
          },
          {
            "step": 14280,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6728762518383641,
            "MicroF1": 0.6728762518383641,
            "MacroF1": 0.6689356443053496,
            "Memory in Mb": 3.5720291137695312,
            "Time in s": 530.0355519999999
          },
          {
            "step": 14688,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6762442976782188,
            "MicroF1": 0.6762442976782188,
            "MacroF1": 0.6753292472514647,
            "Memory in Mb": 3.66759204864502,
            "Time in s": 571.9481539999999
          },
          {
            "step": 15096,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6830076184166942,
            "MicroF1": 0.6830076184166942,
            "MacroF1": 0.6822311287838643,
            "Memory in Mb": 3.763277053833008,
            "Time in s": 616.057788
          },
          {
            "step": 15504,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6818035218989873,
            "MicroF1": 0.6818035218989873,
            "MacroF1": 0.6788656596145115,
            "Memory in Mb": 3.858839988708496,
            "Time in s": 662.434182
          },
          {
            "step": 15912,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6816039218150964,
            "MicroF1": 0.6816039218150964,
            "MacroF1": 0.6801525397911032,
            "Memory in Mb": 0.2779512405395508,
            "Time in s": 710.461266
          },
          {
            "step": 16320,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6858263373981249,
            "MicroF1": 0.6858263373981249,
            "MacroF1": 0.6851912800185752,
            "Memory in Mb": 0.4695272445678711,
            "Time in s": 759.2533930000001
          },
          {
            "step": 16728,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6896634184253004,
            "MicroF1": 0.6896634184253004,
            "MacroF1": 0.6890226069872225,
            "Memory in Mb": 0.6609811782836914,
            "Time in s": 808.8937860000001
          },
          {
            "step": 17136,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6925007295010213,
            "MicroF1": 0.6925007295010213,
            "MacroF1": 0.691863544221197,
            "Memory in Mb": 0.9803314208984376,
            "Time in s": 859.476205
          },
          {
            "step": 17544,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.6990252522373597,
            "MicroF1": 0.6990252522373597,
            "MacroF1": 0.6986638608261282,
            "Memory in Mb": 0.2722988128662109,
            "Time in s": 910.506546
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7040833379756003,
            "MicroF1": 0.7040833379756003,
            "MacroF1": 0.7034973599095433,
            "Memory in Mb": 0.1428661346435547,
            "Time in s": 961.768777
          },
          {
            "step": 18360,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7102783376000872,
            "MicroF1": 0.7102783376000872,
            "MacroF1": 0.7096708693716106,
            "Memory in Mb": 0.2385511398315429,
            "Time in s": 1013.187005
          },
          {
            "step": 18768,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7155645548036447,
            "MicroF1": 0.7155645548036447,
            "MacroF1": 0.714820465744771,
            "Memory in Mb": 0.3341751098632812,
            "Time in s": 1064.826228
          },
          {
            "step": 19176,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7183833116036505,
            "MicroF1": 0.7183833116036505,
            "MacroF1": 0.7174783905571958,
            "Memory in Mb": 0.4296159744262695,
            "Time in s": 1116.742805
          },
          {
            "step": 19584,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7229229433692489,
            "MicroF1": 0.7229229433692489,
            "MacroF1": 0.7220221994049509,
            "Memory in Mb": 0.5253620147705078,
            "Time in s": 1168.993402
          },
          {
            "step": 19992,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7224751138012105,
            "MicroF1": 0.7224751138012104,
            "MacroF1": 0.7211832505275634,
            "Memory in Mb": 0.6323385238647461,
            "Time in s": 1221.636335
          },
          {
            "step": 20400,
            "track": "Multiclass classification",
            "model": "Hoeffding Adaptive Tree",
            "dataset": "Keystroke",
            "Accuracy": 0.7237119466640521,
            "MicroF1": 0.7237119466640521,
            "MacroF1": 0.7223930256436224,
            "Memory in Mb": 0.7279014587402344,
            "Time in s": 1274.7271140000005
          },
          {
            "step": 46,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.4222222222222222,
            "MicroF1": 0.4222222222222222,
            "MacroF1": 0.3590236094437775,
            "Memory in Mb": 0.9732446670532228,
            "Time in s": 0.50913
          },
          {
            "step": 92,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.5604395604395604,
            "MicroF1": 0.5604395604395604,
            "MacroF1": 0.5746538615446178,
            "Memory in Mb": 1.0627803802490234,
            "Time in s": 1.324173
          },
          {
            "step": 138,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.5766423357664233,
            "MicroF1": 0.5766423357664233,
            "MacroF1": 0.598257695340355,
            "Memory in Mb": 1.355058670043945,
            "Time in s": 2.399595
          },
          {
            "step": 184,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.6229508196721312,
            "MicroF1": 0.6229508196721312,
            "MacroF1": 0.6451744040758779,
            "Memory in Mb": 1.424909591674805,
            "Time in s": 3.72413
          },
          {
            "step": 230,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.6506550218340611,
            "MicroF1": 0.6506550218340611,
            "MacroF1": 0.668065528002595,
            "Memory in Mb": 1.5721073150634766,
            "Time in s": 5.289042
          },
          {
            "step": 276,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.6727272727272727,
            "MicroF1": 0.6727272727272727,
            "MacroF1": 0.6900672130049011,
            "Memory in Mb": 1.7710065841674805,
            "Time in s": 7.016464
          },
          {
            "step": 322,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7040498442367601,
            "MicroF1": 0.7040498442367601,
            "MacroF1": 0.7087861936875777,
            "Memory in Mb": 1.8489313125610352,
            "Time in s": 8.876771
          },
          {
            "step": 368,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7302452316076294,
            "MicroF1": 0.7302452316076294,
            "MacroF1": 0.7285991575377422,
            "Memory in Mb": 1.987476348876953,
            "Time in s": 10.883345
          },
          {
            "step": 414,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7457627118644068,
            "MicroF1": 0.7457627118644068,
            "MacroF1": 0.7430362907281778,
            "Memory in Mb": 2.008787155151367,
            "Time in s": 13.045158
          },
          {
            "step": 460,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7342047930283224,
            "MicroF1": 0.7342047930283224,
            "MacroF1": 0.7271744800226859,
            "Memory in Mb": 1.8246965408325195,
            "Time in s": 15.362221000000002
          },
          {
            "step": 506,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7405940594059406,
            "MicroF1": 0.7405940594059406,
            "MacroF1": 0.7304322149686578,
            "Memory in Mb": 1.7282800674438477,
            "Time in s": 17.823546
          },
          {
            "step": 552,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7368421052631579,
            "MicroF1": 0.7368421052631579,
            "MacroF1": 0.7267508109083203,
            "Memory in Mb": 1.5214414596557615,
            "Time in s": 20.437237
          },
          {
            "step": 598,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7403685092127303,
            "MicroF1": 0.7403685092127302,
            "MacroF1": 0.7318978254380312,
            "Memory in Mb": 1.6621322631835938,
            "Time in s": 23.204591
          },
          {
            "step": 644,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7325038880248833,
            "MicroF1": 0.7325038880248833,
            "MacroF1": 0.7248107612258206,
            "Memory in Mb": 1.7895660400390625,
            "Time in s": 26.125323
          },
          {
            "step": 690,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7242380261248186,
            "MicroF1": 0.7242380261248187,
            "MacroF1": 0.7153272190465999,
            "Memory in Mb": 1.929594039916992,
            "Time in s": 29.195315
          },
          {
            "step": 736,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7251700680272108,
            "MicroF1": 0.725170068027211,
            "MacroF1": 0.7148466398758337,
            "Memory in Mb": 2.079819679260254,
            "Time in s": 32.416287000000004
          },
          {
            "step": 782,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7259923175416133,
            "MicroF1": 0.7259923175416134,
            "MacroF1": 0.7134712280209222,
            "Memory in Mb": 2.0407657623291016,
            "Time in s": 35.785816000000004
          },
          {
            "step": 828,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.727932285368803,
            "MicroF1": 0.727932285368803,
            "MacroF1": 0.7177600265828429,
            "Memory in Mb": 2.245401382446289,
            "Time in s": 39.307391
          },
          {
            "step": 874,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7353951890034365,
            "MicroF1": 0.7353951890034366,
            "MacroF1": 0.7262567978322628,
            "Memory in Mb": 2.3208675384521484,
            "Time in s": 42.982338000000006
          },
          {
            "step": 920,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7431991294885746,
            "MicroF1": 0.7431991294885745,
            "MacroF1": 0.7345004589126253,
            "Memory in Mb": 2.463038444519043,
            "Time in s": 46.81357700000001
          },
          {
            "step": 966,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7471502590673575,
            "MicroF1": 0.7471502590673575,
            "MacroF1": 0.7368855656689401,
            "Memory in Mb": 2.4979677200317383,
            "Time in s": 50.81254500000001
          },
          {
            "step": 1012,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7546983184965381,
            "MicroF1": 0.754698318496538,
            "MacroF1": 0.7446216664767904,
            "Memory in Mb": 2.589772224426269,
            "Time in s": 54.96693400000001
          },
          {
            "step": 1058,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.760643330179754,
            "MicroF1": 0.760643330179754,
            "MacroF1": 0.7502594177262459,
            "Memory in Mb": 2.824686050415039,
            "Time in s": 59.28579400000001
          },
          {
            "step": 1104,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7624660018132366,
            "MicroF1": 0.7624660018132366,
            "MacroF1": 0.7523020427630668,
            "Memory in Mb": 2.512765884399414,
            "Time in s": 63.76907700000001
          },
          {
            "step": 1150,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7650130548302873,
            "MicroF1": 0.7650130548302874,
            "MacroF1": 0.7555087521342715,
            "Memory in Mb": 2.350802421569824,
            "Time in s": 68.40298100000001
          },
          {
            "step": 1196,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7690376569037657,
            "MicroF1": 0.7690376569037657,
            "MacroF1": 0.7603504370239863,
            "Memory in Mb": 2.0774078369140625,
            "Time in s": 73.17908000000001
          },
          {
            "step": 1242,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7719580983078163,
            "MicroF1": 0.7719580983078163,
            "MacroF1": 0.7638249032322542,
            "Memory in Mb": 2.143113136291504,
            "Time in s": 78.09633200000002
          },
          {
            "step": 1288,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7746697746697747,
            "MicroF1": 0.7746697746697747,
            "MacroF1": 0.7668828628349821,
            "Memory in Mb": 2.3053293228149414,
            "Time in s": 83.14236000000002
          },
          {
            "step": 1334,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7771942985746436,
            "MicroF1": 0.7771942985746436,
            "MacroF1": 0.7696789046658701,
            "Memory in Mb": 2.4279375076293945,
            "Time in s": 88.31606900000003
          },
          {
            "step": 1380,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7817258883248731,
            "MicroF1": 0.7817258883248731,
            "MacroF1": 0.7754511149783998,
            "Memory in Mb": 2.350360870361328,
            "Time in s": 93.61768400000004
          },
          {
            "step": 1426,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7866666666666666,
            "MicroF1": 0.7866666666666666,
            "MacroF1": 0.7797171864703156,
            "Memory in Mb": 2.461531639099121,
            "Time in s": 99.04413800000005
          },
          {
            "step": 1472,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7912984364377974,
            "MicroF1": 0.7912984364377974,
            "MacroF1": 0.7836430453045393,
            "Memory in Mb": 2.5941333770751958,
            "Time in s": 104.59649900000004
          },
          {
            "step": 1518,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7963085036255768,
            "MicroF1": 0.7963085036255768,
            "MacroF1": 0.7883976288226552,
            "Memory in Mb": 2.7080554962158203,
            "Time in s": 110.30036700000004
          },
          {
            "step": 1564,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7984644913627639,
            "MicroF1": 0.7984644913627639,
            "MacroF1": 0.7915512335737709,
            "Memory in Mb": 2.379396438598633,
            "Time in s": 116.13112600000004
          },
          {
            "step": 1610,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.798011187072716,
            "MicroF1": 0.7980111870727161,
            "MacroF1": 0.7913527809122488,
            "Memory in Mb": 2.557906150817871,
            "Time in s": 122.09210800000004
          },
          {
            "step": 1656,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7987915407854985,
            "MicroF1": 0.7987915407854985,
            "MacroF1": 0.7921693301011166,
            "Memory in Mb": 2.5870275497436523,
            "Time in s": 128.19249900000003
          },
          {
            "step": 1702,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.7995296884185773,
            "MicroF1": 0.7995296884185774,
            "MacroF1": 0.7947635312368726,
            "Memory in Mb": 2.441390991210937,
            "Time in s": 134.42240900000002
          },
          {
            "step": 1748,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.8019461934745278,
            "MicroF1": 0.8019461934745278,
            "MacroF1": 0.7968342396743014,
            "Memory in Mb": 2.619420051574707,
            "Time in s": 140.777643
          },
          {
            "step": 1794,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.8059118795315114,
            "MicroF1": 0.8059118795315114,
            "MacroF1": 0.8002313091513137,
            "Memory in Mb": 2.70393180847168,
            "Time in s": 147.25850100000002
          },
          {
            "step": 1840,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.8058727569331158,
            "MicroF1": 0.8058727569331158,
            "MacroF1": 0.8006185305294855,
            "Memory in Mb": 3.167543411254883,
            "Time in s": 153.88034500000003
          },
          {
            "step": 1886,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.8084880636604774,
            "MicroF1": 0.8084880636604774,
            "MacroF1": 0.8041348438460234,
            "Memory in Mb": 3.187774658203125,
            "Time in s": 160.64099900000002
          },
          {
            "step": 1932,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.8089073019161056,
            "MicroF1": 0.8089073019161055,
            "MacroF1": 0.8042053366874767,
            "Memory in Mb": 3.4328765869140625,
            "Time in s": 167.537774
          },
          {
            "step": 1978,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.8108244815376834,
            "MicroF1": 0.8108244815376834,
            "MacroF1": 0.8062422218151643,
            "Memory in Mb": 3.621993064880371,
            "Time in s": 174.56717600000002
          },
          {
            "step": 2024,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.8111715274345032,
            "MicroF1": 0.8111715274345032,
            "MacroF1": 0.805670935248126,
            "Memory in Mb": 3.783546447753906,
            "Time in s": 181.730034
          },
          {
            "step": 2070,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.8134364427259546,
            "MicroF1": 0.8134364427259546,
            "MacroF1": 0.8085538776813638,
            "Memory in Mb": 3.740958213806152,
            "Time in s": 189.029553
          },
          {
            "step": 2116,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.816548463356974,
            "MicroF1": 0.816548463356974,
            "MacroF1": 0.8113031614777911,
            "Memory in Mb": 3.760796546936035,
            "Time in s": 196.461518
          },
          {
            "step": 2162,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.8167515039333642,
            "MicroF1": 0.8167515039333642,
            "MacroF1": 0.8113905234748385,
            "Memory in Mb": 4.035200119018555,
            "Time in s": 204.024576
          },
          {
            "step": 2208,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.818305391934753,
            "MicroF1": 0.818305391934753,
            "MacroF1": 0.8126353495892602,
            "Memory in Mb": 4.192110061645508,
            "Time in s": 211.717055
          },
          {
            "step": 2254,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.8184642698624057,
            "MicroF1": 0.8184642698624057,
            "MacroF1": 0.8136291554244021,
            "Memory in Mb": 4.486760139465332,
            "Time in s": 219.553296
          },
          {
            "step": 2300,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "ImageSegments",
            "Accuracy": 0.8190517616354936,
            "MicroF1": 0.8190517616354936,
            "MacroF1": 0.8144252010220491,
            "Memory in Mb": 4.66081428527832,
            "Time in s": 227.540685
          },
          {
            "step": 1056,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.6701421800947868,
            "MicroF1": 0.6701421800947868,
            "MacroF1": 0.6068786932307204,
            "Memory in Mb": 6.831533432006836,
            "Time in s": 4.539822
          },
          {
            "step": 2112,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.6887730933207011,
            "MicroF1": 0.6887730933207011,
            "MacroF1": 0.6229217946585527,
            "Memory in Mb": 10.195775032043455,
            "Time in s": 12.638872
          },
          {
            "step": 3168,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.6962425007893905,
            "MicroF1": 0.6962425007893905,
            "MacroF1": 0.622910390568452,
            "Memory in Mb": 16.60274887084961,
            "Time in s": 24.371093
          },
          {
            "step": 4224,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7054226852948141,
            "MicroF1": 0.7054226852948141,
            "MacroF1": 0.6279874627708885,
            "Memory in Mb": 17.471903800964355,
            "Time in s": 39.890989
          },
          {
            "step": 5280,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7099829513165372,
            "MicroF1": 0.7099829513165372,
            "MacroF1": 0.6301031937879839,
            "Memory in Mb": 20.888835906982425,
            "Time in s": 58.938014
          },
          {
            "step": 6336,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7108129439621153,
            "MicroF1": 0.7108129439621153,
            "MacroF1": 0.6300557461749893,
            "Memory in Mb": 23.72772216796875,
            "Time in s": 81.51329799999999
          },
          {
            "step": 7392,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7126234609660398,
            "MicroF1": 0.7126234609660397,
            "MacroF1": 0.6287819651813062,
            "Memory in Mb": 28.070199966430664,
            "Time in s": 107.680368
          },
          {
            "step": 8448,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7168225405469397,
            "MicroF1": 0.7168225405469397,
            "MacroF1": 0.6299159911335922,
            "Memory in Mb": 31.613859176635746,
            "Time in s": 137.61638299999998
          },
          {
            "step": 9504,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7214563821950963,
            "MicroF1": 0.7214563821950963,
            "MacroF1": 0.6314635817104112,
            "Memory in Mb": 35.43206214904785,
            "Time in s": 171.294053
          },
          {
            "step": 10560,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7230798371057865,
            "MicroF1": 0.7230798371057865,
            "MacroF1": 0.6311670445333952,
            "Memory in Mb": 35.676584243774414,
            "Time in s": 208.900765
          },
          {
            "step": 11616,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7247524752475247,
            "MicroF1": 0.7247524752475247,
            "MacroF1": 0.6314302971551563,
            "Memory in Mb": 41.6592378616333,
            "Time in s": 250.157475
          },
          {
            "step": 12672,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7252781943019493,
            "MicroF1": 0.7252781943019494,
            "MacroF1": 0.6359647238599803,
            "Memory in Mb": 42.66780757904053,
            "Time in s": 295.204683
          },
          {
            "step": 13728,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7402928535004006,
            "MicroF1": 0.7402928535004006,
            "MacroF1": 0.7348419335996624,
            "Memory in Mb": 24.025733947753903,
            "Time in s": 342.696854
          },
          {
            "step": 14784,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7468714063451262,
            "MicroF1": 0.7468714063451262,
            "MacroF1": 0.7455387452701401,
            "Memory in Mb": 2.272738456726074,
            "Time in s": 392.697558
          },
          {
            "step": 15840,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7388092682618852,
            "MicroF1": 0.7388092682618853,
            "MacroF1": 0.7393651674564367,
            "Memory in Mb": 6.547223091125488,
            "Time in s": 445.901262
          },
          {
            "step": 16896,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7343000887836638,
            "MicroF1": 0.7343000887836638,
            "MacroF1": 0.7364396291092657,
            "Memory in Mb": 11.097336769104004,
            "Time in s": 502.213806
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7271461199933151,
            "MicroF1": 0.7271461199933151,
            "MacroF1": 0.7303078098029304,
            "Memory in Mb": 16.88492202758789,
            "Time in s": 561.702561
          },
          {
            "step": 19008,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7373073078339559,
            "MicroF1": 0.7373073078339558,
            "MacroF1": 0.7369507693389319,
            "Memory in Mb": 6.800461769104004,
            "Time in s": 623.6949609999999
          },
          {
            "step": 20064,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7412650152021133,
            "MicroF1": 0.7412650152021133,
            "MacroF1": 0.7370000710650216,
            "Memory in Mb": 3.369675636291504,
            "Time in s": 688.4094249999999
          },
          {
            "step": 21120,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7425540982054074,
            "MicroF1": 0.7425540982054074,
            "MacroF1": 0.7353012584659039,
            "Memory in Mb": 5.941231727600098,
            "Time in s": 756.7149029999999
          },
          {
            "step": 22176,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7435851183765502,
            "MicroF1": 0.7435851183765501,
            "MacroF1": 0.7334480812988377,
            "Memory in Mb": 8.846389770507812,
            "Time in s": 828.508796
          },
          {
            "step": 23232,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7452111402866859,
            "MicroF1": 0.7452111402866859,
            "MacroF1": 0.7324964055744654,
            "Memory in Mb": 9.471121788024902,
            "Time in s": 903.877525
          },
          {
            "step": 24288,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7463663688392967,
            "MicroF1": 0.7463663688392966,
            "MacroF1": 0.7310050929424414,
            "Memory in Mb": 12.406947135925291,
            "Time in s": 982.696529
          },
          {
            "step": 25344,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7474647831748412,
            "MicroF1": 0.7474647831748412,
            "MacroF1": 0.7298615493429103,
            "Memory in Mb": 15.979948043823242,
            "Time in s": 1064.886272
          },
          {
            "step": 26400,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7483616803666806,
            "MicroF1": 0.7483616803666806,
            "MacroF1": 0.7285096183890708,
            "Memory in Mb": 19.665884017944336,
            "Time in s": 1150.466447
          },
          {
            "step": 27456,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.749626661810235,
            "MicroF1": 0.749626661810235,
            "MacroF1": 0.7275235594970662,
            "Memory in Mb": 24.26569175720215,
            "Time in s": 1239.482455
          },
          {
            "step": 28512,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7465188874469503,
            "MicroF1": 0.7465188874469504,
            "MacroF1": 0.7258897093263847,
            "Memory in Mb": 8.914395332336426,
            "Time in s": 1332.2242660000002
          },
          {
            "step": 29568,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7451550715324518,
            "MicroF1": 0.7451550715324519,
            "MacroF1": 0.7292330017207805,
            "Memory in Mb": 8.459691047668457,
            "Time in s": 1428.2455670000002
          },
          {
            "step": 30624,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7443751428664729,
            "MicroF1": 0.7443751428664729,
            "MacroF1": 0.7327893612754602,
            "Memory in Mb": 12.943696022033691,
            "Time in s": 1527.20672
          },
          {
            "step": 31680,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7437103443921841,
            "MicroF1": 0.7437103443921841,
            "MacroF1": 0.7357305076230832,
            "Memory in Mb": 18.80640697479248,
            "Time in s": 1629.363326
          },
          {
            "step": 32736,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7432717275087827,
            "MicroF1": 0.7432717275087827,
            "MacroF1": 0.7381285892142362,
            "Memory in Mb": 16.162379264831543,
            "Time in s": 1734.350472
          },
          {
            "step": 33792,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7377408185611554,
            "MicroF1": 0.7377408185611554,
            "MacroF1": 0.7340057348640155,
            "Memory in Mb": 11.18346881866455,
            "Time in s": 1842.731877
          },
          {
            "step": 34848,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7340373633311332,
            "MicroF1": 0.7340373633311332,
            "MacroF1": 0.7302084976112027,
            "Memory in Mb": 5.613262176513672,
            "Time in s": 1955.023471
          },
          {
            "step": 35904,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7312759379439044,
            "MicroF1": 0.7312759379439044,
            "MacroF1": 0.7271196230245338,
            "Memory in Mb": 9.756120681762695,
            "Time in s": 2071.1719540000004
          },
          {
            "step": 36960,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7278335452799047,
            "MicroF1": 0.7278335452799047,
            "MacroF1": 0.7234434079919367,
            "Memory in Mb": 11.990450859069824,
            "Time in s": 2191.1586210000005
          },
          {
            "step": 38016,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7254241746678942,
            "MicroF1": 0.7254241746678942,
            "MacroF1": 0.7207605796154644,
            "Memory in Mb": 14.404197692871094,
            "Time in s": 2314.6725880000004
          },
          {
            "step": 39072,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7250390315067441,
            "MicroF1": 0.7250390315067441,
            "MacroF1": 0.7205508934526729,
            "Memory in Mb": 7.651473045349121,
            "Time in s": 2441.666332
          },
          {
            "step": 40128,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7236524036185112,
            "MicroF1": 0.7236524036185111,
            "MacroF1": 0.7196200887167502,
            "Memory in Mb": 7.583705902099609,
            "Time in s": 2572.286548
          },
          {
            "step": 41184,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7235995435009591,
            "MicroF1": 0.7235995435009591,
            "MacroF1": 0.7199895911465058,
            "Memory in Mb": 12.209360122680664,
            "Time in s": 2706.295895
          },
          {
            "step": 42240,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7235966760576719,
            "MicroF1": 0.7235966760576719,
            "MacroF1": 0.7203672841246517,
            "Memory in Mb": 15.002169609069824,
            "Time in s": 2843.875227
          },
          {
            "step": 43296,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7241713823767179,
            "MicroF1": 0.7241713823767179,
            "MacroF1": 0.7213145862540888,
            "Memory in Mb": 17.433518409729004,
            "Time in s": 2985.305163
          },
          {
            "step": 44352,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7245608892696895,
            "MicroF1": 0.7245608892696895,
            "MacroF1": 0.7219384327675483,
            "Memory in Mb": 20.337363243103027,
            "Time in s": 3130.446815
          },
          {
            "step": 45408,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7253947629220164,
            "MicroF1": 0.7253947629220163,
            "MacroF1": 0.7227741676779873,
            "Memory in Mb": 20.507991790771484,
            "Time in s": 3279.062075
          },
          {
            "step": 46464,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7263198674213891,
            "MicroF1": 0.7263198674213891,
            "MacroF1": 0.7236028172229397,
            "Memory in Mb": 24.947001457214355,
            "Time in s": 3431.085655
          },
          {
            "step": 47520,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7259622466802753,
            "MicroF1": 0.7259622466802753,
            "MacroF1": 0.7234132526915972,
            "Memory in Mb": 9.389252662658691,
            "Time in s": 3586.979459
          },
          {
            "step": 48576,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7297581060216161,
            "MicroF1": 0.7297581060216161,
            "MacroF1": 0.7273884829439242,
            "Memory in Mb": 9.12541389465332,
            "Time in s": 3745.95156
          },
          {
            "step": 49632,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7336543692450284,
            "MicroF1": 0.7336543692450284,
            "MacroF1": 0.7312645046388119,
            "Memory in Mb": 8.804935455322266,
            "Time in s": 3907.504223
          },
          {
            "step": 50688,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7372501824925524,
            "MicroF1": 0.7372501824925524,
            "MacroF1": 0.7346466630802606,
            "Memory in Mb": 12.506796836853027,
            "Time in s": 4071.296369
          },
          {
            "step": 51744,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.741182382157973,
            "MicroF1": 0.741182382157973,
            "MacroF1": 0.7382896911640772,
            "Memory in Mb": 15.31224250793457,
            "Time in s": 4237.296913
          },
          {
            "step": 52800,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Insects",
            "Accuracy": 0.7442565200098487,
            "MicroF1": 0.7442565200098487,
            "MacroF1": 0.7419321396565435,
            "Memory in Mb": 0.3696470260620117,
            "Time in s": 4404.707415
          },
          {
            "step": 408,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9803439803439804,
            "MicroF1": 0.9803439803439804,
            "MacroF1": 0.4950372208436724,
            "Memory in Mb": 0.3514842987060547,
            "Time in s": 0.595241
          },
          {
            "step": 816,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9730061349693252,
            "MicroF1": 0.9730061349693252,
            "MacroF1": 0.7867307803099512,
            "Memory in Mb": 1.312638282775879,
            "Time in s": 2.098583
          },
          {
            "step": 1224,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9705641864268192,
            "MicroF1": 0.9705641864268192,
            "MacroF1": 0.93705029195588,
            "Memory in Mb": 2.2586374282836914,
            "Time in s": 4.31371
          },
          {
            "step": 1632,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9711833231146536,
            "MicroF1": 0.9711833231146536,
            "MacroF1": 0.9377953913100076,
            "Memory in Mb": 3.394951820373535,
            "Time in s": 7.338651
          },
          {
            "step": 2040,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.969592937714566,
            "MicroF1": 0.969592937714566,
            "MacroF1": 0.9445939973353388,
            "Memory in Mb": 5.254854202270508,
            "Time in s": 11.230817
          },
          {
            "step": 2448,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.970167552104618,
            "MicroF1": 0.970167552104618,
            "MacroF1": 0.9654865811906564,
            "Memory in Mb": 2.048126220703125,
            "Time in s": 15.860121
          },
          {
            "step": 2856,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9726795096322242,
            "MicroF1": 0.9726795096322242,
            "MacroF1": 0.9705770446236132,
            "Memory in Mb": 2.732625961303711,
            "Time in s": 21.099127
          },
          {
            "step": 3264,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.971805087342936,
            "MicroF1": 0.971805087342936,
            "MacroF1": 0.9627836140542232,
            "Memory in Mb": 2.790935516357422,
            "Time in s": 26.971176
          },
          {
            "step": 3672,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9733042767638246,
            "MicroF1": 0.9733042767638246,
            "MacroF1": 0.9719148371902758,
            "Memory in Mb": 2.987569808959961,
            "Time in s": 33.477629
          },
          {
            "step": 4080,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9698455503799952,
            "MicroF1": 0.9698455503799952,
            "MacroF1": 0.958802050565698,
            "Memory in Mb": 4.571287155151367,
            "Time in s": 40.764845
          },
          {
            "step": 4488,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9710274125250724,
            "MicroF1": 0.9710274125250724,
            "MacroF1": 0.970190142555116,
            "Memory in Mb": 1.7459039688110352,
            "Time in s": 48.697823
          },
          {
            "step": 4896,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9722165474974463,
            "MicroF1": 0.9722165474974463,
            "MacroF1": 0.971936417428158,
            "Memory in Mb": 2.7892093658447266,
            "Time in s": 57.250967
          },
          {
            "step": 5304,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9720912690929664,
            "MicroF1": 0.9720912690929664,
            "MacroF1": 0.970282662152698,
            "Memory in Mb": 2.895453453063965,
            "Time in s": 66.54505900000001
          },
          {
            "step": 5712,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9723340921029592,
            "MicroF1": 0.9723340921029592,
            "MacroF1": 0.9718828908328702,
            "Memory in Mb": 4.064221382141113,
            "Time in s": 76.51574600000001
          },
          {
            "step": 6120,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9718908318352673,
            "MicroF1": 0.9718908318352673,
            "MacroF1": 0.9703726237787478,
            "Memory in Mb": 5.130434989929199,
            "Time in s": 87.20768000000001
          },
          {
            "step": 6528,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.97196261682243,
            "MicroF1": 0.97196261682243,
            "MacroF1": 0.9714458378209956,
            "Memory in Mb": 2.455193519592285,
            "Time in s": 98.596053
          },
          {
            "step": 6936,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9733237202595528,
            "MicroF1": 0.9733237202595528,
            "MacroF1": 0.9740372626056704,
            "Memory in Mb": 2.4587574005126958,
            "Time in s": 110.618692
          },
          {
            "step": 7344,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9735802805392892,
            "MicroF1": 0.9735802805392892,
            "MacroF1": 0.973376514333954,
            "Memory in Mb": 3.893580436706543,
            "Time in s": 123.504669
          },
          {
            "step": 7752,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9729067217133271,
            "MicroF1": 0.9729067217133271,
            "MacroF1": 0.972110994169212,
            "Memory in Mb": 4.61766529083252,
            "Time in s": 137.14807100000002
          },
          {
            "step": 8160,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.97217796298566,
            "MicroF1": 0.97217796298566,
            "MacroF1": 0.9713389113158796,
            "Memory in Mb": 5.2350358963012695,
            "Time in s": 151.61678500000002
          },
          {
            "step": 8568,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9725691607330454,
            "MicroF1": 0.9725691607330454,
            "MacroF1": 0.9726516232305996,
            "Memory in Mb": 3.659168243408203,
            "Time in s": 166.92608600000003
          },
          {
            "step": 8976,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9733704735376044,
            "MicroF1": 0.9733704735376044,
            "MacroF1": 0.973745927183376,
            "Memory in Mb": 5.072476387023926,
            "Time in s": 182.940013
          },
          {
            "step": 9384,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9733560694873707,
            "MicroF1": 0.9733560694873707,
            "MacroF1": 0.9732604538569352,
            "Memory in Mb": 5.722126007080078,
            "Time in s": 199.767059
          },
          {
            "step": 9792,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9731385966704116,
            "MicroF1": 0.9731385966704116,
            "MacroF1": 0.9729642609350584,
            "Memory in Mb": 4.404660224914551,
            "Time in s": 217.43706000000003
          },
          {
            "step": 10200,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9725463280713796,
            "MicroF1": 0.9725463280713796,
            "MacroF1": 0.9722080895483168,
            "Memory in Mb": 2.652709007263184,
            "Time in s": 235.89310200000003
          },
          {
            "step": 10608,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9726595644385784,
            "MicroF1": 0.9726595644385784,
            "MacroF1": 0.972708084817296,
            "Memory in Mb": 1.1834087371826172,
            "Time in s": 254.958035
          },
          {
            "step": 11016,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9729459827507944,
            "MicroF1": 0.9729459827507944,
            "MacroF1": 0.973083018444042,
            "Memory in Mb": 1.520833969116211,
            "Time in s": 274.61158
          },
          {
            "step": 11424,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9724240567276548,
            "MicroF1": 0.9724240567276548,
            "MacroF1": 0.972236101273467,
            "Memory in Mb": 2.823396682739258,
            "Time in s": 294.969552
          },
          {
            "step": 11832,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9720226523539852,
            "MicroF1": 0.9720226523539852,
            "MacroF1": 0.9719096687987197,
            "Memory in Mb": 2.493410110473633,
            "Time in s": 316.08929700000004
          },
          {
            "step": 12240,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9726284827191763,
            "MicroF1": 0.9726284827191763,
            "MacroF1": 0.9728780734732722,
            "Memory in Mb": 2.3478269577026367,
            "Time in s": 337.99406600000003
          },
          {
            "step": 12648,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9727998734877836,
            "MicroF1": 0.9727998734877836,
            "MacroF1": 0.9729097588140672,
            "Memory in Mb": 2.5516576766967773,
            "Time in s": 360.586004
          },
          {
            "step": 13056,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9726541554959786,
            "MicroF1": 0.9726541554959786,
            "MacroF1": 0.9726709194030316,
            "Memory in Mb": 3.304943084716797,
            "Time in s": 383.87924
          },
          {
            "step": 13464,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9722201589541708,
            "MicroF1": 0.9722201589541708,
            "MacroF1": 0.9721650267620996,
            "Memory in Mb": 4.003572463989258,
            "Time in s": 407.932153
          },
          {
            "step": 13872,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.97246052916156,
            "MicroF1": 0.97246052916156,
            "MacroF1": 0.972591005704606,
            "Memory in Mb": 4.270735740661621,
            "Time in s": 432.78626
          },
          {
            "step": 14280,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9713565375726592,
            "MicroF1": 0.9713565375726592,
            "MacroF1": 0.9711654862365112,
            "Memory in Mb": 4.102839469909668,
            "Time in s": 458.452707
          },
          {
            "step": 14688,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9718118063593654,
            "MicroF1": 0.9718118063593654,
            "MacroF1": 0.9719655808676524,
            "Memory in Mb": 3.971695899963379,
            "Time in s": 484.888355
          },
          {
            "step": 15096,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9724412056972508,
            "MicroF1": 0.9724412056972508,
            "MacroF1": 0.9726138064055022,
            "Memory in Mb": 4.612870216369629,
            "Time in s": 512.116119
          },
          {
            "step": 15504,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.972327936528414,
            "MicroF1": 0.972327936528414,
            "MacroF1": 0.9723669009986284,
            "Memory in Mb": 3.2941598892211914,
            "Time in s": 540.133825
          },
          {
            "step": 15912,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.97240902520269,
            "MicroF1": 0.97240902520269,
            "MacroF1": 0.9724748506273226,
            "Memory in Mb": 5.215278625488281,
            "Time in s": 569.064511
          },
          {
            "step": 16320,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9718119982842086,
            "MicroF1": 0.9718119982842086,
            "MacroF1": 0.9717822259045504,
            "Memory in Mb": 2.705050468444824,
            "Time in s": 598.831932
          },
          {
            "step": 16728,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9713636635379924,
            "MicroF1": 0.9713636635379924,
            "MacroF1": 0.971358198091739,
            "Memory in Mb": 1.4999914169311523,
            "Time in s": 629.214947
          },
          {
            "step": 17136,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9716953603735046,
            "MicroF1": 0.9716953603735046,
            "MacroF1": 0.9717778191727772,
            "Memory in Mb": 1.5952835083007812,
            "Time in s": 660.25813
          },
          {
            "step": 17544,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9716696118109788,
            "MicroF1": 0.9716696118109788,
            "MacroF1": 0.971712982907841,
            "Memory in Mb": 2.6761178970336914,
            "Time in s": 692.002463
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9709765472675616,
            "MicroF1": 0.9709765472675616,
            "MacroF1": 0.970966525204854,
            "Memory in Mb": 3.671113014221192,
            "Time in s": 724.550435
          },
          {
            "step": 18360,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9709679176425732,
            "MicroF1": 0.9709679176425732,
            "MacroF1": 0.9710033330464194,
            "Memory in Mb": 4.91295337677002,
            "Time in s": 757.867405
          },
          {
            "step": 18768,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.971012948260244,
            "MicroF1": 0.971012948260244,
            "MacroF1": 0.9710485326344032,
            "Memory in Mb": 5.05178165435791,
            "Time in s": 792.058633
          },
          {
            "step": 19176,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.97116036505867,
            "MicroF1": 0.97116036505867,
            "MacroF1": 0.9711938240802872,
            "Memory in Mb": 5.466279983520508,
            "Time in s": 827.148813
          },
          {
            "step": 19584,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9707909921871012,
            "MicroF1": 0.9707909921871012,
            "MacroF1": 0.9708057459916865,
            "Memory in Mb": 5.881702423095703,
            "Time in s": 863.142165
          },
          {
            "step": 19992,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9705867640438196,
            "MicroF1": 0.9705867640438196,
            "MacroF1": 0.9706070593086332,
            "Memory in Mb": 5.831451416015625,
            "Time in s": 900.075184
          },
          {
            "step": 20400,
            "track": "Multiclass classification",
            "model": "Adaptive Random Forest",
            "dataset": "Keystroke",
            "Accuracy": 0.9698514633070248,
            "MicroF1": 0.9698514633070248,
            "MacroF1": 0.9698673821244655,
            "Memory in Mb": 2.3371658325195312,
            "Time in s": 937.846308
          },
          {
            "step": 46,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.3111111111111111,
            "MicroF1": 0.3111111111111111,
            "MacroF1": 0.2220238095238095,
            "Memory in Mb": 2.606511116027832,
            "Time in s": 1.431937
          },
          {
            "step": 92,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.4945054945054945,
            "MicroF1": 0.4945054945054945,
            "MacroF1": 0.5053729602697932,
            "Memory in Mb": 2.609585762023926,
            "Time in s": 3.886705
          },
          {
            "step": 138,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.5109489051094891,
            "MicroF1": 0.5109489051094891,
            "MacroF1": 0.5310665055578762,
            "Memory in Mb": 2.6113672256469727,
            "Time in s": 7.103774
          },
          {
            "step": 184,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.5737704918032787,
            "MicroF1": 0.5737704918032787,
            "MacroF1": 0.5886643910747036,
            "Memory in Mb": 2.6136903762817383,
            "Time in s": 11.095617
          },
          {
            "step": 230,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.6026200873362445,
            "MicroF1": 0.6026200873362445,
            "MacroF1": 0.6106719627755607,
            "Memory in Mb": 2.614529609680176,
            "Time in s": 15.833752
          },
          {
            "step": 276,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.6181818181818182,
            "MicroF1": 0.6181818181818182,
            "MacroF1": 0.6264208209498925,
            "Memory in Mb": 2.6147661209106445,
            "Time in s": 21.302563
          },
          {
            "step": 322,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.6448598130841121,
            "MicroF1": 0.6448598130841121,
            "MacroF1": 0.6378728366046057,
            "Memory in Mb": 2.616147041320801,
            "Time in s": 27.471138
          },
          {
            "step": 368,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.667574931880109,
            "MicroF1": 0.667574931880109,
            "MacroF1": 0.6581306320431076,
            "Memory in Mb": 2.6166696548461914,
            "Time in s": 34.32642
          },
          {
            "step": 414,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.6803874092009685,
            "MicroF1": 0.6803874092009685,
            "MacroF1": 0.6704325632692101,
            "Memory in Mb": 2.6175050735473637,
            "Time in s": 41.86551
          },
          {
            "step": 460,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.6884531590413944,
            "MicroF1": 0.6884531590413944,
            "MacroF1": 0.6760149332924277,
            "Memory in Mb": 2.617680549621582,
            "Time in s": 50.055222
          },
          {
            "step": 506,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.691089108910891,
            "MicroF1": 0.691089108910891,
            "MacroF1": 0.6769247074861785,
            "Memory in Mb": 2.617680549621582,
            "Time in s": 58.91147
          },
          {
            "step": 552,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.691470054446461,
            "MicroF1": 0.691470054446461,
            "MacroF1": 0.6803521213965826,
            "Memory in Mb": 2.6178178787231445,
            "Time in s": 68.422832
          },
          {
            "step": 598,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.6968174204355109,
            "MicroF1": 0.6968174204355109,
            "MacroF1": 0.6854975219125513,
            "Memory in Mb": 2.617863655090332,
            "Time in s": 78.595742
          },
          {
            "step": 644,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.6936236391912908,
            "MicroF1": 0.6936236391912908,
            "MacroF1": 0.6835764097697864,
            "Memory in Mb": 2.6192026138305664,
            "Time in s": 89.423453
          },
          {
            "step": 690,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.6966618287373004,
            "MicroF1": 0.6966618287373004,
            "MacroF1": 0.6871604229696352,
            "Memory in Mb": 2.6194162368774414,
            "Time in s": 100.907013
          },
          {
            "step": 736,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.6965986394557823,
            "MicroF1": 0.6965986394557823,
            "MacroF1": 0.6884795420777536,
            "Memory in Mb": 2.6194887161254883,
            "Time in s": 113.056901
          },
          {
            "step": 782,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7016645326504481,
            "MicroF1": 0.7016645326504481,
            "MacroF1": 0.6927955715819348,
            "Memory in Mb": 2.6197519302368164,
            "Time in s": 125.871728
          },
          {
            "step": 828,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7037484885126964,
            "MicroF1": 0.7037484885126964,
            "MacroF1": 0.6971811816445675,
            "Memory in Mb": 2.61989688873291,
            "Time in s": 139.34805
          },
          {
            "step": 874,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7124856815578465,
            "MicroF1": 0.7124856815578465,
            "MacroF1": 0.7027179013602759,
            "Memory in Mb": 2.61989688873291,
            "Time in s": 153.488556
          },
          {
            "step": 920,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7127312295973884,
            "MicroF1": 0.7127312295973884,
            "MacroF1": 0.7019247882761857,
            "Memory in Mb": 2.61989688873291,
            "Time in s": 168.286881
          },
          {
            "step": 966,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7119170984455958,
            "MicroF1": 0.7119170984455958,
            "MacroF1": 0.7013991197312313,
            "Memory in Mb": 2.61989688873291,
            "Time in s": 183.731934
          },
          {
            "step": 1012,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7111770524233432,
            "MicroF1": 0.7111770524233432,
            "MacroF1": 0.7000689942734505,
            "Memory in Mb": 2.61989688873291,
            "Time in s": 199.829198
          },
          {
            "step": 1058,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7123935666982024,
            "MicroF1": 0.7123935666982024,
            "MacroF1": 0.700757485135609,
            "Memory in Mb": 2.620041847229004,
            "Time in s": 216.594079
          },
          {
            "step": 1104,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7116953762466002,
            "MicroF1": 0.7116953762466002,
            "MacroF1": 0.6997536275311635,
            "Memory in Mb": 2.6201601028442383,
            "Time in s": 234.008444
          },
          {
            "step": 1150,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7136640557006092,
            "MicroF1": 0.7136640557006092,
            "MacroF1": 0.7002507718266925,
            "Memory in Mb": 2.6201601028442383,
            "Time in s": 252.079326
          },
          {
            "step": 1196,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7154811715481172,
            "MicroF1": 0.7154811715481171,
            "MacroF1": 0.7029614354817431,
            "Memory in Mb": 2.530026435852051,
            "Time in s": 270.790044
          },
          {
            "step": 1242,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.717163577759871,
            "MicroF1": 0.717163577759871,
            "MacroF1": 0.7059650228666394,
            "Memory in Mb": 2.753697395324707,
            "Time in s": 290.035925
          },
          {
            "step": 1288,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7187257187257188,
            "MicroF1": 0.7187257187257188,
            "MacroF1": 0.706699668165461,
            "Memory in Mb": 3.664814949035645,
            "Time in s": 309.61323300000004
          },
          {
            "step": 1334,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.719429857464366,
            "MicroF1": 0.719429857464366,
            "MacroF1": 0.7094425115390415,
            "Memory in Mb": 4.463783264160156,
            "Time in s": 329.52005700000007
          },
          {
            "step": 1380,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7251631617113851,
            "MicroF1": 0.725163161711385,
            "MacroF1": 0.7174387625572534,
            "Memory in Mb": 4.938790321350098,
            "Time in s": 349.76822000000004
          },
          {
            "step": 1426,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7319298245614035,
            "MicroF1": 0.7319298245614035,
            "MacroF1": 0.7244482628352659,
            "Memory in Mb": 5.045901298522949,
            "Time in s": 370.340987
          },
          {
            "step": 1472,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7335146159075459,
            "MicroF1": 0.7335146159075459,
            "MacroF1": 0.7247675805597543,
            "Memory in Mb": 5.884430885314941,
            "Time in s": 391.25731
          },
          {
            "step": 1518,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7251153592617007,
            "MicroF1": 0.7251153592617007,
            "MacroF1": 0.7184902268106362,
            "Memory in Mb": 6.2875261306762695,
            "Time in s": 412.53433
          },
          {
            "step": 1564,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7204094689699296,
            "MicroF1": 0.7204094689699295,
            "MacroF1": 0.7171509654034274,
            "Memory in Mb": 6.316588401794434,
            "Time in s": 434.180383
          },
          {
            "step": 1610,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7165941578620261,
            "MicroF1": 0.7165941578620262,
            "MacroF1": 0.7136076251491865,
            "Memory in Mb": 6.364602088928223,
            "Time in s": 456.188448
          },
          {
            "step": 1656,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7202416918429003,
            "MicroF1": 0.7202416918429003,
            "MacroF1": 0.7179265770125135,
            "Memory in Mb": 6.460197448730469,
            "Time in s": 478.554693
          },
          {
            "step": 1702,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.721928277483833,
            "MicroF1": 0.7219282774838331,
            "MacroF1": 0.7220156076184944,
            "Memory in Mb": 6.666633605957031,
            "Time in s": 501.28713
          },
          {
            "step": 1748,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7263880938752146,
            "MicroF1": 0.7263880938752146,
            "MacroF1": 0.7263874723147012,
            "Memory in Mb": 6.882956504821777,
            "Time in s": 524.384863
          },
          {
            "step": 1794,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7328499721137758,
            "MicroF1": 0.7328499721137758,
            "MacroF1": 0.7320714565315939,
            "Memory in Mb": 6.874361991882324,
            "Time in s": 547.829739
          },
          {
            "step": 1840,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.734094616639478,
            "MicroF1": 0.734094616639478,
            "MacroF1": 0.7334477172925166,
            "Memory in Mb": 7.857270240783691,
            "Time in s": 571.634536
          },
          {
            "step": 1886,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7358090185676393,
            "MicroF1": 0.7358090185676393,
            "MacroF1": 0.736235296466255,
            "Memory in Mb": 8.041683197021484,
            "Time in s": 595.832215
          },
          {
            "step": 1932,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7369238736406007,
            "MicroF1": 0.7369238736406007,
            "MacroF1": 0.7364098924240724,
            "Memory in Mb": 8.212060928344727,
            "Time in s": 620.406916
          },
          {
            "step": 1978,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7369752149721801,
            "MicroF1": 0.73697521497218,
            "MacroF1": 0.7356260672719533,
            "Memory in Mb": 8.416284561157227,
            "Time in s": 645.365163
          },
          {
            "step": 2024,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7409787444389521,
            "MicroF1": 0.7409787444389521,
            "MacroF1": 0.7385453010661254,
            "Memory in Mb": 8.869349479675293,
            "Time in s": 670.737914
          },
          {
            "step": 2070,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7438376027066216,
            "MicroF1": 0.7438376027066217,
            "MacroF1": 0.7418803204845174,
            "Memory in Mb": 9.001053810119629,
            "Time in s": 696.540108
          },
          {
            "step": 2116,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7475177304964539,
            "MicroF1": 0.7475177304964539,
            "MacroF1": 0.7450940881618369,
            "Memory in Mb": 9.427652359008787,
            "Time in s": 722.759269
          },
          {
            "step": 2162,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7482646922720962,
            "MicroF1": 0.7482646922720962,
            "MacroF1": 0.7457425826498583,
            "Memory in Mb": 9.724228858947754,
            "Time in s": 749.427824
          },
          {
            "step": 2208,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7521522428636158,
            "MicroF1": 0.7521522428636158,
            "MacroF1": 0.7492034954191574,
            "Memory in Mb": 9.71615219116211,
            "Time in s": 776.532359
          },
          {
            "step": 2254,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7532179316466933,
            "MicroF1": 0.7532179316466933,
            "MacroF1": 0.7508205496072249,
            "Memory in Mb": 10.198495864868164,
            "Time in s": 804.090452
          },
          {
            "step": 2300,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "ImageSegments",
            "Accuracy": 0.7546759460635059,
            "MicroF1": 0.754675946063506,
            "MacroF1": 0.7527273841922961,
            "Memory in Mb": 10.425667762756348,
            "Time in s": 832.069921
          },
          {
            "step": 1056,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.6265402843601896,
            "MicroF1": 0.6265402843601896,
            "MacroF1": 0.5882776540607534,
            "Memory in Mb": 10.90817928314209,
            "Time in s": 22.534162
          },
          {
            "step": 2112,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.6570345807674088,
            "MicroF1": 0.6570345807674088,
            "MacroF1": 0.61544126739188,
            "Memory in Mb": 21.709880828857425,
            "Time in s": 65.027277
          },
          {
            "step": 3168,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.6684559520050521,
            "MicroF1": 0.6684559520050521,
            "MacroF1": 0.6242294974630811,
            "Memory in Mb": 28.635205268859863,
            "Time in s": 131.537407
          },
          {
            "step": 4224,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.6810324413923751,
            "MicroF1": 0.6810324413923751,
            "MacroF1": 0.6325456686453049,
            "Memory in Mb": 36.43542194366455,
            "Time in s": 221.773783
          },
          {
            "step": 5280,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.6910399696912294,
            "MicroF1": 0.6910399696912294,
            "MacroF1": 0.6411255615252124,
            "Memory in Mb": 45.614484786987305,
            "Time in s": 335.089181
          },
          {
            "step": 6336,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.6937647987371744,
            "MicroF1": 0.6937647987371744,
            "MacroF1": 0.6440375279924044,
            "Memory in Mb": 53.59738254547119,
            "Time in s": 471.883196
          },
          {
            "step": 7392,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.6988228927073468,
            "MicroF1": 0.6988228927073468,
            "MacroF1": 0.6494865599203364,
            "Memory in Mb": 66.1818675994873,
            "Time in s": 633.159586
          },
          {
            "step": 8448,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7001302237480762,
            "MicroF1": 0.7001302237480762,
            "MacroF1": 0.6494906800979877,
            "Memory in Mb": 76.50763607025146,
            "Time in s": 819.592493
          },
          {
            "step": 9504,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7055666631590024,
            "MicroF1": 0.7055666631590024,
            "MacroF1": 0.6515748182594757,
            "Memory in Mb": 80.03414821624756,
            "Time in s": 1031.228873
          },
          {
            "step": 10560,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7099157117151246,
            "MicroF1": 0.7099157117151246,
            "MacroF1": 0.6536141909419667,
            "Memory in Mb": 75.23120212554932,
            "Time in s": 1268.49353
          },
          {
            "step": 11616,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7112354713732243,
            "MicroF1": 0.7112354713732243,
            "MacroF1": 0.6532930257397846,
            "Memory in Mb": 87.85937118530273,
            "Time in s": 1530.837041
          },
          {
            "step": 12672,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7140715018546286,
            "MicroF1": 0.7140715018546285,
            "MacroF1": 0.6586632134486646,
            "Memory in Mb": 96.90367698669434,
            "Time in s": 1818.32799
          },
          {
            "step": 13728,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7196765498652291,
            "MicroF1": 0.7196765498652291,
            "MacroF1": 0.7110222921473365,
            "Memory in Mb": 56.69392013549805,
            "Time in s": 2124.564995
          },
          {
            "step": 14784,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7275248596360685,
            "MicroF1": 0.7275248596360685,
            "MacroF1": 0.7243727970733626,
            "Memory in Mb": 23.290308952331543,
            "Time in s": 2446.995357
          },
          {
            "step": 15840,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7219521434433992,
            "MicroF1": 0.7219521434433992,
            "MacroF1": 0.7204121258981635,
            "Memory in Mb": 12.419946670532228,
            "Time in s": 2789.3598920000004
          },
          {
            "step": 16896,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7180822728617934,
            "MicroF1": 0.7180822728617934,
            "MacroF1": 0.7177336146344276,
            "Memory in Mb": 18.459078788757324,
            "Time in s": 3150.9630650000004
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7130521976491561,
            "MicroF1": 0.713052197649156,
            "MacroF1": 0.7136298242976093,
            "Memory in Mb": 32.946556091308594,
            "Time in s": 3531.3081450000004
          },
          {
            "step": 19008,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7221023833324565,
            "MicroF1": 0.7221023833324565,
            "MacroF1": 0.7193994629254835,
            "Memory in Mb": 14.42181396484375,
            "Time in s": 3928.2542010000006
          },
          {
            "step": 20064,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7273089767233215,
            "MicroF1": 0.7273089767233214,
            "MacroF1": 0.721146893328104,
            "Memory in Mb": 20.33617401123047,
            "Time in s": 4340.108142000001
          },
          {
            "step": 21120,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7289170888773142,
            "MicroF1": 0.7289170888773142,
            "MacroF1": 0.7201390592471967,
            "Memory in Mb": 29.71843242645264,
            "Time in s": 4775.145968000001
          },
          {
            "step": 22176,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7305524239007892,
            "MicroF1": 0.7305524239007891,
            "MacroF1": 0.719265816341323,
            "Memory in Mb": 21.72282314300537,
            "Time in s": 5233.162328
          },
          {
            "step": 23232,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7328569583745856,
            "MicroF1": 0.7328569583745856,
            "MacroF1": 0.7192472788421966,
            "Memory in Mb": 31.907146453857425,
            "Time in s": 5712.0379140000005
          },
          {
            "step": 24288,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7349610902952196,
            "MicroF1": 0.7349610902952196,
            "MacroF1": 0.7190161489472059,
            "Memory in Mb": 36.71180248260498,
            "Time in s": 6211.046404000001
          },
          {
            "step": 25344,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7382314643096713,
            "MicroF1": 0.7382314643096713,
            "MacroF1": 0.7202655895968563,
            "Memory in Mb": 44.65183067321777,
            "Time in s": 6729.485654000001
          },
          {
            "step": 26400,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7397249895829388,
            "MicroF1": 0.7397249895829386,
            "MacroF1": 0.7198095986730461,
            "Memory in Mb": 54.357375144958496,
            "Time in s": 7266.820082000001
          },
          {
            "step": 27456,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7418685121107267,
            "MicroF1": 0.7418685121107267,
            "MacroF1": 0.7199133187431289,
            "Memory in Mb": 60.99125003814697,
            "Time in s": 7822.667714000001
          },
          {
            "step": 28512,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7388727157939041,
            "MicroF1": 0.7388727157939041,
            "MacroF1": 0.7182833957431396,
            "Memory in Mb": 26.944812774658203,
            "Time in s": 8398.518895000001
          },
          {
            "step": 29568,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7376128792234586,
            "MicroF1": 0.7376128792234586,
            "MacroF1": 0.7214769633664444,
            "Memory in Mb": 24.290247917175293,
            "Time in s": 8990.352305
          },
          {
            "step": 30624,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7372889658100121,
            "MicroF1": 0.7372889658100121,
            "MacroF1": 0.7255972176885724,
            "Memory in Mb": 19.85909652709961,
            "Time in s": 9598.262147
          },
          {
            "step": 31680,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7371444805707251,
            "MicroF1": 0.737144480570725,
            "MacroF1": 0.7291466686667684,
            "Memory in Mb": 32.85751724243164,
            "Time in s": 10221.399895
          },
          {
            "step": 32736,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7374064457003208,
            "MicroF1": 0.7374064457003208,
            "MacroF1": 0.7322831246511409,
            "Memory in Mb": 38.75182342529297,
            "Time in s": 10860.215509
          },
          {
            "step": 33792,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7329170489183511,
            "MicroF1": 0.7329170489183511,
            "MacroF1": 0.7291423789419403,
            "Memory in Mb": 76.79454803466797,
            "Time in s": 11517.947008
          },
          {
            "step": 34848,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7290728039716475,
            "MicroF1": 0.7290728039716475,
            "MacroF1": 0.7252059051088736,
            "Memory in Mb": 37.93787670135498,
            "Time in s": 12198.376918999998
          },
          {
            "step": 35904,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.726791633011169,
            "MicroF1": 0.7267916330111689,
            "MacroF1": 0.72277521319889,
            "Memory in Mb": 30.2938232421875,
            "Time in s": 12898.613101
          },
          {
            "step": 36960,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7233150247571634,
            "MicroF1": 0.7233150247571634,
            "MacroF1": 0.7191521630945247,
            "Memory in Mb": 34.07670021057129,
            "Time in s": 13619.767749
          },
          {
            "step": 38016,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7210837827173484,
            "MicroF1": 0.7210837827173484,
            "MacroF1": 0.7166085958184295,
            "Memory in Mb": 39.338196754455566,
            "Time in s": 14364.864177
          },
          {
            "step": 39072,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7203040618361445,
            "MicroF1": 0.7203040618361445,
            "MacroF1": 0.7160627724850469,
            "Memory in Mb": 41.774664878845215,
            "Time in s": 15133.386085
          },
          {
            "step": 40128,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7183193361078576,
            "MicroF1": 0.7183193361078576,
            "MacroF1": 0.7145670483840382,
            "Memory in Mb": 44.649410247802734,
            "Time in s": 15926.835433
          },
          {
            "step": 41184,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7176990505791224,
            "MicroF1": 0.7176990505791223,
            "MacroF1": 0.7142800617937591,
            "Memory in Mb": 38.580246925354,
            "Time in s": 16742.181512
          },
          {
            "step": 42240,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7177489997395772,
            "MicroF1": 0.7177489997395772,
            "MacroF1": 0.7147225222929322,
            "Memory in Mb": 44.2959041595459,
            "Time in s": 17577.282826
          },
          {
            "step": 43296,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7185818223813374,
            "MicroF1": 0.7185818223813374,
            "MacroF1": 0.7159354160738768,
            "Memory in Mb": 44.74843406677246,
            "Time in s": 18431.175396
          },
          {
            "step": 44352,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7192171540664246,
            "MicroF1": 0.7192171540664247,
            "MacroF1": 0.7168891106233332,
            "Memory in Mb": 50.63485240936279,
            "Time in s": 19303.296412
          },
          {
            "step": 45408,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7197128196093113,
            "MicroF1": 0.7197128196093113,
            "MacroF1": 0.7173999204613543,
            "Memory in Mb": 48.77041816711426,
            "Time in s": 20195.421521
          },
          {
            "step": 46464,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7207240169597314,
            "MicroF1": 0.7207240169597314,
            "MacroF1": 0.7184187872009821,
            "Memory in Mb": 56.04546070098877,
            "Time in s": 21107.071122
          },
          {
            "step": 47520,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7212062543403691,
            "MicroF1": 0.7212062543403692,
            "MacroF1": 0.7191280088329424,
            "Memory in Mb": 48.19489002227783,
            "Time in s": 22037.88723
          },
          {
            "step": 48576,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7252084405558414,
            "MicroF1": 0.7252084405558414,
            "MacroF1": 0.7232782847500743,
            "Memory in Mb": 54.32844257354736,
            "Time in s": 22988.011007
          },
          {
            "step": 49632,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7291813584251778,
            "MicroF1": 0.7291813584251778,
            "MacroF1": 0.7271951034706091,
            "Memory in Mb": 53.6518030166626,
            "Time in s": 23956.634362
          },
          {
            "step": 50688,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7326928009154221,
            "MicroF1": 0.7326928009154221,
            "MacroF1": 0.7304439468758875,
            "Memory in Mb": 27.42653465270996,
            "Time in s": 24940.316756
          },
          {
            "step": 51744,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7367180101656262,
            "MicroF1": 0.7367180101656263,
            "MacroF1": 0.7341247480346391,
            "Memory in Mb": 36.39958953857422,
            "Time in s": 25936.798718
          },
          {
            "step": 52800,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Insects",
            "Accuracy": 0.7395784011060815,
            "MicroF1": 0.7395784011060814,
            "MacroF1": 0.737512125998823,
            "Memory in Mb": 8.341936111450195,
            "Time in s": 26942.262482
          },
          {
            "step": 408,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9803439803439804,
            "MicroF1": 0.9803439803439804,
            "MacroF1": 0.4950372208436724,
            "Memory in Mb": 1.551915168762207,
            "Time in s": 2.576854
          },
          {
            "step": 816,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.974233128834356,
            "MicroF1": 0.974233128834356,
            "MacroF1": 0.8747406597440331,
            "Memory in Mb": 4.161267280578613,
            "Time in s": 7.918686
          },
          {
            "step": 1224,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9672935404742437,
            "MicroF1": 0.9672935404742437,
            "MacroF1": 0.9345378451161834,
            "Memory in Mb": 7.904744148254394,
            "Time in s": 16.17177
          },
          {
            "step": 1632,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9662783568362968,
            "MicroF1": 0.9662783568362968,
            "MacroF1": 0.920078959712528,
            "Memory in Mb": 12.156608581542969,
            "Time in s": 27.617862
          },
          {
            "step": 2040,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9632172633643944,
            "MicroF1": 0.9632172633643944,
            "MacroF1": 0.9392069284616192,
            "Memory in Mb": 18.052184104919437,
            "Time in s": 42.519185
          },
          {
            "step": 2448,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9591336330200244,
            "MicroF1": 0.9591336330200244,
            "MacroF1": 0.952707267188964,
            "Memory in Mb": 17.44593620300293,
            "Time in s": 61.058317
          },
          {
            "step": 2856,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9600700525394046,
            "MicroF1": 0.9600700525394046,
            "MacroF1": 0.9487475492194613,
            "Memory in Mb": 23.22895908355713,
            "Time in s": 83.383302
          },
          {
            "step": 3264,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9589334967821024,
            "MicroF1": 0.9589334967821024,
            "MacroF1": 0.9481804303110768,
            "Memory in Mb": 30.014172554016117,
            "Time in s": 110.001368
          },
          {
            "step": 3672,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.957504767093435,
            "MicroF1": 0.957504767093435,
            "MacroF1": 0.948270905442242,
            "Memory in Mb": 37.68842315673828,
            "Time in s": 141.307336
          },
          {
            "step": 4080,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9529296396175532,
            "MicroF1": 0.9529296396175532,
            "MacroF1": 0.9350591426916868,
            "Memory in Mb": 43.92499256134033,
            "Time in s": 178.285036
          },
          {
            "step": 4488,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9558725206151104,
            "MicroF1": 0.9558725206151104,
            "MacroF1": 0.958348874105129,
            "Memory in Mb": 23.460043907165527,
            "Time in s": 220.514164
          },
          {
            "step": 4896,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.957711950970378,
            "MicroF1": 0.957711950970378,
            "MacroF1": 0.9572545884780326,
            "Memory in Mb": 21.815909385681152,
            "Time in s": 267.806579
          },
          {
            "step": 5304,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9581369036394494,
            "MicroF1": 0.9581369036394494,
            "MacroF1": 0.9564558175945328,
            "Memory in Mb": 29.116984367370605,
            "Time in s": 320.079148
          },
          {
            "step": 5712,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.959376641568902,
            "MicroF1": 0.959376641568902,
            "MacroF1": 0.9590743474150508,
            "Memory in Mb": 34.215229988098145,
            "Time in s": 377.359359
          },
          {
            "step": 6120,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9576728223565942,
            "MicroF1": 0.9576728223565942,
            "MacroF1": 0.9540539138154064,
            "Memory in Mb": 42.24546051025391,
            "Time in s": 440.303036
          },
          {
            "step": 6528,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.957254481385016,
            "MicroF1": 0.9572544813850162,
            "MacroF1": 0.9569914463415944,
            "Memory in Mb": 19.71925640106201,
            "Time in s": 508.401183
          },
          {
            "step": 6936,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9586157173756308,
            "MicroF1": 0.9586157173756308,
            "MacroF1": 0.9593505106134974,
            "Memory in Mb": 22.396859169006348,
            "Time in s": 580.778744
          },
          {
            "step": 7344,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9592809478414815,
            "MicroF1": 0.9592809478414815,
            "MacroF1": 0.9593459120031488,
            "Memory in Mb": 26.322874069213867,
            "Time in s": 657.980924
          },
          {
            "step": 7752,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9600051606244356,
            "MicroF1": 0.9600051606244356,
            "MacroF1": 0.9601169971762602,
            "Memory in Mb": 30.259758949279785,
            "Time in s": 740.2744439999999
          },
          {
            "step": 8160,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.95747027822037,
            "MicroF1": 0.95747027822037,
            "MacroF1": 0.9549133730963548,
            "Memory in Mb": 37.68336868286133,
            "Time in s": 827.6717389999999
          },
          {
            "step": 8568,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9564608380996849,
            "MicroF1": 0.9564608380996849,
            "MacroF1": 0.9560990529914856,
            "Memory in Mb": 43.737112045288086,
            "Time in s": 921.0092
          },
          {
            "step": 8976,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9568802228412256,
            "MicroF1": 0.9568802228412256,
            "MacroF1": 0.9569984740230398,
            "Memory in Mb": 33.59728527069092,
            "Time in s": 1020.6233179999998
          },
          {
            "step": 9384,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9575828626238942,
            "MicroF1": 0.9575828626238942,
            "MacroF1": 0.9578510301970172,
            "Memory in Mb": 34.01332950592041,
            "Time in s": 1125.214274
          },
          {
            "step": 9792,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9576141354304974,
            "MicroF1": 0.9576141354304974,
            "MacroF1": 0.95758927245962,
            "Memory in Mb": 40.18074893951416,
            "Time in s": 1234.457491
          },
          {
            "step": 10200,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9558780272575744,
            "MicroF1": 0.9558780272575744,
            "MacroF1": 0.954787839223492,
            "Memory in Mb": 48.97087860107422,
            "Time in s": 1349.23618
          },
          {
            "step": 10608,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9524842085415292,
            "MicroF1": 0.9524842085415292,
            "MacroF1": 0.9506853107984292,
            "Memory in Mb": 30.6993989944458,
            "Time in s": 1470.124694
          },
          {
            "step": 11016,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9539718565592374,
            "MicroF1": 0.9539718565592374,
            "MacroF1": 0.9545620457235888,
            "Memory in Mb": 26.549206733703613,
            "Time in s": 1595.074551
          },
          {
            "step": 11424,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9543902652543116,
            "MicroF1": 0.9543902652543116,
            "MacroF1": 0.9545363240408884,
            "Memory in Mb": 33.6107063293457,
            "Time in s": 1724.543611
          },
          {
            "step": 11832,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9541881497760122,
            "MicroF1": 0.9541881497760122,
            "MacroF1": 0.954140840579052,
            "Memory in Mb": 25.182985305786133,
            "Time in s": 1859.013318
          },
          {
            "step": 12240,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.955061688046409,
            "MicroF1": 0.955061688046409,
            "MacroF1": 0.9554321262858616,
            "Memory in Mb": 27.34038543701172,
            "Time in s": 1997.870855
          },
          {
            "step": 12648,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9546928125247094,
            "MicroF1": 0.9546928125247094,
            "MacroF1": 0.9546233453975912,
            "Memory in Mb": 35.30395698547363,
            "Time in s": 2141.6119080000003
          },
          {
            "step": 13056,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.953887399463807,
            "MicroF1": 0.953887399463807,
            "MacroF1": 0.9537532269202632,
            "Memory in Mb": 33.51621055603027,
            "Time in s": 2290.943904
          },
          {
            "step": 13464,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9540221347396568,
            "MicroF1": 0.9540221347396568,
            "MacroF1": 0.954138309472004,
            "Memory in Mb": 33.38596153259277,
            "Time in s": 2445.0119170000003
          },
          {
            "step": 13872,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9546535938288516,
            "MicroF1": 0.9546535938288516,
            "MacroF1": 0.9549190485054234,
            "Memory in Mb": 31.36033725738525,
            "Time in s": 2603.685647
          },
          {
            "step": 14280,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9534281112122698,
            "MicroF1": 0.9534281112122698,
            "MacroF1": 0.9532093226981456,
            "Memory in Mb": 38.61776542663574,
            "Time in s": 2767.1788560000005
          },
          {
            "step": 14688,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9540409886294,
            "MicroF1": 0.9540409886294,
            "MacroF1": 0.9542688403803362,
            "Memory in Mb": 42.8822660446167,
            "Time in s": 2935.9642900000003
          },
          {
            "step": 15096,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9547532295462072,
            "MicroF1": 0.9547532295462072,
            "MacroF1": 0.9549723528375392,
            "Memory in Mb": 41.949758529663086,
            "Time in s": 3110.4045410000003
          },
          {
            "step": 15504,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9549764561697736,
            "MicroF1": 0.9549764561697736,
            "MacroF1": 0.9551012466300322,
            "Memory in Mb": 36.29027271270752,
            "Time in s": 3290.0043080000005
          },
          {
            "step": 15912,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9551253849538056,
            "MicroF1": 0.9551253849538056,
            "MacroF1": 0.955237279627336,
            "Memory in Mb": 33.26945877075195,
            "Time in s": 3474.8467850000006
          },
          {
            "step": 16320,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9555119799007292,
            "MicroF1": 0.9555119799007292,
            "MacroF1": 0.9556369370454034,
            "Memory in Mb": 38.47606945037842,
            "Time in s": 3664.7140100000006
          },
          {
            "step": 16728,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.954923178095295,
            "MicroF1": 0.954923178095295,
            "MacroF1": 0.9549151106032768,
            "Memory in Mb": 38.78229522705078,
            "Time in s": 3859.856695
          },
          {
            "step": 17136,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.955587977823169,
            "MicroF1": 0.955587977823169,
            "MacroF1": 0.9557184838324558,
            "Memory in Mb": 44.56228828430176,
            "Time in s": 4060.267757000001
          },
          {
            "step": 17544,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9550817990081514,
            "MicroF1": 0.9550817990081514,
            "MacroF1": 0.9550944582439086,
            "Memory in Mb": 49.72221755981445,
            "Time in s": 4266.497535
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9547657512116317,
            "MicroF1": 0.9547657512116317,
            "MacroF1": 0.9547923955213532,
            "Memory in Mb": 44.72002029418945,
            "Time in s": 4478.609232000001
          },
          {
            "step": 18360,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9553897271093196,
            "MicroF1": 0.9553897271093196,
            "MacroF1": 0.955476322048541,
            "Memory in Mb": 52.00297737121582,
            "Time in s": 4696.983244000001
          },
          {
            "step": 18768,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.955507006980338,
            "MicroF1": 0.955507006980338,
            "MacroF1": 0.9555572955831596,
            "Memory in Mb": 59.27475929260254,
            "Time in s": 4921.837298000001
          },
          {
            "step": 19176,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9548891786179922,
            "MicroF1": 0.9548891786179922,
            "MacroF1": 0.9549038695373788,
            "Memory in Mb": 71.70181655883789,
            "Time in s": 5153.196879000001
          },
          {
            "step": 19584,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.954858806107338,
            "MicroF1": 0.954858806107338,
            "MacroF1": 0.9548865417655428,
            "Memory in Mb": 78.58561515808105,
            "Time in s": 5390.578061000001
          },
          {
            "step": 19992,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9539292681706768,
            "MicroF1": 0.9539292681706768,
            "MacroF1": 0.9539347026376764,
            "Memory in Mb": 85.24763870239258,
            "Time in s": 5634.910465000001
          },
          {
            "step": 20400,
            "track": "Multiclass classification",
            "model": "Streaming Random Patches",
            "dataset": "Keystroke",
            "Accuracy": 0.9532330016177264,
            "MicroF1": 0.9532330016177264,
            "MacroF1": 0.9532392337717848,
            "Memory in Mb": 74.55205345153809,
            "Time in s": 5886.477404000001
          },
          {
            "step": 46,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.5555555555555556,
            "MicroF1": 0.5555555555555556,
            "MacroF1": 0.4458032432860809,
            "Memory in Mb": 0.061410903930664,
            "Time in s": 0.019654
          },
          {
            "step": 92,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.6483516483516484,
            "MicroF1": 0.6483516483516484,
            "MacroF1": 0.646491610589355,
            "Memory in Mb": 0.1154079437255859,
            "Time in s": 0.0623149999999999
          },
          {
            "step": 138,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.708029197080292,
            "MicroF1": 0.708029197080292,
            "MacroF1": 0.7216654146545566,
            "Memory in Mb": 0.1258535385131836,
            "Time in s": 0.135768
          },
          {
            "step": 184,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.7431693989071039,
            "MicroF1": 0.743169398907104,
            "MacroF1": 0.7576794034998369,
            "Memory in Mb": 0.1263303756713867,
            "Time in s": 0.24048
          },
          {
            "step": 230,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.7641921397379913,
            "MicroF1": 0.7641921397379913,
            "MacroF1": 0.7751275973499576,
            "Memory in Mb": 0.126317024230957,
            "Time in s": 0.376459
          },
          {
            "step": 276,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.7672727272727272,
            "MicroF1": 0.7672727272727272,
            "MacroF1": 0.7799448750812884,
            "Memory in Mb": 0.1262655258178711,
            "Time in s": 0.543817
          },
          {
            "step": 322,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.7757009345794392,
            "MicroF1": 0.7757009345794392,
            "MacroF1": 0.781311030606134,
            "Memory in Mb": 0.1267433166503906,
            "Time in s": 0.742198
          },
          {
            "step": 368,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.782016348773842,
            "MicroF1": 0.782016348773842,
            "MacroF1": 0.7830988277979799,
            "Memory in Mb": 0.1267471313476562,
            "Time in s": 0.972069
          },
          {
            "step": 414,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.7893462469733656,
            "MicroF1": 0.7893462469733655,
            "MacroF1": 0.7891834545778567,
            "Memory in Mb": 0.1262693405151367,
            "Time in s": 1.233319
          },
          {
            "step": 460,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.7799564270152506,
            "MicroF1": 0.7799564270152506,
            "MacroF1": 0.778762654261754,
            "Memory in Mb": 0.1262626647949218,
            "Time in s": 1.526064
          },
          {
            "step": 506,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.7841584158415842,
            "MicroF1": 0.7841584158415842,
            "MacroF1": 0.7830263284725031,
            "Memory in Mb": 0.126779556274414,
            "Time in s": 1.849975
          },
          {
            "step": 552,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.7840290381125227,
            "MicroF1": 0.7840290381125228,
            "MacroF1": 0.7833214841514466,
            "Memory in Mb": 0.1267738342285156,
            "Time in s": 2.2053070000000004
          },
          {
            "step": 598,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.7839195979899497,
            "MicroF1": 0.7839195979899497,
            "MacroF1": 0.7851401823229054,
            "Memory in Mb": 0.1262836456298828,
            "Time in s": 2.5919950000000003
          },
          {
            "step": 644,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.7884914463452566,
            "MicroF1": 0.7884914463452566,
            "MacroF1": 0.790931132142264,
            "Memory in Mb": 0.1262893676757812,
            "Time in s": 3.010019
          },
          {
            "step": 690,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.795355587808418,
            "MicroF1": 0.795355587808418,
            "MacroF1": 0.7973717331367783,
            "Memory in Mb": 0.1267967224121093,
            "Time in s": 3.459217
          },
          {
            "step": 736,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.7918367346938775,
            "MicroF1": 0.7918367346938775,
            "MacroF1": 0.79371924750244,
            "Memory in Mb": 0.1262922286987304,
            "Time in s": 3.93961
          },
          {
            "step": 782,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8015364916773368,
            "MicroF1": 0.8015364916773368,
            "MacroF1": 0.8027236936866887,
            "Memory in Mb": 0.1262769699096679,
            "Time in s": 4.451398
          },
          {
            "step": 828,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.7980652962515115,
            "MicroF1": 0.7980652962515115,
            "MacroF1": 0.8001612113332863,
            "Memory in Mb": 0.1267776489257812,
            "Time in s": 4.994585
          },
          {
            "step": 874,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8041237113402062,
            "MicroF1": 0.8041237113402062,
            "MacroF1": 0.8058476562214167,
            "Memory in Mb": 0.1267652511596679,
            "Time in s": 5.568929
          },
          {
            "step": 920,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8063112078346029,
            "MicroF1": 0.8063112078346029,
            "MacroF1": 0.8071524109530731,
            "Memory in Mb": 0.1262378692626953,
            "Time in s": 6.174556
          },
          {
            "step": 966,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8072538860103627,
            "MicroF1": 0.8072538860103627,
            "MacroF1": 0.8069383576906736,
            "Memory in Mb": 0.1262502670288086,
            "Time in s": 6.811568
          },
          {
            "step": 1012,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8120672601384767,
            "MicroF1": 0.8120672601384767,
            "MacroF1": 0.8103691514865562,
            "Memory in Mb": 0.1267623901367187,
            "Time in s": 7.479958
          },
          {
            "step": 1058,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8117313150425733,
            "MicroF1": 0.8117313150425733,
            "MacroF1": 0.8093057999862455,
            "Memory in Mb": 0.1267585754394531,
            "Time in s": 8.179363
          },
          {
            "step": 1104,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8105167724388033,
            "MicroF1": 0.8105167724388033,
            "MacroF1": 0.8087453181575575,
            "Memory in Mb": 0.126260757446289,
            "Time in s": 8.909729
          },
          {
            "step": 1150,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8120104438642297,
            "MicroF1": 0.8120104438642298,
            "MacroF1": 0.8093458779132273,
            "Memory in Mb": 0.1267480850219726,
            "Time in s": 9.671618
          },
          {
            "step": 1196,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8125523012552301,
            "MicroF1": 0.8125523012552303,
            "MacroF1": 0.8098995946687924,
            "Memory in Mb": 0.1267566680908203,
            "Time in s": 10.464471
          },
          {
            "step": 1242,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8170829975825947,
            "MicroF1": 0.8170829975825946,
            "MacroF1": 0.8146737825459542,
            "Memory in Mb": 0.1263046264648437,
            "Time in s": 11.288498
          },
          {
            "step": 1288,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8174048174048174,
            "MicroF1": 0.8174048174048174,
            "MacroF1": 0.8149699191034137,
            "Memory in Mb": 0.1263151168823242,
            "Time in s": 12.143769999999998
          },
          {
            "step": 1334,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8169542385596399,
            "MicroF1": 0.8169542385596399,
            "MacroF1": 0.8144172630221828,
            "Memory in Mb": 0.1267910003662109,
            "Time in s": 13.030243999999998
          },
          {
            "step": 1380,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8165337200870196,
            "MicroF1": 0.8165337200870196,
            "MacroF1": 0.8142638589810781,
            "Memory in Mb": 0.1267881393432617,
            "Time in s": 13.947682999999998
          },
          {
            "step": 1426,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8210526315789474,
            "MicroF1": 0.8210526315789475,
            "MacroF1": 0.8177443463463022,
            "Memory in Mb": 0.1262807846069336,
            "Time in s": 14.896382999999998
          },
          {
            "step": 1472,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.822569680489463,
            "MicroF1": 0.822569680489463,
            "MacroF1": 0.8180682540474884,
            "Memory in Mb": 0.1267719268798828,
            "Time in s": 15.876335
          },
          {
            "step": 1518,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8233355306526038,
            "MicroF1": 0.8233355306526038,
            "MacroF1": 0.8183049909694801,
            "Memory in Mb": 0.1267585754394531,
            "Time in s": 16.887748
          },
          {
            "step": 1564,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8227767114523352,
            "MicroF1": 0.8227767114523352,
            "MacroF1": 0.8180063024943973,
            "Memory in Mb": 0.1262645721435547,
            "Time in s": 17.930424
          },
          {
            "step": 1610,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8228713486637663,
            "MicroF1": 0.8228713486637663,
            "MacroF1": 0.818440484251979,
            "Memory in Mb": 0.1262655258178711,
            "Time in s": 19.004458
          },
          {
            "step": 1656,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.824773413897281,
            "MicroF1": 0.824773413897281,
            "MacroF1": 0.8207684581521858,
            "Memory in Mb": 0.1267824172973632,
            "Time in s": 20.10985
          },
          {
            "step": 1702,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.824808935920047,
            "MicroF1": 0.824808935920047,
            "MacroF1": 0.8222541912553749,
            "Memory in Mb": 0.1268024444580078,
            "Time in s": 21.24685
          },
          {
            "step": 1748,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8259874069834001,
            "MicroF1": 0.8259874069834001,
            "MacroF1": 0.8228660744170171,
            "Memory in Mb": 0.1263065338134765,
            "Time in s": 22.415483
          },
          {
            "step": 1794,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.826547685443391,
            "MicroF1": 0.826547685443391,
            "MacroF1": 0.8226613560637924,
            "Memory in Mb": 0.126774787902832,
            "Time in s": 23.615758
          },
          {
            "step": 1840,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8254486133768353,
            "MicroF1": 0.8254486133768353,
            "MacroF1": 0.8217381124058762,
            "Memory in Mb": 0.1267585754394531,
            "Time in s": 24.847279
          },
          {
            "step": 1886,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8249336870026526,
            "MicroF1": 0.8249336870026526,
            "MacroF1": 0.8216008133499116,
            "Memory in Mb": 0.1262845993041992,
            "Time in s": 26.110083
          },
          {
            "step": 1932,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8234075608493009,
            "MicroF1": 0.8234075608493009,
            "MacroF1": 0.8193527544316537,
            "Memory in Mb": 0.1262779235839843,
            "Time in s": 27.403915
          },
          {
            "step": 1978,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8234699038947901,
            "MicroF1": 0.8234699038947901,
            "MacroF1": 0.8195124114516217,
            "Memory in Mb": 0.1267585754394531,
            "Time in s": 28.728915
          },
          {
            "step": 2024,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8220464656450815,
            "MicroF1": 0.8220464656450815,
            "MacroF1": 0.8172381305352,
            "Memory in Mb": 0.126774787902832,
            "Time in s": 30.084907
          },
          {
            "step": 2070,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8206863218946351,
            "MicroF1": 0.8206863218946351,
            "MacroF1": 0.8164336862763343,
            "Memory in Mb": 0.1262893676757812,
            "Time in s": 31.472199
          },
          {
            "step": 2116,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8217494089834515,
            "MicroF1": 0.8217494089834515,
            "MacroF1": 0.8168455585843762,
            "Memory in Mb": 0.1262645721435547,
            "Time in s": 32.891028
          },
          {
            "step": 2162,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8204534937528922,
            "MicroF1": 0.8204534937528921,
            "MacroF1": 0.8154843900985335,
            "Memory in Mb": 0.1267728805541992,
            "Time in s": 34.340933
          },
          {
            "step": 2208,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.822383325781604,
            "MicroF1": 0.822383325781604,
            "MacroF1": 0.8171788245797035,
            "Memory in Mb": 0.1262683868408203,
            "Time in s": 35.822171
          },
          {
            "step": 2254,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.821127385707945,
            "MicroF1": 0.821127385707945,
            "MacroF1": 0.8170261701336431,
            "Memory in Mb": 0.126255989074707,
            "Time in s": 37.335093
          },
          {
            "step": 2300,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "ImageSegments",
            "Accuracy": 0.8199217050891692,
            "MicroF1": 0.8199217050891693,
            "MacroF1": 0.8158945802523674,
            "Memory in Mb": 0.1267604827880859,
            "Time in s": 38.879423
          },
          {
            "step": 1056,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6322274881516587,
            "MicroF1": 0.6322274881516587,
            "MacroF1": 0.5639948035153092,
            "Memory in Mb": 0.2159481048583984,
            "Time in s": 1.014879
          },
          {
            "step": 2112,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.636191378493605,
            "MicroF1": 0.636191378493605,
            "MacroF1": 0.5686546251961576,
            "Memory in Mb": 0.2164621353149414,
            "Time in s": 3.11274
          },
          {
            "step": 3168,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6274076413009156,
            "MicroF1": 0.6274076413009156,
            "MacroF1": 0.5664829980315041,
            "Memory in Mb": 0.2159862518310547,
            "Time in s": 6.352824
          },
          {
            "step": 4224,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6317783566185177,
            "MicroF1": 0.6317783566185177,
            "MacroF1": 0.5676004628647836,
            "Memory in Mb": 0.216461181640625,
            "Time in s": 10.71472
          },
          {
            "step": 5280,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6277704110627013,
            "MicroF1": 0.6277704110627013,
            "MacroF1": 0.5651907052085646,
            "Memory in Mb": 0.215977668762207,
            "Time in s": 16.153805
          },
          {
            "step": 6336,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6244672454617206,
            "MicroF1": 0.6244672454617206,
            "MacroF1": 0.5642758642399058,
            "Memory in Mb": 0.2164630889892578,
            "Time in s": 22.622426
          },
          {
            "step": 7392,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.621160871329996,
            "MicroF1": 0.621160871329996,
            "MacroF1": 0.5621999618118433,
            "Memory in Mb": 0.2159862518310547,
            "Time in s": 30.095944
          },
          {
            "step": 8448,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6183260329110927,
            "MicroF1": 0.6183260329110927,
            "MacroF1": 0.560545956984929,
            "Memory in Mb": 0.2164678573608398,
            "Time in s": 38.53514199999999
          },
          {
            "step": 9504,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6195938124802693,
            "MicroF1": 0.6195938124802693,
            "MacroF1": 0.5612689785887882,
            "Memory in Mb": 0.2159566879272461,
            "Time in s": 47.930055
          },
          {
            "step": 10560,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6209868358746093,
            "MicroF1": 0.6209868358746093,
            "MacroF1": 0.5626902992589761,
            "Memory in Mb": 0.2164936065673828,
            "Time in s": 58.279528
          },
          {
            "step": 11616,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6196297890658631,
            "MicroF1": 0.6196297890658631,
            "MacroF1": 0.5618958864151227,
            "Memory in Mb": 0.215947151184082,
            "Time in s": 69.583507
          },
          {
            "step": 12672,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6235498382132428,
            "MicroF1": 0.6235498382132428,
            "MacroF1": 0.577401509815314,
            "Memory in Mb": 0.2165937423706054,
            "Time in s": 81.841211
          },
          {
            "step": 13728,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6524368033801996,
            "MicroF1": 0.6524368033801996,
            "MacroF1": 0.656066758247117,
            "Memory in Mb": 0.2161521911621093,
            "Time in s": 95.05061
          },
          {
            "step": 14784,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6639383075153893,
            "MicroF1": 0.6639383075153893,
            "MacroF1": 0.6656513873636037,
            "Memory in Mb": 0.2165174484252929,
            "Time in s": 109.208982
          },
          {
            "step": 15840,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6599532798787803,
            "MicroF1": 0.6599532798787803,
            "MacroF1": 0.6660828271082423,
            "Memory in Mb": 0.2159938812255859,
            "Time in s": 124.323768
          },
          {
            "step": 16896,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6583012725658479,
            "MicroF1": 0.6583012725658479,
            "MacroF1": 0.6678320995738946,
            "Memory in Mb": 0.2164812088012695,
            "Time in s": 140.393226
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6558966074313409,
            "MicroF1": 0.6558966074313409,
            "MacroF1": 0.6676009154715022,
            "Memory in Mb": 0.2160320281982422,
            "Time in s": 157.416689
          },
          {
            "step": 19008,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6731730415110223,
            "MicroF1": 0.6731730415110223,
            "MacroF1": 0.6774302820037228,
            "Memory in Mb": 0.2165126800537109,
            "Time in s": 175.38966399999998
          },
          {
            "step": 20064,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6798086029008623,
            "MicroF1": 0.6798086029008623,
            "MacroF1": 0.6780616401383449,
            "Memory in Mb": 0.2158927917480468,
            "Time in s": 194.316978
          },
          {
            "step": 21120,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.680382593872816,
            "MicroF1": 0.680382593872816,
            "MacroF1": 0.6752117016598617,
            "Memory in Mb": 0.2164011001586914,
            "Time in s": 214.200546
          },
          {
            "step": 22176,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6805862457722661,
            "MicroF1": 0.6805862457722661,
            "MacroF1": 0.6722568877045599,
            "Memory in Mb": 0.2158823013305664,
            "Time in s": 235.037825
          },
          {
            "step": 23232,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6813740260858336,
            "MicroF1": 0.6813740260858336,
            "MacroF1": 0.6702824994179433,
            "Memory in Mb": 0.2163906097412109,
            "Time in s": 256.831775
          },
          {
            "step": 24288,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6815992094536172,
            "MicroF1": 0.6815992094536172,
            "MacroF1": 0.6677450869096582,
            "Memory in Mb": 0.2159175872802734,
            "Time in s": 279.574148
          },
          {
            "step": 25344,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6821212958213313,
            "MicroF1": 0.6821212958213313,
            "MacroF1": 0.6660355323582295,
            "Memory in Mb": 0.2163667678833007,
            "Time in s": 303.273869
          },
          {
            "step": 26400,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6831319368157884,
            "MicroF1": 0.6831319368157884,
            "MacroF1": 0.6646803813034555,
            "Memory in Mb": 0.2159061431884765,
            "Time in s": 327.922368
          },
          {
            "step": 27456,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6836277545073757,
            "MicroF1": 0.6836277545073757,
            "MacroF1": 0.6627124931293528,
            "Memory in Mb": 0.2164020538330078,
            "Time in s": 353.52686
          },
          {
            "step": 28512,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6834905825821612,
            "MicroF1": 0.6834905825821612,
            "MacroF1": 0.664548122616301,
            "Memory in Mb": 0.2161540985107422,
            "Time in s": 380.080866
          },
          {
            "step": 29568,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6814692055331958,
            "MicroF1": 0.6814692055331958,
            "MacroF1": 0.6671975305669872,
            "Memory in Mb": 0.2166757583618164,
            "Time in s": 407.595828
          },
          {
            "step": 30624,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6796525487378767,
            "MicroF1": 0.6796525487378767,
            "MacroF1": 0.669471411791397,
            "Memory in Mb": 0.2161798477172851,
            "Time in s": 436.062936
          },
          {
            "step": 31680,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6779570062186306,
            "MicroF1": 0.6779570062186306,
            "MacroF1": 0.6711290718417154,
            "Memory in Mb": 0.216679573059082,
            "Time in s": 465.4881830000001
          },
          {
            "step": 32736,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6768901787078051,
            "MicroF1": 0.6768901787078051,
            "MacroF1": 0.6727094382078547,
            "Memory in Mb": 0.2161359786987304,
            "Time in s": 495.86473400000006
          },
          {
            "step": 33792,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6734337545500281,
            "MicroF1": 0.6734337545500281,
            "MacroF1": 0.6702378074852682,
            "Memory in Mb": 0.2164754867553711,
            "Time in s": 527.2018280000001
          },
          {
            "step": 34848,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6690676385341636,
            "MicroF1": 0.6690676385341636,
            "MacroF1": 0.6661382581729155,
            "Memory in Mb": 0.215947151184082,
            "Time in s": 559.4897450000001
          },
          {
            "step": 35904,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6663510013090828,
            "MicroF1": 0.6663510013090828,
            "MacroF1": 0.6633778558128317,
            "Memory in Mb": 0.2165002822875976,
            "Time in s": 592.7366190000001
          },
          {
            "step": 36960,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.662409697232068,
            "MicroF1": 0.662409697232068,
            "MacroF1": 0.6597878724618786,
            "Memory in Mb": 0.215972900390625,
            "Time in s": 626.9366260000002
          },
          {
            "step": 38016,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6594239116138366,
            "MicroF1": 0.6594239116138366,
            "MacroF1": 0.6567102170776443,
            "Memory in Mb": 0.2164802551269531,
            "Time in s": 662.1000680000002
          },
          {
            "step": 39072,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.662409459701569,
            "MicroF1": 0.662409459701569,
            "MacroF1": 0.6591983036871739,
            "Memory in Mb": 0.2159795761108398,
            "Time in s": 698.2204010000002
          },
          {
            "step": 40128,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6615495800832357,
            "MicroF1": 0.6615495800832357,
            "MacroF1": 0.658372148729009,
            "Memory in Mb": 0.2165098190307617,
            "Time in s": 735.3021140000002
          },
          {
            "step": 41184,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6616079450258602,
            "MicroF1": 0.6616079450258602,
            "MacroF1": 0.6583203582230679,
            "Memory in Mb": 0.2160120010375976,
            "Time in s": 773.3352610000002
          },
          {
            "step": 42240,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6620895381045953,
            "MicroF1": 0.6620895381045953,
            "MacroF1": 0.6586855795305535,
            "Memory in Mb": 0.216496467590332,
            "Time in s": 812.3294790000002
          },
          {
            "step": 43296,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6626862224275321,
            "MicroF1": 0.6626862224275321,
            "MacroF1": 0.6591267371039767,
            "Memory in Mb": 0.216012954711914,
            "Time in s": 852.2739140000002
          },
          {
            "step": 44352,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6625104281752384,
            "MicroF1": 0.6625104281752384,
            "MacroF1": 0.6587853710847982,
            "Memory in Mb": 0.2164974212646484,
            "Time in s": 893.1794370000002
          },
          {
            "step": 45408,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6629374325544519,
            "MicroF1": 0.6629374325544519,
            "MacroF1": 0.6587077344895959,
            "Memory in Mb": 0.2159938812255859,
            "Time in s": 935.0370030000004
          },
          {
            "step": 46464,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6634311172330671,
            "MicroF1": 0.6634311172330671,
            "MacroF1": 0.6587873315408634,
            "Memory in Mb": 0.2165002822875976,
            "Time in s": 977.8531990000002
          },
          {
            "step": 47520,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.666217723436941,
            "MicroF1": 0.666217723436941,
            "MacroF1": 0.6621071051846,
            "Memory in Mb": 0.2159204483032226,
            "Time in s": 1021.6188110000004
          },
          {
            "step": 48576,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6698507462686567,
            "MicroF1": 0.6698507462686567,
            "MacroF1": 0.6663907774790556,
            "Memory in Mb": 0.2164478302001953,
            "Time in s": 1066.3421490000003
          },
          {
            "step": 49632,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6739940762829683,
            "MicroF1": 0.6739940762829683,
            "MacroF1": 0.6709516060662618,
            "Memory in Mb": 0.2159433364868164,
            "Time in s": 1112.0155100000002
          },
          {
            "step": 50688,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6774715410262987,
            "MicroF1": 0.6774715410262987,
            "MacroF1": 0.6745572423992897,
            "Memory in Mb": 0.2164840698242187,
            "Time in s": 1158.6520200000002
          },
          {
            "step": 51744,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6814834856888855,
            "MicroF1": 0.6814834856888855,
            "MacroF1": 0.6786206144243011,
            "Memory in Mb": 0.2159938812255859,
            "Time in s": 1206.2391940000002
          },
          {
            "step": 52800,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Insects",
            "Accuracy": 0.6865470936949564,
            "MicroF1": 0.6865470936949564,
            "MacroF1": 0.6836613373539585,
            "Memory in Mb": 0.2166557312011718,
            "Time in s": 1254.784849
          },
          {
            "step": 408,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9828009828009828,
            "MicroF1": 0.9828009828009828,
            "MacroF1": 0.6067632850241546,
            "Memory in Mb": 0.2092580795288086,
            "Time in s": 0.626636
          },
          {
            "step": 816,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9828220858895704,
            "MicroF1": 0.9828220858895704,
            "MacroF1": 0.9550926410288756,
            "Memory in Mb": 0.2098121643066406,
            "Time in s": 1.943188
          },
          {
            "step": 1224,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9852820932134096,
            "MicroF1": 0.9852820932134096,
            "MacroF1": 0.9672695079711996,
            "Memory in Mb": 0.2093591690063476,
            "Time in s": 3.654696
          },
          {
            "step": 1632,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9840588595953402,
            "MicroF1": 0.9840588595953402,
            "MacroF1": 0.9604409213604836,
            "Memory in Mb": 0.2098979949951172,
            "Time in s": 5.778702
          },
          {
            "step": 2040,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.984796468857283,
            "MicroF1": 0.984796468857283,
            "MacroF1": 0.9791423790442798,
            "Memory in Mb": 0.2104520797729492,
            "Time in s": 8.327232
          },
          {
            "step": 2448,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9861054352268084,
            "MicroF1": 0.9861054352268084,
            "MacroF1": 0.9837809767868474,
            "Memory in Mb": 0.2099990844726562,
            "Time in s": 11.304767
          },
          {
            "step": 2856,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9859894921190894,
            "MicroF1": 0.9859894921190894,
            "MacroF1": 0.9813641447908844,
            "Memory in Mb": 0.2105531692504882,
            "Time in s": 14.706237
          },
          {
            "step": 3264,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9871284094391664,
            "MicroF1": 0.9871284094391664,
            "MacroF1": 0.9868437405314092,
            "Memory in Mb": 0.2106037139892578,
            "Time in s": 18.525325
          },
          {
            "step": 3672,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9880141650776356,
            "MicroF1": 0.9880141650776356,
            "MacroF1": 0.9878382173613446,
            "Memory in Mb": 0.2101507186889648,
            "Time in s": 22.76123
          },
          {
            "step": 4080,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9877420936504046,
            "MicroF1": 0.9877420936504046,
            "MacroF1": 0.9857777629944036,
            "Memory in Mb": 0.2107048034667968,
            "Time in s": 27.408752
          },
          {
            "step": 4488,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9881880989525296,
            "MicroF1": 0.9881880989525296,
            "MacroF1": 0.9878235870948694,
            "Memory in Mb": 0.2102518081665039,
            "Time in s": 32.463238
          },
          {
            "step": 4896,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9885597548518896,
            "MicroF1": 0.9885597548518896,
            "MacroF1": 0.9882962361329112,
            "Memory in Mb": 0.2103023529052734,
            "Time in s": 37.912156
          },
          {
            "step": 5304,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9867999245709976,
            "MicroF1": 0.9867999245709976,
            "MacroF1": 0.9836140972543967,
            "Memory in Mb": 0.210906982421875,
            "Time in s": 43.748231
          },
          {
            "step": 5712,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9873927508317284,
            "MicroF1": 0.9873927508317284,
            "MacroF1": 0.9875632488318824,
            "Memory in Mb": 0.210453987121582,
            "Time in s": 49.983611
          },
          {
            "step": 6120,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9872528190880864,
            "MicroF1": 0.9872528190880864,
            "MacroF1": 0.986679154193125,
            "Memory in Mb": 0.211008071899414,
            "Time in s": 56.586139
          },
          {
            "step": 6528,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.987130381492263,
            "MicroF1": 0.987130381492263,
            "MacroF1": 0.9866769113371192,
            "Memory in Mb": 0.2110586166381836,
            "Time in s": 63.546493
          },
          {
            "step": 6936,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9875991348233598,
            "MicroF1": 0.9875991348233598,
            "MacroF1": 0.9877805463370743,
            "Memory in Mb": 0.2106056213378906,
            "Time in s": 70.86485
          },
          {
            "step": 7344,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.986926324390576,
            "MicroF1": 0.986926324390576,
            "MacroF1": 0.9861386596476128,
            "Memory in Mb": 0.2126245498657226,
            "Time in s": 78.54037100000001
          },
          {
            "step": 7752,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9865823764675524,
            "MicroF1": 0.9865823764675524,
            "MacroF1": 0.986151116916088,
            "Memory in Mb": 0.2121715545654297,
            "Time in s": 86.582909
          },
          {
            "step": 8160,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9861502635126854,
            "MicroF1": 0.9861502635126854,
            "MacroF1": 0.9857089041873668,
            "Memory in Mb": 0.2122220993041992,
            "Time in s": 94.988236
          },
          {
            "step": 8568,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9863429438543247,
            "MicroF1": 0.9863429438543247,
            "MacroF1": 0.986382977302644,
            "Memory in Mb": 0.2127761840820312,
            "Time in s": 103.754213
          },
          {
            "step": 8976,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9865181058495822,
            "MicroF1": 0.9865181058495822,
            "MacroF1": 0.9865643235024212,
            "Memory in Mb": 0.2123231887817382,
            "Time in s": 112.879589
          },
          {
            "step": 9384,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9858254289672812,
            "MicroF1": 0.9858254289672812,
            "MacroF1": 0.9853734936692788,
            "Memory in Mb": 0.2128772735595703,
            "Time in s": 122.365291
          },
          {
            "step": 9792,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9857011541211316,
            "MicroF1": 0.9857011541211316,
            "MacroF1": 0.9856081881161904,
            "Memory in Mb": 0.2129278182983398,
            "Time in s": 132.212374
          },
          {
            "step": 10200,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9860770663790568,
            "MicroF1": 0.9860770663790568,
            "MacroF1": 0.9862471716083434,
            "Memory in Mb": 0.2124748229980468,
            "Time in s": 142.422221
          },
          {
            "step": 10608,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.98538700857924,
            "MicroF1": 0.98538700857924,
            "MacroF1": 0.9850628829106896,
            "Memory in Mb": 0.2130289077758789,
            "Time in s": 152.99283200000002
          },
          {
            "step": 11016,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9855651384475717,
            "MicroF1": 0.9855651384475717,
            "MacroF1": 0.9856470830770891,
            "Memory in Mb": 0.2125759124755859,
            "Time in s": 163.926329
          },
          {
            "step": 11424,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9857305436400244,
            "MicroF1": 0.9857305436400244,
            "MacroF1": 0.9858087969497248,
            "Memory in Mb": 0.2126264572143554,
            "Time in s": 175.222393
          },
          {
            "step": 11832,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9858845406136422,
            "MicroF1": 0.9858845406136422,
            "MacroF1": 0.9859589489459036,
            "Memory in Mb": 0.2131805419921875,
            "Time in s": 186.880683
          },
          {
            "step": 12240,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9861099763052537,
            "MicroF1": 0.9861099763052537,
            "MacroF1": 0.9862068987479334,
            "Memory in Mb": 0.2127275466918945,
            "Time in s": 198.905241
          },
          {
            "step": 12648,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.986241796473472,
            "MicroF1": 0.986241796473472,
            "MacroF1": 0.9863073128720756,
            "Memory in Mb": 0.2132816314697265,
            "Time in s": 211.292323
          },
          {
            "step": 13056,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.985905783224818,
            "MicroF1": 0.985905783224818,
            "MacroF1": 0.9858386074980298,
            "Memory in Mb": 0.2133321762084961,
            "Time in s": 224.041487
          },
          {
            "step": 13464,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9857386912278095,
            "MicroF1": 0.9857386912278095,
            "MacroF1": 0.985725098817589,
            "Memory in Mb": 0.2128791809082031,
            "Time in s": 237.153453
          },
          {
            "step": 13872,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.985725614591594,
            "MicroF1": 0.985725614591594,
            "MacroF1": 0.9857526199764752,
            "Memory in Mb": 0.2134332656860351,
            "Time in s": 250.627393
          },
          {
            "step": 14280,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9845927585965404,
            "MicroF1": 0.9845927585965404,
            "MacroF1": 0.9843691165759658,
            "Memory in Mb": 0.2129802703857422,
            "Time in s": 264.463365
          },
          {
            "step": 14688,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9848845918158916,
            "MicroF1": 0.9848845918158916,
            "MacroF1": 0.9849709956409892,
            "Memory in Mb": 0.2130308151245117,
            "Time in s": 278.661824
          },
          {
            "step": 15096,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9851606492215964,
            "MicroF1": 0.9851606492215964,
            "MacroF1": 0.9852374033885688,
            "Memory in Mb": 0.2135848999023437,
            "Time in s": 293.222186
          },
          {
            "step": 15504,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9843901180416692,
            "MicroF1": 0.9843901180416692,
            "MacroF1": 0.9842921251481088,
            "Memory in Mb": 0.2131319046020507,
            "Time in s": 308.144297
          },
          {
            "step": 15912,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9840362013701212,
            "MicroF1": 0.9840362013701212,
            "MacroF1": 0.9840127534225096,
            "Memory in Mb": 0.2136859893798828,
            "Time in s": 323.428421
          },
          {
            "step": 16320,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.984067651204118,
            "MicroF1": 0.984067651204118,
            "MacroF1": 0.98409717640125,
            "Memory in Mb": 0.2137365341186523,
            "Time in s": 339.073895
          },
          {
            "step": 16728,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9838584324744424,
            "MicroF1": 0.9838584324744424,
            "MacroF1": 0.9838587519327452,
            "Memory in Mb": 0.2132835388183593,
            "Time in s": 355.082258
          },
          {
            "step": 17136,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9840676976947768,
            "MicroF1": 0.9840676976947768,
            "MacroF1": 0.9841085979018744,
            "Memory in Mb": 0.2138376235961914,
            "Time in s": 371.45447
          },
          {
            "step": 17544,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9840962207148152,
            "MicroF1": 0.9840962207148152,
            "MacroF1": 0.9841170088782344,
            "Memory in Mb": 0.2133846282958984,
            "Time in s": 388.187765
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9840120327558354,
            "MicroF1": 0.9840120327558354,
            "MacroF1": 0.98402212072501,
            "Memory in Mb": 0.2134351730346679,
            "Time in s": 405.281273
          },
          {
            "step": 18360,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9842039326760716,
            "MicroF1": 0.9842039326760716,
            "MacroF1": 0.9842275892846344,
            "Memory in Mb": 0.2139892578125,
            "Time in s": 422.735746
          },
          {
            "step": 18768,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.984280918633772,
            "MicroF1": 0.984280918633772,
            "MacroF1": 0.9842944297848302,
            "Memory in Mb": 0.213536262512207,
            "Time in s": 440.550971
          },
          {
            "step": 19176,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9843024771838332,
            "MicroF1": 0.9843024771838332,
            "MacroF1": 0.9843104669951572,
            "Memory in Mb": 0.214090347290039,
            "Time in s": 458.72564299999993
          },
          {
            "step": 19584,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9843742021140786,
            "MicroF1": 0.9843742021140786,
            "MacroF1": 0.9843801024949196,
            "Memory in Mb": 0.2141408920288086,
            "Time in s": 477.2614029999999
          },
          {
            "step": 19992,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.9845430443699664,
            "MicroF1": 0.9845430443699664,
            "MacroF1": 0.984546236206973,
            "Memory in Mb": 0.2136878967285156,
            "Time in s": 496.1581819999999
          },
          {
            "step": 20400,
            "track": "Multiclass classification",
            "model": "k-Nearest Neighbors",
            "dataset": "Keystroke",
            "Accuracy": 0.984509044561008,
            "MicroF1": 0.984509044561008,
            "MacroF1": 0.984507607652182,
            "Memory in Mb": 0.2142419815063476,
            "Time in s": 515.4151939999999
          },
          {
            "step": 46,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.3111111111111111,
            "MicroF1": 0.3111111111111111,
            "MacroF1": 0.2457649726557289,
            "Memory in Mb": 4.137397766113281,
            "Time in s": 1.064188
          },
          {
            "step": 92,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.4835164835164835,
            "MicroF1": 0.4835164835164835,
            "MacroF1": 0.4934752395581889,
            "Memory in Mb": 4.140613555908203,
            "Time in s": 2.631663
          },
          {
            "step": 138,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.5328467153284672,
            "MicroF1": 0.5328467153284672,
            "MacroF1": 0.5528821792646677,
            "Memory in Mb": 4.140277862548828,
            "Time in s": 4.836076
          },
          {
            "step": 184,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.5956284153005464,
            "MicroF1": 0.5956284153005464,
            "MacroF1": 0.614143164890895,
            "Memory in Mb": 4.141227722167969,
            "Time in s": 7.573955
          },
          {
            "step": 230,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.62882096069869,
            "MicroF1": 0.62882096069869,
            "MacroF1": 0.6441389332893815,
            "Memory in Mb": 3.913887023925781,
            "Time in s": 10.681288
          },
          {
            "step": 276,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.64,
            "MicroF1": 0.64,
            "MacroF1": 0.6559607038460422,
            "Memory in Mb": 4.028352737426758,
            "Time in s": 14.131211
          },
          {
            "step": 322,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.6697819314641744,
            "MicroF1": 0.6697819314641744,
            "MacroF1": 0.6706320385346652,
            "Memory in Mb": 4.144774436950684,
            "Time in s": 17.917192
          },
          {
            "step": 368,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.6948228882833788,
            "MicroF1": 0.6948228882833788,
            "MacroF1": 0.6897433526546474,
            "Memory in Mb": 4.144762992858887,
            "Time in s": 22.052519
          },
          {
            "step": 414,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.711864406779661,
            "MicroF1": 0.711864406779661,
            "MacroF1": 0.706570530482581,
            "Memory in Mb": 4.148934364318848,
            "Time in s": 26.547524
          },
          {
            "step": 460,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7145969498910676,
            "MicroF1": 0.7145969498910676,
            "MacroF1": 0.7071122267088653,
            "Memory in Mb": 4.148022651672363,
            "Time in s": 31.388583
          },
          {
            "step": 506,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7247524752475247,
            "MicroF1": 0.7247524752475247,
            "MacroF1": 0.7147973207987898,
            "Memory in Mb": 4.147336006164551,
            "Time in s": 36.558334
          },
          {
            "step": 552,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7295825771324864,
            "MicroF1": 0.7295825771324864,
            "MacroF1": 0.7210771168277493,
            "Memory in Mb": 4.147068977355957,
            "Time in s": 42.057708
          },
          {
            "step": 598,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7336683417085427,
            "MicroF1": 0.7336683417085426,
            "MacroF1": 0.7250288715672424,
            "Memory in Mb": 4.14684009552002,
            "Time in s": 47.89146
          },
          {
            "step": 644,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7325038880248833,
            "MicroF1": 0.7325038880248833,
            "MacroF1": 0.725892488365903,
            "Memory in Mb": 4.150084495544434,
            "Time in s": 54.057887
          },
          {
            "step": 690,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.737300435413643,
            "MicroF1": 0.737300435413643,
            "MacroF1": 0.730253637873586,
            "Memory in Mb": 4.149851799011231,
            "Time in s": 60.540489
          },
          {
            "step": 736,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7387755102040816,
            "MicroF1": 0.7387755102040816,
            "MacroF1": 0.7329631379486717,
            "Memory in Mb": 4.149523735046387,
            "Time in s": 67.34570099999999
          },
          {
            "step": 782,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7439180537772087,
            "MicroF1": 0.7439180537772088,
            "MacroF1": 0.7387105187530085,
            "Memory in Mb": 4.149043083190918,
            "Time in s": 74.478308
          },
          {
            "step": 828,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7460701330108828,
            "MicroF1": 0.7460701330108827,
            "MacroF1": 0.7425025596154724,
            "Memory in Mb": 4.1487531661987305,
            "Time in s": 81.934305
          },
          {
            "step": 874,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7514318442153494,
            "MicroF1": 0.7514318442153494,
            "MacroF1": 0.7467163857842192,
            "Memory in Mb": 4.148730278015137,
            "Time in s": 89.700464
          },
          {
            "step": 920,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.750816104461371,
            "MicroF1": 0.750816104461371,
            "MacroF1": 0.7453933609147307,
            "Memory in Mb": 4.148531913757324,
            "Time in s": 97.776444
          },
          {
            "step": 966,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7512953367875648,
            "MicroF1": 0.7512953367875648,
            "MacroF1": 0.7451117895470661,
            "Memory in Mb": 4.148127555847168,
            "Time in s": 106.157006
          },
          {
            "step": 1012,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7507418397626113,
            "MicroF1": 0.7507418397626113,
            "MacroF1": 0.7449630804815479,
            "Memory in Mb": 4.147826194763184,
            "Time in s": 114.848603
          },
          {
            "step": 1058,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7511825922421949,
            "MicroF1": 0.7511825922421949,
            "MacroF1": 0.7446315489945474,
            "Memory in Mb": 4.149008750915527,
            "Time in s": 123.845956
          },
          {
            "step": 1104,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7533998186763372,
            "MicroF1": 0.7533998186763373,
            "MacroF1": 0.7466082689908061,
            "Memory in Mb": 4.149382591247559,
            "Time in s": 133.146638
          },
          {
            "step": 1150,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7563098346388164,
            "MicroF1": 0.7563098346388164,
            "MacroF1": 0.7491651771194965,
            "Memory in Mb": 4.148917198181152,
            "Time in s": 142.738156
          },
          {
            "step": 1196,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7589958158995815,
            "MicroF1": 0.7589958158995815,
            "MacroF1": 0.7526420027035882,
            "Memory in Mb": 4.148730278015137,
            "Time in s": 152.636035
          },
          {
            "step": 1242,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.75825946817083,
            "MicroF1": 0.7582594681708301,
            "MacroF1": 0.7524016178277559,
            "Memory in Mb": 4.148566246032715,
            "Time in s": 162.845279
          },
          {
            "step": 1288,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7637917637917638,
            "MicroF1": 0.7637917637917638,
            "MacroF1": 0.75666252908711,
            "Memory in Mb": 4.14877986907959,
            "Time in s": 173.368823
          },
          {
            "step": 1334,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7636909227306826,
            "MicroF1": 0.7636909227306825,
            "MacroF1": 0.7569484848610158,
            "Memory in Mb": 4.148688316345215,
            "Time in s": 184.200835
          },
          {
            "step": 1380,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7650471356055112,
            "MicroF1": 0.7650471356055112,
            "MacroF1": 0.7590436403579585,
            "Memory in Mb": 4.1487226486206055,
            "Time in s": 195.341139
          },
          {
            "step": 1426,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.767719298245614,
            "MicroF1": 0.767719298245614,
            "MacroF1": 0.7612112896959209,
            "Memory in Mb": 4.148562431335449,
            "Time in s": 206.790743
          },
          {
            "step": 1472,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7722637661454793,
            "MicroF1": 0.7722637661454793,
            "MacroF1": 0.7640566966433581,
            "Memory in Mb": 4.148623466491699,
            "Time in s": 218.546701
          },
          {
            "step": 1518,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7732366512854317,
            "MicroF1": 0.7732366512854317,
            "MacroF1": 0.7642341334147652,
            "Memory in Mb": 4.148673057556152,
            "Time in s": 230.6041
          },
          {
            "step": 1564,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7735124760076776,
            "MicroF1": 0.7735124760076776,
            "MacroF1": 0.7653316001442942,
            "Memory in Mb": 4.148703575134277,
            "Time in s": 242.961148
          },
          {
            "step": 1610,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7737725295214419,
            "MicroF1": 0.7737725295214419,
            "MacroF1": 0.7647353044337892,
            "Memory in Mb": 4.148566246032715,
            "Time in s": 255.638001
          },
          {
            "step": 1656,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7734138972809668,
            "MicroF1": 0.7734138972809667,
            "MacroF1": 0.7645730180903106,
            "Memory in Mb": 4.148055076599121,
            "Time in s": 268.628995
          },
          {
            "step": 1702,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7724867724867724,
            "MicroF1": 0.7724867724867724,
            "MacroF1": 0.7656182355666586,
            "Memory in Mb": 4.148245811462402,
            "Time in s": 281.916269
          },
          {
            "step": 1748,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7750429307384087,
            "MicroF1": 0.7750429307384087,
            "MacroF1": 0.7677424040514297,
            "Memory in Mb": 4.148360252380371,
            "Time in s": 295.50082
          },
          {
            "step": 1794,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7763524818739542,
            "MicroF1": 0.7763524818739542,
            "MacroF1": 0.7677176136548695,
            "Memory in Mb": 4.148287773132324,
            "Time in s": 309.399686
          },
          {
            "step": 1840,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7775965198477434,
            "MicroF1": 0.7775965198477434,
            "MacroF1": 0.7691578918725354,
            "Memory in Mb": 4.147894859313965,
            "Time in s": 323.61456
          },
          {
            "step": 1886,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7761273209549071,
            "MicroF1": 0.7761273209549071,
            "MacroF1": 0.7681560201617949,
            "Memory in Mb": 4.147856712341309,
            "Time in s": 338.130858
          },
          {
            "step": 1932,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7762817193164163,
            "MicroF1": 0.7762817193164163,
            "MacroF1": 0.7674170460709654,
            "Memory in Mb": 4.147791862487793,
            "Time in s": 352.957512
          },
          {
            "step": 1978,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7769347496206374,
            "MicroF1": 0.7769347496206374,
            "MacroF1": 0.7672843880004774,
            "Memory in Mb": 4.147627830505371,
            "Time in s": 368.093168
          },
          {
            "step": 2024,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7790410281759763,
            "MicroF1": 0.7790410281759763,
            "MacroF1": 0.7681802739952505,
            "Memory in Mb": 4.147582054138184,
            "Time in s": 383.545184
          },
          {
            "step": 2070,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.778153697438376,
            "MicroF1": 0.7781536974383759,
            "MacroF1": 0.7675304391667319,
            "Memory in Mb": 4.147578239440918,
            "Time in s": 399.300197
          },
          {
            "step": 2116,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7787234042553192,
            "MicroF1": 0.778723404255319,
            "MacroF1": 0.7673415220519754,
            "Memory in Mb": 4.147555351257324,
            "Time in s": 415.3667640000001
          },
          {
            "step": 2162,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7797316057380842,
            "MicroF1": 0.7797316057380842,
            "MacroF1": 0.7679341969633587,
            "Memory in Mb": 4.147627830505371,
            "Time in s": 431.738527
          },
          {
            "step": 2208,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7816039873130947,
            "MicroF1": 0.7816039873130947,
            "MacroF1": 0.7687944234581563,
            "Memory in Mb": 4.1476240158081055,
            "Time in s": 448.4359490000001
          },
          {
            "step": 2254,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7785175321793165,
            "MicroF1": 0.7785175321793165,
            "MacroF1": 0.7657018899401807,
            "Memory in Mb": 4.147597312927246,
            "Time in s": 465.434175
          },
          {
            "step": 2300,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7777294475859069,
            "MicroF1": 0.7777294475859068,
            "MacroF1": 0.7649119672933201,
            "Memory in Mb": 4.14768123626709,
            "Time in s": 482.736348
          },
          {
            "step": 1056,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6360189573459716,
            "MicroF1": 0.6360189573459716,
            "MacroF1": 0.5970323052762561,
            "Memory in Mb": 6.54005241394043,
            "Time in s": 11.482596
          },
          {
            "step": 2112,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.62482235907153,
            "MicroF1": 0.62482235907153,
            "MacroF1": 0.5890580890213498,
            "Memory in Mb": 6.540731430053711,
            "Time in s": 32.930503
          },
          {
            "step": 3168,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6157246605620461,
            "MicroF1": 0.6157246605620461,
            "MacroF1": 0.5802533923244892,
            "Memory in Mb": 6.541685104370117,
            "Time in s": 64.407359
          },
          {
            "step": 4224,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6107032914989344,
            "MicroF1": 0.6107032914989344,
            "MacroF1": 0.574850135712032,
            "Memory in Mb": 6.54176139831543,
            "Time in s": 105.889955
          },
          {
            "step": 5280,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.614889183557492,
            "MicroF1": 0.614889183557492,
            "MacroF1": 0.5777842549225518,
            "Memory in Mb": 6.542509078979492,
            "Time in s": 157.317574
          },
          {
            "step": 6336,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.608997632202052,
            "MicroF1": 0.608997632202052,
            "MacroF1": 0.5733157350789627,
            "Memory in Mb": 6.541296005249023,
            "Time in s": 218.706259
          },
          {
            "step": 7392,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6057367068055743,
            "MicroF1": 0.6057367068055743,
            "MacroF1": 0.5703382690867538,
            "Memory in Mb": 6.541265487670898,
            "Time in s": 290.118972
          },
          {
            "step": 8448,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6069610512608027,
            "MicroF1": 0.6069610512608027,
            "MacroF1": 0.5711427916016896,
            "Memory in Mb": 6.541204452514648,
            "Time in s": 371.535386
          },
          {
            "step": 9504,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6039145532989583,
            "MicroF1": 0.6039145532989583,
            "MacroF1": 0.5678102867297488,
            "Memory in Mb": 6.541570663452148,
            "Time in s": 462.975881
          },
          {
            "step": 10560,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6034662373330808,
            "MicroF1": 0.6034662373330808,
            "MacroF1": 0.567425153452482,
            "Memory in Mb": 6.541746139526367,
            "Time in s": 564.403412
          },
          {
            "step": 11616,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6005165733964701,
            "MicroF1": 0.6005165733964701,
            "MacroF1": 0.5651283239572901,
            "Memory in Mb": 6.541906356811523,
            "Time in s": 675.845474
          },
          {
            "step": 12672,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6031883829216321,
            "MicroF1": 0.6031883829216321,
            "MacroF1": 0.5703828979306639,
            "Memory in Mb": 6.542104721069336,
            "Time in s": 797.3461219999999
          },
          {
            "step": 13728,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6152108982297662,
            "MicroF1": 0.6152108982297662,
            "MacroF1": 0.5959760515786451,
            "Memory in Mb": 6.026429176330566,
            "Time in s": 928.242385
          },
          {
            "step": 14784,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6060339579246432,
            "MicroF1": 0.6060339579246432,
            "MacroF1": 0.5869142505177357,
            "Memory in Mb": 6.546758651733398,
            "Time in s": 1068.594912
          },
          {
            "step": 15840,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5713744554580465,
            "MicroF1": 0.5713744554580465,
            "MacroF1": 0.5537658591956377,
            "Memory in Mb": 6.547109603881836,
            "Time in s": 1218.89566
          },
          {
            "step": 16896,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.545546019532406,
            "MicroF1": 0.545546019532406,
            "MacroF1": 0.5286479939306437,
            "Memory in Mb": 6.431658744812012,
            "Time in s": 1379.1626680000002
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.526767311013314,
            "MicroF1": 0.526767311013314,
            "MacroF1": 0.509587529402725,
            "Memory in Mb": 6.54762077331543,
            "Time in s": 1549.227957
          },
          {
            "step": 19008,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.517756615983585,
            "MicroF1": 0.517756615983585,
            "MacroF1": 0.4976462434137419,
            "Memory in Mb": 4.743686676025391,
            "Time in s": 1728.135143
          },
          {
            "step": 20064,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5296815032647162,
            "MicroF1": 0.5296815032647162,
            "MacroF1": 0.5080882715573688,
            "Memory in Mb": 10.447637557983398,
            "Time in s": 1914.608483
          },
          {
            "step": 21120,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.539750935176855,
            "MicroF1": 0.539750935176855,
            "MacroF1": 0.5184934777423561,
            "Memory in Mb": 11.000249862670898,
            "Time in s": 2110.852221
          },
          {
            "step": 22176,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5468771138669674,
            "MicroF1": 0.5468771138669674,
            "MacroF1": 0.5259709774382829,
            "Memory in Mb": 10.998456954956056,
            "Time in s": 2316.603266
          },
          {
            "step": 23232,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5551633593043778,
            "MicroF1": 0.5551633593043778,
            "MacroF1": 0.5340735310276195,
            "Memory in Mb": 12.317106246948242,
            "Time in s": 2531.7014360000003
          },
          {
            "step": 24288,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5615761518507844,
            "MicroF1": 0.5615761518507844,
            "MacroF1": 0.5396852076547555,
            "Memory in Mb": 12.966436386108398,
            "Time in s": 2756.048529
          },
          {
            "step": 25344,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5679280274632048,
            "MicroF1": 0.5679280274632048,
            "MacroF1": 0.5455634192548012,
            "Memory in Mb": 13.622279167175291,
            "Time in s": 2989.801767
          },
          {
            "step": 26400,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5727868479866661,
            "MicroF1": 0.5727868479866661,
            "MacroF1": 0.5496374434570931,
            "Memory in Mb": 13.72577953338623,
            "Time in s": 3232.991513
          },
          {
            "step": 27456,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5754143143325442,
            "MicroF1": 0.5754143143325442,
            "MacroF1": 0.5513680135969626,
            "Memory in Mb": 13.724169731140137,
            "Time in s": 3485.670293
          },
          {
            "step": 28512,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5772859598049875,
            "MicroF1": 0.5772859598049875,
            "MacroF1": 0.5551350356863173,
            "Memory in Mb": 13.7214994430542,
            "Time in s": 3747.946766
          },
          {
            "step": 29568,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.577772516657084,
            "MicroF1": 0.577772516657084,
            "MacroF1": 0.559086133229251,
            "Memory in Mb": 13.720248222351074,
            "Time in s": 4020.178573
          },
          {
            "step": 30624,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.578225516768442,
            "MicroF1": 0.578225516768442,
            "MacroF1": 0.5625516131192055,
            "Memory in Mb": 12.85925006866455,
            "Time in s": 4302.520044
          },
          {
            "step": 31680,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5795637488557088,
            "MicroF1": 0.5795637488557088,
            "MacroF1": 0.5663363640160618,
            "Memory in Mb": 12.858540534973145,
            "Time in s": 4594.920222
          },
          {
            "step": 32736,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5811211241790133,
            "MicroF1": 0.5811211241790133,
            "MacroF1": 0.5696723582178382,
            "Memory in Mb": 12.857670783996582,
            "Time in s": 4897.257331
          },
          {
            "step": 33792,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.575804208221124,
            "MicroF1": 0.575804208221124,
            "MacroF1": 0.5647934119551398,
            "Memory in Mb": 13.070902824401855,
            "Time in s": 5209.806865
          },
          {
            "step": 34848,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5701495107182828,
            "MicroF1": 0.5701495107182828,
            "MacroF1": 0.559068023359177,
            "Memory in Mb": 13.0708646774292,
            "Time in s": 5532.790806
          },
          {
            "step": 35904,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5657744478177311,
            "MicroF1": 0.5657744478177311,
            "MacroF1": 0.5542573482740074,
            "Memory in Mb": 13.072970390319824,
            "Time in s": 5866.228602
          },
          {
            "step": 36960,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5611894261208366,
            "MicroF1": 0.5611894261208366,
            "MacroF1": 0.5493152777162592,
            "Memory in Mb": 13.618464469909668,
            "Time in s": 6210.1901960000005
          },
          {
            "step": 38016,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.558779429172695,
            "MicroF1": 0.558779429172695,
            "MacroF1": 0.5463982360776033,
            "Memory in Mb": 13.620196342468262,
            "Time in s": 6564.708960000001
          },
          {
            "step": 39072,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5546825010877633,
            "MicroF1": 0.5546825010877633,
            "MacroF1": 0.5426283860139581,
            "Memory in Mb": 14.406596183776855,
            "Time in s": 6929.528370000001
          },
          {
            "step": 40128,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5542153662122761,
            "MicroF1": 0.5542153662122761,
            "MacroF1": 0.5429626632180721,
            "Memory in Mb": 15.257904052734377,
            "Time in s": 7304.280479000001
          },
          {
            "step": 41184,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5541364155112547,
            "MicroF1": 0.5541364155112547,
            "MacroF1": 0.5435420562964656,
            "Memory in Mb": 15.358447074890137,
            "Time in s": 7688.491709000001
          },
          {
            "step": 42240,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5542981604678141,
            "MicroF1": 0.5542981604678141,
            "MacroF1": 0.5443914000180358,
            "Memory in Mb": 15.357035636901855,
            "Time in s": 8082.149982000001
          },
          {
            "step": 43296,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.554151749624668,
            "MicroF1": 0.554151749624668,
            "MacroF1": 0.5448486588729108,
            "Memory in Mb": 13.518179893493652,
            "Time in s": 8485.318247000001
          },
          {
            "step": 44352,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5536290049829767,
            "MicroF1": 0.5536290049829767,
            "MacroF1": 0.5448029815059025,
            "Memory in Mb": 13.742095947265623,
            "Time in s": 8897.925593000002
          },
          {
            "step": 45408,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5541436342414165,
            "MicroF1": 0.5541436342414165,
            "MacroF1": 0.5454957405719211,
            "Memory in Mb": 14.286569595336914,
            "Time in s": 9319.685836000002
          },
          {
            "step": 46464,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5553020683124207,
            "MicroF1": 0.5553020683124207,
            "MacroF1": 0.546961663735647,
            "Memory in Mb": 15.266200065612791,
            "Time in s": 9750.326660000002
          },
          {
            "step": 47520,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5579662871693428,
            "MicroF1": 0.5579662871693428,
            "MacroF1": 0.5498636684303295,
            "Memory in Mb": 14.323851585388184,
            "Time in s": 10190.236686000002
          },
          {
            "step": 48576,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5627586206896552,
            "MicroF1": 0.5627586206896552,
            "MacroF1": 0.5545030394801858,
            "Memory in Mb": 14.950955390930176,
            "Time in s": 10639.519510000002
          },
          {
            "step": 49632,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5677701436602124,
            "MicroF1": 0.5677701436602124,
            "MacroF1": 0.5591808574875289,
            "Memory in Mb": 15.350643157958984,
            "Time in s": 11098.085262000002
          },
          {
            "step": 50688,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5730463432438297,
            "MicroF1": 0.5730463432438297,
            "MacroF1": 0.5639878919164368,
            "Memory in Mb": 16.015583038330078,
            "Time in s": 11565.627356000005
          },
          {
            "step": 51744,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5791894555785324,
            "MicroF1": 0.5791894555785324,
            "MacroF1": 0.5695807960578061,
            "Memory in Mb": 16.33325481414795,
            "Time in s": 12041.828478000005
          },
          {
            "step": 52800,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5794238527244834,
            "MicroF1": 0.5794238527244834,
            "MacroF1": 0.5701364277094956,
            "Memory in Mb": 15.444610595703123,
            "Time in s": 12525.893185000004
          },
          {
            "step": 408,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9828009828009828,
            "MicroF1": 0.9828009828009828,
            "MacroF1": 0.6067632850241546,
            "Memory in Mb": 2.2430334091186523,
            "Time in s": 1.187339
          },
          {
            "step": 816,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9411042944785276,
            "MicroF1": 0.9411042944785276,
            "MacroF1": 0.7377235942917068,
            "Memory in Mb": 3.19038200378418,
            "Time in s": 4.698555
          },
          {
            "step": 1224,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8879803761242846,
            "MicroF1": 0.8879803761242846,
            "MacroF1": 0.873420796574987,
            "Memory in Mb": 4.134529113769531,
            "Time in s": 10.466629
          },
          {
            "step": 1632,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8988350705088902,
            "MicroF1": 0.8988350705088902,
            "MacroF1": 0.8792834531664682,
            "Memory in Mb": 5.086630821228027,
            "Time in s": 18.824542
          },
          {
            "step": 2040,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8950465914664051,
            "MicroF1": 0.8950465914664051,
            "MacroF1": 0.8828407845486113,
            "Memory in Mb": 6.147420883178711,
            "Time in s": 30.316821
          },
          {
            "step": 2448,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.856559051900286,
            "MicroF1": 0.856559051900286,
            "MacroF1": 0.8543242501248514,
            "Memory in Mb": 6.513773918151856,
            "Time in s": 45.490026
          },
          {
            "step": 2856,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8640980735551663,
            "MicroF1": 0.8640980735551663,
            "MacroF1": 0.8525227127090282,
            "Memory in Mb": 7.461577415466309,
            "Time in s": 64.600537
          },
          {
            "step": 3264,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.855654305853509,
            "MicroF1": 0.855654305853509,
            "MacroF1": 0.8307453339686874,
            "Memory in Mb": 8.407819747924805,
            "Time in s": 88.28106700000001
          },
          {
            "step": 3672,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8469081994007083,
            "MicroF1": 0.8469081994007084,
            "MacroF1": 0.8445950801753395,
            "Memory in Mb": 9.06855297088623,
            "Time in s": 117.068452
          },
          {
            "step": 4080,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.839911743074283,
            "MicroF1": 0.839911743074283,
            "MacroF1": 0.8273018519986841,
            "Memory in Mb": 10.241823196411133,
            "Time in s": 151.440392
          },
          {
            "step": 4488,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8279474036104302,
            "MicroF1": 0.8279474036104302,
            "MacroF1": 0.8381848634946416,
            "Memory in Mb": 11.187081336975098,
            "Time in s": 191.919368
          },
          {
            "step": 4896,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8294177732379979,
            "MicroF1": 0.8294177732379979,
            "MacroF1": 0.8370944525285466,
            "Memory in Mb": 8.72095775604248,
            "Time in s": 237.683057
          },
          {
            "step": 5304,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.832736187063926,
            "MicroF1": 0.832736187063926,
            "MacroF1": 0.8304665020850452,
            "Memory in Mb": 9.573864936828612,
            "Time in s": 288.722663
          },
          {
            "step": 5712,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8254246191560147,
            "MicroF1": 0.8254246191560147,
            "MacroF1": 0.8318293629616008,
            "Memory in Mb": 10.363824844360352,
            "Time in s": 345.486273
          },
          {
            "step": 6120,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8238274227815002,
            "MicroF1": 0.8238274227815002,
            "MacroF1": 0.8134447828524414,
            "Memory in Mb": 11.40614128112793,
            "Time in s": 408.45138
          },
          {
            "step": 6528,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8043511567335683,
            "MicroF1": 0.8043511567335683,
            "MacroF1": 0.8054460603633147,
            "Memory in Mb": 12.15697956085205,
            "Time in s": 478.205713
          },
          {
            "step": 6936,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8005767844268205,
            "MicroF1": 0.8005767844268206,
            "MacroF1": 0.8067791986535922,
            "Memory in Mb": 11.58870792388916,
            "Time in s": 555.142755
          },
          {
            "step": 7344,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8081165736075173,
            "MicroF1": 0.8081165736075173,
            "MacroF1": 0.8106639227074198,
            "Memory in Mb": 12.00939655303955,
            "Time in s": 638.5654069999999
          },
          {
            "step": 7752,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8097019739388466,
            "MicroF1": 0.8097019739388466,
            "MacroF1": 0.8127585051729247,
            "Memory in Mb": 13.156713485717772,
            "Time in s": 728.9317779999999
          },
          {
            "step": 8160,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8134575315602403,
            "MicroF1": 0.8134575315602401,
            "MacroF1": 0.8148392057777913,
            "Memory in Mb": 13.88283348083496,
            "Time in s": 826.9412199999999
          },
          {
            "step": 8568,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.817672464106455,
            "MicroF1": 0.817672464106455,
            "MacroF1": 0.8208026583224199,
            "Memory in Mb": 15.191060066223145,
            "Time in s": 933.116233
          },
          {
            "step": 8976,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8213927576601672,
            "MicroF1": 0.8213927576601672,
            "MacroF1": 0.8243856825821874,
            "Memory in Mb": 16.25563907623291,
            "Time in s": 1048.180156
          },
          {
            "step": 9384,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8219119684535863,
            "MicroF1": 0.8219119684535864,
            "MacroF1": 0.8243183344026902,
            "Memory in Mb": 17.071918487548828,
            "Time in s": 1172.5966959999998
          },
          {
            "step": 9792,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8224900418751915,
            "MicroF1": 0.8224900418751915,
            "MacroF1": 0.8248306232761192,
            "Memory in Mb": 18.233381271362305,
            "Time in s": 1306.7477949999998
          },
          {
            "step": 10200,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.819197960584371,
            "MicroF1": 0.819197960584371,
            "MacroF1": 0.8170259665463304,
            "Memory in Mb": 19.36789894104004,
            "Time in s": 1451.4292829999995
          },
          {
            "step": 10608,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.805788630149901,
            "MicroF1": 0.8057886301499011,
            "MacroF1": 0.8022367569175978,
            "Memory in Mb": 20.506345748901367,
            "Time in s": 1607.1296199999997
          },
          {
            "step": 11016,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.802088061733999,
            "MicroF1": 0.8020880617339992,
            "MacroF1": 0.8038074645550285,
            "Memory in Mb": 19.16464138031006,
            "Time in s": 1773.9352609999996
          },
          {
            "step": 11424,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8018909218243894,
            "MicroF1": 0.8018909218243894,
            "MacroF1": 0.8005729972530424,
            "Memory in Mb": 19.133995056152344,
            "Time in s": 1951.246828
          },
          {
            "step": 11832,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8002704758684811,
            "MicroF1": 0.800270475868481,
            "MacroF1": 0.8004166941842216,
            "Memory in Mb": 20.079543113708496,
            "Time in s": 2139.48669
          },
          {
            "step": 12240,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8035787237519405,
            "MicroF1": 0.8035787237519405,
            "MacroF1": 0.8060123607032721,
            "Memory in Mb": 17.827110290527344,
            "Time in s": 2338.9088759999995
          },
          {
            "step": 12648,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8088084130623864,
            "MicroF1": 0.8088084130623864,
            "MacroF1": 0.8108606005777994,
            "Memory in Mb": 15.629319190979004,
            "Time in s": 2547.8446989999998
          },
          {
            "step": 13056,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8079662964381463,
            "MicroF1": 0.8079662964381463,
            "MacroF1": 0.8077709771623751,
            "Memory in Mb": 16.39698314666748,
            "Time in s": 2766.5130659999995
          },
          {
            "step": 13464,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8068038327267325,
            "MicroF1": 0.8068038327267325,
            "MacroF1": 0.807905549135964,
            "Memory in Mb": 17.24907112121582,
            "Time in s": 2995.4784789999994
          },
          {
            "step": 13872,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.810107418354841,
            "MicroF1": 0.810107418354841,
            "MacroF1": 0.8115061911206084,
            "Memory in Mb": 18.023069381713867,
            "Time in s": 3234.9291429999994
          },
          {
            "step": 14280,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.813432313187198,
            "MicroF1": 0.813432313187198,
            "MacroF1": 0.814709519180665,
            "Memory in Mb": 19.25601577758789,
            "Time in s": 3485.414837
          },
          {
            "step": 14688,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8169810036086335,
            "MicroF1": 0.8169810036086335,
            "MacroF1": 0.8183348126971706,
            "Memory in Mb": 19.79203414916992,
            "Time in s": 3747.476331
          },
          {
            "step": 15096,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8210665783371978,
            "MicroF1": 0.8210665783371978,
            "MacroF1": 0.8224533109934684,
            "Memory in Mb": 20.76410961151123,
            "Time in s": 4021.7266099999993
          },
          {
            "step": 15504,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8227439850351544,
            "MicroF1": 0.8227439850351544,
            "MacroF1": 0.8236860076332361,
            "Memory in Mb": 21.70703220367432,
            "Time in s": 4308.908110999999
          },
          {
            "step": 15912,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8177361573754006,
            "MicroF1": 0.8177361573754006,
            "MacroF1": 0.8170714187961161,
            "Memory in Mb": 22.97433376312256,
            "Time in s": 4609.388637999999
          },
          {
            "step": 16320,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8135915190881794,
            "MicroF1": 0.8135915190881794,
            "MacroF1": 0.8136474897036394,
            "Memory in Mb": 23.700613975524902,
            "Time in s": 4923.859395999999
          },
          {
            "step": 16728,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8133556525378132,
            "MicroF1": 0.8133556525378132,
            "MacroF1": 0.8142218072403056,
            "Memory in Mb": 24.764389038085938,
            "Time in s": 5252.522612999999
          },
          {
            "step": 17136,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8092792529909542,
            "MicroF1": 0.8092792529909542,
            "MacroF1": 0.8090411402278314,
            "Memory in Mb": 26.198601722717285,
            "Time in s": 5595.909021
          },
          {
            "step": 17544,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8062475061278003,
            "MicroF1": 0.8062475061278003,
            "MacroF1": 0.8065701979489333,
            "Memory in Mb": 25.60452175140381,
            "Time in s": 5954.543495
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8078101498523759,
            "MicroF1": 0.8078101498523759,
            "MacroF1": 0.8084559739072698,
            "Memory in Mb": 26.552149772644043,
            "Time in s": 6328.536349
          },
          {
            "step": 18360,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8103927229151915,
            "MicroF1": 0.8103927229151915,
            "MacroF1": 0.8111272646261444,
            "Memory in Mb": 27.50138759613037,
            "Time in s": 6718.347567
          },
          {
            "step": 18768,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.813022859274258,
            "MicroF1": 0.813022859274258,
            "MacroF1": 0.8138485204649677,
            "Memory in Mb": 28.353083610534668,
            "Time in s": 7124.657721
          },
          {
            "step": 19176,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8090743155149935,
            "MicroF1": 0.8090743155149935,
            "MacroF1": 0.8093701596568051,
            "Memory in Mb": 29.32584285736084,
            "Time in s": 7547.838048
          },
          {
            "step": 19584,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8110606137976817,
            "MicroF1": 0.8110606137976817,
            "MacroF1": 0.8116953495842238,
            "Memory in Mb": 30.33370780944824,
            "Time in s": 7988.342591
          },
          {
            "step": 19992,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8081136511430144,
            "MicroF1": 0.8081136511430144,
            "MacroF1": 0.8084718836746521,
            "Memory in Mb": 31.28043556213379,
            "Time in s": 8446.728501
          },
          {
            "step": 20400,
            "track": "Multiclass classification",
            "model": "ADWIN Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8058238148928869,
            "MicroF1": 0.805823814892887,
            "MacroF1": 0.8062504565207905,
            "Memory in Mb": 32.1812219619751,
            "Time in s": 8923.60629
          },
          {
            "step": 46,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.1111111111111111,
            "MicroF1": 0.1111111111111111,
            "MacroF1": 0.0815018315018315,
            "Memory in Mb": 3.44619369506836,
            "Time in s": 0.804099
          },
          {
            "step": 92,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.2307692307692307,
            "MicroF1": 0.2307692307692307,
            "MacroF1": 0.2226391771283412,
            "Memory in Mb": 4.129319190979004,
            "Time in s": 2.027411
          },
          {
            "step": 138,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.4233576642335766,
            "MicroF1": 0.4233576642335766,
            "MacroF1": 0.4463537718619156,
            "Memory in Mb": 4.129193305969238,
            "Time in s": 3.599985
          },
          {
            "step": 184,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.5355191256830601,
            "MicroF1": 0.5355191256830601,
            "MacroF1": 0.5617062146473911,
            "Memory in Mb": 4.129368782043457,
            "Time in s": 5.452412
          },
          {
            "step": 230,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.5938864628820961,
            "MicroF1": 0.5938864628820961,
            "MacroF1": 0.6236530662596055,
            "Memory in Mb": 4.12935733795166,
            "Time in s": 7.651963
          },
          {
            "step": 276,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.6290909090909091,
            "MicroF1": 0.6290909090909091,
            "MacroF1": 0.6558170665459355,
            "Memory in Mb": 4.129300117492676,
            "Time in s": 10.207424
          },
          {
            "step": 322,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.660436137071651,
            "MicroF1": 0.660436137071651,
            "MacroF1": 0.6785747202615152,
            "Memory in Mb": 4.128628730773926,
            "Time in s": 13.08877
          },
          {
            "step": 368,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.6920980926430518,
            "MicroF1": 0.6920980926430518,
            "MacroF1": 0.7041680355881775,
            "Memory in Mb": 4.12868595123291,
            "Time in s": 16.291428
          },
          {
            "step": 414,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.7167070217917676,
            "MicroF1": 0.7167070217917676,
            "MacroF1": 0.7259075149442815,
            "Memory in Mb": 4.128170967102051,
            "Time in s": 19.832325
          },
          {
            "step": 460,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.7254901960784313,
            "MicroF1": 0.7254901960784313,
            "MacroF1": 0.732501171084948,
            "Memory in Mb": 4.128491401672363,
            "Time in s": 23.76135
          },
          {
            "step": 506,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.7386138613861386,
            "MicroF1": 0.7386138613861386,
            "MacroF1": 0.7428621938273078,
            "Memory in Mb": 4.128743171691895,
            "Time in s": 28.024553
          },
          {
            "step": 552,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.7422867513611615,
            "MicroF1": 0.7422867513611615,
            "MacroF1": 0.7453719085253248,
            "Memory in Mb": 4.128548622131348,
            "Time in s": 32.646215
          },
          {
            "step": 598,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.7487437185929648,
            "MicroF1": 0.7487437185929648,
            "MacroF1": 0.7504522188790484,
            "Memory in Mb": 4.128659248352051,
            "Time in s": 37.596323
          },
          {
            "step": 644,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.7465007776049767,
            "MicroF1": 0.7465007776049767,
            "MacroF1": 0.7482323503576439,
            "Memory in Mb": 4.128731727600098,
            "Time in s": 42.92857
          },
          {
            "step": 690,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.7489114658925979,
            "MicroF1": 0.748911465892598,
            "MacroF1": 0.7488472102580619,
            "Memory in Mb": 4.128785133361816,
            "Time in s": 48.576103
          },
          {
            "step": 736,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.7523809523809524,
            "MicroF1": 0.7523809523809524,
            "MacroF1": 0.75182837230991,
            "Memory in Mb": 4.1286211013793945,
            "Time in s": 54.551686
          },
          {
            "step": 782,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.7541613316261203,
            "MicroF1": 0.7541613316261204,
            "MacroF1": 0.7531089046321313,
            "Memory in Mb": 4.128552436828613,
            "Time in s": 60.838379
          },
          {
            "step": 828,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.7557436517533253,
            "MicroF1": 0.7557436517533253,
            "MacroF1": 0.7552013614952863,
            "Memory in Mb": 4.128499031066895,
            "Time in s": 67.464563
          },
          {
            "step": 874,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.7617411225658648,
            "MicroF1": 0.7617411225658649,
            "MacroF1": 0.7601066395856337,
            "Memory in Mb": 4.128571510314941,
            "Time in s": 74.378096
          },
          {
            "step": 920,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.763873775843308,
            "MicroF1": 0.763873775843308,
            "MacroF1": 0.7623480483274478,
            "Memory in Mb": 4.1285905838012695,
            "Time in s": 81.591089
          },
          {
            "step": 966,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.7678756476683938,
            "MicroF1": 0.7678756476683938,
            "MacroF1": 0.7646598072570266,
            "Memory in Mb": 4.128613471984863,
            "Time in s": 89.10581599999999
          },
          {
            "step": 1012,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.7705242334322453,
            "MicroF1": 0.7705242334322453,
            "MacroF1": 0.7668271197983112,
            "Memory in Mb": 4.128720283508301,
            "Time in s": 96.930863
          },
          {
            "step": 1058,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.7757805108798487,
            "MicroF1": 0.7757805108798487,
            "MacroF1": 0.7714920336037776,
            "Memory in Mb": 4.128579139709473,
            "Time in s": 105.051956
          },
          {
            "step": 1104,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.7760652765185857,
            "MicroF1": 0.7760652765185856,
            "MacroF1": 0.7719206139767609,
            "Memory in Mb": 4.128727912902832,
            "Time in s": 113.491292
          },
          {
            "step": 1150,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.7789382071366405,
            "MicroF1": 0.7789382071366405,
            "MacroF1": 0.7750313949659529,
            "Memory in Mb": 4.128632545471191,
            "Time in s": 122.217257
          },
          {
            "step": 1196,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.7849372384937239,
            "MicroF1": 0.7849372384937239,
            "MacroF1": 0.782000389047251,
            "Memory in Mb": 4.128678321838379,
            "Time in s": 131.239072
          },
          {
            "step": 1242,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.7856567284448026,
            "MicroF1": 0.7856567284448026,
            "MacroF1": 0.7827470902102025,
            "Memory in Mb": 4.128628730773926,
            "Time in s": 140.604336
          },
          {
            "step": 1288,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.7894327894327894,
            "MicroF1": 0.7894327894327894,
            "MacroF1": 0.785982924599392,
            "Memory in Mb": 4.128533363342285,
            "Time in s": 150.25346299999998
          },
          {
            "step": 1334,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.7906976744186046,
            "MicroF1": 0.7906976744186046,
            "MacroF1": 0.7876424482584368,
            "Memory in Mb": 4.128628730773926,
            "Time in s": 160.232262
          },
          {
            "step": 1380,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.7933284989122552,
            "MicroF1": 0.7933284989122552,
            "MacroF1": 0.7906471924204203,
            "Memory in Mb": 4.128582954406738,
            "Time in s": 170.496442
          },
          {
            "step": 1426,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.7978947368421052,
            "MicroF1": 0.7978947368421052,
            "MacroF1": 0.7945020166797493,
            "Memory in Mb": 4.128670692443848,
            "Time in s": 181.024488
          },
          {
            "step": 1472,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.8028552005438477,
            "MicroF1": 0.8028552005438477,
            "MacroF1": 0.7982243751921435,
            "Memory in Mb": 4.128663063049316,
            "Time in s": 191.820653
          },
          {
            "step": 1518,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.8035596572181938,
            "MicroF1": 0.8035596572181938,
            "MacroF1": 0.7981876534181911,
            "Memory in Mb": 4.1286821365356445,
            "Time in s": 202.942587
          },
          {
            "step": 1564,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.8035828534868842,
            "MicroF1": 0.8035828534868842,
            "MacroF1": 0.798634974540431,
            "Memory in Mb": 4.128708839416504,
            "Time in s": 214.370001
          },
          {
            "step": 1610,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.8048477315102548,
            "MicroF1": 0.8048477315102549,
            "MacroF1": 0.7997380784882049,
            "Memory in Mb": 4.128571510314941,
            "Time in s": 226.09596
          },
          {
            "step": 1656,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.8066465256797583,
            "MicroF1": 0.8066465256797583,
            "MacroF1": 0.80161945439383,
            "Memory in Mb": 4.128567695617676,
            "Time in s": 238.115396
          },
          {
            "step": 1702,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.8059964726631393,
            "MicroF1": 0.8059964726631393,
            "MacroF1": 0.8024858564723996,
            "Memory in Mb": 4.128705024719238,
            "Time in s": 250.457722
          },
          {
            "step": 1748,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.8070978820835718,
            "MicroF1": 0.8070978820835718,
            "MacroF1": 0.8029124203507954,
            "Memory in Mb": 4.128613471984863,
            "Time in s": 263.107237
          },
          {
            "step": 1794,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.8081427774679308,
            "MicroF1": 0.8081427774679307,
            "MacroF1": 0.8029834045630978,
            "Memory in Mb": 4.12865161895752,
            "Time in s": 276.028168
          },
          {
            "step": 1840,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.8069603045133225,
            "MicroF1": 0.8069603045133223,
            "MacroF1": 0.8019276227162541,
            "Memory in Mb": 4.128785133361816,
            "Time in s": 289.253241
          },
          {
            "step": 1886,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.8053050397877984,
            "MicroF1": 0.8053050397877984,
            "MacroF1": 0.8006727596367826,
            "Memory in Mb": 4.1285905838012695,
            "Time in s": 302.793979
          },
          {
            "step": 1932,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.8047643707923355,
            "MicroF1": 0.8047643707923355,
            "MacroF1": 0.7995493059800364,
            "Memory in Mb": 4.128586769104004,
            "Time in s": 316.637791
          },
          {
            "step": 1978,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.8057663125948407,
            "MicroF1": 0.8057663125948407,
            "MacroF1": 0.8003960406612561,
            "Memory in Mb": 4.12862491607666,
            "Time in s": 330.791782
          },
          {
            "step": 2024,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.8072170044488384,
            "MicroF1": 0.8072170044488384,
            "MacroF1": 0.8005625942078284,
            "Memory in Mb": 4.1286211013793945,
            "Time in s": 345.23827800000004
          },
          {
            "step": 2070,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.8066698888351861,
            "MicroF1": 0.8066698888351861,
            "MacroF1": 0.8002110568368,
            "Memory in Mb": 4.128506660461426,
            "Time in s": 360.0031
          },
          {
            "step": 2116,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.807565011820331,
            "MicroF1": 0.807565011820331,
            "MacroF1": 0.8005131307885663,
            "Memory in Mb": 4.128533363342285,
            "Time in s": 375.059377
          },
          {
            "step": 2162,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.8079592781119852,
            "MicroF1": 0.8079592781119852,
            "MacroF1": 0.8006755955605838,
            "Memory in Mb": 4.128510475158691,
            "Time in s": 390.39582400000006
          },
          {
            "step": 2208,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.8087902129587675,
            "MicroF1": 0.8087902129587675,
            "MacroF1": 0.8009921695193861,
            "Memory in Mb": 4.128510475158691,
            "Time in s": 405.9976
          },
          {
            "step": 2254,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.8060363959165557,
            "MicroF1": 0.8060363959165557,
            "MacroF1": 0.7987732120640717,
            "Memory in Mb": 4.128533363342285,
            "Time in s": 421.95304
          },
          {
            "step": 2300,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "ImageSegments",
            "Accuracy": 0.8051326663766856,
            "MicroF1": 0.8051326663766856,
            "MacroF1": 0.7980778928096751,
            "Memory in Mb": 4.128533363342285,
            "Time in s": 438.2190000000001
          },
          {
            "step": 1056,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.6360189573459716,
            "MicroF1": 0.6360189573459716,
            "MacroF1": 0.5992691812827112,
            "Memory in Mb": 6.522543907165527,
            "Time in s": 11.237994
          },
          {
            "step": 2112,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.6110847939365229,
            "MicroF1": 0.6110847939365229,
            "MacroF1": 0.5773210074897359,
            "Memory in Mb": 6.522406578063965,
            "Time in s": 32.531305
          },
          {
            "step": 3168,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.6043574360593622,
            "MicroF1": 0.6043574360593622,
            "MacroF1": 0.5704368753709179,
            "Memory in Mb": 6.521971702575684,
            "Time in s": 63.87417000000001
          },
          {
            "step": 4224,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.6014681506038362,
            "MicroF1": 0.6014681506038362,
            "MacroF1": 0.5676969561642587,
            "Memory in Mb": 6.521697044372559,
            "Time in s": 105.215764
          },
          {
            "step": 5280,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.6057965523773442,
            "MicroF1": 0.6057965523773442,
            "MacroF1": 0.5710016183775801,
            "Memory in Mb": 6.521697044372559,
            "Time in s": 156.441097
          },
          {
            "step": 6336,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5966850828729282,
            "MicroF1": 0.5966850828729282,
            "MacroF1": 0.5635903588556204,
            "Memory in Mb": 6.521857261657715,
            "Time in s": 217.680264
          },
          {
            "step": 7392,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5957245298335814,
            "MicroF1": 0.5957245298335814,
            "MacroF1": 0.5625002603439991,
            "Memory in Mb": 6.52231502532959,
            "Time in s": 288.908384
          },
          {
            "step": 8448,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5982005445720374,
            "MicroF1": 0.5982005445720374,
            "MacroF1": 0.5646892369665863,
            "Memory in Mb": 6.522658348083496,
            "Time in s": 370.082682
          },
          {
            "step": 9504,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.596337998526781,
            "MicroF1": 0.596337998526781,
            "MacroF1": 0.5627085514562804,
            "Memory in Mb": 6.523001670837402,
            "Time in s": 461.259323
          },
          {
            "step": 10560,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5965527038545316,
            "MicroF1": 0.5965527038545316,
            "MacroF1": 0.5631320282838163,
            "Memory in Mb": 6.523184776306152,
            "Time in s": 562.3727710000001
          },
          {
            "step": 11616,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5953508394317693,
            "MicroF1": 0.5953508394317693,
            "MacroF1": 0.562671447170627,
            "Memory in Mb": 6.523184776306152,
            "Time in s": 673.482974
          },
          {
            "step": 12672,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5979796385447084,
            "MicroF1": 0.5979796385447084,
            "MacroF1": 0.5680559575776837,
            "Memory in Mb": 6.522841453552246,
            "Time in s": 794.598712
          },
          {
            "step": 13728,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.610767101333139,
            "MicroF1": 0.610767101333139,
            "MacroF1": 0.5941277335666079,
            "Memory in Mb": 6.522337913513184,
            "Time in s": 925.382156
          },
          {
            "step": 14784,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.6019752418318338,
            "MicroF1": 0.6019752418318338,
            "MacroF1": 0.5851264744797858,
            "Memory in Mb": 6.522246360778809,
            "Time in s": 1066.407983
          },
          {
            "step": 15840,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5705536965717533,
            "MicroF1": 0.5705536965717533,
            "MacroF1": 0.5545059657048704,
            "Memory in Mb": 6.522475242614746,
            "Time in s": 1217.74448
          },
          {
            "step": 16896,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.548091151228174,
            "MicroF1": 0.548091151228174,
            "MacroF1": 0.5320735507355622,
            "Memory in Mb": 6.522887229919434,
            "Time in s": 1378.917078
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5307225224221492,
            "MicroF1": 0.5307225224221492,
            "MacroF1": 0.5138536287616571,
            "Memory in Mb": 6.523138999938965,
            "Time in s": 1549.794627
          },
          {
            "step": 19008,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5182827379386542,
            "MicroF1": 0.5182827379386542,
            "MacroF1": 0.4990809738484312,
            "Memory in Mb": 6.523367881774902,
            "Time in s": 1730.586936
          },
          {
            "step": 20064,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5182176145142801,
            "MicroF1": 0.5182176145142801,
            "MacroF1": 0.497867701567998,
            "Memory in Mb": 8.711265563964844,
            "Time in s": 1921.389763
          },
          {
            "step": 21120,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5272503432927695,
            "MicroF1": 0.5272503432927695,
            "MacroF1": 0.5067114684709674,
            "Memory in Mb": 15.55071258544922,
            "Time in s": 2121.77156
          },
          {
            "step": 22176,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.533032694475761,
            "MicroF1": 0.533032694475761,
            "MacroF1": 0.5127471323280748,
            "Memory in Mb": 16.9340763092041,
            "Time in s": 2331.759567
          },
          {
            "step": 23232,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5410442942619775,
            "MicroF1": 0.5410442942619775,
            "MacroF1": 0.5207771198745245,
            "Memory in Mb": 17.15799903869629,
            "Time in s": 2551.0181620000003
          },
          {
            "step": 24288,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5459710956478775,
            "MicroF1": 0.5459710956478775,
            "MacroF1": 0.5251711652768186,
            "Memory in Mb": 17.155046463012695,
            "Time in s": 2778.8449820000005
          },
          {
            "step": 25344,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5532099593576135,
            "MicroF1": 0.5532099593576135,
            "MacroF1": 0.5314216535856217,
            "Memory in Mb": 17.15360450744629,
            "Time in s": 3015.0560080000005
          },
          {
            "step": 26400,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5607788173794462,
            "MicroF1": 0.5607788173794462,
            "MacroF1": 0.5375130024626694,
            "Memory in Mb": 17.265982627868652,
            "Time in s": 3259.2678140000003
          },
          {
            "step": 27456,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5667091604443635,
            "MicroF1": 0.5667091604443635,
            "MacroF1": 0.5418496825562071,
            "Memory in Mb": 17.378803253173828,
            "Time in s": 3511.9238520000004
          },
          {
            "step": 28512,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5692890463329943,
            "MicroF1": 0.5692890463329943,
            "MacroF1": 0.5455529487931667,
            "Memory in Mb": 17.379924774169922,
            "Time in s": 3773.544178
          },
          {
            "step": 29568,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5688436432509216,
            "MicroF1": 0.5688436432509216,
            "MacroF1": 0.5481992899375988,
            "Memory in Mb": 17.380359649658203,
            "Time in s": 4045.250953
          },
          {
            "step": 30624,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5687228553701467,
            "MicroF1": 0.5687228553701467,
            "MacroF1": 0.5505043481720591,
            "Memory in Mb": 17.380290985107422,
            "Time in s": 4327.317909
          },
          {
            "step": 31680,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5691467533697402,
            "MicroF1": 0.5691467533697402,
            "MacroF1": 0.5529220328647554,
            "Memory in Mb": 17.37978744506836,
            "Time in s": 4619.673811000001
          },
          {
            "step": 32736,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5703986558729189,
            "MicroF1": 0.5703986558729189,
            "MacroF1": 0.5556828084411201,
            "Memory in Mb": 17.37948989868164,
            "Time in s": 4922.186926
          },
          {
            "step": 33792,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5650025154626972,
            "MicroF1": 0.5650025154626972,
            "MacroF1": 0.5507695387439543,
            "Memory in Mb": 17.604686737060547,
            "Time in s": 5235.061695
          },
          {
            "step": 34848,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5587568513788849,
            "MicroF1": 0.5587568513788849,
            "MacroF1": 0.5445559443415654,
            "Memory in Mb": 18.05030918121338,
            "Time in s": 5558.17951
          },
          {
            "step": 35904,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.554215525165028,
            "MicroF1": 0.554215525165028,
            "MacroF1": 0.5396701176441828,
            "Memory in Mb": 18.150463104248047,
            "Time in s": 5891.808229
          },
          {
            "step": 36960,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5490408290267594,
            "MicroF1": 0.5490408290267594,
            "MacroF1": 0.5342475234810463,
            "Memory in Mb": 19.505443572998047,
            "Time in s": 6235.988455000001
          },
          {
            "step": 38016,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5476522425358411,
            "MicroF1": 0.5476522425358411,
            "MacroF1": 0.5324130893403342,
            "Memory in Mb": 21.324423789978027,
            "Time in s": 6590.662011
          },
          {
            "step": 39072,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5427810908346344,
            "MicroF1": 0.5427810908346344,
            "MacroF1": 0.5280992603544316,
            "Memory in Mb": 22.076537132263184,
            "Time in s": 6955.865912
          },
          {
            "step": 40128,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5417300072270541,
            "MicroF1": 0.5417300072270541,
            "MacroF1": 0.5282649533846114,
            "Memory in Mb": 22.738471031188965,
            "Time in s": 7330.794870000001
          },
          {
            "step": 41184,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5417283830706845,
            "MicroF1": 0.5417283830706845,
            "MacroF1": 0.5295529576867488,
            "Memory in Mb": 23.16494464874268,
            "Time in s": 7714.553666000001
          },
          {
            "step": 42240,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5419635881531286,
            "MicroF1": 0.5419635881531286,
            "MacroF1": 0.5308394560628455,
            "Memory in Mb": 23.606464385986328,
            "Time in s": 8107.062243
          },
          {
            "step": 43296,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5438734264926666,
            "MicroF1": 0.5438734264926666,
            "MacroF1": 0.5334569208328087,
            "Memory in Mb": 23.706249237060547,
            "Time in s": 8507.305821
          },
          {
            "step": 44352,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5453315596040675,
            "MicroF1": 0.5453315596040675,
            "MacroF1": 0.5354029875943346,
            "Memory in Mb": 24.299342155456543,
            "Time in s": 8915.303832
          },
          {
            "step": 45408,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.547140308762966,
            "MicroF1": 0.547140308762966,
            "MacroF1": 0.5374156745075451,
            "Memory in Mb": 24.818781852722168,
            "Time in s": 9330.915411
          },
          {
            "step": 46464,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5492327228116997,
            "MicroF1": 0.5492327228116997,
            "MacroF1": 0.5397202270950943,
            "Memory in Mb": 25.030532836914062,
            "Time in s": 9754.309095
          },
          {
            "step": 47520,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5481596834950231,
            "MicroF1": 0.5481596834950231,
            "MacroF1": 0.5387960204161004,
            "Memory in Mb": 25.669864654541016,
            "Time in s": 10186.398449
          },
          {
            "step": 48576,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5455275347400926,
            "MicroF1": 0.5455275347400926,
            "MacroF1": 0.5361266295596548,
            "Memory in Mb": 25.668834686279297,
            "Time in s": 10627.539344
          },
          {
            "step": 49632,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5440752755334367,
            "MicroF1": 0.5440752755334367,
            "MacroF1": 0.534604738581891,
            "Memory in Mb": 26.10423469543457,
            "Time in s": 11077.630938
          },
          {
            "step": 50688,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5484443742971571,
            "MicroF1": 0.5484443742971571,
            "MacroF1": 0.538570218508335,
            "Memory in Mb": 27.224528312683105,
            "Time in s": 11536.66667
          },
          {
            "step": 51744,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5534081904798717,
            "MicroF1": 0.5534081904798717,
            "MacroF1": 0.5429607704191827,
            "Memory in Mb": 28.06478881835937,
            "Time in s": 12004.483564
          },
          {
            "step": 52800,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Insects",
            "Accuracy": 0.5540824636830243,
            "MicroF1": 0.5540824636830243,
            "MacroF1": 0.543927330204892,
            "Memory in Mb": 28.290183067321777,
            "Time in s": 12481.284297
          },
          {
            "step": 408,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.9877149877149876,
            "MicroF1": 0.9877149877149876,
            "MacroF1": 0.7696139476961394,
            "Memory in Mb": 2.1705713272094727,
            "Time in s": 1.169245
          },
          {
            "step": 816,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.988957055214724,
            "MicroF1": 0.988957055214724,
            "MacroF1": 0.9592655637573824,
            "Memory in Mb": 2.994051933288574,
            "Time in s": 4.333642
          },
          {
            "step": 1224,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.9852820932134096,
            "MicroF1": 0.9852820932134096,
            "MacroF1": 0.9482751483180804,
            "Memory in Mb": 4.729727745056152,
            "Time in s": 12.346709
          },
          {
            "step": 1632,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.9822194972409564,
            "MicroF1": 0.9822194972409564,
            "MacroF1": 0.9509896151723368,
            "Memory in Mb": 5.999781608581543,
            "Time in s": 25.091939
          },
          {
            "step": 2040,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.9725355566454144,
            "MicroF1": 0.9725355566454144,
            "MacroF1": 0.928775026512405,
            "Memory in Mb": 7.915155410766602,
            "Time in s": 43.495481
          },
          {
            "step": 2448,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.9550469963220268,
            "MicroF1": 0.9550469963220268,
            "MacroF1": 0.9404929408648164,
            "Memory in Mb": 9.98500156402588,
            "Time in s": 68.010219
          },
          {
            "step": 2856,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.9516637478108582,
            "MicroF1": 0.9516637478108582,
            "MacroF1": 0.9265706247083844,
            "Memory in Mb": 13.281692504882812,
            "Time in s": 95.835033
          },
          {
            "step": 3264,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.9457554397793442,
            "MicroF1": 0.9457554397793442,
            "MacroF1": 0.9273434636455652,
            "Memory in Mb": 16.35391616821289,
            "Time in s": 127.205902
          },
          {
            "step": 3672,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.9417052574230454,
            "MicroF1": 0.9417052574230454,
            "MacroF1": 0.925978466853896,
            "Memory in Mb": 18.91156578063965,
            "Time in s": 163.340984
          },
          {
            "step": 4080,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.9355234126011276,
            "MicroF1": 0.9355234126011276,
            "MacroF1": 0.9181372267911062,
            "Memory in Mb": 22.338744163513184,
            "Time in s": 205.235807
          },
          {
            "step": 4488,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.931580120347671,
            "MicroF1": 0.931580120347671,
            "MacroF1": 0.9327276252021246,
            "Memory in Mb": 25.31070709228516,
            "Time in s": 252.959286
          },
          {
            "step": 4896,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.9303370786516854,
            "MicroF1": 0.9303370786516854,
            "MacroF1": 0.9257176086775136,
            "Memory in Mb": 28.274658203125,
            "Time in s": 306.688529
          },
          {
            "step": 5304,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.925325287573072,
            "MicroF1": 0.925325287573072,
            "MacroF1": 0.9165251784293146,
            "Memory in Mb": 32.214202880859375,
            "Time in s": 367.4329340000001
          },
          {
            "step": 5712,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.9226054981614428,
            "MicroF1": 0.9226054981614428,
            "MacroF1": 0.9209111845314156,
            "Memory in Mb": 34.49547863006592,
            "Time in s": 436.878813
          },
          {
            "step": 6120,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.9181238764504004,
            "MicroF1": 0.9181238764504004,
            "MacroF1": 0.9091206319047904,
            "Memory in Mb": 38.86995029449463,
            "Time in s": 513.269664
          },
          {
            "step": 6528,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.9129768653286348,
            "MicroF1": 0.9129768653286348,
            "MacroF1": 0.9114007831703168,
            "Memory in Mb": 42.32428169250488,
            "Time in s": 602.212119
          },
          {
            "step": 6936,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.9114635904830568,
            "MicroF1": 0.9114635904830568,
            "MacroF1": 0.9134311944430068,
            "Memory in Mb": 44.19167232513428,
            "Time in s": 700.8190040000001
          },
          {
            "step": 7344,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.9116165055154568,
            "MicroF1": 0.9116165055154568,
            "MacroF1": 0.9097332482243848,
            "Memory in Mb": 44.84274196624756,
            "Time in s": 807.0835780000001
          },
          {
            "step": 7752,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.9112372597084248,
            "MicroF1": 0.9112372597084248,
            "MacroF1": 0.9111242959524108,
            "Memory in Mb": 46.84857273101807,
            "Time in s": 921.356409
          },
          {
            "step": 8160,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.9094251746537566,
            "MicroF1": 0.9094251746537566,
            "MacroF1": 0.9076128910354778,
            "Memory in Mb": 51.16739654541016,
            "Time in s": 1044.662094
          },
          {
            "step": 8568,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.9066184195167504,
            "MicroF1": 0.9066184195167504,
            "MacroF1": 0.9066450469749988,
            "Memory in Mb": 55.86186981201172,
            "Time in s": 1177.672199
          },
          {
            "step": 8976,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.9056267409470752,
            "MicroF1": 0.9056267409470752,
            "MacroF1": 0.906335380756654,
            "Memory in Mb": 58.41574668884277,
            "Time in s": 1322.047745
          },
          {
            "step": 9384,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.9030160929340296,
            "MicroF1": 0.9030160929340296,
            "MacroF1": 0.9022077684947396,
            "Memory in Mb": 62.26175117492676,
            "Time in s": 1477.298762
          },
          {
            "step": 9792,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.8986824634868757,
            "MicroF1": 0.8986824634868757,
            "MacroF1": 0.8984090939041232,
            "Memory in Mb": 65.20822143554688,
            "Time in s": 1646.0177680000002
          },
          {
            "step": 10200,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.8947936072163938,
            "MicroF1": 0.8947936072163937,
            "MacroF1": 0.8926613887647973,
            "Memory in Mb": 69.96524620056152,
            "Time in s": 1829.707382
          },
          {
            "step": 10608,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.8881870462901857,
            "MicroF1": 0.8881870462901857,
            "MacroF1": 0.8865702773222168,
            "Memory in Mb": 73.60436820983887,
            "Time in s": 2032.787653
          },
          {
            "step": 11016,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.8849750340444847,
            "MicroF1": 0.8849750340444847,
            "MacroF1": 0.8859866133359942,
            "Memory in Mb": 77.43610191345215,
            "Time in s": 2252.324419
          },
          {
            "step": 11424,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.8823426420379935,
            "MicroF1": 0.8823426420379935,
            "MacroF1": 0.8811651142625456,
            "Memory in Mb": 81.9732666015625,
            "Time in s": 2486.561888
          },
          {
            "step": 11832,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.8789620488547037,
            "MicroF1": 0.8789620488547037,
            "MacroF1": 0.8783725809837211,
            "Memory in Mb": 87.50129985809326,
            "Time in s": 2738.033946
          },
          {
            "step": 12240,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.8803823841817142,
            "MicroF1": 0.8803823841817142,
            "MacroF1": 0.8815469015078649,
            "Memory in Mb": 88.71501064300537,
            "Time in s": 3002.626444
          },
          {
            "step": 12648,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.878231991776706,
            "MicroF1": 0.878231991776706,
            "MacroF1": 0.8774838611192476,
            "Memory in Mb": 93.73236656188963,
            "Time in s": 3282.43662
          },
          {
            "step": 13056,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.8737648410570663,
            "MicroF1": 0.8737648410570663,
            "MacroF1": 0.8731746930338653,
            "Memory in Mb": 98.2464723587036,
            "Time in s": 3580.919978
          },
          {
            "step": 13464,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.872168164599272,
            "MicroF1": 0.872168164599272,
            "MacroF1": 0.8726091982990895,
            "Memory in Mb": 102.17141056060792,
            "Time in s": 3896.583194
          },
          {
            "step": 13872,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.8693677456564054,
            "MicroF1": 0.8693677456564054,
            "MacroF1": 0.869586320653203,
            "Memory in Mb": 106.91088581085204,
            "Time in s": 4229.831886
          },
          {
            "step": 14280,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.8653967364661391,
            "MicroF1": 0.8653967364661391,
            "MacroF1": 0.8650950015616714,
            "Memory in Mb": 111.61231708526611,
            "Time in s": 4581.249889
          },
          {
            "step": 14688,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.8650507251310683,
            "MicroF1": 0.8650507251310683,
            "MacroF1": 0.8661026270223636,
            "Memory in Mb": 115.34651947021484,
            "Time in s": 4948.063453
          },
          {
            "step": 15096,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.8667770785028155,
            "MicroF1": 0.8667770785028155,
            "MacroF1": 0.8680260482593213,
            "Memory in Mb": 118.24315452575684,
            "Time in s": 5329.445739
          },
          {
            "step": 15504,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.8673805069986454,
            "MicroF1": 0.8673805069986454,
            "MacroF1": 0.8683176015232197,
            "Memory in Mb": 121.64087104797365,
            "Time in s": 5727.804435999999
          },
          {
            "step": 15912,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.8630507196279303,
            "MicroF1": 0.8630507196279303,
            "MacroF1": 0.8630078198630903,
            "Memory in Mb": 126.92564392089844,
            "Time in s": 6152.632774
          },
          {
            "step": 16320,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.8588761566272444,
            "MicroF1": 0.8588761566272444,
            "MacroF1": 0.8591617021179835,
            "Memory in Mb": 132.20891761779785,
            "Time in s": 6602.417036
          },
          {
            "step": 16728,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.8546661086865547,
            "MicroF1": 0.8546661086865547,
            "MacroF1": 0.8551483908890608,
            "Memory in Mb": 137.6782627105713,
            "Time in s": 7072.757626
          },
          {
            "step": 17136,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.850539830755763,
            "MicroF1": 0.850539830755763,
            "MacroF1": 0.8506967692771638,
            "Memory in Mb": 139.40350437164307,
            "Time in s": 7574.584865
          },
          {
            "step": 17544,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.8456934389785099,
            "MicroF1": 0.8456934389785099,
            "MacroF1": 0.845900693576748,
            "Memory in Mb": 144.32332038879397,
            "Time in s": 8103.947966
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.8442983677789538,
            "MicroF1": 0.8442983677789538,
            "MacroF1": 0.8449597117669952,
            "Memory in Mb": 149.55280876159668,
            "Time in s": 8653.396068
          },
          {
            "step": 18360,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.8462879241788769,
            "MicroF1": 0.8462879241788769,
            "MacroF1": 0.8472097835611015,
            "Memory in Mb": 154.20918083190918,
            "Time in s": 9220.958091
          },
          {
            "step": 18768,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.8472851281504769,
            "MicroF1": 0.8472851281504769,
            "MacroF1": 0.8482490326871865,
            "Memory in Mb": 158.50969696044922,
            "Time in s": 9805.946569
          },
          {
            "step": 19176,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.845632333767927,
            "MicroF1": 0.845632333767927,
            "MacroF1": 0.8464906194356719,
            "Memory in Mb": 163.69990730285645,
            "Time in s": 10414.74471
          },
          {
            "step": 19584,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.8470612265740693,
            "MicroF1": 0.8470612265740693,
            "MacroF1": 0.8481337525939465,
            "Memory in Mb": 167.61861419677734,
            "Time in s": 11041.283472
          },
          {
            "step": 19992,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.8455805112300535,
            "MicroF1": 0.8455805112300535,
            "MacroF1": 0.8467102165017473,
            "Memory in Mb": 172.522611618042,
            "Time in s": 11691.217597
          },
          {
            "step": 20400,
            "track": "Multiclass classification",
            "model": "AdaBoost",
            "dataset": "Keystroke",
            "Accuracy": 0.8424922790332859,
            "MicroF1": 0.8424922790332859,
            "MacroF1": 0.8436347186891262,
            "Memory in Mb": 177.38492488861084,
            "Time in s": 12366.859336
          },
          {
            "step": 46,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.3111111111111111,
            "MicroF1": 0.3111111111111111,
            "MacroF1": 0.2457649726557289,
            "Memory in Mb": 4.181334495544434,
            "Time in s": 1.091398
          },
          {
            "step": 92,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.4835164835164835,
            "MicroF1": 0.4835164835164835,
            "MacroF1": 0.4934752395581889,
            "Memory in Mb": 4.184550285339356,
            "Time in s": 2.768304
          },
          {
            "step": 138,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.5328467153284672,
            "MicroF1": 0.5328467153284672,
            "MacroF1": 0.5528821792646677,
            "Memory in Mb": 4.184275627136231,
            "Time in s": 4.818241
          },
          {
            "step": 184,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.5956284153005464,
            "MicroF1": 0.5956284153005464,
            "MacroF1": 0.614143164890895,
            "Memory in Mb": 4.184859275817871,
            "Time in s": 7.25269
          },
          {
            "step": 230,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.62882096069869,
            "MicroF1": 0.62882096069869,
            "MacroF1": 0.6441389332893815,
            "Memory in Mb": 4.184233665466309,
            "Time in s": 10.061851
          },
          {
            "step": 276,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.64,
            "MicroF1": 0.64,
            "MacroF1": 0.6559607038460422,
            "Memory in Mb": 4.184771537780762,
            "Time in s": 13.254704
          },
          {
            "step": 322,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.6666666666666666,
            "MicroF1": 0.6666666666666666,
            "MacroF1": 0.6673617488913626,
            "Memory in Mb": 4.184481620788574,
            "Time in s": 16.809438
          },
          {
            "step": 368,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.6948228882833788,
            "MicroF1": 0.6948228882833788,
            "MacroF1": 0.6911959597548877,
            "Memory in Mb": 4.184699058532715,
            "Time in s": 20.719577
          },
          {
            "step": 414,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.711864406779661,
            "MicroF1": 0.711864406779661,
            "MacroF1": 0.7079630503641953,
            "Memory in Mb": 4.185038566589356,
            "Time in s": 25.02075
          },
          {
            "step": 460,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7124183006535948,
            "MicroF1": 0.7124183006535948,
            "MacroF1": 0.7065500352371009,
            "Memory in Mb": 4.184954643249512,
            "Time in s": 29.692393
          },
          {
            "step": 506,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7207920792079208,
            "MicroF1": 0.7207920792079208,
            "MacroF1": 0.7127593158348896,
            "Memory in Mb": 4.184813499450684,
            "Time in s": 34.691523
          },
          {
            "step": 552,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7259528130671506,
            "MicroF1": 0.7259528130671506,
            "MacroF1": 0.7192025503807162,
            "Memory in Mb": 4.184779167175293,
            "Time in s": 40.034435
          },
          {
            "step": 598,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7319932998324958,
            "MicroF1": 0.7319932998324957,
            "MacroF1": 0.7251188986558661,
            "Memory in Mb": 4.185019493103027,
            "Time in s": 45.725681
          },
          {
            "step": 644,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7309486780715396,
            "MicroF1": 0.7309486780715396,
            "MacroF1": 0.7259740406437202,
            "Memory in Mb": 4.184813499450684,
            "Time in s": 51.77256499999999
          },
          {
            "step": 690,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7358490566037735,
            "MicroF1": 0.7358490566037735,
            "MacroF1": 0.7304359912942561,
            "Memory in Mb": 4.184943199157715,
            "Time in s": 58.14505499999999
          },
          {
            "step": 736,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7374149659863946,
            "MicroF1": 0.7374149659863947,
            "MacroF1": 0.7331499347170709,
            "Memory in Mb": 4.185004234313965,
            "Time in s": 64.846683
          },
          {
            "step": 782,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7426376440460948,
            "MicroF1": 0.7426376440460948,
            "MacroF1": 0.7385597120510639,
            "Memory in Mb": 4.184893608093262,
            "Time in s": 71.895225
          },
          {
            "step": 828,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7436517533252721,
            "MicroF1": 0.7436517533252721,
            "MacroF1": 0.7412375783772317,
            "Memory in Mb": 4.184882164001465,
            "Time in s": 79.27328899999999
          },
          {
            "step": 874,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7491408934707904,
            "MicroF1": 0.7491408934707904,
            "MacroF1": 0.7454343548790068,
            "Memory in Mb": 4.185431480407715,
            "Time in s": 86.96748099999999
          },
          {
            "step": 920,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7486398258977149,
            "MicroF1": 0.7486398258977149,
            "MacroF1": 0.7441307384051415,
            "Memory in Mb": 4.185576438903809,
            "Time in s": 94.98591
          },
          {
            "step": 966,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7492227979274612,
            "MicroF1": 0.749222797927461,
            "MacroF1": 0.7439306216964365,
            "Memory in Mb": 4.185370445251465,
            "Time in s": 103.316786
          },
          {
            "step": 1012,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7487636003956478,
            "MicroF1": 0.7487636003956478,
            "MacroF1": 0.7437900284473965,
            "Memory in Mb": 4.185484886169434,
            "Time in s": 111.977749
          },
          {
            "step": 1058,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.750236518448439,
            "MicroF1": 0.7502365184484389,
            "MacroF1": 0.7448138061687654,
            "Memory in Mb": 4.185519218444824,
            "Time in s": 120.953125
          },
          {
            "step": 1104,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7524932003626473,
            "MicroF1": 0.7524932003626473,
            "MacroF1": 0.7468314646869902,
            "Memory in Mb": 4.185484886169434,
            "Time in s": 130.25113499999998
          },
          {
            "step": 1150,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7554395126196692,
            "MicroF1": 0.7554395126196692,
            "MacroF1": 0.7493227137357602,
            "Memory in Mb": 4.185664176940918,
            "Time in s": 139.83544499999996
          },
          {
            "step": 1196,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7581589958158996,
            "MicroF1": 0.7581589958158996,
            "MacroF1": 0.7527652773681007,
            "Memory in Mb": 4.185568809509277,
            "Time in s": 149.73636899999997
          },
          {
            "step": 1242,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7574536663980661,
            "MicroF1": 0.7574536663980661,
            "MacroF1": 0.7525915384194215,
            "Memory in Mb": 4.185683250427246,
            "Time in s": 159.94850499999995
          },
          {
            "step": 1288,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7622377622377622,
            "MicroF1": 0.7622377622377621,
            "MacroF1": 0.7563448085202399,
            "Memory in Mb": 4.185866355895996,
            "Time in s": 170.48539899999994
          },
          {
            "step": 1334,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7621905476369092,
            "MicroF1": 0.7621905476369092,
            "MacroF1": 0.7566636999776912,
            "Memory in Mb": 4.186026573181152,
            "Time in s": 181.34153899999995
          },
          {
            "step": 1380,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7635968092820885,
            "MicroF1": 0.7635968092820886,
            "MacroF1": 0.7587252257765656,
            "Memory in Mb": 4.1860761642456055,
            "Time in s": 192.51808799999995
          },
          {
            "step": 1426,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7663157894736842,
            "MicroF1": 0.7663157894736842,
            "MacroF1": 0.7609139797315135,
            "Memory in Mb": 4.186099052429199,
            "Time in s": 204.01178799999997
          },
          {
            "step": 1472,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7709041468388851,
            "MicroF1": 0.7709041468388851,
            "MacroF1": 0.763768994920769,
            "Memory in Mb": 4.186240196228027,
            "Time in s": 215.81223399999996
          },
          {
            "step": 1518,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7719182597231378,
            "MicroF1": 0.7719182597231378,
            "MacroF1": 0.7639714255563932,
            "Memory in Mb": 4.186617851257324,
            "Time in s": 227.92141499999997
          },
          {
            "step": 1564,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7722328854766475,
            "MicroF1": 0.7722328854766475,
            "MacroF1": 0.765072133508071,
            "Memory in Mb": 4.186800956726074,
            "Time in s": 240.33772499999995
          },
          {
            "step": 1610,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7725295214418894,
            "MicroF1": 0.7725295214418892,
            "MacroF1": 0.764505787280341,
            "Memory in Mb": 4.186892509460449,
            "Time in s": 253.082857
          },
          {
            "step": 1656,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7716012084592145,
            "MicroF1": 0.7716012084592145,
            "MacroF1": 0.7634170612719107,
            "Memory in Mb": 4.1867780685424805,
            "Time in s": 266.155857
          },
          {
            "step": 1702,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7713109935332157,
            "MicroF1": 0.7713109935332157,
            "MacroF1": 0.7652815676598499,
            "Memory in Mb": 4.187075614929199,
            "Time in s": 279.522325
          },
          {
            "step": 1748,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.77389811104751,
            "MicroF1": 0.77389811104751,
            "MacroF1": 0.7674409436090757,
            "Memory in Mb": 4.187258720397949,
            "Time in s": 293.187056
          },
          {
            "step": 1794,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7752370329057445,
            "MicroF1": 0.7752370329057446,
            "MacroF1": 0.7674318582149376,
            "Memory in Mb": 4.1872968673706055,
            "Time in s": 307.180551
          },
          {
            "step": 1840,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7765089722675367,
            "MicroF1": 0.7765089722675368,
            "MacroF1": 0.7688731808749575,
            "Memory in Mb": 4.187228202819824,
            "Time in s": 321.497423
          },
          {
            "step": 1886,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7750663129973475,
            "MicroF1": 0.7750663129973475,
            "MacroF1": 0.7678921362145585,
            "Memory in Mb": 4.187155723571777,
            "Time in s": 336.11504499999995
          },
          {
            "step": 1932,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7752459865354738,
            "MicroF1": 0.7752459865354739,
            "MacroF1": 0.7671636716269125,
            "Memory in Mb": 4.187251091003418,
            "Time in s": 351.053915
          },
          {
            "step": 1978,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7759231158320687,
            "MicroF1": 0.7759231158320687,
            "MacroF1": 0.7670573130332382,
            "Memory in Mb": 4.187151908874512,
            "Time in s": 366.316945
          },
          {
            "step": 2024,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7775580820563519,
            "MicroF1": 0.7775580820563519,
            "MacroF1": 0.7671264358471986,
            "Memory in Mb": 4.187129020690918,
            "Time in s": 381.896141
          },
          {
            "step": 2070,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.77670372160464,
            "MicroF1": 0.7767037216046399,
            "MacroF1": 0.7665050383810529,
            "Memory in Mb": 4.187205314636231,
            "Time in s": 397.783945
          },
          {
            "step": 2116,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7773049645390071,
            "MicroF1": 0.7773049645390071,
            "MacroF1": 0.7663404166149341,
            "Memory in Mb": 4.187205314636231,
            "Time in s": 413.995271
          },
          {
            "step": 2162,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7783433595557612,
            "MicroF1": 0.7783433595557612,
            "MacroF1": 0.7669657147488861,
            "Memory in Mb": 4.187277793884277,
            "Time in s": 430.524022
          },
          {
            "step": 2208,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.780244676030811,
            "MicroF1": 0.780244676030811,
            "MacroF1": 0.7678552364681829,
            "Memory in Mb": 4.187273979187012,
            "Time in s": 447.387318
          },
          {
            "step": 2254,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7776298268974701,
            "MicroF1": 0.7776298268974701,
            "MacroF1": 0.7652407320979201,
            "Memory in Mb": 4.187224388122559,
            "Time in s": 464.558569
          },
          {
            "step": 2300,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7768595041322314,
            "MicroF1": 0.7768595041322314,
            "MacroF1": 0.7644610611003249,
            "Memory in Mb": 4.18729305267334,
            "Time in s": 482.036228
          },
          {
            "step": 1056,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6360189573459716,
            "MicroF1": 0.6360189573459716,
            "MacroF1": 0.5970323052762561,
            "Memory in Mb": 6.583989143371582,
            "Time in s": 11.628861
          },
          {
            "step": 2112,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.62482235907153,
            "MicroF1": 0.62482235907153,
            "MacroF1": 0.5890580890213498,
            "Memory in Mb": 6.584485054016113,
            "Time in s": 33.423105
          },
          {
            "step": 3168,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6157246605620461,
            "MicroF1": 0.6157246605620461,
            "MacroF1": 0.5802533923244892,
            "Memory in Mb": 6.58519458770752,
            "Time in s": 65.495981
          },
          {
            "step": 4224,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6107032914989344,
            "MicroF1": 0.6107032914989344,
            "MacroF1": 0.574850135712032,
            "Memory in Mb": 6.585576057434082,
            "Time in s": 107.816588
          },
          {
            "step": 5280,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.615078613373745,
            "MicroF1": 0.615078613373745,
            "MacroF1": 0.5779184071248228,
            "Memory in Mb": 6.5863847732543945,
            "Time in s": 160.258759
          },
          {
            "step": 6336,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6091554853985793,
            "MicroF1": 0.6091554853985793,
            "MacroF1": 0.5734262289926554,
            "Memory in Mb": 6.586209297180176,
            "Time in s": 222.866435
          },
          {
            "step": 7392,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6058720064943851,
            "MicroF1": 0.6058720064943851,
            "MacroF1": 0.5704339658550047,
            "Memory in Mb": 6.585629463195801,
            "Time in s": 295.76759400000003
          },
          {
            "step": 8448,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6070794364863265,
            "MicroF1": 0.6070794364863265,
            "MacroF1": 0.5712261057542335,
            "Memory in Mb": 6.585507392883301,
            "Time in s": 378.90789100000006
          },
          {
            "step": 9504,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6040197832263495,
            "MicroF1": 0.6040197832263495,
            "MacroF1": 0.567883906637128,
            "Memory in Mb": 6.585629463195801,
            "Time in s": 472.31115100000005
          },
          {
            "step": 10560,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6035609432711431,
            "MicroF1": 0.6035609432711431,
            "MacroF1": 0.5674913890030829,
            "Memory in Mb": 6.5859880447387695,
            "Time in s": 575.935454
          },
          {
            "step": 11616,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6006026689625484,
            "MicroF1": 0.6006026689625484,
            "MacroF1": 0.5651886352361905,
            "Memory in Mb": 6.585965156555176,
            "Time in s": 689.792083
          },
          {
            "step": 12672,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6032673032909794,
            "MicroF1": 0.6032673032909794,
            "MacroF1": 0.5704386423232538,
            "Memory in Mb": 6.585919380187988,
            "Time in s": 813.953016
          },
          {
            "step": 13728,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6147738034530488,
            "MicroF1": 0.6147738034530488,
            "MacroF1": 0.5955647708468143,
            "Memory in Mb": 6.584591865539551,
            "Time in s": 948.29318
          },
          {
            "step": 14784,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6052222147060813,
            "MicroF1": 0.6052222147060813,
            "MacroF1": 0.586323857604342,
            "Memory in Mb": 6.583569526672363,
            "Time in s": 1092.793426
          },
          {
            "step": 15840,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.570427425973862,
            "MicroF1": 0.570427425973862,
            "MacroF1": 0.5530515395071289,
            "Memory in Mb": 6.584805488586426,
            "Time in s": 1247.37522
          },
          {
            "step": 16896,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5441254809115122,
            "MicroF1": 0.5441254809115122,
            "MacroF1": 0.5274626123277456,
            "Memory in Mb": 6.5833024978637695,
            "Time in s": 1412.138663
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5247061445044844,
            "MicroF1": 0.5247061445044844,
            "MacroF1": 0.5077849244821269,
            "Memory in Mb": 6.58421802520752,
            "Time in s": 1587.005582
          },
          {
            "step": 19008,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5143368232756353,
            "MicroF1": 0.5143368232756353,
            "MacroF1": 0.4945891921842289,
            "Memory in Mb": 5.490016937255859,
            "Time in s": 1771.530278
          },
          {
            "step": 20064,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5203110202860988,
            "MicroF1": 0.5203110202860988,
            "MacroF1": 0.4996705647403201,
            "Memory in Mb": 13.561949729919434,
            "Time in s": 1966.283082
          },
          {
            "step": 21120,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5285288129172783,
            "MicroF1": 0.5285288129172783,
            "MacroF1": 0.5082662721949724,
            "Memory in Mb": 14.33274745941162,
            "Time in s": 2175.200087
          },
          {
            "step": 22176,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5345208568207441,
            "MicroF1": 0.5345208568207441,
            "MacroF1": 0.5149076376433322,
            "Memory in Mb": 14.875703811645508,
            "Time in s": 2397.3423569999995
          },
          {
            "step": 23232,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5431104989023288,
            "MicroF1": 0.5431104989023288,
            "MacroF1": 0.5234265380967914,
            "Memory in Mb": 14.787662506103516,
            "Time in s": 2632.304625
          },
          {
            "step": 24288,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.550253221888253,
            "MicroF1": 0.550253221888253,
            "MacroF1": 0.5298759738824472,
            "Memory in Mb": 16.32009983062744,
            "Time in s": 2880.082024
          },
          {
            "step": 25344,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5564455668231859,
            "MicroF1": 0.5564455668231859,
            "MacroF1": 0.5355827199778521,
            "Memory in Mb": 16.310463905334473,
            "Time in s": 3141.997661
          },
          {
            "step": 26400,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5614985416114247,
            "MicroF1": 0.5614985416114247,
            "MacroF1": 0.5398687013453174,
            "Memory in Mb": 16.303704261779785,
            "Time in s": 3417.905768
          },
          {
            "step": 27456,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5647787288289929,
            "MicroF1": 0.5647787288289929,
            "MacroF1": 0.5421799635248432,
            "Memory in Mb": 15.336288452148438,
            "Time in s": 3708.118695
          },
          {
            "step": 28512,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5680965241485743,
            "MicroF1": 0.5680965241485743,
            "MacroF1": 0.5473162851674372,
            "Memory in Mb": 13.81827449798584,
            "Time in s": 4011.685358
          },
          {
            "step": 29568,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5701626813677411,
            "MicroF1": 0.5701626813677411,
            "MacroF1": 0.5529817475842932,
            "Memory in Mb": 11.429868698120115,
            "Time in s": 4328.316425
          },
          {
            "step": 30624,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5724455474643242,
            "MicroF1": 0.5724455474643242,
            "MacroF1": 0.5586057023406553,
            "Memory in Mb": 9.4625883102417,
            "Time in s": 4656.70214
          },
          {
            "step": 31680,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5750181508254679,
            "MicroF1": 0.5750181508254679,
            "MacroF1": 0.5636300266647484,
            "Memory in Mb": 9.46137523651123,
            "Time in s": 4996.240557
          },
          {
            "step": 32736,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5782190316175347,
            "MicroF1": 0.5782190316175347,
            "MacroF1": 0.5684825891024486,
            "Memory in Mb": 9.4603910446167,
            "Time in s": 5346.796886
          },
          {
            "step": 33792,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5756858335059631,
            "MicroF1": 0.5756858335059631,
            "MacroF1": 0.5663669622675245,
            "Memory in Mb": 7.919375419616699,
            "Time in s": 5710.001071
          },
          {
            "step": 34848,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5754871294516027,
            "MicroF1": 0.5754871294516027,
            "MacroF1": 0.5660193869557423,
            "Memory in Mb": 7.260138511657715,
            "Time in s": 6084.365793
          },
          {
            "step": 35904,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5764142272233518,
            "MicroF1": 0.5764142272233518,
            "MacroF1": 0.5665362650344427,
            "Memory in Mb": 6.60003662109375,
            "Time in s": 6469.590309
          },
          {
            "step": 36960,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.575908439081144,
            "MicroF1": 0.575908439081144,
            "MacroF1": 0.5657420280651625,
            "Memory in Mb": 6.597938537597656,
            "Time in s": 6865.284174
          },
          {
            "step": 38016,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5767723267131396,
            "MicroF1": 0.5767723267131396,
            "MacroF1": 0.5661330182942309,
            "Memory in Mb": 6.597076416015625,
            "Time in s": 7271.517818
          },
          {
            "step": 39072,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5764889560031737,
            "MicroF1": 0.5764889560031737,
            "MacroF1": 0.5659501482422926,
            "Memory in Mb": 6.593917846679688,
            "Time in s": 7688.178552
          },
          {
            "step": 40128,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5734792035287961,
            "MicroF1": 0.5734792035287961,
            "MacroF1": 0.5636824355748769,
            "Memory in Mb": 8.576746940612793,
            "Time in s": 8115.281794
          },
          {
            "step": 41184,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5726634776485443,
            "MicroF1": 0.5726634776485443,
            "MacroF1": 0.563417094879665,
            "Memory in Mb": 8.779536247253418,
            "Time in s": 8552.341759
          },
          {
            "step": 42240,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5723383602831507,
            "MicroF1": 0.5723383602831507,
            "MacroF1": 0.5635995837049609,
            "Memory in Mb": 10.21227741241455,
            "Time in s": 8998.890652
          },
          {
            "step": 43296,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5718212264695692,
            "MicroF1": 0.5718212264695692,
            "MacroF1": 0.5636175088230181,
            "Memory in Mb": 10.21111011505127,
            "Time in s": 9455.260723
          },
          {
            "step": 44352,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.571125791977633,
            "MicroF1": 0.571125791977633,
            "MacroF1": 0.5633830644644046,
            "Memory in Mb": 10.209759712219238,
            "Time in s": 9921.572486
          },
          {
            "step": 45408,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5712555332878191,
            "MicroF1": 0.5712555332878191,
            "MacroF1": 0.5638292127585011,
            "Memory in Mb": 11.4169340133667,
            "Time in s": 10398.038321
          },
          {
            "step": 46464,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5728429072595399,
            "MicroF1": 0.5728429072595399,
            "MacroF1": 0.565898409914518,
            "Memory in Mb": 12.449102401733398,
            "Time in s": 10884.980897
          },
          {
            "step": 47520,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5768850354595004,
            "MicroF1": 0.5768850354595004,
            "MacroF1": 0.57039744062367,
            "Memory in Mb": 16.244935989379883,
            "Time in s": 11383.832017
          },
          {
            "step": 48576,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5828718476582604,
            "MicroF1": 0.5828718476582604,
            "MacroF1": 0.5764217258661826,
            "Memory in Mb": 15.377230644226074,
            "Time in s": 11896.186788
          },
          {
            "step": 49632,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.5890471681005823,
            "MicroF1": 0.5890471681005823,
            "MacroF1": 0.5823842044431963,
            "Memory in Mb": 14.950640678405762,
            "Time in s": 12421.939031
          },
          {
            "step": 50688,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.594728431353207,
            "MicroF1": 0.594728431353207,
            "MacroF1": 0.5876258810149467,
            "Memory in Mb": 12.564711570739746,
            "Time in s": 12959.806662
          },
          {
            "step": 51744,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6007382641130201,
            "MicroF1": 0.6007382641130201,
            "MacroF1": 0.5930524375976366,
            "Memory in Mb": 11.987659454345703,
            "Time in s": 13508.895544
          },
          {
            "step": 52800,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Insects",
            "Accuracy": 0.606053144945927,
            "MicroF1": 0.606053144945927,
            "MacroF1": 0.5982224401760299,
            "Memory in Mb": 3.750063896179199,
            "Time in s": 14067.159334999998
          },
          {
            "step": 408,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9828009828009828,
            "MicroF1": 0.9828009828009828,
            "MacroF1": 0.6067632850241546,
            "Memory in Mb": 2.2869701385498047,
            "Time in s": 1.329477
          },
          {
            "step": 816,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9411042944785276,
            "MicroF1": 0.9411042944785276,
            "MacroF1": 0.7377235942917068,
            "Memory in Mb": 3.233952522277832,
            "Time in s": 4.406974
          },
          {
            "step": 1224,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8879803761242846,
            "MicroF1": 0.8879803761242846,
            "MacroF1": 0.873420796574987,
            "Memory in Mb": 4.178282737731934,
            "Time in s": 9.866805
          },
          {
            "step": 1632,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8988350705088902,
            "MicroF1": 0.8988350705088902,
            "MacroF1": 0.8792834531664682,
            "Memory in Mb": 5.13068962097168,
            "Time in s": 18.093903
          },
          {
            "step": 2040,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8950465914664051,
            "MicroF1": 0.8950465914664051,
            "MacroF1": 0.8828407845486113,
            "Memory in Mb": 6.191357612609863,
            "Time in s": 29.527087
          },
          {
            "step": 2448,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8561503882304863,
            "MicroF1": 0.8561503882304863,
            "MacroF1": 0.8521381720173345,
            "Memory in Mb": 7.13577938079834,
            "Time in s": 44.765436
          },
          {
            "step": 2856,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8623467600700525,
            "MicroF1": 0.8623467600700525,
            "MacroF1": 0.8461129037988256,
            "Memory in Mb": 8.082098007202148,
            "Time in s": 64.393699
          },
          {
            "step": 3264,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8528961078761875,
            "MicroF1": 0.8528961078761875,
            "MacroF1": 0.828204357625989,
            "Memory in Mb": 9.02734088897705,
            "Time in s": 89.05171
          },
          {
            "step": 3672,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8428221193135386,
            "MicroF1": 0.8428221193135386,
            "MacroF1": 0.8381978174360706,
            "Memory in Mb": 9.972960472106934,
            "Time in s": 119.28991
          },
          {
            "step": 4080,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8350085805344447,
            "MicroF1": 0.8350085805344447,
            "MacroF1": 0.8208915725311207,
            "Memory in Mb": 11.171079635620115,
            "Time in s": 155.799996
          },
          {
            "step": 4488,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.822821484287943,
            "MicroF1": 0.822821484287943,
            "MacroF1": 0.832874806475175,
            "Memory in Mb": 12.116948127746582,
            "Time in s": 198.957143
          },
          {
            "step": 4896,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8212461695607763,
            "MicroF1": 0.8212461695607763,
            "MacroF1": 0.8275900848879882,
            "Memory in Mb": 13.061814308166504,
            "Time in s": 249.493693
          },
          {
            "step": 5304,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.8178389590797661,
            "MicroF1": 0.8178389590797661,
            "MacroF1": 0.8022229037941512,
            "Memory in Mb": 14.007365226745604,
            "Time in s": 307.913741
          },
          {
            "step": 5712,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.7974085098931886,
            "MicroF1": 0.7974085098931886,
            "MacroF1": 0.8005324816804641,
            "Memory in Mb": 14.95396614074707,
            "Time in s": 374.813654
          },
          {
            "step": 6120,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.7947377022389279,
            "MicroF1": 0.7947377022389279,
            "MacroF1": 0.7763699164747573,
            "Memory in Mb": 15.899590492248535,
            "Time in s": 450.788506
          },
          {
            "step": 6528,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.7695725448138502,
            "MicroF1": 0.7695725448138502,
            "MacroF1": 0.7646092489325799,
            "Memory in Mb": 16.84642791748047,
            "Time in s": 536.475639
          },
          {
            "step": 6936,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.7614996395097332,
            "MicroF1": 0.7614996395097332,
            "MacroF1": 0.7633186803137438,
            "Memory in Mb": 17.791536331176758,
            "Time in s": 632.653421
          },
          {
            "step": 7344,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.770393572109492,
            "MicroF1": 0.770393572109492,
            "MacroF1": 0.7679684376178252,
            "Memory in Mb": 18.753963470458984,
            "Time in s": 739.829913
          },
          {
            "step": 7752,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.7709972906721714,
            "MicroF1": 0.7709972906721715,
            "MacroF1": 0.7694364393340193,
            "Memory in Mb": 19.70015239715576,
            "Time in s": 858.3068800000001
          },
          {
            "step": 8160,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.7739919107733791,
            "MicroF1": 0.7739919107733791,
            "MacroF1": 0.7702264725589797,
            "Memory in Mb": 20.646059036254883,
            "Time in s": 989.0801780000002
          },
          {
            "step": 8568,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.770631492938018,
            "MicroF1": 0.770631492938018,
            "MacroF1": 0.7706502591714904,
            "Memory in Mb": 22.072673797607425,
            "Time in s": 1132.67323
          },
          {
            "step": 8976,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.7691364902506964,
            "MicroF1": 0.7691364902506964,
            "MacroF1": 0.7697475673922982,
            "Memory in Mb": 23.017619132995605,
            "Time in s": 1289.930532
          },
          {
            "step": 9384,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.7679846530960247,
            "MicroF1": 0.7679846530960248,
            "MacroF1": 0.7675735514139922,
            "Memory in Mb": 23.96499538421631,
            "Time in s": 1461.233929
          },
          {
            "step": 9792,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.7634562353181493,
            "MicroF1": 0.7634562353181493,
            "MacroF1": 0.7626887405791724,
            "Memory in Mb": 24.91041660308838,
            "Time in s": 1647.051427
          },
          {
            "step": 10200,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.7552701245220119,
            "MicroF1": 0.7552701245220118,
            "MacroF1": 0.7474447650479976,
            "Memory in Mb": 25.855456352233887,
            "Time in s": 1848.111209
          },
          {
            "step": 10608,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.734326388234185,
            "MicroF1": 0.734326388234185,
            "MacroF1": 0.7218544335091276,
            "Memory in Mb": 26.80277729034424,
            "Time in s": 2064.993893
          },
          {
            "step": 11016,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.727099409895597,
            "MicroF1": 0.727099409895597,
            "MacroF1": 0.7232704418570853,
            "Memory in Mb": 27.74752426147461,
            "Time in s": 2298.467217
          },
          {
            "step": 11424,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.7203011468090694,
            "MicroF1": 0.7203011468090693,
            "MacroF1": 0.7069709690618045,
            "Memory in Mb": 28.693537712097168,
            "Time in s": 2549.106467
          },
          {
            "step": 11832,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.7107598681430141,
            "MicroF1": 0.7107598681430141,
            "MacroF1": 0.7032019097144009,
            "Memory in Mb": 29.638681411743164,
            "Time in s": 2817.5197820000003
          },
          {
            "step": 12240,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.7152545142577008,
            "MicroF1": 0.7152545142577007,
            "MacroF1": 0.7117335483783439,
            "Memory in Mb": 30.584209442138672,
            "Time in s": 3104.4910310000005
          },
          {
            "step": 12648,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.7121056377006405,
            "MicroF1": 0.7121056377006404,
            "MacroF1": 0.7043178518121461,
            "Memory in Mb": 31.53076934814453,
            "Time in s": 3410.5360360000004
          },
          {
            "step": 13056,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.7005744925315971,
            "MicroF1": 0.7005744925315971,
            "MacroF1": 0.6932522175542292,
            "Memory in Mb": 32.476640701293945,
            "Time in s": 3735.542008
          },
          {
            "step": 13464,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.6985070192379114,
            "MicroF1": 0.6985070192379114,
            "MacroF1": 0.6945196760058037,
            "Memory in Mb": 33.421990394592285,
            "Time in s": 4080.277425
          },
          {
            "step": 13872,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.6980751207555331,
            "MicroF1": 0.6980751207555331,
            "MacroF1": 0.6949558493849793,
            "Memory in Mb": 34.36870098114014,
            "Time in s": 4445.365121
          },
          {
            "step": 14280,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.6936760277330345,
            "MicroF1": 0.6936760277330345,
            "MacroF1": 0.6891645690411646,
            "Memory in Mb": 35.313669204711914,
            "Time in s": 4831.313826
          },
          {
            "step": 14688,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.6963300878327773,
            "MicroF1": 0.6963300878327773,
            "MacroF1": 0.6946500105809528,
            "Memory in Mb": 36.259453773498535,
            "Time in s": 5238.529619
          },
          {
            "step": 15096,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.7024180192116595,
            "MicroF1": 0.7024180192116595,
            "MacroF1": 0.7008836593188431,
            "Memory in Mb": 37.20665740966797,
            "Time in s": 5667.803474
          },
          {
            "step": 15504,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.702509191769335,
            "MicroF1": 0.702509191769335,
            "MacroF1": 0.6995855030221436,
            "Memory in Mb": 38.15153884887695,
            "Time in s": 6119.5687720000005
          },
          {
            "step": 15912,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.6934824963861479,
            "MicroF1": 0.6934824963861479,
            "MacroF1": 0.687175788748239,
            "Memory in Mb": 39.09754180908203,
            "Time in s": 6594.573754000001
          },
          {
            "step": 16320,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.6848458851645322,
            "MicroF1": 0.6848458851645322,
            "MacroF1": 0.6802460349069701,
            "Memory in Mb": 40.04375648498535,
            "Time in s": 7093.395211000001
          },
          {
            "step": 16728,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.6819513361630896,
            "MicroF1": 0.6819513361630896,
            "MacroF1": 0.6795788912922722,
            "Memory in Mb": 40.98903942108154,
            "Time in s": 7616.277784000001
          },
          {
            "step": 17136,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.6779107090749927,
            "MicroF1": 0.6779107090749927,
            "MacroF1": 0.6747648209169417,
            "Memory in Mb": 42.91780757904053,
            "Time in s": 8164.158576000001
          },
          {
            "step": 17544,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.6705808584620646,
            "MicroF1": 0.6705808584620646,
            "MacroF1": 0.6680341530684186,
            "Memory in Mb": 43.864423751831055,
            "Time in s": 8737.125467000002
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.6695448721519692,
            "MicroF1": 0.6695448721519692,
            "MacroF1": 0.6687363294804706,
            "Memory in Mb": 44.8110933303833,
            "Time in s": 9335.759485000002
          },
          {
            "step": 18360,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.674927828313089,
            "MicroF1": 0.674927828313089,
            "MacroF1": 0.6747300618557481,
            "Memory in Mb": 45.75679397583008,
            "Time in s": 9961.084735000002
          },
          {
            "step": 18768,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.6799701603879149,
            "MicroF1": 0.6799701603879149,
            "MacroF1": 0.6801519832282531,
            "Memory in Mb": 46.703369140625,
            "Time in s": 10614.122071000002
          },
          {
            "step": 19176,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.6720730117340287,
            "MicroF1": 0.6720730117340287,
            "MacroF1": 0.6711666831354974,
            "Memory in Mb": 47.648451805114746,
            "Time in s": 11294.958186000002
          },
          {
            "step": 19584,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.6760455497114845,
            "MicroF1": 0.6760455497114845,
            "MacroF1": 0.6762772840246767,
            "Memory in Mb": 48.59414768218994,
            "Time in s": 12004.137364000002
          },
          {
            "step": 19992,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.6715521984893202,
            "MicroF1": 0.6715521984893202,
            "MacroF1": 0.6718362805013157,
            "Memory in Mb": 49.5405502319336,
            "Time in s": 12741.947206000004
          },
          {
            "step": 20400,
            "track": "Multiclass classification",
            "model": "Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.6679739202902103,
            "MicroF1": 0.6679739202902103,
            "MacroF1": 0.6688529665037395,
            "Memory in Mb": 50.48721218109131,
            "Time in s": 13509.112779000005
          },
          {
            "step": 46,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.3777777777777777,
            "MicroF1": 0.3777777777777777,
            "MacroF1": 0.2811210847975554,
            "Memory in Mb": 4.12965202331543,
            "Time in s": 2.162623
          },
          {
            "step": 92,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.5164835164835165,
            "MicroF1": 0.5164835164835165,
            "MacroF1": 0.5316649744849407,
            "Memory in Mb": 4.130231857299805,
            "Time in s": 5.385739
          },
          {
            "step": 138,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.5547445255474452,
            "MicroF1": 0.5547445255474452,
            "MacroF1": 0.5804654781117262,
            "Memory in Mb": 4.130353927612305,
            "Time in s": 9.498091
          },
          {
            "step": 184,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.6174863387978142,
            "MicroF1": 0.6174863387978142,
            "MacroF1": 0.6394923756219437,
            "Memory in Mb": 4.130964279174805,
            "Time in s": 14.44781
          },
          {
            "step": 230,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.6506550218340611,
            "MicroF1": 0.6506550218340611,
            "MacroF1": 0.66859135700569,
            "Memory in Mb": 4.130964279174805,
            "Time in s": 20.213143
          },
          {
            "step": 276,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.6618181818181819,
            "MicroF1": 0.6618181818181819,
            "MacroF1": 0.6795855359270878,
            "Memory in Mb": 4.131082534790039,
            "Time in s": 26.807818
          },
          {
            "step": 322,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.6853582554517134,
            "MicroF1": 0.6853582554517134,
            "MacroF1": 0.6872635633687633,
            "Memory in Mb": 4.131624221801758,
            "Time in s": 34.210100999999995
          },
          {
            "step": 368,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7111716621253406,
            "MicroF1": 0.7111716621253404,
            "MacroF1": 0.7098417316927395,
            "Memory in Mb": 4.131597518920898,
            "Time in s": 42.42663699999999
          },
          {
            "step": 414,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7215496368038741,
            "MicroF1": 0.7215496368038742,
            "MacroF1": 0.7201557312728714,
            "Memory in Mb": 4.13151741027832,
            "Time in s": 51.47481599999999
          },
          {
            "step": 460,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7211328976034859,
            "MicroF1": 0.721132897603486,
            "MacroF1": 0.7175330036146421,
            "Memory in Mb": 4.131570816040039,
            "Time in s": 61.34488599999999
          },
          {
            "step": 506,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7287128712871287,
            "MicroF1": 0.7287128712871287,
            "MacroF1": 0.7233455022590812,
            "Memory in Mb": 4.131570816040039,
            "Time in s": 72.04824799999999
          },
          {
            "step": 552,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7295825771324864,
            "MicroF1": 0.7295825771324864,
            "MacroF1": 0.7255599965917697,
            "Memory in Mb": 4.131490707397461,
            "Time in s": 83.59783199999998
          },
          {
            "step": 598,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7353433835845896,
            "MicroF1": 0.7353433835845896,
            "MacroF1": 0.7308494254186014,
            "Memory in Mb": 4.131513595581055,
            "Time in s": 95.96874199999998
          },
          {
            "step": 644,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7340590979782271,
            "MicroF1": 0.7340590979782271,
            "MacroF1": 0.7314183982762247,
            "Memory in Mb": 4.132074356079102,
            "Time in s": 109.15469699999996
          },
          {
            "step": 690,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.737300435413643,
            "MicroF1": 0.737300435413643,
            "MacroF1": 0.7343909641298695,
            "Memory in Mb": 4.132074356079102,
            "Time in s": 123.16042199999995
          },
          {
            "step": 736,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7387755102040816,
            "MicroF1": 0.7387755102040816,
            "MacroF1": 0.7369557659594496,
            "Memory in Mb": 4.132101058959961,
            "Time in s": 138.00293199999996
          },
          {
            "step": 782,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7439180537772087,
            "MicroF1": 0.7439180537772088,
            "MacroF1": 0.7419020281650245,
            "Memory in Mb": 4.132101058959961,
            "Time in s": 153.67249999999996
          },
          {
            "step": 828,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7436517533252721,
            "MicroF1": 0.7436517533252721,
            "MacroF1": 0.7432199627682998,
            "Memory in Mb": 4.132101058959961,
            "Time in s": 170.15907299999995
          },
          {
            "step": 874,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7502863688430699,
            "MicroF1": 0.7502863688430699,
            "MacroF1": 0.7482089866208982,
            "Memory in Mb": 4.132101058959961,
            "Time in s": 187.48903599999997
          },
          {
            "step": 920,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.750816104461371,
            "MicroF1": 0.750816104461371,
            "MacroF1": 0.7477650187313973,
            "Memory in Mb": 4.132074356079102,
            "Time in s": 205.64141899999996
          },
          {
            "step": 966,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7512953367875648,
            "MicroF1": 0.7512953367875648,
            "MacroF1": 0.747322646811651,
            "Memory in Mb": 4.132074356079102,
            "Time in s": 224.60247399999992
          },
          {
            "step": 1012,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7507418397626113,
            "MicroF1": 0.7507418397626113,
            "MacroF1": 0.7469783619055548,
            "Memory in Mb": 4.132074356079102,
            "Time in s": 244.38522099999992
          },
          {
            "step": 1058,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7530747398297067,
            "MicroF1": 0.7530747398297066,
            "MacroF1": 0.7482363934596314,
            "Memory in Mb": 4.132074356079102,
            "Time in s": 264.9944789999999
          },
          {
            "step": 1104,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7552130553037172,
            "MicroF1": 0.7552130553037172,
            "MacroF1": 0.750118495060715,
            "Memory in Mb": 4.132123947143555,
            "Time in s": 286.4318919999999
          },
          {
            "step": 1150,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7571801566579635,
            "MicroF1": 0.7571801566579635,
            "MacroF1": 0.7516199800653578,
            "Memory in Mb": 4.132123947143555,
            "Time in s": 308.6933539999999
          },
          {
            "step": 1196,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7598326359832636,
            "MicroF1": 0.7598326359832636,
            "MacroF1": 0.7548841797367704,
            "Memory in Mb": 4.132123947143555,
            "Time in s": 331.8138849999999
          },
          {
            "step": 1242,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7598710717163578,
            "MicroF1": 0.7598710717163577,
            "MacroF1": 0.7553301531902636,
            "Memory in Mb": 4.132123947143555,
            "Time in s": 355.7543559999999
          },
          {
            "step": 1288,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7645687645687645,
            "MicroF1": 0.7645687645687647,
            "MacroF1": 0.7590078532621816,
            "Memory in Mb": 4.132734298706055,
            "Time in s": 380.5041089999999
          },
          {
            "step": 1334,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7644411102775694,
            "MicroF1": 0.7644411102775694,
            "MacroF1": 0.7591993978414527,
            "Memory in Mb": 4.132757186889648,
            "Time in s": 406.0948849999999
          },
          {
            "step": 1380,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7650471356055112,
            "MicroF1": 0.7650471356055112,
            "MacroF1": 0.7601575050520946,
            "Memory in Mb": 4.132757186889648,
            "Time in s": 432.504415
          },
          {
            "step": 1426,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7670175438596492,
            "MicroF1": 0.7670175438596492,
            "MacroF1": 0.7613339877221927,
            "Memory in Mb": 4.132757186889648,
            "Time in s": 459.75425999999993
          },
          {
            "step": 1472,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7715839564921821,
            "MicroF1": 0.7715839564921821,
            "MacroF1": 0.76413964752182,
            "Memory in Mb": 4.132802963256836,
            "Time in s": 487.813313
          },
          {
            "step": 1518,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7732366512854317,
            "MicroF1": 0.7732366512854317,
            "MacroF1": 0.7648275341801108,
            "Memory in Mb": 4.132802963256836,
            "Time in s": 516.6922259999999
          },
          {
            "step": 1564,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7735124760076776,
            "MicroF1": 0.7735124760076776,
            "MacroF1": 0.7657569341108763,
            "Memory in Mb": 4.132802963256836,
            "Time in s": 546.3975239999999
          },
          {
            "step": 1610,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7737725295214419,
            "MicroF1": 0.7737725295214419,
            "MacroF1": 0.7651494083475014,
            "Memory in Mb": 4.13282585144043,
            "Time in s": 576.9286479999998
          },
          {
            "step": 1656,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7740181268882175,
            "MicroF1": 0.7740181268882175,
            "MacroF1": 0.7654813489818475,
            "Memory in Mb": 4.132780075073242,
            "Time in s": 608.2971799999998
          },
          {
            "step": 1702,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7730746619635509,
            "MicroF1": 0.7730746619635509,
            "MacroF1": 0.7664930279619061,
            "Memory in Mb": 4.132780075073242,
            "Time in s": 640.4789209999998
          },
          {
            "step": 1748,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7756153405838581,
            "MicroF1": 0.7756153405838581,
            "MacroF1": 0.7686072256536652,
            "Memory in Mb": 4.132780075073242,
            "Time in s": 673.4948829999998
          },
          {
            "step": 1794,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7769102063580591,
            "MicroF1": 0.7769102063580591,
            "MacroF1": 0.7685414235990153,
            "Memory in Mb": 4.132753372192383,
            "Time in s": 707.3642379999999
          },
          {
            "step": 1840,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7781402936378466,
            "MicroF1": 0.7781402936378466,
            "MacroF1": 0.7699957723931324,
            "Memory in Mb": 4.132753372192383,
            "Time in s": 742.0792459999999
          },
          {
            "step": 1886,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7761273209549071,
            "MicroF1": 0.7761273209549071,
            "MacroF1": 0.7684985598909853,
            "Memory in Mb": 4.132753372192383,
            "Time in s": 777.6304799999999
          },
          {
            "step": 1932,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7762817193164163,
            "MicroF1": 0.7762817193164163,
            "MacroF1": 0.767743441804642,
            "Memory in Mb": 4.132753372192383,
            "Time in s": 814.0157899999999
          },
          {
            "step": 1978,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7774405665149215,
            "MicroF1": 0.7774405665149215,
            "MacroF1": 0.7684788817649146,
            "Memory in Mb": 4.132753372192383,
            "Time in s": 851.2372119999999
          },
          {
            "step": 2024,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7790410281759763,
            "MicroF1": 0.7790410281759763,
            "MacroF1": 0.7689103339153599,
            "Memory in Mb": 4.132753372192383,
            "Time in s": 889.2963629999999
          },
          {
            "step": 2070,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7786370227162881,
            "MicroF1": 0.7786370227162881,
            "MacroF1": 0.7686288077529282,
            "Memory in Mb": 4.132753372192383,
            "Time in s": 928.20257
          },
          {
            "step": 2116,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7791962174940898,
            "MicroF1": 0.7791962174940898,
            "MacroF1": 0.768391950800897,
            "Memory in Mb": 4.132753372192383,
            "Time in s": 967.945094
          },
          {
            "step": 2162,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7801943544655252,
            "MicroF1": 0.7801943544655253,
            "MacroF1": 0.768962628827985,
            "Memory in Mb": 4.132776260375977,
            "Time in s": 1008.503462
          },
          {
            "step": 2208,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7820570910738559,
            "MicroF1": 0.7820570910738559,
            "MacroF1": 0.7698068761587117,
            "Memory in Mb": 4.132749557495117,
            "Time in s": 1049.8865549999998
          },
          {
            "step": 2254,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7789613848202397,
            "MicroF1": 0.7789613848202397,
            "MacroF1": 0.7667173742344939,
            "Memory in Mb": 4.132749557495117,
            "Time in s": 1092.1021469999998
          },
          {
            "step": 2300,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "ImageSegments",
            "Accuracy": 0.7781644193127447,
            "MicroF1": 0.7781644193127447,
            "MacroF1": 0.7659138381656089,
            "Memory in Mb": 4.132749557495117,
            "Time in s": 1135.1554539999995
          },
          {
            "step": 1056,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6218009478672986,
            "MicroF1": 0.6218009478672986,
            "MacroF1": 0.5857016652718547,
            "Memory in Mb": 6.522056579589844,
            "Time in s": 30.289418
          },
          {
            "step": 2112,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6196115585030791,
            "MicroF1": 0.6196115585030791,
            "MacroF1": 0.5856756432415232,
            "Memory in Mb": 10.389650344848633,
            "Time in s": 88.669185
          },
          {
            "step": 3168,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.628986422481844,
            "MicroF1": 0.628986422481844,
            "MacroF1": 0.5949930595607558,
            "Memory in Mb": 19.16711711883545,
            "Time in s": 174.550284
          },
          {
            "step": 4224,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6294103717736207,
            "MicroF1": 0.6294103717736207,
            "MacroF1": 0.5952675443708706,
            "Memory in Mb": 19.668034553527832,
            "Time in s": 287.918965
          },
          {
            "step": 5280,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6364841826103429,
            "MicroF1": 0.6364841826103429,
            "MacroF1": 0.5994911272790604,
            "Memory in Mb": 18.96163558959961,
            "Time in s": 428.854975
          },
          {
            "step": 6336,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6352012628255722,
            "MicroF1": 0.6352012628255722,
            "MacroF1": 0.5993891820807257,
            "Memory in Mb": 20.14603328704834,
            "Time in s": 597.190787
          },
          {
            "step": 7392,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.638749830875389,
            "MicroF1": 0.638749830875389,
            "MacroF1": 0.6030343276880051,
            "Memory in Mb": 21.10132884979248,
            "Time in s": 793.049033
          },
          {
            "step": 8448,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6405824553095774,
            "MicroF1": 0.6405824553095774,
            "MacroF1": 0.6028521616895871,
            "Memory in Mb": 24.15276908874512,
            "Time in s": 1016.521492
          },
          {
            "step": 9504,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6449542249815847,
            "MicroF1": 0.6449542249815847,
            "MacroF1": 0.6055705492028415,
            "Memory in Mb": 24.86981773376465,
            "Time in s": 1266.726445
          },
          {
            "step": 10560,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6485462638507434,
            "MicroF1": 0.6485462638507434,
            "MacroF1": 0.6081614166360886,
            "Memory in Mb": 28.971991539001465,
            "Time in s": 1544.137884
          },
          {
            "step": 11616,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6490744726646578,
            "MicroF1": 0.6490744726646578,
            "MacroF1": 0.6078786452761632,
            "Memory in Mb": 31.018654823303223,
            "Time in s": 1848.95866
          },
          {
            "step": 12672,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6514876489621971,
            "MicroF1": 0.6514876489621971,
            "MacroF1": 0.6111938480023121,
            "Memory in Mb": 35.39500713348389,
            "Time in s": 2179.477442
          },
          {
            "step": 13728,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6707947840023312,
            "MicroF1": 0.6707947840023312,
            "MacroF1": 0.6607574394823456,
            "Memory in Mb": 17.66313648223877,
            "Time in s": 2527.754593
          },
          {
            "step": 14784,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6814584319826829,
            "MicroF1": 0.6814584319826829,
            "MacroF1": 0.6724584381879511,
            "Memory in Mb": 11.128533363342283,
            "Time in s": 2896.354015
          },
          {
            "step": 15840,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6762421870067554,
            "MicroF1": 0.6762421870067554,
            "MacroF1": 0.6688785181435096,
            "Memory in Mb": 14.811795234680176,
            "Time in s": 3290.113809
          },
          {
            "step": 16896,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6741639538324948,
            "MicroF1": 0.6741639538324948,
            "MacroF1": 0.6676833597101233,
            "Memory in Mb": 15.36542510986328,
            "Time in s": 3710.779607
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.670491894601972,
            "MicroF1": 0.670491894601972,
            "MacroF1": 0.6643621029883554,
            "Memory in Mb": 15.98740005493164,
            "Time in s": 4158.727334
          },
          {
            "step": 19008,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6754353659178197,
            "MicroF1": 0.6754353659178197,
            "MacroF1": 0.6656526175716114,
            "Memory in Mb": 17.100504875183105,
            "Time in s": 4627.925426
          },
          {
            "step": 20064,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6800079748791308,
            "MicroF1": 0.6800079748791308,
            "MacroF1": 0.6670489534490986,
            "Memory in Mb": 26.370519638061523,
            "Time in s": 5118.761149
          },
          {
            "step": 21120,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6835550925706709,
            "MicroF1": 0.6835550925706709,
            "MacroF1": 0.6685883462655132,
            "Memory in Mb": 32.78877353668213,
            "Time in s": 5635.605831
          },
          {
            "step": 22176,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6869447576099211,
            "MicroF1": 0.6869447576099211,
            "MacroF1": 0.6701495347804184,
            "Memory in Mb": 36.29740715026856,
            "Time in s": 6178.169973
          },
          {
            "step": 23232,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6912745899875167,
            "MicroF1": 0.6912745899875167,
            "MacroF1": 0.6726358783249661,
            "Memory in Mb": 38.26123523712158,
            "Time in s": 6746.46947
          },
          {
            "step": 24288,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6940338452670153,
            "MicroF1": 0.6940338452670153,
            "MacroF1": 0.673442702110033,
            "Memory in Mb": 39.100372314453125,
            "Time in s": 7340.085466
          },
          {
            "step": 25344,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6976679951071302,
            "MicroF1": 0.6976679951071302,
            "MacroF1": 0.67525701759611,
            "Memory in Mb": 42.24958515167236,
            "Time in s": 7958.602362
          },
          {
            "step": 26400,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.7000643963786507,
            "MicroF1": 0.7000643963786507,
            "MacroF1": 0.6759116206749555,
            "Memory in Mb": 41.52747917175293,
            "Time in s": 8602.02664
          },
          {
            "step": 27456,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.7027135312329266,
            "MicroF1": 0.7027135312329266,
            "MacroF1": 0.6765494742782628,
            "Memory in Mb": 43.56198120117188,
            "Time in s": 9269.998307
          },
          {
            "step": 28512,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.7018343797130931,
            "MicroF1": 0.7018343797130931,
            "MacroF1": 0.6771545550561098,
            "Memory in Mb": 24.23386573791504,
            "Time in s": 9962.817287
          },
          {
            "step": 29568,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.7013224202658369,
            "MicroF1": 0.7013224202658369,
            "MacroF1": 0.681362451564682,
            "Memory in Mb": 5.156903266906738,
            "Time in s": 10676.641313
          },
          {
            "step": 30624,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.699702837736342,
            "MicroF1": 0.699702837736342,
            "MacroF1": 0.6839521261644582,
            "Memory in Mb": 8.359548568725586,
            "Time in s": 11409.958608
          },
          {
            "step": 31680,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6993907635973358,
            "MicroF1": 0.6993907635973358,
            "MacroF1": 0.6874853197903658,
            "Memory in Mb": 12.837088584899902,
            "Time in s": 12162.299998
          },
          {
            "step": 32736,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.7005651443409195,
            "MicroF1": 0.7005651443409195,
            "MacroF1": 0.692127614099415,
            "Memory in Mb": 14.392640113830566,
            "Time in s": 12933.190651
          },
          {
            "step": 33792,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6971678849397769,
            "MicroF1": 0.6971678849397769,
            "MacroF1": 0.6903104823999882,
            "Memory in Mb": 22.11440753936768,
            "Time in s": 13726.605059
          },
          {
            "step": 34848,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6941487072057853,
            "MicroF1": 0.6941487072057853,
            "MacroF1": 0.6871648754350796,
            "Memory in Mb": 16.369569778442383,
            "Time in s": 14546.496494
          },
          {
            "step": 35904,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6917527783193606,
            "MicroF1": 0.6917527783193606,
            "MacroF1": 0.684473708604621,
            "Memory in Mb": 15.783265113830566,
            "Time in s": 15393.169285
          },
          {
            "step": 36960,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6883303119673151,
            "MicroF1": 0.6883303119673151,
            "MacroF1": 0.6807777972894504,
            "Memory in Mb": 18.195876121521,
            "Time in s": 16266.308937
          },
          {
            "step": 38016,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6865973957648297,
            "MicroF1": 0.6865973957648297,
            "MacroF1": 0.6786744939637405,
            "Memory in Mb": 21.092598915100098,
            "Time in s": 17165.821217
          },
          {
            "step": 39072,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6857259860254409,
            "MicroF1": 0.6857259860254409,
            "MacroF1": 0.6778492437957201,
            "Memory in Mb": 16.29904079437256,
            "Time in s": 18090.697656
          },
          {
            "step": 40128,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6837540807934807,
            "MicroF1": 0.6837540807934807,
            "MacroF1": 0.6766238977666043,
            "Memory in Mb": 13.538718223571776,
            "Time in s": 19041.492123
          },
          {
            "step": 41184,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6814462278124469,
            "MicroF1": 0.6814462278124469,
            "MacroF1": 0.675074837604149,
            "Memory in Mb": 15.844508171081545,
            "Time in s": 20015.794843
          },
          {
            "step": 42240,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6790643717891048,
            "MicroF1": 0.6790643717891048,
            "MacroF1": 0.6733686277261395,
            "Memory in Mb": 15.962260246276855,
            "Time in s": 21014.495285
          },
          {
            "step": 43296,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6762443700196328,
            "MicroF1": 0.6762443700196328,
            "MacroF1": 0.6713719096586489,
            "Memory in Mb": 17.128825187683105,
            "Time in s": 22038.291411
          },
          {
            "step": 44352,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6738066785416338,
            "MicroF1": 0.6738066785416338,
            "MacroF1": 0.6696205967919768,
            "Memory in Mb": 16.462289810180664,
            "Time in s": 23087.153485000003
          },
          {
            "step": 45408,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6717686700288502,
            "MicroF1": 0.6717686700288502,
            "MacroF1": 0.6680705737277651,
            "Memory in Mb": 17.22057342529297,
            "Time in s": 24160.866409
          },
          {
            "step": 46464,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6708994253492025,
            "MicroF1": 0.6708994253492025,
            "MacroF1": 0.6677330044499646,
            "Memory in Mb": 17.752578735351562,
            "Time in s": 25258.281990000003
          },
          {
            "step": 47520,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6729939603106126,
            "MicroF1": 0.6729939603106126,
            "MacroF1": 0.6699611714455135,
            "Memory in Mb": 18.93515110015869,
            "Time in s": 26380.044308000004
          },
          {
            "step": 48576,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6767061245496655,
            "MicroF1": 0.6767061245496655,
            "MacroF1": 0.6733691077464542,
            "Memory in Mb": 20.549713134765625,
            "Time in s": 27525.907792000005
          },
          {
            "step": 49632,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6807237412101308,
            "MicroF1": 0.6807237412101308,
            "MacroF1": 0.6769109137483648,
            "Memory in Mb": 20.974443435668945,
            "Time in s": 28695.60352400001
          },
          {
            "step": 50688,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6845147671000453,
            "MicroF1": 0.6845147671000453,
            "MacroF1": 0.6800104952374638,
            "Memory in Mb": 22.97932243347168,
            "Time in s": 29888.26411600001
          },
          {
            "step": 51744,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6885182536768258,
            "MicroF1": 0.6885182536768258,
            "MacroF1": 0.6832561756017089,
            "Memory in Mb": 24.11430263519287,
            "Time in s": 31103.175398000007
          },
          {
            "step": 52800,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Insects",
            "Accuracy": 0.6915471883937195,
            "MicroF1": 0.6915471883937195,
            "MacroF1": 0.6864107325641782,
            "Memory in Mb": 18.141328811645508,
            "Time in s": 32334.061725000007
          },
          {
            "step": 408,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9828009828009828,
            "MicroF1": 0.9828009828009828,
            "MacroF1": 0.6067632850241546,
            "Memory in Mb": 2.238800048828125,
            "Time in s": 2.971982
          },
          {
            "step": 816,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9460122699386504,
            "MicroF1": 0.9460122699386504,
            "MacroF1": 0.8367492469040564,
            "Memory in Mb": 4.44326114654541,
            "Time in s": 9.847778
          },
          {
            "step": 1224,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9411283728536386,
            "MicroF1": 0.9411283728536386,
            "MacroF1": 0.9276213812296338,
            "Memory in Mb": 6.153376579284668,
            "Time in s": 20.932147
          },
          {
            "step": 1632,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.950337216431637,
            "MicroF1": 0.950337216431637,
            "MacroF1": 0.9330502878949444,
            "Memory in Mb": 7.8991851806640625,
            "Time in s": 36.717851
          },
          {
            "step": 2040,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9494850416871016,
            "MicroF1": 0.9494850416871016,
            "MacroF1": 0.932928877406915,
            "Memory in Mb": 10.965654373168944,
            "Time in s": 57.922702
          },
          {
            "step": 2448,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9525950143032283,
            "MicroF1": 0.9525950143032283,
            "MacroF1": 0.9502305130509756,
            "Memory in Mb": 10.694184303283691,
            "Time in s": 83.450682
          },
          {
            "step": 2856,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9544658493870404,
            "MicroF1": 0.9544658493870404,
            "MacroF1": 0.943855127765724,
            "Memory in Mb": 15.53213119506836,
            "Time in s": 113.412194
          },
          {
            "step": 3264,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9515783021759118,
            "MicroF1": 0.9515783021759118,
            "MacroF1": 0.944582727256988,
            "Memory in Mb": 15.652314186096191,
            "Time in s": 149.290909
          },
          {
            "step": 3672,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9526014709888314,
            "MicroF1": 0.9526014709888314,
            "MacroF1": 0.9497542235388344,
            "Memory in Mb": 16.11695098876953,
            "Time in s": 190.984607
          },
          {
            "step": 4080,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9499877420936504,
            "MicroF1": 0.9499877420936504,
            "MacroF1": 0.9391633661003512,
            "Memory in Mb": 16.578293800354004,
            "Time in s": 238.599211
          },
          {
            "step": 4488,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9474036104301314,
            "MicroF1": 0.9474036104301314,
            "MacroF1": 0.9496969875723204,
            "Memory in Mb": 13.190230369567873,
            "Time in s": 292.48392399999994
          },
          {
            "step": 4896,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9493360572012256,
            "MicroF1": 0.9493360572012256,
            "MacroF1": 0.9494027577495958,
            "Memory in Mb": 12.864276885986328,
            "Time in s": 353.18946299999993
          },
          {
            "step": 5304,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.951159720912691,
            "MicroF1": 0.951159720912691,
            "MacroF1": 0.9518992835106976,
            "Memory in Mb": 11.604743957519531,
            "Time in s": 419.932067
          },
          {
            "step": 5712,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.951146909472947,
            "MicroF1": 0.951146909472947,
            "MacroF1": 0.9505351682914018,
            "Memory in Mb": 13.577879905700684,
            "Time in s": 492.702395
          },
          {
            "step": 6120,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9478672985781992,
            "MicroF1": 0.9478672985781992,
            "MacroF1": 0.9429356622084736,
            "Memory in Mb": 16.35688304901123,
            "Time in s": 572.32886
          },
          {
            "step": 6528,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9480618967366324,
            "MicroF1": 0.9480618967366324,
            "MacroF1": 0.9478348775735732,
            "Memory in Mb": 10.670846939086914,
            "Time in s": 659.058815
          },
          {
            "step": 6936,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9495313626532084,
            "MicroF1": 0.9495313626532084,
            "MacroF1": 0.9511497142125284,
            "Memory in Mb": 10.614124298095703,
            "Time in s": 751.777349
          },
          {
            "step": 7344,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9500204276181398,
            "MicroF1": 0.9500204276181398,
            "MacroF1": 0.9502583235097112,
            "Memory in Mb": 12.824416160583496,
            "Time in s": 851.689186
          },
          {
            "step": 7752,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9505870210295446,
            "MicroF1": 0.9505870210295446,
            "MacroF1": 0.9508630550075082,
            "Memory in Mb": 11.99438190460205,
            "Time in s": 957.810381
          },
          {
            "step": 8160,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9487682314009068,
            "MicroF1": 0.9487682314009068,
            "MacroF1": 0.9466937008923912,
            "Memory in Mb": 16.34542465209961,
            "Time in s": 1069.487963
          },
          {
            "step": 8568,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9491070386366288,
            "MicroF1": 0.9491070386366288,
            "MacroF1": 0.9496258519963297,
            "Memory in Mb": 14.096193313598633,
            "Time in s": 1187.33597
          },
          {
            "step": 8976,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.95041782729805,
            "MicroF1": 0.95041782729805,
            "MacroF1": 0.95112303337496,
            "Memory in Mb": 8.487105369567871,
            "Time in s": 1310.539589
          },
          {
            "step": 9384,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9507620164126612,
            "MicroF1": 0.9507620164126612,
            "MacroF1": 0.9509680125568912,
            "Memory in Mb": 10.491826057434082,
            "Time in s": 1439.276187
          },
          {
            "step": 9792,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9505668471044836,
            "MicroF1": 0.9505668471044836,
            "MacroF1": 0.9508008066421794,
            "Memory in Mb": 12.578843116760254,
            "Time in s": 1574.391157
          },
          {
            "step": 10200,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9496029022453182,
            "MicroF1": 0.9496029022453182,
            "MacroF1": 0.9490825188137642,
            "Memory in Mb": 15.329971313476562,
            "Time in s": 1716.648577
          },
          {
            "step": 10608,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9462619025172057,
            "MicroF1": 0.9462619025172057,
            "MacroF1": 0.9448381382156612,
            "Memory in Mb": 14.149526596069336,
            "Time in s": 1865.207745
          },
          {
            "step": 11016,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.94734453018611,
            "MicroF1": 0.94734453018611,
            "MacroF1": 0.9480489849360164,
            "Memory in Mb": 15.43016529083252,
            "Time in s": 2018.851918
          },
          {
            "step": 11424,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9476494791210716,
            "MicroF1": 0.9476494791210716,
            "MacroF1": 0.947763256048792,
            "Memory in Mb": 19.33940696716309,
            "Time in s": 2178.069416
          },
          {
            "step": 11832,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9466655396838812,
            "MicroF1": 0.9466655396838812,
            "MacroF1": 0.9465646854570324,
            "Memory in Mb": 20.836685180664062,
            "Time in s": 2343.241513
          },
          {
            "step": 12240,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9473813220034316,
            "MicroF1": 0.9473813220034316,
            "MacroF1": 0.9477056335712672,
            "Memory in Mb": 22.594761848449707,
            "Time in s": 2516.191517
          },
          {
            "step": 12648,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9481299913022851,
            "MicroF1": 0.9481299913022851,
            "MacroF1": 0.9484695727303012,
            "Memory in Mb": 17.12001132965088,
            "Time in s": 2696.111746
          },
          {
            "step": 13056,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9465338950593642,
            "MicroF1": 0.9465338950593642,
            "MacroF1": 0.9461537407653536,
            "Memory in Mb": 16.317899703979492,
            "Time in s": 2881.285632
          },
          {
            "step": 13464,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.947559979202258,
            "MicroF1": 0.947559979202258,
            "MacroF1": 0.9479124389900307,
            "Memory in Mb": 17.175325393676758,
            "Time in s": 3071.301855
          },
          {
            "step": 13872,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9480931439694328,
            "MicroF1": 0.9480931439694328,
            "MacroF1": 0.9483129032895908,
            "Memory in Mb": 20.13454818725586,
            "Time in s": 3267.097668
          },
          {
            "step": 14280,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.947265214650886,
            "MicroF1": 0.9472652146508858,
            "MacroF1": 0.9472495958535088,
            "Memory in Mb": 23.271190643310547,
            "Time in s": 3470.238619
          },
          {
            "step": 14688,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9470960713556206,
            "MicroF1": 0.9470960713556206,
            "MacroF1": 0.9472715831304288,
            "Memory in Mb": 24.70554256439209,
            "Time in s": 3681.555614
          },
          {
            "step": 15096,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9477310367671414,
            "MicroF1": 0.9477310367671414,
            "MacroF1": 0.948023523282346,
            "Memory in Mb": 12.28943920135498,
            "Time in s": 3901.326497
          },
          {
            "step": 15504,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9481390698574468,
            "MicroF1": 0.9481390698574468,
            "MacroF1": 0.9483821660022894,
            "Memory in Mb": 8.167351722717285,
            "Time in s": 4127.852482
          },
          {
            "step": 15912,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9489661240651122,
            "MicroF1": 0.9489661240651122,
            "MacroF1": 0.949259317367439,
            "Memory in Mb": 10.055158615112305,
            "Time in s": 4359.152379
          },
          {
            "step": 16320,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9489552055885776,
            "MicroF1": 0.9489552055885776,
            "MacroF1": 0.949102505659295,
            "Memory in Mb": 8.927614212036133,
            "Time in s": 4595.973694
          },
          {
            "step": 16728,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.948645901835356,
            "MicroF1": 0.948645901835356,
            "MacroF1": 0.9487532899546076,
            "Memory in Mb": 9.772565841674805,
            "Time in s": 4838.16456
          },
          {
            "step": 17136,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9491683688357164,
            "MicroF1": 0.9491683688357164,
            "MacroF1": 0.9493664270655614,
            "Memory in Mb": 8.885259628295898,
            "Time in s": 5086.602422
          },
          {
            "step": 17544,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9497235364532862,
            "MicroF1": 0.9497235364532862,
            "MacroF1": 0.9498997400720456,
            "Memory in Mb": 7.432655334472656,
            "Time in s": 5340.103049
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9492507381204388,
            "MicroF1": 0.9492507381204388,
            "MacroF1": 0.94932994822919,
            "Memory in Mb": 6.564939498901367,
            "Time in s": 5598.267976
          },
          {
            "step": 18360,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9498883381447792,
            "MicroF1": 0.949888338144779,
            "MacroF1": 0.9500369712738612,
            "Memory in Mb": 9.445448875427246,
            "Time in s": 5861.742373
          },
          {
            "step": 18768,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.950125219800714,
            "MicroF1": 0.950125219800714,
            "MacroF1": 0.950244810756275,
            "Memory in Mb": 8.985580444335938,
            "Time in s": 6130.741927
          },
          {
            "step": 19176,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9502477183833116,
            "MicroF1": 0.9502477183833116,
            "MacroF1": 0.950357710715448,
            "Memory in Mb": 10.539984703063965,
            "Time in s": 6405.708121
          },
          {
            "step": 19584,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9504672419956084,
            "MicroF1": 0.9504672419956084,
            "MacroF1": 0.9505675543483478,
            "Memory in Mb": 12.433100700378418,
            "Time in s": 6686.90918
          },
          {
            "step": 19992,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.9504777149717372,
            "MicroF1": 0.9504777149717372,
            "MacroF1": 0.95056596570352,
            "Memory in Mb": 12.245397567749023,
            "Time in s": 6973.48828
          },
          {
            "step": 20400,
            "track": "Multiclass classification",
            "model": "Leveraging Bagging",
            "dataset": "Keystroke",
            "Accuracy": 0.950389724986519,
            "MicroF1": 0.950389724986519,
            "MacroF1": 0.9504675266923704,
            "Memory in Mb": 10.420146942138672,
            "Time in s": 7265.024106
          },
          {
            "step": 46,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.4,
            "MicroF1": 0.4000000000000001,
            "MacroF1": 0.3289160825620571,
            "Memory in Mb": 1.8703498840332031,
            "Time in s": 0.761019
          },
          {
            "step": 92,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.5494505494505495,
            "MicroF1": 0.5494505494505495,
            "MacroF1": 0.5607526488856412,
            "Memory in Mb": 2.0432376861572266,
            "Time in s": 2.058459
          },
          {
            "step": 138,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.5620437956204379,
            "MicroF1": 0.5620437956204379,
            "MacroF1": 0.5814352652080846,
            "Memory in Mb": 2.2601184844970703,
            "Time in s": 3.877596
          },
          {
            "step": 184,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.6174863387978142,
            "MicroF1": 0.6174863387978142,
            "MacroF1": 0.6349823285289026,
            "Memory in Mb": 2.5773630142211914,
            "Time in s": 5.988367
          },
          {
            "step": 230,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.6550218340611353,
            "MicroF1": 0.6550218340611353,
            "MacroF1": 0.6697464616246889,
            "Memory in Mb": 2.673569679260254,
            "Time in s": 8.36838
          },
          {
            "step": 276,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.68,
            "MicroF1": 0.68,
            "MacroF1": 0.6977451412884614,
            "Memory in Mb": 2.705929756164551,
            "Time in s": 11.015152
          },
          {
            "step": 322,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.7040498442367601,
            "MicroF1": 0.7040498442367601,
            "MacroF1": 0.708655608864303,
            "Memory in Mb": 2.747677803039551,
            "Time in s": 13.922748
          },
          {
            "step": 368,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.7302452316076294,
            "MicroF1": 0.7302452316076294,
            "MacroF1": 0.731555248839775,
            "Memory in Mb": 2.91958236694336,
            "Time in s": 17.085856
          },
          {
            "step": 414,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.7481840193704601,
            "MicroF1": 0.7481840193704601,
            "MacroF1": 0.7498869297449521,
            "Memory in Mb": 3.2087087631225586,
            "Time in s": 20.515594
          },
          {
            "step": 460,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.7429193899782135,
            "MicroF1": 0.7429193899782135,
            "MacroF1": 0.7431113090395209,
            "Memory in Mb": 2.874252319335937,
            "Time in s": 24.226579
          },
          {
            "step": 506,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.7465346534653465,
            "MicroF1": 0.7465346534653465,
            "MacroF1": 0.7453691625646783,
            "Memory in Mb": 3.051929473876953,
            "Time in s": 28.205653
          },
          {
            "step": 552,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.7531760435571688,
            "MicroF1": 0.7531760435571688,
            "MacroF1": 0.7537204076398122,
            "Memory in Mb": 3.133829116821289,
            "Time in s": 32.454626000000005
          },
          {
            "step": 598,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.7587939698492462,
            "MicroF1": 0.7587939698492462,
            "MacroF1": 0.7612399908296416,
            "Memory in Mb": 3.14900016784668,
            "Time in s": 36.970245000000006
          },
          {
            "step": 644,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.7589424572317263,
            "MicroF1": 0.7589424572317262,
            "MacroF1": 0.7628637146980985,
            "Memory in Mb": 3.4707136154174805,
            "Time in s": 41.75427400000001
          },
          {
            "step": 690,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.7619738751814223,
            "MicroF1": 0.7619738751814223,
            "MacroF1": 0.76530464273308,
            "Memory in Mb": 3.455944061279297,
            "Time in s": 46.80492900000001
          },
          {
            "step": 736,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.7687074829931972,
            "MicroF1": 0.7687074829931972,
            "MacroF1": 0.7727990926768868,
            "Memory in Mb": 3.680045127868652,
            "Time in s": 52.11999800000001
          },
          {
            "step": 782,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.7733674775928298,
            "MicroF1": 0.7733674775928298,
            "MacroF1": 0.7767963295410655,
            "Memory in Mb": 3.8801565170288086,
            "Time in s": 57.706528000000006
          },
          {
            "step": 828,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.7738814993954051,
            "MicroF1": 0.7738814993954051,
            "MacroF1": 0.7787678467755003,
            "Memory in Mb": 3.867655754089356,
            "Time in s": 63.567513000000005
          },
          {
            "step": 874,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.7812142038946163,
            "MicroF1": 0.7812142038946163,
            "MacroF1": 0.7848289172220594,
            "Memory in Mb": 3.691183090209961,
            "Time in s": 69.704029
          },
          {
            "step": 920,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.7878128400435256,
            "MicroF1": 0.7878128400435256,
            "MacroF1": 0.7905661589338376,
            "Memory in Mb": 3.770216941833496,
            "Time in s": 76.10799700000001
          },
          {
            "step": 966,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.7917098445595855,
            "MicroF1": 0.7917098445595855,
            "MacroF1": 0.7936972979049142,
            "Memory in Mb": 3.8226003646850586,
            "Time in s": 82.78424000000001
          },
          {
            "step": 1012,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.7952522255192879,
            "MicroF1": 0.7952522255192878,
            "MacroF1": 0.796484514345152,
            "Memory in Mb": 4.098711967468262,
            "Time in s": 89.73293500000001
          },
          {
            "step": 1058,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8032166508987701,
            "MicroF1": 0.8032166508987703,
            "MacroF1": 0.8038465931831994,
            "Memory in Mb": 4.173297882080078,
            "Time in s": 96.952593
          },
          {
            "step": 1104,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8041704442429737,
            "MicroF1": 0.8041704442429737,
            "MacroF1": 0.8051724065917674,
            "Memory in Mb": 4.39574146270752,
            "Time in s": 104.442345
          },
          {
            "step": 1150,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8102697998259356,
            "MicroF1": 0.8102697998259357,
            "MacroF1": 0.8109646011887589,
            "Memory in Mb": 4.552497863769531,
            "Time in s": 112.202361
          },
          {
            "step": 1196,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8142259414225942,
            "MicroF1": 0.8142259414225941,
            "MacroF1": 0.8149917549940485,
            "Memory in Mb": 4.571473121643066,
            "Time in s": 120.234064
          },
          {
            "step": 1242,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8186946011281225,
            "MicroF1": 0.8186946011281225,
            "MacroF1": 0.8196592056494876,
            "Memory in Mb": 4.626148223876953,
            "Time in s": 128.53886899999998
          },
          {
            "step": 1288,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8212898212898213,
            "MicroF1": 0.8212898212898213,
            "MacroF1": 0.822176577441966,
            "Memory in Mb": 5.001523017883301,
            "Time in s": 137.11683999999997
          },
          {
            "step": 1334,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8229557389347337,
            "MicroF1": 0.8229557389347337,
            "MacroF1": 0.8237863794336502,
            "Memory in Mb": 5.142135620117188,
            "Time in s": 145.97222099999996
          },
          {
            "step": 1380,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8245105148658448,
            "MicroF1": 0.8245105148658448,
            "MacroF1": 0.8256018780761997,
            "Memory in Mb": 5.339564323425293,
            "Time in s": 155.11012299999996
          },
          {
            "step": 1426,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8287719298245614,
            "MicroF1": 0.8287719298245614,
            "MacroF1": 0.8290084946618356,
            "Memory in Mb": 5.337568283081055,
            "Time in s": 164.53666099999995
          },
          {
            "step": 1472,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8334466349422162,
            "MicroF1": 0.8334466349422162,
            "MacroF1": 0.8325983603187124,
            "Memory in Mb": 5.299435615539551,
            "Time in s": 174.25653999999994
          },
          {
            "step": 1518,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8358602504943968,
            "MicroF1": 0.8358602504943968,
            "MacroF1": 0.8344617749849152,
            "Memory in Mb": 5.345264434814453,
            "Time in s": 184.26794299999997
          },
          {
            "step": 1564,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8387715930902111,
            "MicroF1": 0.8387715930902111,
            "MacroF1": 0.837784263767798,
            "Memory in Mb": 4.9267168045043945,
            "Time in s": 194.57830699999997
          },
          {
            "step": 1610,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.839030453697949,
            "MicroF1": 0.839030453697949,
            "MacroF1": 0.838065870841574,
            "Memory in Mb": 4.685762405395508,
            "Time in s": 205.19165999999996
          },
          {
            "step": 1656,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8416918429003021,
            "MicroF1": 0.8416918429003022,
            "MacroF1": 0.8408915736149335,
            "Memory in Mb": 4.737677574157715,
            "Time in s": 216.09684699999997
          },
          {
            "step": 1702,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8418577307466196,
            "MicroF1": 0.8418577307466195,
            "MacroF1": 0.8423710518418951,
            "Memory in Mb": 4.180461883544922,
            "Time in s": 227.29100499999996
          },
          {
            "step": 1748,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8431597023468803,
            "MicroF1": 0.8431597023468804,
            "MacroF1": 0.8432643493367186,
            "Memory in Mb": 4.331151962280273,
            "Time in s": 238.7660419999999
          },
          {
            "step": 1794,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8455103179029559,
            "MicroF1": 0.8455103179029559,
            "MacroF1": 0.8449435902582664,
            "Memory in Mb": 4.424459457397461,
            "Time in s": 250.52640599999992
          },
          {
            "step": 1840,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8466557911908646,
            "MicroF1": 0.8466557911908648,
            "MacroF1": 0.8462222022075542,
            "Memory in Mb": 4.3217973709106445,
            "Time in s": 262.5845499999999
          },
          {
            "step": 1886,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8477453580901857,
            "MicroF1": 0.8477453580901856,
            "MacroF1": 0.84772474367672,
            "Memory in Mb": 4.364754676818848,
            "Time in s": 274.94178099999993
          },
          {
            "step": 1932,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8487830139823925,
            "MicroF1": 0.8487830139823925,
            "MacroF1": 0.8484572581714136,
            "Memory in Mb": 4.410244941711426,
            "Time in s": 287.5965719999999
          },
          {
            "step": 1978,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.849772382397572,
            "MicroF1": 0.849772382397572,
            "MacroF1": 0.8495372758679525,
            "Memory in Mb": 4.436578750610352,
            "Time in s": 300.5458569999999
          },
          {
            "step": 2024,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8507167572911517,
            "MicroF1": 0.8507167572911517,
            "MacroF1": 0.8496927624131454,
            "Memory in Mb": 4.292850494384766,
            "Time in s": 313.7872069999999
          },
          {
            "step": 2070,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8511358144030933,
            "MicroF1": 0.8511358144030933,
            "MacroF1": 0.8503705992191455,
            "Memory in Mb": 4.422323226928711,
            "Time in s": 327.3192999999999
          },
          {
            "step": 2116,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8515366430260047,
            "MicroF1": 0.8515366430260047,
            "MacroF1": 0.850305284692234,
            "Memory in Mb": 4.440757751464844,
            "Time in s": 341.1384159999999
          },
          {
            "step": 2162,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8505321610365572,
            "MicroF1": 0.850532161036557,
            "MacroF1": 0.84908675540822,
            "Memory in Mb": 4.611151695251465,
            "Time in s": 355.24865599999987
          },
          {
            "step": 2208,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.851835070231083,
            "MicroF1": 0.851835070231083,
            "MacroF1": 0.8501011345319502,
            "Memory in Mb": 4.809813499450684,
            "Time in s": 369.64566299999984
          },
          {
            "step": 2254,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8490901020861075,
            "MicroF1": 0.8490901020861075,
            "MacroF1": 0.847799327251759,
            "Memory in Mb": 4.949430465698242,
            "Time in s": 384.3275549999999
          },
          {
            "step": 2300,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "ImageSegments",
            "Accuracy": 0.8490648107872988,
            "MicroF1": 0.8490648107872988,
            "MacroF1": 0.8479218608351832,
            "Memory in Mb": 5.295671463012695,
            "Time in s": 399.2890019999999
          },
          {
            "step": 1056,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.6454976303317536,
            "MicroF1": 0.6454976303317536,
            "MacroF1": 0.5867724425586438,
            "Memory in Mb": 7.441350936889648,
            "Time in s": 8.583896
          },
          {
            "step": 2112,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.6826148744670772,
            "MicroF1": 0.6826148744670772,
            "MacroF1": 0.6053874539212664,
            "Memory in Mb": 11.234929084777832,
            "Time in s": 25.378115
          },
          {
            "step": 3168,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.6896116198294916,
            "MicroF1": 0.6896116198294916,
            "MacroF1": 0.6083758872885286,
            "Memory in Mb": 13.40891933441162,
            "Time in s": 50.237849
          },
          {
            "step": 4224,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.6954771489462468,
            "MicroF1": 0.6954771489462468,
            "MacroF1": 0.6085129807470798,
            "Memory in Mb": 18.096717834472656,
            "Time in s": 83.102519
          },
          {
            "step": 5280,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7014586095851487,
            "MicroF1": 0.7014586095851487,
            "MacroF1": 0.6122692721162352,
            "Memory in Mb": 21.446727752685547,
            "Time in s": 123.914838
          },
          {
            "step": 6336,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7021310181531176,
            "MicroF1": 0.7021310181531176,
            "MacroF1": 0.6116513676781078,
            "Memory in Mb": 27.182113647460938,
            "Time in s": 172.776504
          },
          {
            "step": 7392,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7054525774590719,
            "MicroF1": 0.7054525774590719,
            "MacroF1": 0.6129808753663538,
            "Memory in Mb": 26.876797676086422,
            "Time in s": 229.478895
          },
          {
            "step": 8448,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.70853557476027,
            "MicroF1": 0.70853557476027,
            "MacroF1": 0.6147213044531655,
            "Memory in Mb": 31.851184844970703,
            "Time in s": 293.916345
          },
          {
            "step": 9504,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7137745974955277,
            "MicroF1": 0.7137745974955277,
            "MacroF1": 0.6175531178778296,
            "Memory in Mb": 22.848219871521,
            "Time in s": 366.230765
          },
          {
            "step": 10560,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7174921867601098,
            "MicroF1": 0.7174921867601098,
            "MacroF1": 0.619713417782018,
            "Memory in Mb": 19.317788124084476,
            "Time in s": 446.248603
          },
          {
            "step": 11616,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.717606543263022,
            "MicroF1": 0.717606543263022,
            "MacroF1": 0.618960125586482,
            "Memory in Mb": 19.568995475769043,
            "Time in s": 533.7141509999999
          },
          {
            "step": 12672,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7184121221687317,
            "MicroF1": 0.7184121221687317,
            "MacroF1": 0.6302774396409263,
            "Memory in Mb": 23.59817409515381,
            "Time in s": 628.5255009999998
          },
          {
            "step": 13728,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7373060391928317,
            "MicroF1": 0.7373060391928317,
            "MacroF1": 0.7337291247132964,
            "Memory in Mb": 6.318717002868652,
            "Time in s": 729.5138609999999
          },
          {
            "step": 14784,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.744774403030508,
            "MicroF1": 0.7447744030305079,
            "MacroF1": 0.7439388578060665,
            "Memory in Mb": 4.747281074523926,
            "Time in s": 836.7314339999999
          },
          {
            "step": 15840,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7374834269840268,
            "MicroF1": 0.7374834269840268,
            "MacroF1": 0.7388535634976899,
            "Memory in Mb": 8.635688781738281,
            "Time in s": 951.47119
          },
          {
            "step": 16896,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7324652263983427,
            "MicroF1": 0.7324652263983427,
            "MacroF1": 0.736003592775451,
            "Memory in Mb": 13.742908477783203,
            "Time in s": 1073.59215
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7253077822962509,
            "MicroF1": 0.7253077822962509,
            "MacroF1": 0.7305072565778182,
            "Memory in Mb": 20.00777053833008,
            "Time in s": 1203.171307
          },
          {
            "step": 19008,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7387804493081497,
            "MicroF1": 0.7387804493081497,
            "MacroF1": 0.7395324944779035,
            "Memory in Mb": 6.087196350097656,
            "Time in s": 1339.825862
          },
          {
            "step": 20064,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7439066939141704,
            "MicroF1": 0.7439066939141704,
            "MacroF1": 0.7399287274487314,
            "Memory in Mb": 7.841000556945801,
            "Time in s": 1483.477757
          },
          {
            "step": 21120,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7456792461764288,
            "MicroF1": 0.7456792461764288,
            "MacroF1": 0.738136498436516,
            "Memory in Mb": 11.804688453674316,
            "Time in s": 1634.941518
          },
          {
            "step": 22176,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7464712514092446,
            "MicroF1": 0.7464712514092445,
            "MacroF1": 0.7355899333520025,
            "Memory in Mb": 17.346091270446777,
            "Time in s": 1793.971452
          },
          {
            "step": 23232,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7486548146872714,
            "MicroF1": 0.7486548146872714,
            "MacroF1": 0.7347795423630049,
            "Memory in Mb": 20.6541748046875,
            "Time in s": 1960.617459
          },
          {
            "step": 24288,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7502367521719439,
            "MicroF1": 0.7502367521719437,
            "MacroF1": 0.7334324471857778,
            "Memory in Mb": 21.727876663208008,
            "Time in s": 2134.847723
          },
          {
            "step": 25344,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7523576530008287,
            "MicroF1": 0.7523576530008288,
            "MacroF1": 0.7330792890892175,
            "Memory in Mb": 28.11653995513916,
            "Time in s": 2316.991759
          },
          {
            "step": 26400,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7535891511042085,
            "MicroF1": 0.7535891511042085,
            "MacroF1": 0.731955812013067,
            "Memory in Mb": 28.993709564208984,
            "Time in s": 2507.189995
          },
          {
            "step": 27456,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7540338736113641,
            "MicroF1": 0.7540338736113641,
            "MacroF1": 0.7298780765329144,
            "Memory in Mb": 35.52100467681885,
            "Time in s": 2705.597193
          },
          {
            "step": 28512,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7522710532776823,
            "MicroF1": 0.7522710532776823,
            "MacroF1": 0.7301216768723076,
            "Memory in Mb": 19.597237586975098,
            "Time in s": 2912.282888
          },
          {
            "step": 29568,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7502621165488551,
            "MicroF1": 0.7502621165488552,
            "MacroF1": 0.733319854895679,
            "Memory in Mb": 15.064599990844728,
            "Time in s": 3126.360865
          },
          {
            "step": 30624,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7500244913953564,
            "MicroF1": 0.7500244913953564,
            "MacroF1": 0.7381499467352403,
            "Memory in Mb": 21.998522758483887,
            "Time in s": 3348.070871
          },
          {
            "step": 31680,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7493292086240096,
            "MicroF1": 0.7493292086240096,
            "MacroF1": 0.7414716120706107,
            "Memory in Mb": 28.99252319335937,
            "Time in s": 3577.2433
          },
          {
            "step": 32736,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7494424927447686,
            "MicroF1": 0.7494424927447686,
            "MacroF1": 0.7447602446394828,
            "Memory in Mb": 36.39131259918213,
            "Time in s": 3813.876585
          },
          {
            "step": 33792,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7448137077920156,
            "MicroF1": 0.7448137077920156,
            "MacroF1": 0.7415559043607837,
            "Memory in Mb": 7.406244277954102,
            "Time in s": 4059.132247
          },
          {
            "step": 34848,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7397193445633771,
            "MicroF1": 0.739719344563377,
            "MacroF1": 0.7363475181006618,
            "Memory in Mb": 8.795232772827148,
            "Time in s": 4312.474973
          },
          {
            "step": 35904,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7365679748210456,
            "MicroF1": 0.7365679748210455,
            "MacroF1": 0.7329849736783064,
            "Memory in Mb": 11.10138702392578,
            "Time in s": 4573.597154
          },
          {
            "step": 36960,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7330014340214832,
            "MicroF1": 0.7330014340214832,
            "MacroF1": 0.7293557861681861,
            "Memory in Mb": 15.830912590026855,
            "Time in s": 4842.477951
          },
          {
            "step": 38016,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7302643693278968,
            "MicroF1": 0.7302643693278967,
            "MacroF1": 0.7264691718738406,
            "Memory in Mb": 19.795815467834476,
            "Time in s": 5119.095735
          },
          {
            "step": 39072,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7309513449873307,
            "MicroF1": 0.7309513449873307,
            "MacroF1": 0.7270525503986339,
            "Memory in Mb": 9.05908489227295,
            "Time in s": 5403.358583
          },
          {
            "step": 40128,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.729284521643781,
            "MicroF1": 0.729284521643781,
            "MacroF1": 0.7256952486493923,
            "Memory in Mb": 11.242535591125488,
            "Time in s": 5695.769582
          },
          {
            "step": 41184,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7294029089672923,
            "MicroF1": 0.7294029089672922,
            "MacroF1": 0.7260996194485368,
            "Memory in Mb": 9.506610870361328,
            "Time in s": 5995.177374
          },
          {
            "step": 42240,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7298941736310045,
            "MicroF1": 0.7298941736310045,
            "MacroF1": 0.7269475794208268,
            "Memory in Mb": 15.258689880371094,
            "Time in s": 6301.374087
          },
          {
            "step": 43296,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7306848365862109,
            "MicroF1": 0.7306848365862109,
            "MacroF1": 0.7280100891072271,
            "Memory in Mb": 20.2097225189209,
            "Time in s": 6614.284978
          },
          {
            "step": 44352,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7312574688282113,
            "MicroF1": 0.7312574688282113,
            "MacroF1": 0.7287466644577517,
            "Memory in Mb": 26.378506660461422,
            "Time in s": 6934.278340999999
          },
          {
            "step": 45408,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7317814433897857,
            "MicroF1": 0.7317814433897857,
            "MacroF1": 0.7291491859846939,
            "Memory in Mb": 32.061384201049805,
            "Time in s": 7261.498196999999
          },
          {
            "step": 46464,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.732776617954071,
            "MicroF1": 0.732776617954071,
            "MacroF1": 0.7299865007540453,
            "Memory in Mb": 32.25613784790039,
            "Time in s": 7595.964266999999
          },
          {
            "step": 47520,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7334329426124288,
            "MicroF1": 0.7334329426124286,
            "MacroF1": 0.7309449816547512,
            "Memory in Mb": 16.162960052490234,
            "Time in s": 7937.466337999999
          },
          {
            "step": 48576,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7373751930005147,
            "MicroF1": 0.7373751930005147,
            "MacroF1": 0.7352697035426822,
            "Memory in Mb": 12.93554973602295,
            "Time in s": 8285.412244
          },
          {
            "step": 49632,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.741210130765046,
            "MicroF1": 0.741210130765046,
            "MacroF1": 0.739269872700679,
            "Memory in Mb": 15.798639297485352,
            "Time in s": 8639.863329999998
          },
          {
            "step": 50688,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7446682581332491,
            "MicroF1": 0.7446682581332491,
            "MacroF1": 0.7426657147430288,
            "Memory in Mb": 14.252553939819336,
            "Time in s": 9000.792304999999
          },
          {
            "step": 51744,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.7485650232881742,
            "MicroF1": 0.7485650232881743,
            "MacroF1": 0.7463959215624629,
            "Memory in Mb": 16.087495803833008,
            "Time in s": 9368.035475
          },
          {
            "step": 52800,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Insects",
            "Accuracy": 0.752154396863577,
            "MicroF1": 0.752154396863577,
            "MacroF1": 0.7502511872752614,
            "Memory in Mb": 11.339015007019045,
            "Time in s": 9741.135739
          },
          {
            "step": 408,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9803439803439804,
            "MicroF1": 0.9803439803439804,
            "MacroF1": 0.4950372208436724,
            "Memory in Mb": 1.0347824096679688,
            "Time in s": 1.562947
          },
          {
            "step": 816,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9840490797546012,
            "MicroF1": 0.9840490797546012,
            "MacroF1": 0.9559273479637392,
            "Memory in Mb": 2.137660026550293,
            "Time in s": 5.08284
          },
          {
            "step": 1224,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.983646770237122,
            "MicroF1": 0.983646770237122,
            "MacroF1": 0.9660207101584454,
            "Memory in Mb": 3.2939910888671875,
            "Time in s": 10.794145
          },
          {
            "step": 1632,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9803801348865726,
            "MicroF1": 0.9803801348865726,
            "MacroF1": 0.9452685517164728,
            "Memory in Mb": 4.760180473327637,
            "Time in s": 18.719227
          },
          {
            "step": 2040,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.973516429622364,
            "MicroF1": 0.973516429622364,
            "MacroF1": 0.9361195161551138,
            "Memory in Mb": 6.6425981521606445,
            "Time in s": 28.938428
          },
          {
            "step": 2448,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.973028197793216,
            "MicroF1": 0.973028197793216,
            "MacroF1": 0.9615988180290456,
            "Memory in Mb": 5.552071571350098,
            "Time in s": 41.439403
          },
          {
            "step": 2856,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9747810858143608,
            "MicroF1": 0.9747810858143608,
            "MacroF1": 0.9713591464752812,
            "Memory in Mb": 6.8436784744262695,
            "Time in s": 56.378536
          },
          {
            "step": 3264,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.974869751762182,
            "MicroF1": 0.974869751762182,
            "MacroF1": 0.9692034094625394,
            "Memory in Mb": 6.567030906677246,
            "Time in s": 73.934123
          },
          {
            "step": 3672,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9743938981204032,
            "MicroF1": 0.9743938981204032,
            "MacroF1": 0.9689232613591288,
            "Memory in Mb": 8.48116397857666,
            "Time in s": 94.245901
          },
          {
            "step": 4080,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9722971316499142,
            "MicroF1": 0.9722971316499142,
            "MacroF1": 0.96426610548244,
            "Memory in Mb": 5.934853553771973,
            "Time in s": 117.581237
          },
          {
            "step": 4488,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9732560731000668,
            "MicroF1": 0.9732560731000668,
            "MacroF1": 0.9722719909296184,
            "Memory in Mb": 3.644045829772949,
            "Time in s": 143.695153
          },
          {
            "step": 4896,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9748723186925434,
            "MicroF1": 0.9748723186925434,
            "MacroF1": 0.9754037061196345,
            "Memory in Mb": 4.787886619567871,
            "Time in s": 172.738198
          },
          {
            "step": 5304,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9741655666603808,
            "MicroF1": 0.9741655666603808,
            "MacroF1": 0.9716360242916738,
            "Memory in Mb": 5.742301940917969,
            "Time in s": 204.85114700000003
          },
          {
            "step": 5712,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9746104009805638,
            "MicroF1": 0.9746104009805638,
            "MacroF1": 0.9740216295290516,
            "Memory in Mb": 6.782421112060547,
            "Time in s": 240.11050800000004
          },
          {
            "step": 6120,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9738519365909464,
            "MicroF1": 0.9738519365909464,
            "MacroF1": 0.9722333406974256,
            "Memory in Mb": 8.176769256591797,
            "Time in s": 278.69027800000003
          },
          {
            "step": 6528,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9742607629845258,
            "MicroF1": 0.9742607629845258,
            "MacroF1": 0.9741504405159308,
            "Memory in Mb": 4.433716773986816,
            "Time in s": 320.74209900000005
          },
          {
            "step": 6936,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9751982696467196,
            "MicroF1": 0.9751982696467196,
            "MacroF1": 0.9755523782693606,
            "Memory in Mb": 5.133135795593262,
            "Time in s": 366.1265460000001
          },
          {
            "step": 7344,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9757592264741932,
            "MicroF1": 0.9757592264741932,
            "MacroF1": 0.9758485662267348,
            "Memory in Mb": 5.760107040405273,
            "Time in s": 415.04014400000005
          },
          {
            "step": 7752,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9760030963746612,
            "MicroF1": 0.9760030963746612,
            "MacroF1": 0.9758957983961688,
            "Memory in Mb": 6.842521667480469,
            "Time in s": 467.5877
          },
          {
            "step": 8160,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9758548841769824,
            "MicroF1": 0.9758548841769824,
            "MacroF1": 0.9755087152005796,
            "Memory in Mb": 8.403876304626465,
            "Time in s": 523.9395460000001
          },
          {
            "step": 8568,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9751371541963348,
            "MicroF1": 0.9751371541963348,
            "MacroF1": 0.9744422302091884,
            "Memory in Mb": 8.9804105758667,
            "Time in s": 584.299164
          },
          {
            "step": 8976,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.975598885793872,
            "MicroF1": 0.975598885793872,
            "MacroF1": 0.9757626053423432,
            "Memory in Mb": 8.348807334899902,
            "Time in s": 648.764254
          },
          {
            "step": 9384,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.97527443248428,
            "MicroF1": 0.97527443248428,
            "MacroF1": 0.9749874884381716,
            "Memory in Mb": 8.780474662780762,
            "Time in s": 717.435188
          },
          {
            "step": 9792,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9751812889388214,
            "MicroF1": 0.9751812889388214,
            "MacroF1": 0.9751287694103772,
            "Memory in Mb": 7.27089786529541,
            "Time in s": 790.366072
          },
          {
            "step": 10200,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9748994999509756,
            "MicroF1": 0.9748994999509756,
            "MacroF1": 0.9747198913701116,
            "Memory in Mb": 8.182950019836426,
            "Time in s": 867.658962
          },
          {
            "step": 10608,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9744508343546714,
            "MicroF1": 0.9744508343546714,
            "MacroF1": 0.9742218409220016,
            "Memory in Mb": 8.212386131286621,
            "Time in s": 949.401841
          },
          {
            "step": 11016,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9747616886064456,
            "MicroF1": 0.9747616886064456,
            "MacroF1": 0.9748981365239816,
            "Memory in Mb": 7.736974716186523,
            "Time in s": 1035.676852
          },
          {
            "step": 11424,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9748752516851964,
            "MicroF1": 0.9748752516851964,
            "MacroF1": 0.9749367981815978,
            "Memory in Mb": 8.583486557006836,
            "Time in s": 1126.650079
          },
          {
            "step": 11832,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9749809821654974,
            "MicroF1": 0.9749809821654974,
            "MacroF1": 0.9750463661723392,
            "Memory in Mb": 8.887914657592773,
            "Time in s": 1222.478726
          },
          {
            "step": 12240,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9755698995015932,
            "MicroF1": 0.9755698995015932,
            "MacroF1": 0.9757989853757532,
            "Memory in Mb": 9.402573585510254,
            "Time in s": 1323.224256
          },
          {
            "step": 12648,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9760417490313908,
            "MicroF1": 0.9760417490313908,
            "MacroF1": 0.9762258400907322,
            "Memory in Mb": 9.833843231201172,
            "Time in s": 1429.020384
          },
          {
            "step": 13056,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9760245116813482,
            "MicroF1": 0.9760245116813482,
            "MacroF1": 0.9760626338918788,
            "Memory in Mb": 9.939188957214355,
            "Time in s": 1539.962955
          },
          {
            "step": 13464,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9762311520463492,
            "MicroF1": 0.9762311520463492,
            "MacroF1": 0.976330045562598,
            "Memory in Mb": 10.46349811553955,
            "Time in s": 1656.191893
          },
          {
            "step": 13872,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9763535433638526,
            "MicroF1": 0.9763535433638526,
            "MacroF1": 0.9764287224231292,
            "Memory in Mb": 11.428705215454102,
            "Time in s": 1777.90617
          },
          {
            "step": 14280,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9758386441627565,
            "MicroF1": 0.9758386441627565,
            "MacroF1": 0.9757700210755772,
            "Memory in Mb": 10.687178611755373,
            "Time in s": 1905.20097
          },
          {
            "step": 14688,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.976101314087288,
            "MicroF1": 0.976101314087288,
            "MacroF1": 0.9761996431080104,
            "Memory in Mb": 10.750173568725586,
            "Time in s": 2038.11614
          },
          {
            "step": 15096,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9764822789002982,
            "MicroF1": 0.9764822789002982,
            "MacroF1": 0.9765941858257003,
            "Memory in Mb": 10.87511920928955,
            "Time in s": 2176.860193
          },
          {
            "step": 15504,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9762626588402245,
            "MicroF1": 0.9762626588402245,
            "MacroF1": 0.9762697293829714,
            "Memory in Mb": 11.144217491149902,
            "Time in s": 2321.518715
          },
          {
            "step": 15912,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.976305700458802,
            "MicroF1": 0.976305700458802,
            "MacroF1": 0.9763523962033862,
            "Memory in Mb": 10.881969451904297,
            "Time in s": 2472.196158
          },
          {
            "step": 16320,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9761627550707764,
            "MicroF1": 0.9761627550707764,
            "MacroF1": 0.9761821898526978,
            "Memory in Mb": 10.72462272644043,
            "Time in s": 2629.0181850000004
          },
          {
            "step": 16728,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9760267830453756,
            "MicroF1": 0.9760267830453756,
            "MacroF1": 0.9760462981867312,
            "Memory in Mb": 10.397873878479004,
            "Time in s": 2792.0314960000005
          },
          {
            "step": 17136,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9763641669098336,
            "MicroF1": 0.9763641669098336,
            "MacroF1": 0.976427628373518,
            "Memory in Mb": 11.5784912109375,
            "Time in s": 2961.4220420000006
          },
          {
            "step": 17544,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9762868380550648,
            "MicroF1": 0.9762868380550648,
            "MacroF1": 0.9763077393136288,
            "Memory in Mb": 10.780138969421388,
            "Time in s": 3137.2839260000005
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9763801459528716,
            "MicroF1": 0.9763801459528716,
            "MacroF1": 0.9764101118400772,
            "Memory in Mb": 11.24526309967041,
            "Time in s": 3319.622350000001
          },
          {
            "step": 18360,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.97663271420012,
            "MicroF1": 0.97663271420012,
            "MacroF1": 0.976666198082788,
            "Memory in Mb": 11.450252532958984,
            "Time in s": 3508.4984120000004
          },
          {
            "step": 18768,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9769808706772526,
            "MicroF1": 0.9769808706772526,
            "MacroF1": 0.9770112706505792,
            "Memory in Mb": 12.824438095092772,
            "Time in s": 3704.110850000001
          },
          {
            "step": 19176,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9767926988265972,
            "MicroF1": 0.9767926988265972,
            "MacroF1": 0.976797459665624,
            "Memory in Mb": 13.463789939880373,
            "Time in s": 3906.654607
          },
          {
            "step": 19584,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9766123678700914,
            "MicroF1": 0.9766123678700914,
            "MacroF1": 0.97661532368473,
            "Memory in Mb": 12.60595417022705,
            "Time in s": 4116.138220000001
          },
          {
            "step": 19992,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9765894652593669,
            "MicroF1": 0.9765894652593669,
            "MacroF1": 0.976591825772772,
            "Memory in Mb": 11.445868492126465,
            "Time in s": 4332.655771000001
          },
          {
            "step": 20400,
            "track": "Multiclass classification",
            "model": "Stacking",
            "dataset": "Keystroke",
            "Accuracy": 0.9765184567870974,
            "MicroF1": 0.9765184567870974,
            "MacroF1": 0.9765167109502484,
            "Memory in Mb": 12.220311164855955,
            "Time in s": 4556.333966000001
          },
          {
            "step": 46,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.4888888888888889,
            "MicroF1": 0.4888888888888889,
            "MacroF1": 0.4138888888888889,
            "Memory in Mb": 0.8855724334716797,
            "Time in s": 0.380739
          },
          {
            "step": 92,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.6263736263736264,
            "MicroF1": 0.6263736263736264,
            "MacroF1": 0.6295417331131617,
            "Memory in Mb": 0.9400959014892578,
            "Time in s": 0.906366
          },
          {
            "step": 138,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.6788321167883211,
            "MicroF1": 0.6788321167883211,
            "MacroF1": 0.6955125455614023,
            "Memory in Mb": 0.9512205123901368,
            "Time in s": 1.596335
          },
          {
            "step": 184,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.7158469945355191,
            "MicroF1": 0.7158469945355191,
            "MacroF1": 0.7293605295181818,
            "Memory in Mb": 0.9506902694702148,
            "Time in s": 2.451892
          },
          {
            "step": 230,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.74235807860262,
            "MicroF1": 0.74235807860262,
            "MacroF1": 0.7560849066334576,
            "Memory in Mb": 0.9507265090942384,
            "Time in s": 3.478975
          },
          {
            "step": 276,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.7490909090909091,
            "MicroF1": 0.7490909090909091,
            "MacroF1": 0.7654899494294127,
            "Memory in Mb": 0.9522123336791992,
            "Time in s": 4.6524790000000005
          },
          {
            "step": 322,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.7632398753894081,
            "MicroF1": 0.7632398753894081,
            "MacroF1": 0.7699967547900484,
            "Memory in Mb": 0.9522132873535156,
            "Time in s": 5.915859
          },
          {
            "step": 368,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.782016348773842,
            "MicroF1": 0.782016348773842,
            "MacroF1": 0.7847454642968661,
            "Memory in Mb": 0.9517135620117188,
            "Time in s": 7.268692
          },
          {
            "step": 414,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.7869249394673123,
            "MicroF1": 0.7869249394673122,
            "MacroF1": 0.7891209865588749,
            "Memory in Mb": 0.952162742614746,
            "Time in s": 8.714221
          },
          {
            "step": 460,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.7821350762527233,
            "MicroF1": 0.7821350762527233,
            "MacroF1": 0.7829889615631377,
            "Memory in Mb": 0.9522056579589844,
            "Time in s": 10.249346
          },
          {
            "step": 506,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.7861386138613862,
            "MicroF1": 0.7861386138613862,
            "MacroF1": 0.7872755051739567,
            "Memory in Mb": 0.9517154693603516,
            "Time in s": 11.874447
          },
          {
            "step": 552,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.7858439201451906,
            "MicroF1": 0.7858439201451906,
            "MacroF1": 0.7876565639439724,
            "Memory in Mb": 0.9515762329101562,
            "Time in s": 13.588949
          },
          {
            "step": 598,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.7872696817420436,
            "MicroF1": 0.7872696817420435,
            "MacroF1": 0.7897468061485311,
            "Memory in Mb": 0.9521427154541016,
            "Time in s": 15.393404
          },
          {
            "step": 644,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.7822706065318819,
            "MicroF1": 0.7822706065318819,
            "MacroF1": 0.7858452362125997,
            "Memory in Mb": 0.9521217346191406,
            "Time in s": 17.2908
          },
          {
            "step": 690,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.7851959361393324,
            "MicroF1": 0.7851959361393324,
            "MacroF1": 0.788215888108031,
            "Memory in Mb": 0.9515953063964844,
            "Time in s": 19.280843
          },
          {
            "step": 736,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.7836734693877551,
            "MicroF1": 0.783673469387755,
            "MacroF1": 0.7873581098337732,
            "Memory in Mb": 0.9521245956420898,
            "Time in s": 21.362069
          },
          {
            "step": 782,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.7861715749039693,
            "MicroF1": 0.7861715749039692,
            "MacroF1": 0.7892834149474556,
            "Memory in Mb": 0.9521360397338868,
            "Time in s": 23.534270000000003
          },
          {
            "step": 828,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.7847642079806529,
            "MicroF1": 0.7847642079806529,
            "MacroF1": 0.7891292080670234,
            "Memory in Mb": 0.951629638671875,
            "Time in s": 25.799776
          },
          {
            "step": 874,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.7892325315005727,
            "MicroF1": 0.7892325315005727,
            "MacroF1": 0.7922023317831084,
            "Memory in Mb": 0.9516172409057616,
            "Time in s": 28.155901
          },
          {
            "step": 920,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.7889009793253536,
            "MicroF1": 0.7889009793253536,
            "MacroF1": 0.7905862723276574,
            "Memory in Mb": 0.9520702362060548,
            "Time in s": 30.602138
          },
          {
            "step": 966,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.78860103626943,
            "MicroF1": 0.78860103626943,
            "MacroF1": 0.7894031693051725,
            "Memory in Mb": 0.952082633972168,
            "Time in s": 33.138731
          },
          {
            "step": 1012,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.7873392680514342,
            "MicroF1": 0.7873392680514342,
            "MacroF1": 0.7878835011583499,
            "Memory in Mb": 0.9515609741210938,
            "Time in s": 35.768379
          },
          {
            "step": 1058,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.7899716177861873,
            "MicroF1": 0.7899716177861873,
            "MacroF1": 0.7897146415510686,
            "Memory in Mb": 0.9520339965820312,
            "Time in s": 38.488246
          },
          {
            "step": 1104,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.7905711695376246,
            "MicroF1": 0.7905711695376246,
            "MacroF1": 0.7902707663283154,
            "Memory in Mb": 0.9521427154541016,
            "Time in s": 41.298010000000005
          },
          {
            "step": 1150,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.7919930374238469,
            "MicroF1": 0.7919930374238469,
            "MacroF1": 0.7910217164829003,
            "Memory in Mb": 0.9516496658325196,
            "Time in s": 44.198117
          },
          {
            "step": 1196,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.793305439330544,
            "MicroF1": 0.793305439330544,
            "MacroF1": 0.7926565595792737,
            "Memory in Mb": 0.9516582489013672,
            "Time in s": 47.188338
          },
          {
            "step": 1242,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.7921031426269137,
            "MicroF1": 0.7921031426269137,
            "MacroF1": 0.791644431462719,
            "Memory in Mb": 0.9522132873535156,
            "Time in s": 50.271512
          },
          {
            "step": 1288,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.7964257964257965,
            "MicroF1": 0.7964257964257965,
            "MacroF1": 0.7949172523959339,
            "Memory in Mb": 0.952223777770996,
            "Time in s": 53.444669000000005
          },
          {
            "step": 1334,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.795198799699925,
            "MicroF1": 0.7951987996999249,
            "MacroF1": 0.7938516970082157,
            "Memory in Mb": 0.9516925811767578,
            "Time in s": 56.707980000000006
          },
          {
            "step": 1380,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.7955039883973894,
            "MicroF1": 0.7955039883973894,
            "MacroF1": 0.794312731896104,
            "Memory in Mb": 0.9516897201538086,
            "Time in s": 60.06121
          },
          {
            "step": 1426,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.7971929824561403,
            "MicroF1": 0.7971929824561403,
            "MacroF1": 0.7952130436298935,
            "Memory in Mb": 0.9521360397338868,
            "Time in s": 63.507281000000006
          },
          {
            "step": 1472,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.8008157715839564,
            "MicroF1": 0.8008157715839563,
            "MacroF1": 0.7971305683653547,
            "Memory in Mb": 0.9521236419677734,
            "Time in s": 67.043451
          },
          {
            "step": 1518,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.8015820698747528,
            "MicroF1": 0.8015820698747528,
            "MacroF1": 0.7969787037511136,
            "Memory in Mb": 0.95166015625,
            "Time in s": 70.670162
          },
          {
            "step": 1564,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.8016634676903391,
            "MicroF1": 0.8016634676903392,
            "MacroF1": 0.7975983332578384,
            "Memory in Mb": 0.9521465301513672,
            "Time in s": 74.387065
          },
          {
            "step": 1610,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.8017402113113735,
            "MicroF1": 0.8017402113113735,
            "MacroF1": 0.7969541458804642,
            "Memory in Mb": 0.9521703720092772,
            "Time in s": 78.19725000000001
          },
          {
            "step": 1656,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.8018126888217523,
            "MicroF1": 0.8018126888217523,
            "MacroF1": 0.7970318311622571,
            "Memory in Mb": 0.9516267776489258,
            "Time in s": 82.09764400000002
          },
          {
            "step": 1702,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.8018812463256908,
            "MicroF1": 0.8018812463256908,
            "MacroF1": 0.7992301124377234,
            "Memory in Mb": 0.9516735076904296,
            "Time in s": 86.09102400000002
          },
          {
            "step": 1748,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.8036634230108758,
            "MicroF1": 0.8036634230108759,
            "MacroF1": 0.8004815801809151,
            "Memory in Mb": 0.952157974243164,
            "Time in s": 90.17551900000002
          },
          {
            "step": 1794,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.8042387060791969,
            "MicroF1": 0.8042387060791969,
            "MacroF1": 0.799787639242423,
            "Memory in Mb": 0.9521493911743164,
            "Time in s": 94.35067000000002
          },
          {
            "step": 1840,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.8053289831430125,
            "MicroF1": 0.8053289831430125,
            "MacroF1": 0.8009597766649573,
            "Memory in Mb": 0.9516563415527344,
            "Time in s": 98.61893200000004
          },
          {
            "step": 1886,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.803183023872679,
            "MicroF1": 0.8031830238726789,
            "MacroF1": 0.799227837217116,
            "Memory in Mb": 0.9521894454956056,
            "Time in s": 102.97762100000004
          },
          {
            "step": 1932,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.8032107716209218,
            "MicroF1": 0.8032107716209218,
            "MacroF1": 0.7985344176802335,
            "Memory in Mb": 0.9521827697753906,
            "Time in s": 107.42655600000003
          },
          {
            "step": 1978,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.8042488619119879,
            "MicroF1": 0.8042488619119877,
            "MacroF1": 0.7992002826592023,
            "Memory in Mb": 0.9516563415527344,
            "Time in s": 111.96600400000004
          },
          {
            "step": 2024,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.8057340583292141,
            "MicroF1": 0.8057340583292142,
            "MacroF1": 0.799488243695578,
            "Memory in Mb": 0.9516725540161132,
            "Time in s": 116.59842900000002
          },
          {
            "step": 2070,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.80521991300145,
            "MicroF1": 0.80521991300145,
            "MacroF1": 0.7990099218703556,
            "Memory in Mb": 0.9521942138671876,
            "Time in s": 121.31866500000002
          },
          {
            "step": 2116,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.8056737588652483,
            "MicroF1": 0.8056737588652483,
            "MacroF1": 0.798658845250099,
            "Memory in Mb": 0.9521694183349608,
            "Time in s": 126.12612700000004
          },
          {
            "step": 2162,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.8061082832022212,
            "MicroF1": 0.8061082832022212,
            "MacroF1": 0.7986518526284686,
            "Memory in Mb": 0.9516706466674804,
            "Time in s": 131.02082400000003
          },
          {
            "step": 2208,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.8078840054372451,
            "MicroF1": 0.8078840054372451,
            "MacroF1": 0.7995103660963299,
            "Memory in Mb": 0.9521732330322266,
            "Time in s": 136.00538900000004
          },
          {
            "step": 2254,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.8047048379937861,
            "MicroF1": 0.8047048379937861,
            "MacroF1": 0.7963417515999387,
            "Memory in Mb": 0.9521608352661132,
            "Time in s": 141.07727400000005
          },
          {
            "step": 2300,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "ImageSegments",
            "Accuracy": 0.8033927794693345,
            "MicroF1": 0.8033927794693345,
            "MacroF1": 0.7949752803158223,
            "Memory in Mb": 0.9516582489013672,
            "Time in s": 146.23634000000004
          },
          {
            "step": 1056,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6293838862559241,
            "MicroF1": 0.6293838862559241,
            "MacroF1": 0.5939725193500994,
            "Memory in Mb": 1.5110340118408203,
            "Time in s": 3.200552
          },
          {
            "step": 2112,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.62482235907153,
            "MicroF1": 0.62482235907153,
            "MacroF1": 0.5894737350922559,
            "Memory in Mb": 1.5110177993774414,
            "Time in s": 9.230037
          },
          {
            "step": 3168,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6198294916324597,
            "MicroF1": 0.6198294916324597,
            "MacroF1": 0.5838888884930272,
            "Memory in Mb": 1.5110721588134766,
            "Time in s": 17.854201
          },
          {
            "step": 4224,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6192280369405636,
            "MicroF1": 0.6192280369405636,
            "MacroF1": 0.5835519631382228,
            "Memory in Mb": 1.5110435485839844,
            "Time in s": 28.950406
          },
          {
            "step": 5280,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6256866830839174,
            "MicroF1": 0.6256866830839174,
            "MacroF1": 0.5887468172490868,
            "Memory in Mb": 1.511063575744629,
            "Time in s": 42.489771000000005
          },
          {
            "step": 6336,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6187845303867403,
            "MicroF1": 0.6187845303867403,
            "MacroF1": 0.5833486573822239,
            "Memory in Mb": 1.5110454559326172,
            "Time in s": 58.469284
          },
          {
            "step": 7392,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6180489784873495,
            "MicroF1": 0.6180489784873495,
            "MacroF1": 0.5826198728106428,
            "Memory in Mb": 1.5110721588134766,
            "Time in s": 76.881409
          },
          {
            "step": 8448,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.619746655617379,
            "MicroF1": 0.619746655617379,
            "MacroF1": 0.5840081546383048,
            "Memory in Mb": 1.5110502243041992,
            "Time in s": 97.728436
          },
          {
            "step": 9504,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6190676628433126,
            "MicroF1": 0.6190676628433126,
            "MacroF1": 0.5828637425505069,
            "Memory in Mb": 1.511042594909668,
            "Time in s": 121.006042
          },
          {
            "step": 10560,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6198503646178616,
            "MicroF1": 0.6198503646178616,
            "MacroF1": 0.5836946750940745,
            "Memory in Mb": 1.5110759735107422,
            "Time in s": 146.711441
          },
          {
            "step": 11616,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6175634954799828,
            "MicroF1": 0.6175634954799828,
            "MacroF1": 0.5822534545682404,
            "Memory in Mb": 1.511033058166504,
            "Time in s": 174.85184600000002
          },
          {
            "step": 12672,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6204719438086971,
            "MicroF1": 0.6204719438086971,
            "MacroF1": 0.5879866433279776,
            "Memory in Mb": 1.5111761093139648,
            "Time in s": 205.425851
          },
          {
            "step": 13728,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6369199388067313,
            "MicroF1": 0.6369199388067313,
            "MacroF1": 0.618745437324273,
            "Memory in Mb": 1.5112380981445312,
            "Time in s": 238.425571
          },
          {
            "step": 14784,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.630386254481499,
            "MicroF1": 0.630386254481499,
            "MacroF1": 0.6115259179282228,
            "Memory in Mb": 1.5110998153686523,
            "Time in s": 273.85181700000004
          },
          {
            "step": 15840,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.5992171222930741,
            "MicroF1": 0.5992171222930741,
            "MacroF1": 0.581747071745844,
            "Memory in Mb": 1.5110797882080078,
            "Time in s": 311.71395800000005
          },
          {
            "step": 16896,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.5783959751405742,
            "MicroF1": 0.5783959751405742,
            "MacroF1": 0.5619501594422388,
            "Memory in Mb": 1.511063575744629,
            "Time in s": 352.00364300000007
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.5631998217369506,
            "MicroF1": 0.5631998217369506,
            "MacroF1": 0.5464708450044057,
            "Memory in Mb": 1.511117935180664,
            "Time in s": 394.7217830000001
          },
          {
            "step": 19008,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.565528489503867,
            "MicroF1": 0.565528489503867,
            "MacroF1": 0.5447789723081985,
            "Memory in Mb": 1.5110950469970703,
            "Time in s": 439.87157300000007
          },
          {
            "step": 20064,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.5725464785924338,
            "MicroF1": 0.5725464785924338,
            "MacroF1": 0.5493312346450109,
            "Memory in Mb": 2.16702938079834,
            "Time in s": 487.436087
          },
          {
            "step": 21120,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.5819404327856432,
            "MicroF1": 0.5819404327856432,
            "MacroF1": 0.5575973426297249,
            "Memory in Mb": 2.167789459228516,
            "Time in s": 537.344767
          },
          {
            "step": 22176,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.5905298759864712,
            "MicroF1": 0.5905298759864712,
            "MacroF1": 0.5648531785235197,
            "Memory in Mb": 2.167774200439453,
            "Time in s": 589.565521
          },
          {
            "step": 23232,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.5995867590719297,
            "MicroF1": 0.5995867590719297,
            "MacroF1": 0.5728007753824246,
            "Memory in Mb": 2.167778968811035,
            "Time in s": 644.107989
          },
          {
            "step": 24288,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6068678717009099,
            "MicroF1": 0.6068678717009099,
            "MacroF1": 0.578555560305262,
            "Memory in Mb": 2.16780948638916,
            "Time in s": 700.966552
          },
          {
            "step": 25344,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6143313735548278,
            "MicroF1": 0.6143313735548278,
            "MacroF1": 0.5848116898462843,
            "Memory in Mb": 2.167755126953125,
            "Time in s": 760.153761
          },
          {
            "step": 26400,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.621084131974696,
            "MicroF1": 0.621084131974696,
            "MacroF1": 0.5900605973096019,
            "Memory in Mb": 2.1677980422973637,
            "Time in s": 821.662998
          },
          {
            "step": 27456,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6266618102349298,
            "MicroF1": 0.6266618102349298,
            "MacroF1": 0.5936647802901621,
            "Memory in Mb": 2.167790412902832,
            "Time in s": 885.506266
          },
          {
            "step": 28512,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6295114166462067,
            "MicroF1": 0.6295114166462067,
            "MacroF1": 0.5991480792709615,
            "Memory in Mb": 2.168045997619629,
            "Time in s": 951.65113
          },
          {
            "step": 29568,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6294517536442655,
            "MicroF1": 0.6294517536442655,
            "MacroF1": 0.6037001563215106,
            "Memory in Mb": 2.1680641174316406,
            "Time in s": 1020.249901
          },
          {
            "step": 30624,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6287104463964993,
            "MicroF1": 0.6287104463964993,
            "MacroF1": 0.6068237930795873,
            "Memory in Mb": 2.168071746826172,
            "Time in s": 1091.329507
          },
          {
            "step": 31680,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6292496606584804,
            "MicroF1": 0.6292496606584804,
            "MacroF1": 0.6106666463743293,
            "Memory in Mb": 2.1680679321289062,
            "Time in s": 1164.886578
          },
          {
            "step": 32736,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6302734076676341,
            "MicroF1": 0.6302734076676341,
            "MacroF1": 0.614251388937007,
            "Memory in Mb": 2.168027877807617,
            "Time in s": 1240.9182979999998
          },
          {
            "step": 33792,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6266165547039152,
            "MicroF1": 0.6266165547039152,
            "MacroF1": 0.6112639299818544,
            "Memory in Mb": 2.1678638458251958,
            "Time in s": 1319.4287269999998
          },
          {
            "step": 34848,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6216604011823113,
            "MicroF1": 0.6216604011823113,
            "MacroF1": 0.6060150865308916,
            "Memory in Mb": 2.1678390502929688,
            "Time in s": 1400.4187529999997
          },
          {
            "step": 35904,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6181377600757597,
            "MicroF1": 0.6181377600757597,
            "MacroF1": 0.6018714875673907,
            "Memory in Mb": 2.167888641357422,
            "Time in s": 1483.8935999999997
          },
          {
            "step": 36960,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6138153088557591,
            "MicroF1": 0.6138153088557591,
            "MacroF1": 0.5971057932031453,
            "Memory in Mb": 2.167864799499512,
            "Time in s": 1569.8367669999996
          },
          {
            "step": 38016,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6116796001578324,
            "MicroF1": 0.6116796001578324,
            "MacroF1": 0.5945381289951768,
            "Memory in Mb": 2.709075927734375,
            "Time in s": 1658.2515679999997
          },
          {
            "step": 39072,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6122187811932124,
            "MicroF1": 0.6122187811932124,
            "MacroF1": 0.5950787740952911,
            "Memory in Mb": 2.823759078979492,
            "Time in s": 1749.1539719999994
          },
          {
            "step": 40128,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6125052956861963,
            "MicroF1": 0.6125052956861963,
            "MacroF1": 0.5964110573184415,
            "Memory in Mb": 2.823785781860352,
            "Time in s": 1842.4635979999996
          },
          {
            "step": 41184,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6145254109705461,
            "MicroF1": 0.6145254109705461,
            "MacroF1": 0.5992770713855892,
            "Memory in Mb": 2.82379150390625,
            "Time in s": 1938.0560169999997
          },
          {
            "step": 42240,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6163024692819432,
            "MicroF1": 0.6163024692819432,
            "MacroF1": 0.601670854132613,
            "Memory in Mb": 2.823772430419922,
            "Time in s": 2035.9261369999997
          },
          {
            "step": 43296,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6181776186626631,
            "MicroF1": 0.6181776186626631,
            "MacroF1": 0.6041281005310094,
            "Memory in Mb": 2.8237924575805664,
            "Time in s": 2136.072454
          },
          {
            "step": 44352,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6197605465491195,
            "MicroF1": 0.6197605465491195,
            "MacroF1": 0.6062005996937425,
            "Memory in Mb": 2.824528694152832,
            "Time in s": 2238.51149
          },
          {
            "step": 45408,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6214019864778558,
            "MicroF1": 0.6214019864778558,
            "MacroF1": 0.607792464273323,
            "Memory in Mb": 2.824528694152832,
            "Time in s": 2343.241606
          },
          {
            "step": 46464,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6233992639304393,
            "MicroF1": 0.6233992639304393,
            "MacroF1": 0.6097993182820672,
            "Memory in Mb": 2.824531555175781,
            "Time in s": 2450.26566
          },
          {
            "step": 47520,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6260864075422463,
            "MicroF1": 0.6260864075422463,
            "MacroF1": 0.6129939002712749,
            "Memory in Mb": 2.8244552612304688,
            "Time in s": 2559.63137
          },
          {
            "step": 48576,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6300154400411735,
            "MicroF1": 0.6300154400411735,
            "MacroF1": 0.6173873766747581,
            "Memory in Mb": 2.824479103088379,
            "Time in s": 2671.379304
          },
          {
            "step": 49632,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6343011424311418,
            "MicroF1": 0.6343011424311418,
            "MacroF1": 0.621931196280001,
            "Memory in Mb": 2.8244781494140625,
            "Time in s": 2785.498386
          },
          {
            "step": 50688,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.638506914988064,
            "MicroF1": 0.638506914988064,
            "MacroF1": 0.6263145143911814,
            "Memory in Mb": 2.947113037109375,
            "Time in s": 2902.0058780000004
          },
          {
            "step": 51744,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6434686817540537,
            "MicroF1": 0.6434686817540537,
            "MacroF1": 0.6313977027921706,
            "Memory in Mb": 3.184900283813477,
            "Time in s": 3020.8771950000005
          },
          {
            "step": 52800,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Insects",
            "Accuracy": 0.6479289380480691,
            "MicroF1": 0.6479289380480691,
            "MacroF1": 0.635943324049664,
            "Memory in Mb": 3.3886165618896484,
            "Time in s": 3141.9869480000007
          },
          {
            "step": 408,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.9828009828009828,
            "MicroF1": 0.9828009828009828,
            "MacroF1": 0.6067632850241546,
            "Memory in Mb": 0.6423864364624023,
            "Time in s": 0.703596
          },
          {
            "step": 816,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.9546012269938652,
            "MicroF1": 0.9546012269938652,
            "MacroF1": 0.7993954329623859,
            "Memory in Mb": 0.8351936340332031,
            "Time in s": 2.165771
          },
          {
            "step": 1224,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.9206868356500408,
            "MicroF1": 0.9206868356500408,
            "MacroF1": 0.9055597826779512,
            "Memory in Mb": 1.029007911682129,
            "Time in s": 4.467328
          },
          {
            "step": 1632,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.9307173513182097,
            "MicroF1": 0.9307173513182097,
            "MacroF1": 0.917259757091744,
            "Memory in Mb": 1.2232952117919922,
            "Time in s": 7.6892
          },
          {
            "step": 2040,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.9303580186365864,
            "MicroF1": 0.9303580186365864,
            "MacroF1": 0.919916287137026,
            "Memory in Mb": 1.428065299987793,
            "Time in s": 11.917356
          },
          {
            "step": 2448,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.9060073559460564,
            "MicroF1": 0.9060073559460564,
            "MacroF1": 0.9093956340782632,
            "Memory in Mb": 1.6218795776367188,
            "Time in s": 17.211204
          },
          {
            "step": 2856,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.9103327495621716,
            "MicroF1": 0.9103327495621716,
            "MacroF1": 0.8980697688452707,
            "Memory in Mb": 1.8146867752075195,
            "Time in s": 23.617224
          },
          {
            "step": 3264,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.904382470119522,
            "MicroF1": 0.904382470119522,
            "MacroF1": 0.888202704220525,
            "Memory in Mb": 2.0085010528564453,
            "Time in s": 31.150342
          },
          {
            "step": 3672,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8994824298556252,
            "MicroF1": 0.8994824298556252,
            "MacroF1": 0.8972334256598172,
            "Memory in Mb": 2.2018117904663086,
            "Time in s": 39.868796
          },
          {
            "step": 4080,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8945820053934788,
            "MicroF1": 0.8945820053934787,
            "MacroF1": 0.8851783489415491,
            "Memory in Mb": 2.420787811279297,
            "Time in s": 49.793690000000005
          },
          {
            "step": 4488,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8914642299977713,
            "MicroF1": 0.8914642299977713,
            "MacroF1": 0.898372373723482,
            "Memory in Mb": 2.6146020889282227,
            "Time in s": 61.011475
          },
          {
            "step": 4896,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8880490296220633,
            "MicroF1": 0.8880490296220633,
            "MacroF1": 0.8932697641963906,
            "Memory in Mb": 2.807912826538086,
            "Time in s": 73.572163
          },
          {
            "step": 5304,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.883085046200264,
            "MicroF1": 0.883085046200264,
            "MacroF1": 0.8680917053752625,
            "Memory in Mb": 3.000770568847656,
            "Time in s": 87.52603
          },
          {
            "step": 5712,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8746279110488531,
            "MicroF1": 0.8746279110488531,
            "MacroF1": 0.8792177397015432,
            "Memory in Mb": 3.194584846496582,
            "Time in s": 102.922365
          },
          {
            "step": 6120,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8695865337473443,
            "MicroF1": 0.8695865337473442,
            "MacroF1": 0.8546904737358852,
            "Memory in Mb": 3.387392044067383,
            "Time in s": 119.819987
          },
          {
            "step": 6528,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8579745671824728,
            "MicroF1": 0.8579745671824728,
            "MacroF1": 0.858067415232278,
            "Memory in Mb": 3.5812063217163086,
            "Time in s": 138.28125400000002
          },
          {
            "step": 6936,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8537851478010093,
            "MicroF1": 0.8537851478010093,
            "MacroF1": 0.8590096923865055,
            "Memory in Mb": 3.774517059326172,
            "Time in s": 158.37518500000002
          },
          {
            "step": 7344,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8594579871986926,
            "MicroF1": 0.8594579871986926,
            "MacroF1": 0.8620220702364139,
            "Memory in Mb": 3.9702539443969727,
            "Time in s": 180.164436
          },
          {
            "step": 7752,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8593729841310799,
            "MicroF1": 0.8593729841310799,
            "MacroF1": 0.8617576440335053,
            "Memory in Mb": 4.164068222045898,
            "Time in s": 203.71390800000003
          },
          {
            "step": 8160,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8601544306900355,
            "MicroF1": 0.8601544306900355,
            "MacroF1": 0.8605355806611993,
            "Memory in Mb": 4.357378959655762,
            "Time in s": 229.09093700000005
          },
          {
            "step": 8568,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8596941753239173,
            "MicroF1": 0.8596941753239173,
            "MacroF1": 0.8627767842417701,
            "Memory in Mb": 4.60028076171875,
            "Time in s": 256.36638600000003
          },
          {
            "step": 8976,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8599442896935933,
            "MicroF1": 0.8599442896935933,
            "MacroF1": 0.8629838037923419,
            "Memory in Mb": 4.794095039367676,
            "Time in s": 285.60935500000005
          },
          {
            "step": 9384,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8581477139507621,
            "MicroF1": 0.8581477139507621,
            "MacroF1": 0.8592031021693959,
            "Memory in Mb": 4.986902236938477,
            "Time in s": 316.88668100000007
          },
          {
            "step": 9792,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8539475028087019,
            "MicroF1": 0.8539475028087019,
            "MacroF1": 0.8546213426549989,
            "Memory in Mb": 5.180716514587402,
            "Time in s": 350.2631590000001
          },
          {
            "step": 10200,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8465535836846749,
            "MicroF1": 0.8465535836846749,
            "MacroF1": 0.8431270001478435,
            "Memory in Mb": 5.374000549316406,
            "Time in s": 385.8123490000001
          },
          {
            "step": 10608,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8300179126991609,
            "MicroF1": 0.8300179126991609,
            "MacroF1": 0.8240754775818138,
            "Memory in Mb": 5.566834449768066,
            "Time in s": 423.5940000000001
          },
          {
            "step": 11016,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8254198819791194,
            "MicroF1": 0.8254198819791194,
            "MacroF1": 0.8271925616445298,
            "Memory in Mb": 5.760648727416992,
            "Time in s": 463.6719130000001
          },
          {
            "step": 11424,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.820449969360063,
            "MicroF1": 0.820449969360063,
            "MacroF1": 0.8166393841205931,
            "Memory in Mb": 5.953959465026856,
            "Time in s": 506.1140620000001
          },
          {
            "step": 11832,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8169216465218494,
            "MicroF1": 0.8169216465218494,
            "MacroF1": 0.8172029683603622,
            "Memory in Mb": 6.146766662597656,
            "Time in s": 550.9879090000001
          },
          {
            "step": 12240,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8200016341204347,
            "MicroF1": 0.8200016341204347,
            "MacroF1": 0.8225884010623591,
            "Memory in Mb": 6.340580940246582,
            "Time in s": 598.362302
          },
          {
            "step": 12648,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8167154265833795,
            "MicroF1": 0.8167154265833795,
            "MacroF1": 0.8162987105601626,
            "Memory in Mb": 6.533388137817383,
            "Time in s": 648.30112
          },
          {
            "step": 13056,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8121792416698583,
            "MicroF1": 0.8121792416698584,
            "MacroF1": 0.8136075732214813,
            "Memory in Mb": 6.727202415466309,
            "Time in s": 700.874955
          },
          {
            "step": 13464,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8099234940206492,
            "MicroF1": 0.8099234940206492,
            "MacroF1": 0.8122480630127521,
            "Memory in Mb": 6.920539855957031,
            "Time in s": 756.151971
          },
          {
            "step": 13872,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.810539975488429,
            "MicroF1": 0.810539975488429,
            "MacroF1": 0.8134726777385565,
            "Memory in Mb": 7.113347053527832,
            "Time in s": 814.194174
          },
          {
            "step": 14280,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8103508649065061,
            "MicroF1": 0.810350864906506,
            "MacroF1": 0.8130549704062812,
            "Memory in Mb": 7.307161331176758,
            "Time in s": 875.070018
          },
          {
            "step": 14688,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8133042826989855,
            "MicroF1": 0.8133042826989855,
            "MacroF1": 0.8168484225511677,
            "Memory in Mb": 7.500472068786621,
            "Time in s": 938.856287
          },
          {
            "step": 15096,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8174229877442862,
            "MicroF1": 0.8174229877442862,
            "MacroF1": 0.8208616131428813,
            "Memory in Mb": 7.693279266357422,
            "Time in s": 1005.608239
          },
          {
            "step": 15504,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8175191898342257,
            "MicroF1": 0.8175191898342257,
            "MacroF1": 0.8200404227627133,
            "Memory in Mb": 7.887093544006348,
            "Time in s": 1075.3914499999998
          },
          {
            "step": 15912,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8100685060649865,
            "MicroF1": 0.8100685060649865,
            "MacroF1": 0.8105704783549956,
            "Memory in Mb": 8.079900741577148,
            "Time in s": 1148.2720969999998
          },
          {
            "step": 16320,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8058704577486365,
            "MicroF1": 0.8058704577486365,
            "MacroF1": 0.8082920647955453,
            "Memory in Mb": 8.273715019226074,
            "Time in s": 1224.3182599999998
          },
          {
            "step": 16728,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.8029533090213428,
            "MicroF1": 0.8029533090213428,
            "MacroF1": 0.8061756731743527,
            "Memory in Mb": 8.467025756835938,
            "Time in s": 1303.603477
          },
          {
            "step": 17136,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.7992996790195507,
            "MicroF1": 0.7992996790195507,
            "MacroF1": 0.8021910628966759,
            "Memory in Mb": 8.7622652053833,
            "Time in s": 1386.189622
          },
          {
            "step": 17544,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.7934218776720059,
            "MicroF1": 0.7934218776720059,
            "MacroF1": 0.7969041071406875,
            "Memory in Mb": 8.956079483032227,
            "Time in s": 1472.1458429999998
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.7934933986964514,
            "MicroF1": 0.7934933986964514,
            "MacroF1": 0.7978100866424277,
            "Memory in Mb": 9.14939022064209,
            "Time in s": 1561.5342959999998
          },
          {
            "step": 18360,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.7969933002886868,
            "MicroF1": 0.7969933002886866,
            "MacroF1": 0.8014382450066739,
            "Memory in Mb": 9.34219741821289,
            "Time in s": 1654.4248979999998
          },
          {
            "step": 18768,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.7999147439654714,
            "MicroF1": 0.7999147439654714,
            "MacroF1": 0.8043799341405246,
            "Memory in Mb": 9.536011695861816,
            "Time in s": 1750.8812689999995
          },
          {
            "step": 19176,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.7945241199478488,
            "MicroF1": 0.7945241199478488,
            "MacroF1": 0.7987282715896407,
            "Memory in Mb": 9.728818893432615,
            "Time in s": 1850.973451
          },
          {
            "step": 19584,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.797375274472757,
            "MicroF1": 0.797375274472757,
            "MacroF1": 0.8021140041360401,
            "Memory in Mb": 9.922633171081545,
            "Time in s": 1954.769568
          },
          {
            "step": 19992,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.7945075283877745,
            "MicroF1": 0.7945075283877745,
            "MacroF1": 0.7995475233856788,
            "Memory in Mb": 10.115943908691406,
            "Time in s": 2062.333925
          },
          {
            "step": 20400,
            "track": "Multiclass classification",
            "model": "Voting",
            "dataset": "Keystroke",
            "Accuracy": 0.793274180106868,
            "MicroF1": 0.793274180106868,
            "MacroF1": 0.7984237858213096,
            "Memory in Mb": 10.308751106262209,
            "Time in s": 2173.749777
          },
          {
            "step": 46,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1777777777777777,
            "MicroF1": 0.1777777777777777,
            "MacroF1": 0.1526026604973973,
            "Memory in Mb": 0.0013666152954101,
            "Time in s": 0.007048
          },
          {
            "step": 92,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1318681318681318,
            "MicroF1": 0.1318681318681318,
            "MacroF1": 0.1213108980966124,
            "Memory in Mb": 0.0013637542724609,
            "Time in s": 0.018168
          },
          {
            "step": 138,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1240875912408759,
            "MicroF1": 0.1240875912408759,
            "MacroF1": 0.1187445506554449,
            "Memory in Mb": 0.0013694763183593,
            "Time in s": 0.031716
          },
          {
            "step": 184,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1256830601092896,
            "MicroF1": 0.1256830601092896,
            "MacroF1": 0.1226298342307158,
            "Memory in Mb": 0.0013647079467773,
            "Time in s": 0.047654
          },
          {
            "step": 230,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1266375545851528,
            "MicroF1": 0.1266375545851528,
            "MacroF1": 0.1250385204120806,
            "Memory in Mb": 0.0013637542724609,
            "Time in s": 0.065983
          },
          {
            "step": 276,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1272727272727272,
            "MicroF1": 0.1272727272727272,
            "MacroF1": 0.1242790791814499,
            "Memory in Mb": 0.0013666152954101,
            "Time in s": 0.086242
          },
          {
            "step": 322,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1339563862928348,
            "MicroF1": 0.1339563862928348,
            "MacroF1": 0.1321003659624602,
            "Memory in Mb": 0.0013666152954101,
            "Time in s": 0.108232
          },
          {
            "step": 368,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1389645776566757,
            "MicroF1": 0.1389645776566757,
            "MacroF1": 0.1374501146297296,
            "Memory in Mb": 0.0013675689697265,
            "Time in s": 0.131958
          },
          {
            "step": 414,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1404358353510895,
            "MicroF1": 0.1404358353510895,
            "MacroF1": 0.1403581309694754,
            "Memory in Mb": 0.0013666152954101,
            "Time in s": 0.1574209999999999
          },
          {
            "step": 460,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1459694989106753,
            "MicroF1": 0.1459694989106753,
            "MacroF1": 0.1456314871072794,
            "Memory in Mb": 0.0013656616210937,
            "Time in s": 0.1845859999999999
          },
          {
            "step": 506,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1386138613861386,
            "MicroF1": 0.1386138613861386,
            "MacroF1": 0.1383381610231494,
            "Memory in Mb": 0.0013666152954101,
            "Time in s": 0.2134849999999999
          },
          {
            "step": 552,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1397459165154265,
            "MicroF1": 0.1397459165154265,
            "MacroF1": 0.1393865249177789,
            "Memory in Mb": 0.0013666152954101,
            "Time in s": 0.24411
          },
          {
            "step": 598,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1373534338358459,
            "MicroF1": 0.1373534338358459,
            "MacroF1": 0.1372798104345861,
            "Memory in Mb": 0.0013675689697265,
            "Time in s": 0.276463
          },
          {
            "step": 644,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1399688958009331,
            "MicroF1": 0.1399688958009331,
            "MacroF1": 0.1401757170901796,
            "Memory in Mb": 0.0013666152954101,
            "Time in s": 0.310533
          },
          {
            "step": 690,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1378809869375907,
            "MicroF1": 0.1378809869375907,
            "MacroF1": 0.1380151778455332,
            "Memory in Mb": 0.0013694763183593,
            "Time in s": 0.346313
          },
          {
            "step": 736,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1401360544217687,
            "MicroF1": 0.1401360544217687,
            "MacroF1": 0.1403108892795828,
            "Memory in Mb": 0.0013675689697265,
            "Time in s": 0.38382
          },
          {
            "step": 782,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1421254801536491,
            "MicroF1": 0.1421254801536491,
            "MacroF1": 0.1420930265541123,
            "Memory in Mb": 0.0013647079467773,
            "Time in s": 0.423095
          },
          {
            "step": 828,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1426844014510278,
            "MicroF1": 0.1426844014510278,
            "MacroF1": 0.1422987455304691,
            "Memory in Mb": 0.0013666152954101,
            "Time in s": 0.464082
          },
          {
            "step": 874,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.138602520045819,
            "MicroF1": 0.138602520045819,
            "MacroF1": 0.1384535269459527,
            "Memory in Mb": 0.0013647079467773,
            "Time in s": 0.506788
          },
          {
            "step": 920,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1349292709466811,
            "MicroF1": 0.1349292709466811,
            "MacroF1": 0.1348083913046733,
            "Memory in Mb": 0.0013666152954101,
            "Time in s": 0.551195
          },
          {
            "step": 966,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1336787564766839,
            "MicroF1": 0.1336787564766839,
            "MacroF1": 0.1334917777444527,
            "Memory in Mb": 0.0013637542724609,
            "Time in s": 0.597302
          },
          {
            "step": 1012,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1325420375865479,
            "MicroF1": 0.1325420375865479,
            "MacroF1": 0.1324936677659038,
            "Memory in Mb": 0.0013675689697265,
            "Time in s": 0.645131
          },
          {
            "step": 1058,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1333964049195837,
            "MicroF1": 0.1333964049195837,
            "MacroF1": 0.1331834965440007,
            "Memory in Mb": 0.0013656616210937,
            "Time in s": 0.69466
          },
          {
            "step": 1104,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1341795104261106,
            "MicroF1": 0.1341795104261106,
            "MacroF1": 0.1340282652950153,
            "Memory in Mb": 0.0013666152954101,
            "Time in s": 0.7459020000000001
          },
          {
            "step": 1150,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.134029590948651,
            "MicroF1": 0.134029590948651,
            "MacroF1": 0.1340639115051912,
            "Memory in Mb": 0.0013637542724609,
            "Time in s": 0.7988440000000001
          },
          {
            "step": 1196,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1364016736401673,
            "MicroF1": 0.1364016736401673,
            "MacroF1": 0.1363948420172951,
            "Memory in Mb": 0.0013694763183593,
            "Time in s": 0.8534870000000001
          },
          {
            "step": 1242,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1394037066881547,
            "MicroF1": 0.1394037066881547,
            "MacroF1": 0.1391977238389222,
            "Memory in Mb": 0.0013637542724609,
            "Time in s": 0.909824
          },
          {
            "step": 1288,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1414141414141414,
            "MicroF1": 0.1414141414141414,
            "MacroF1": 0.1411871502321015,
            "Memory in Mb": 0.0013666152954101,
            "Time in s": 0.967868
          },
          {
            "step": 1334,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1432858214553638,
            "MicroF1": 0.1432858214553638,
            "MacroF1": 0.1430255327815666,
            "Memory in Mb": 0.0013637542724609,
            "Time in s": 1.027625
          },
          {
            "step": 1380,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1435823060188542,
            "MicroF1": 0.1435823060188542,
            "MacroF1": 0.1433209000486506,
            "Memory in Mb": 0.0013694763183593,
            "Time in s": 1.089079
          },
          {
            "step": 1426,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1417543859649122,
            "MicroF1": 0.1417543859649122,
            "MacroF1": 0.1414546655929112,
            "Memory in Mb": 0.0013694763183593,
            "Time in s": 1.152253
          },
          {
            "step": 1472,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1393609789259007,
            "MicroF1": 0.1393609789259007,
            "MacroF1": 0.1390762971394262,
            "Memory in Mb": 0.0013647079467773,
            "Time in s": 1.217139
          },
          {
            "step": 1518,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1397495056031641,
            "MicroF1": 0.1397495056031641,
            "MacroF1": 0.1395136668589845,
            "Memory in Mb": 0.0013666152954101,
            "Time in s": 1.283725
          },
          {
            "step": 1564,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1369161868202175,
            "MicroF1": 0.1369161868202175,
            "MacroF1": 0.1366417047439511,
            "Memory in Mb": 0.0013666152954101,
            "Time in s": 1.352073
          },
          {
            "step": 1610,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1361093847110006,
            "MicroF1": 0.1361093847110006,
            "MacroF1": 0.1359768388190307,
            "Memory in Mb": 0.0013637542724609,
            "Time in s": 1.422125
          },
          {
            "step": 1656,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1365558912386707,
            "MicroF1": 0.1365558912386707,
            "MacroF1": 0.1363322462377459,
            "Memory in Mb": 0.0013694763183593,
            "Time in s": 1.493896
          },
          {
            "step": 1702,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1393298059964726,
            "MicroF1": 0.1393298059964726,
            "MacroF1": 0.1390129627439909,
            "Memory in Mb": 0.0013675689697265,
            "Time in s": 1.5673830000000002
          },
          {
            "step": 1748,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1419576416714367,
            "MicroF1": 0.1419576416714367,
            "MacroF1": 0.1414719731272364,
            "Memory in Mb": 0.0013656616210937,
            "Time in s": 1.6425530000000002
          },
          {
            "step": 1794,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1422197434467373,
            "MicroF1": 0.1422197434467373,
            "MacroF1": 0.1419410396611007,
            "Memory in Mb": 0.0013647079467773,
            "Time in s": 1.7194410000000002
          },
          {
            "step": 1840,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1413811854268624,
            "MicroF1": 0.1413811854268624,
            "MacroF1": 0.1411432976659866,
            "Memory in Mb": 0.0013675689697265,
            "Time in s": 1.7980130000000003
          },
          {
            "step": 1886,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.140053050397878,
            "MicroF1": 0.140053050397878,
            "MacroF1": 0.1397325871382075,
            "Memory in Mb": 0.0013666152954101,
            "Time in s": 1.8782870000000005
          },
          {
            "step": 1932,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1429311237700673,
            "MicroF1": 0.1429311237700673,
            "MacroF1": 0.1427522922982585,
            "Memory in Mb": 0.0013666152954101,
            "Time in s": 1.960245
          },
          {
            "step": 1978,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1461810824481537,
            "MicroF1": 0.1461810824481537,
            "MacroF1": 0.1459715815160596,
            "Memory in Mb": 0.0013694763183593,
            "Time in s": 2.043928
          },
          {
            "step": 2024,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1443400889767671,
            "MicroF1": 0.1443400889767671,
            "MacroF1": 0.1441662523776106,
            "Memory in Mb": 0.0013694763183593,
            "Time in s": 2.12929
          },
          {
            "step": 2070,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1440309328177863,
            "MicroF1": 0.1440309328177863,
            "MacroF1": 0.1438554349712762,
            "Memory in Mb": 0.0013666152954101,
            "Time in s": 2.216361
          },
          {
            "step": 2116,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1446808510638297,
            "MicroF1": 0.1446808510638297,
            "MacroF1": 0.1446036231777657,
            "Memory in Mb": 0.0013637542724609,
            "Time in s": 2.305147
          },
          {
            "step": 2162,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1453031004164738,
            "MicroF1": 0.1453031004164738,
            "MacroF1": 0.1452046591382179,
            "Memory in Mb": 0.0013694763183593,
            "Time in s": 2.395629
          },
          {
            "step": 2208,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1449932034435885,
            "MicroF1": 0.1449932034435885,
            "MacroF1": 0.1449110985199169,
            "Memory in Mb": 0.0013694763183593,
            "Time in s": 2.487817
          },
          {
            "step": 2254,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1464713715046604,
            "MicroF1": 0.1464713715046604,
            "MacroF1": 0.146404255341296,
            "Memory in Mb": 0.0013666152954101,
            "Time in s": 2.5817110000000003
          },
          {
            "step": 2300,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "ImageSegments",
            "Accuracy": 0.1478903871248368,
            "MicroF1": 0.1478903871248368,
            "MacroF1": 0.1478868852481029,
            "Memory in Mb": 0.0013675689697265,
            "Time in s": 2.6773210000000005
          },
          {
            "step": 1056,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.1582938388625592,
            "MicroF1": 0.1582938388625592,
            "MacroF1": 0.1376212379233521,
            "Memory in Mb": 0.0013856887817382,
            "Time in s": 0.055672
          },
          {
            "step": 2112,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.1657981999052581,
            "MicroF1": 0.1657981999052581,
            "MacroF1": 0.1511045106411843,
            "Memory in Mb": 0.0013856887817382,
            "Time in s": 0.157206
          },
          {
            "step": 3168,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.1701926113040732,
            "MicroF1": 0.1701926113040732,
            "MacroF1": 0.1568151235503963,
            "Memory in Mb": 0.0013885498046875,
            "Time in s": 0.304619
          },
          {
            "step": 4224,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.1659957376272791,
            "MicroF1": 0.1659957376272791,
            "MacroF1": 0.1525443315605067,
            "Memory in Mb": 0.0013856887817382,
            "Time in s": 0.4978269999999999
          },
          {
            "step": 5280,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.1708656942602765,
            "MicroF1": 0.1708656942602765,
            "MacroF1": 0.1567667911399359,
            "Memory in Mb": 0.0013837814331054,
            "Time in s": 0.736912
          },
          {
            "step": 6336,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.1737963693764798,
            "MicroF1": 0.1737963693764798,
            "MacroF1": 0.1613756819597299,
            "Memory in Mb": 0.0013837814331054,
            "Time in s": 1.021646
          },
          {
            "step": 7392,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.1752130970098769,
            "MicroF1": 0.1752130970098769,
            "MacroF1": 0.1618940790413477,
            "Memory in Mb": 0.0013837814331054,
            "Time in s": 1.351897
          },
          {
            "step": 8448,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.1772226826092103,
            "MicroF1": 0.1772226826092103,
            "MacroF1": 0.163740045170864,
            "Memory in Mb": 0.0013818740844726,
            "Time in s": 1.7276090000000002
          },
          {
            "step": 9504,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.1773124276544249,
            "MicroF1": 0.1773124276544249,
            "MacroF1": 0.1637492974453095,
            "Memory in Mb": 0.0013885498046875,
            "Time in s": 2.148941
          },
          {
            "step": 10560,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.1790889288758405,
            "MicroF1": 0.1790889288758405,
            "MacroF1": 0.1656421076747495,
            "Memory in Mb": 0.0013837814331054,
            "Time in s": 2.615755
          },
          {
            "step": 11616,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.1789926818768833,
            "MicroF1": 0.1789926818768833,
            "MacroF1": 0.1655925383533761,
            "Memory in Mb": 0.0013856887817382,
            "Time in s": 3.128148
          },
          {
            "step": 12672,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.1853050272275274,
            "MicroF1": 0.1853050272275274,
            "MacroF1": 0.182698099884098,
            "Memory in Mb": 0.0013866424560546,
            "Time in s": 3.685883
          },
          {
            "step": 13728,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2479784366576819,
            "MicroF1": 0.2479784366576819,
            "MacroF1": 0.266039368455288,
            "Memory in Mb": 0.0013866424560546,
            "Time in s": 4.288806
          },
          {
            "step": 14784,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2795778935263478,
            "MicroF1": 0.2795778935263478,
            "MacroF1": 0.2822974275171512,
            "Memory in Mb": 0.0013818740844726,
            "Time in s": 4.937051
          },
          {
            "step": 15840,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2761537975882315,
            "MicroF1": 0.2761537975882315,
            "MacroF1": 0.2847375853365436,
            "Memory in Mb": 0.0013818740844726,
            "Time in s": 5.631085000000001
          },
          {
            "step": 16896,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2723290914471737,
            "MicroF1": 0.2723290914471737,
            "MacroF1": 0.2859139704285301,
            "Memory in Mb": 0.0013856887817382,
            "Time in s": 6.370871000000001
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2720739791655061,
            "MicroF1": 0.2720739791655061,
            "MacroF1": 0.2880143206503877,
            "Memory in Mb": 0.0013866424560546,
            "Time in s": 7.156724000000001
          },
          {
            "step": 19008,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2825274898721523,
            "MicroF1": 0.2825274898721523,
            "MacroF1": 0.2877504429321086,
            "Memory in Mb": 0.0013866424560546,
            "Time in s": 7.988744000000001
          },
          {
            "step": 20064,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2872451776902756,
            "MicroF1": 0.2872451776902756,
            "MacroF1": 0.2866739236661926,
            "Memory in Mb": 0.0013818740844726,
            "Time in s": 8.866412000000002
          },
          {
            "step": 21120,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2830626450116009,
            "MicroF1": 0.2830626450116009,
            "MacroF1": 0.2816476602425525,
            "Memory in Mb": 0.0013837814331054,
            "Time in s": 9.789517000000002
          },
          {
            "step": 22176,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2805411499436302,
            "MicroF1": 0.2805411499436302,
            "MacroF1": 0.2786296072528009,
            "Memory in Mb": 0.0013866424560546,
            "Time in s": 10.758806000000002
          },
          {
            "step": 23232,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2797124531875511,
            "MicroF1": 0.2797124531875511,
            "MacroF1": 0.2771941975793341,
            "Memory in Mb": 0.0013856887817382,
            "Time in s": 11.774485000000002
          },
          {
            "step": 24288,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2777205912628155,
            "MicroF1": 0.2777205912628155,
            "MacroF1": 0.2745878480946635,
            "Memory in Mb": 0.0013866424560546,
            "Time in s": 12.836246000000004
          },
          {
            "step": 25344,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2756579726157124,
            "MicroF1": 0.2756579726157124,
            "MacroF1": 0.2723380305202896,
            "Memory in Mb": 0.0013818740844726,
            "Time in s": 13.944171000000004
          },
          {
            "step": 26400,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2739497708246524,
            "MicroF1": 0.2739497708246524,
            "MacroF1": 0.2699690442569991,
            "Memory in Mb": 0.0013837814331054,
            "Time in s": 15.098386000000003
          },
          {
            "step": 27456,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2718994718630486,
            "MicroF1": 0.2718994718630486,
            "MacroF1": 0.2671948532388624,
            "Memory in Mb": 0.0013866424560546,
            "Time in s": 16.299309000000004
          },
          {
            "step": 28512,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2723860965942969,
            "MicroF1": 0.2723860965942969,
            "MacroF1": 0.2686965366571338,
            "Memory in Mb": 0.0013885498046875,
            "Time in s": 17.546694000000006
          },
          {
            "step": 29568,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2738187844556431,
            "MicroF1": 0.2738187844556431,
            "MacroF1": 0.2720266804437783,
            "Memory in Mb": 0.0013885498046875,
            "Time in s": 18.840271000000005
          },
          {
            "step": 30624,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2753812493877151,
            "MicroF1": 0.2753812493877151,
            "MacroF1": 0.2748698663810352,
            "Memory in Mb": 0.0013885498046875,
            "Time in s": 20.179938000000003
          },
          {
            "step": 31680,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2780390795163989,
            "MicroF1": 0.2780390795163989,
            "MacroF1": 0.2784141751235631,
            "Memory in Mb": 0.0013856887817382,
            "Time in s": 21.565237000000003
          },
          {
            "step": 32736,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.279670077898274,
            "MicroF1": 0.279670077898274,
            "MacroF1": 0.2802192251245276,
            "Memory in Mb": 0.0013837814331054,
            "Time in s": 22.996618000000005
          },
          {
            "step": 33792,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2808440117190968,
            "MicroF1": 0.2808440117190968,
            "MacroF1": 0.2811962745371706,
            "Memory in Mb": 0.0013856887817382,
            "Time in s": 24.474295000000005
          },
          {
            "step": 34848,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2772405085086234,
            "MicroF1": 0.2772405085086234,
            "MacroF1": 0.2781905182864757,
            "Memory in Mb": 0.0013837814331054,
            "Time in s": 25.998543000000005
          },
          {
            "step": 35904,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2739325404562293,
            "MicroF1": 0.2739325404562293,
            "MacroF1": 0.2754200456137155,
            "Memory in Mb": 0.0013856887817382,
            "Time in s": 27.569042000000007
          },
          {
            "step": 36960,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.271246516410076,
            "MicroF1": 0.271246516410076,
            "MacroF1": 0.273332837678202,
            "Memory in Mb": 0.0013818740844726,
            "Time in s": 29.18542500000001
          },
          {
            "step": 38016,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2685518874128633,
            "MicroF1": 0.2685518874128633,
            "MacroF1": 0.2710722002891223,
            "Memory in Mb": 0.0013856887817382,
            "Time in s": 30.847968000000005
          },
          {
            "step": 39072,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.277034117376059,
            "MicroF1": 0.277034117376059,
            "MacroF1": 0.2770619820799866,
            "Memory in Mb": 0.0013866424560546,
            "Time in s": 32.556678000000005
          },
          {
            "step": 40128,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2761731502479627,
            "MicroF1": 0.2761731502479627,
            "MacroF1": 0.2760769006623073,
            "Memory in Mb": 0.0013837814331054,
            "Time in s": 34.31145000000001
          },
          {
            "step": 41184,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2756720005827647,
            "MicroF1": 0.2756720005827647,
            "MacroF1": 0.2754352632972117,
            "Memory in Mb": 0.0013837814331054,
            "Time in s": 36.11334600000001
          },
          {
            "step": 42240,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2740121688486943,
            "MicroF1": 0.2740121688486943,
            "MacroF1": 0.2735946193588543,
            "Memory in Mb": 0.0013885498046875,
            "Time in s": 37.96222800000001
          },
          {
            "step": 43296,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2738422450629403,
            "MicroF1": 0.2738422450629403,
            "MacroF1": 0.2731948869083578,
            "Memory in Mb": 0.0013856887817382,
            "Time in s": 39.85759000000001
          },
          {
            "step": 44352,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2729588960790061,
            "MicroF1": 0.2729588960790061,
            "MacroF1": 0.2720911653869048,
            "Memory in Mb": 0.0013866424560546,
            "Time in s": 41.79925400000001
          },
          {
            "step": 45408,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2720505648908758,
            "MicroF1": 0.2720505648908758,
            "MacroF1": 0.2708084959373003,
            "Memory in Mb": 0.0013866424560546,
            "Time in s": 43.79182700000001
          },
          {
            "step": 46464,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.271377224888621,
            "MicroF1": 0.271377224888621,
            "MacroF1": 0.2698631410415436,
            "Memory in Mb": 0.0013837814331054,
            "Time in s": 45.834688000000014
          },
          {
            "step": 47520,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2723542162082535,
            "MicroF1": 0.2723542162082535,
            "MacroF1": 0.2717062798322285,
            "Memory in Mb": 0.0013837814331054,
            "Time in s": 47.92837500000002
          },
          {
            "step": 48576,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2741327843540916,
            "MicroF1": 0.2741327843540916,
            "MacroF1": 0.2744946340974243,
            "Memory in Mb": 0.0013818740844726,
            "Time in s": 50.07231500000002
          },
          {
            "step": 49632,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2753520984868328,
            "MicroF1": 0.2753520984868328,
            "MacroF1": 0.2765036876430403,
            "Memory in Mb": 0.0013818740844726,
            "Time in s": 52.26665400000002
          },
          {
            "step": 50688,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2768362696549411,
            "MicroF1": 0.2768362696549411,
            "MacroF1": 0.2786344091273496,
            "Memory in Mb": 0.0013837814331054,
            "Time in s": 54.51115200000002
          },
          {
            "step": 51744,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2782791875229499,
            "MicroF1": 0.2782791875229499,
            "MacroF1": 0.2805971515128955,
            "Memory in Mb": 0.0013885498046875,
            "Time in s": 56.805577000000014
          },
          {
            "step": 52800,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Insects",
            "Accuracy": 0.2891153241538665,
            "MicroF1": 0.2891153241538665,
            "MacroF1": 0.2892953202729756,
            "Memory in Mb": 0.0013866424560546,
            "Time in s": 59.150289000000015
          },
          {
            "step": 408,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975429975429976,
            "MicroF1": 0.9975429975429976,
            "MacroF1": 0.966040884438882,
            "Memory in Mb": 0.0006122589111328,
            "Time in s": 0.026957
          },
          {
            "step": 816,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975460122699388,
            "MicroF1": 0.9975460122699388,
            "MacroF1": 0.9879967903427672,
            "Memory in Mb": 0.0006628036499023,
            "Time in s": 0.073338
          },
          {
            "step": 1224,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975470155355682,
            "MicroF1": 0.9975470155355682,
            "MacroF1": 0.9931179599499376,
            "Memory in Mb": 0.0007133483886718,
            "Time in s": 0.138405
          },
          {
            "step": 1632,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975475168608215,
            "MicroF1": 0.9975475168608215,
            "MacroF1": 0.9950750839342832,
            "Memory in Mb": 0.0012521743774414,
            "Time in s": 0.2220349999999999
          },
          {
            "step": 2040,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975478175576264,
            "MicroF1": 0.9975478175576264,
            "MacroF1": 0.9960150346160548,
            "Memory in Mb": 0.0013027191162109,
            "Time in s": 0.3242069999999999
          },
          {
            "step": 2448,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975480179812016,
            "MicroF1": 0.9975480179812016,
            "MacroF1": 0.9965317313935652,
            "Memory in Mb": 0.0013532638549804,
            "Time in s": 0.4452569999999999
          },
          {
            "step": 2856,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975481611208408,
            "MicroF1": 0.9975481611208408,
            "MacroF1": 0.996842428316928,
            "Memory in Mb": 0.00140380859375,
            "Time in s": 0.58488
          },
          {
            "step": 3264,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975482684646032,
            "MicroF1": 0.9975482684646032,
            "MacroF1": 0.9970416021996,
            "Memory in Mb": 0.0014543533325195,
            "Time in s": 0.7430509999999999
          },
          {
            "step": 3672,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975483519476982,
            "MicroF1": 0.9975483519476982,
            "MacroF1": 0.9971755428551424,
            "Memory in Mb": 0.001504898071289,
            "Time in s": 0.9196609999999998
          },
          {
            "step": 4080,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975484187300808,
            "MicroF1": 0.9975484187300808,
            "MacroF1": 0.9972690115789392,
            "Memory in Mb": 0.0015554428100585,
            "Time in s": 1.1148029999999998
          },
          {
            "step": 4488,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975484733675062,
            "MicroF1": 0.9975484733675062,
            "MacroF1": 0.9973361791525124,
            "Memory in Mb": 0.0016059875488281,
            "Time in s": 1.3284669999999998
          },
          {
            "step": 4896,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975485188968336,
            "MicroF1": 0.9975485188968336,
            "MacroF1": 0.9973856025730918,
            "Memory in Mb": 0.0016565322875976,
            "Time in s": 1.56082
          },
          {
            "step": 5304,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.997548557420328,
            "MicroF1": 0.997548557420328,
            "MacroF1": 0.997422679833574,
            "Memory in Mb": 0.0017070770263671,
            "Time in s": 1.81175
          },
          {
            "step": 5712,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975485904395028,
            "MicroF1": 0.9975485904395028,
            "MacroF1": 0.99745094204078,
            "Memory in Mb": 0.0017576217651367,
            "Time in s": 2.0815
          },
          {
            "step": 6120,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975486190554012,
            "MicroF1": 0.9975486190554012,
            "MacroF1": 0.9974727709453766,
            "Memory in Mb": 0.0018081665039062,
            "Time in s": 2.369724
          },
          {
            "step": 6528,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975486440937644,
            "MicroF1": 0.9975486440937644,
            "MacroF1": 0.997489815700999,
            "Memory in Mb": 0.0018587112426757,
            "Time in s": 2.6764910000000004
          },
          {
            "step": 6936,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.997548666186013,
            "MicroF1": 0.997548666186013,
            "MacroF1": 0.9975032443691146,
            "Memory in Mb": 0.0019092559814453,
            "Time in s": 3.0019130000000005
          },
          {
            "step": 7344,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.997548685823233,
            "MicroF1": 0.997548685823233,
            "MacroF1": 0.9975139007887864,
            "Memory in Mb": 0.0034246444702148,
            "Time in s": 3.3461420000000004
          },
          {
            "step": 7752,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975487033931104,
            "MicroF1": 0.9975487033931104,
            "MacroF1": 0.9975224052755712,
            "Memory in Mb": 0.0034751892089843,
            "Time in s": 3.708917000000001
          },
          {
            "step": 8160,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.997548719205785,
            "MicroF1": 0.997548719205785,
            "MacroF1": 0.9975292209193422,
            "Memory in Mb": 0.0035257339477539,
            "Time in s": 4.090185000000001
          },
          {
            "step": 8568,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975487335123148,
            "MicroF1": 0.9975487335123148,
            "MacroF1": 0.9975346982235256,
            "Memory in Mb": 0.0035762786865234,
            "Time in s": 4.489997000000001
          },
          {
            "step": 8976,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.997548746518106,
            "MicroF1": 0.997548746518106,
            "MacroF1": 0.9975391057693664,
            "Memory in Mb": 0.0036268234252929,
            "Time in s": 4.908346000000001
          },
          {
            "step": 9384,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.997548758392838,
            "MicroF1": 0.997548758392838,
            "MacroF1": 0.997542651662671,
            "Memory in Mb": 0.0036773681640625,
            "Time in s": 5.3453800000000005
          },
          {
            "step": 9792,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975487692779084,
            "MicroF1": 0.9975487692779084,
            "MacroF1": 0.9975454987794794,
            "Memory in Mb": 0.003727912902832,
            "Time in s": 5.8012950000000005
          },
          {
            "step": 10200,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975487792920874,
            "MicroF1": 0.9975487792920874,
            "MacroF1": 0.9975477757646256,
            "Memory in Mb": 0.0037784576416015,
            "Time in s": 6.275930000000001
          },
          {
            "step": 10608,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975487885358726,
            "MicroF1": 0.9975487885358726,
            "MacroF1": 0.9975495850737114,
            "Memory in Mb": 0.003829002380371,
            "Time in s": 6.769232000000001
          },
          {
            "step": 11016,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975487970948708,
            "MicroF1": 0.9975487970948708,
            "MacroF1": 0.997551008926056,
            "Memory in Mb": 0.0038795471191406,
            "Time in s": 7.281090000000001
          },
          {
            "step": 11424,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975488050424582,
            "MicroF1": 0.9975488050424582,
            "MacroF1": 0.997552113761348,
            "Memory in Mb": 0.0039300918579101,
            "Time in s": 7.811594
          },
          {
            "step": 11832,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.99754881244189,
            "MicroF1": 0.99754881244189,
            "MacroF1": 0.9975529536110198,
            "Memory in Mb": 0.0039806365966796,
            "Time in s": 8.360849
          },
          {
            "step": 12240,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.997548819347986,
            "MicroF1": 0.997548819347986,
            "MacroF1": 0.9975535726732964,
            "Memory in Mb": 0.0040311813354492,
            "Time in s": 8.928801
          },
          {
            "step": 12648,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.997548825808492,
            "MicroF1": 0.997548825808492,
            "MacroF1": 0.9975540072976318,
            "Memory in Mb": 0.0040817260742187,
            "Time in s": 9.515298
          },
          {
            "step": 13056,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975488318651856,
            "MicroF1": 0.9975488318651856,
            "MacroF1": 0.997554287526727,
            "Memory in Mb": 0.0041322708129882,
            "Time in s": 10.120335
          },
          {
            "step": 13464,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975488375547796,
            "MicroF1": 0.9975488375547796,
            "MacroF1": 0.9975544383040468,
            "Memory in Mb": 0.0041828155517578,
            "Time in s": 10.744063
          },
          {
            "step": 13872,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975488429096676,
            "MicroF1": 0.9975488429096676,
            "MacroF1": 0.9975544804262364,
            "Memory in Mb": 0.0042333602905273,
            "Time in s": 11.386615999999998
          },
          {
            "step": 14280,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975488479585404,
            "MicroF1": 0.9975488479585404,
            "MacroF1": 0.9975544312994103,
            "Memory in Mb": 0.0042839050292968,
            "Time in s": 12.048032999999998
          },
          {
            "step": 14688,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975488527269012,
            "MicroF1": 0.9975488527269012,
            "MacroF1": 0.997554305543504,
            "Memory in Mb": 0.0043344497680664,
            "Time in s": 12.728291999999998
          },
          {
            "step": 15096,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.997548857237496,
            "MicroF1": 0.997548857237496,
            "MacroF1": 0.9975541154780816,
            "Memory in Mb": 0.0043849945068359,
            "Time in s": 13.427186999999998
          },
          {
            "step": 15504,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975488615106752,
            "MicroF1": 0.9975488615106752,
            "MacroF1": 0.9975538715150368,
            "Memory in Mb": 0.0044355392456054,
            "Time in s": 14.144781999999998
          },
          {
            "step": 15912,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975488655647036,
            "MicroF1": 0.9975488655647036,
            "MacroF1": 0.997553582477696,
            "Memory in Mb": 0.004486083984375,
            "Time in s": 14.881121999999998
          },
          {
            "step": 16320,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975488694160182,
            "MicroF1": 0.9975488694160182,
            "MacroF1": 0.997553255861403,
            "Memory in Mb": 0.0045366287231445,
            "Time in s": 15.636483999999998
          },
          {
            "step": 16728,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975488730794524,
            "MicroF1": 0.9975488730794524,
            "MacroF1": 0.997552898047314,
            "Memory in Mb": 0.004587173461914,
            "Time in s": 16.410486
          },
          {
            "step": 17136,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975488765684272,
            "MicroF1": 0.9975488765684272,
            "MacroF1": 0.997552514478575,
            "Memory in Mb": 0.0046377182006835,
            "Time in s": 17.203509999999998
          },
          {
            "step": 17544,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975488798951148,
            "MicroF1": 0.9975488798951148,
            "MacroF1": 0.997552109806108,
            "Memory in Mb": 0.0046882629394531,
            "Time in s": 18.015261
          },
          {
            "step": 17952,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.997548883070581,
            "MicroF1": 0.997548883070581,
            "MacroF1": 0.997551688009728,
            "Memory in Mb": 0.0047388076782226,
            "Time in s": 18.845896
          },
          {
            "step": 18360,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975488861049076,
            "MicroF1": 0.9975488861049076,
            "MacroF1": 0.9975512524991372,
            "Memory in Mb": 0.0047893524169921,
            "Time in s": 19.695493
          },
          {
            "step": 18768,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975488890073,
            "MicroF1": 0.9975488890073,
            "MacroF1": 0.9975508061984416,
            "Memory in Mb": 0.0048398971557617,
            "Time in s": 20.563922
          },
          {
            "step": 19176,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.99754889178618,
            "MicroF1": 0.99754889178618,
            "MacroF1": 0.9975503516171184,
            "Memory in Mb": 0.0048904418945312,
            "Time in s": 21.451134
          },
          {
            "step": 19584,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975488944492672,
            "MicroF1": 0.9975488944492672,
            "MacroF1": 0.997549890909789,
            "Memory in Mb": 0.0049409866333007,
            "Time in s": 22.357399
          },
          {
            "step": 19992,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975488970036516,
            "MicroF1": 0.9975488970036516,
            "MacroF1": 0.9975494259267256,
            "Memory in Mb": 0.0049915313720703,
            "Time in s": 23.282656
          },
          {
            "step": 20400,
            "track": "Multiclass classification",
            "model": "[baseline] Last Class",
            "dataset": "Keystroke",
            "Accuracy": 0.9975488994558556,
            "MicroF1": 0.9975488994558556,
            "MacroF1": 0.9975489582566448,
            "Memory in Mb": 0.0050420761108398,
            "Time in s": 24.227046
          }
        ]
      },
      "params": [
        {
          "name": "models",
          "select": {
            "type": "point",
            "fields": [
              "model"
            ]
          },
          "bind": "legend"
        },
        {
          "name": "Dataset",
          "value": "ImageSegments",
          "bind": {
            "input": "select",
            "options": [
              "ImageSegments",
              "Insects",
              "Keystroke"
            ]
          }
        },
        {
          "name": "grid",
          "select": "interval",
          "bind": "scales"
        }
      ],
      "transform": [
        {
          "filter": {
            "field": "dataset",
            "equal": {
              "expr": "Dataset"
            }
          }
        }
      ],
      "repeat": {
        "row": [
          "Accuracy",
          "MicroF1",
          "MacroF1",
          "Memory in Mb",
          "Time in s"
        ]
      },
      "spec": {
        "width": "container",
        "mark": "line",
        "encoding": {
          "x": {
            "field": "step",
            "type": "quantitative",
            "axis": {
              "titleFontSize": 18,
              "labelFontSize": 18,
              "title": "Instance"
            }
          },
          "y": {
            "field": {
              "repeat": "row"
            },
            "type": "quantitative",
            "axis": {
              "titleFontSize": 18,
              "labelFontSize": 18
            }
          },
          "color": {
            "field": "model",
            "type": "ordinal",
            "scale": {
              "scheme": "category20b"
            },
            "title": "Models",
            "legend": {
              "titleFontSize": 18,
              "labelFontSize": 18,
              "labelLimit": 500
            }
          },
          "opacity": {
            "condition": {
              "param": "models",
              "value": 1
            },
            "value": 0.2
          }
        }
      }
    }
    ```

            

## Datasets

???- abstract "ImageSegments"

    Image segments classification.

    This dataset contains features that describe image segments into 7 classes: brickface, sky,
    foliage, cement, window, path, and grass.

        Name  ImageSegments                                              
        Task  Multi-class classification                                 
     Samples  2,310                                                      
    Features  18                                                         
      Sparse  False                                                      
        Path  /home/kulbach/projects/river/river/datasets/segment.csv.zip

<span />

???- abstract "Insects"

    Insects dataset.

    This dataset has different variants, which are:

    - abrupt_balanced
    - abrupt_imbalanced
    - gradual_balanced
    - gradual_imbalanced
    - incremental-abrupt_balanced
    - incremental-abrupt_imbalanced
    - incremental-reoccurring_balanced
    - incremental-reoccurring_imbalanced
    - incremental_balanced
    - incremental_imbalanced
    - out-of-control

    The number of samples and the difficulty change from one variant to another. The number of
    classes is always the same (6), except for the last variant (24).

          Name  Insects                                                                                 
          Task  Multi-class classification                                                              
       Samples  52,848                                                                                  
      Features  33                                                                                      
       Classes  6                                                                                       
        Sparse  False                                                                                   
          Path  /home/kulbach/river_data/Insects/INSECTS-abrupt_balanced_norm.arff                      
           URL  http://sites.labic.icmc.usp.br/vsouza/repository/creme/INSECTS-abrupt_balanced_norm.arff
          Size  15.66 MB                                                                                
    Downloaded  True                                                                                    
       Variant  abrupt_balanced                                                                         

    Parameters
    ----------
        variant
            Indicates which variant of the dataset to load.

<span />

???- abstract "Keystroke"

    CMU keystroke dataset.

    Users are tasked to type in a password. The task is to determine which user is typing in the
    password.

    The only difference with the original dataset is that the "sessionIndex" and "rep" attributes
    have been dropped.

          Name  Keystroke                                                    
          Task  Multi-class classification                                   
       Samples  20,400                                                       
      Features  31                                                           
        Sparse  False                                                        
          Path  /home/kulbach/river_data/Keystroke/DSL-StrongPasswordData.csv
           URL  http://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv  
          Size  4.45 MB                                                      
    Downloaded  True                                                         

<span />

## Models

???- example "Naive Bayes"

    <pre>GaussianNB ()</pre>

<span />

???- example "Hoeffding Tree"

    <pre>HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    )</pre>

<span />

???- example "Hoeffding Adaptive Tree"

    <pre>HoeffdingAdaptiveTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      bootstrap_sampling=True
      drift_window_threshold=300
      drift_detector=ADWIN (
        delta=0.002
        clock=32
        max_buckets=5
        min_window_length=5
        grace_period=10
      )
      switch_significance=0.05
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
      seed=42
    )</pre>

<span />

???- example "Adaptive Random Forest"

    <pre>[]</pre>

<span />

???- example "Streaming Random Patches"

    <pre>SRPClassifier (
      model=HoeffdingTreeClassifier (
        grace_period=50
        max_depth=inf
        split_criterion="info_gain"
        delta=0.01
        tau=0.05
        leaf_prediction="nba"
        nb_threshold=0
        nominal_attributes=None
        splitter=GaussianSplitter (
          n_splits=10
        )
        binary_split=False
        max_size=100.
        memory_estimate_period=1000000
        stop_mem_management=False
        remove_poor_attrs=False
        merit_preprune=True
      )
      n_models=10
      subspace_size=0.6
      training_method="patches"
      lam=6
      drift_detector=ADWIN (
        delta=1e-05
        clock=32
        max_buckets=5
        min_window_length=5
        grace_period=10
      )
      warning_detector=ADWIN (
        delta=0.0001
        clock=32
        max_buckets=5
        min_window_length=5
        grace_period=10
      )
      disable_detector="off"
      disable_weighted_vote=False
      seed=None
      metric=Accuracy (
        cm=ConfusionMatrix (
          classes=[]
        )
      )
    )</pre>

<span />

???- example "k-Nearest Neighbors"

    <pre>Pipeline (
      StandardScaler (
        with_std=True
      ),
      KNNClassifier (
        n_neighbors=5
        window_size=100
        min_distance_keep=0.
        weighted=True
        cleanup_every=0
        distance_func=functools.partial(<function minkowski_distance at 0x7f2d38a59ea0>, p=2)
        softmax=False
      )
    )</pre>

<span />

???- example "ADWIN Bagging"

    <pre>[HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    )]</pre>

<span />

???- example "AdaBoost"

    <pre>[HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    )]</pre>

<span />

???- example "Bagging"

    <pre>[HoeffdingAdaptiveTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      bootstrap_sampling=False
      drift_window_threshold=300
      drift_detector=ADWIN (
        delta=0.002
        clock=32
        max_buckets=5
        min_window_length=5
        grace_period=10
      )
      switch_significance=0.05
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
      seed=None
    ), HoeffdingAdaptiveTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      bootstrap_sampling=False
      drift_window_threshold=300
      drift_detector=ADWIN (
        delta=0.002
        clock=32
        max_buckets=5
        min_window_length=5
        grace_period=10
      )
      switch_significance=0.05
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
      seed=None
    ), HoeffdingAdaptiveTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      bootstrap_sampling=False
      drift_window_threshold=300
      drift_detector=ADWIN (
        delta=0.002
        clock=32
        max_buckets=5
        min_window_length=5
        grace_period=10
      )
      switch_significance=0.05
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
      seed=None
    ), HoeffdingAdaptiveTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      bootstrap_sampling=False
      drift_window_threshold=300
      drift_detector=ADWIN (
        delta=0.002
        clock=32
        max_buckets=5
        min_window_length=5
        grace_period=10
      )
      switch_significance=0.05
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
      seed=None
    ), HoeffdingAdaptiveTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      bootstrap_sampling=False
      drift_window_threshold=300
      drift_detector=ADWIN (
        delta=0.002
        clock=32
        max_buckets=5
        min_window_length=5
        grace_period=10
      )
      switch_significance=0.05
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
      seed=None
    ), HoeffdingAdaptiveTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      bootstrap_sampling=False
      drift_window_threshold=300
      drift_detector=ADWIN (
        delta=0.002
        clock=32
        max_buckets=5
        min_window_length=5
        grace_period=10
      )
      switch_significance=0.05
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
      seed=None
    ), HoeffdingAdaptiveTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      bootstrap_sampling=False
      drift_window_threshold=300
      drift_detector=ADWIN (
        delta=0.002
        clock=32
        max_buckets=5
        min_window_length=5
        grace_period=10
      )
      switch_significance=0.05
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
      seed=None
    ), HoeffdingAdaptiveTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      bootstrap_sampling=False
      drift_window_threshold=300
      drift_detector=ADWIN (
        delta=0.002
        clock=32
        max_buckets=5
        min_window_length=5
        grace_period=10
      )
      switch_significance=0.05
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
      seed=None
    ), HoeffdingAdaptiveTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      bootstrap_sampling=False
      drift_window_threshold=300
      drift_detector=ADWIN (
        delta=0.002
        clock=32
        max_buckets=5
        min_window_length=5
        grace_period=10
      )
      switch_significance=0.05
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
      seed=None
    ), HoeffdingAdaptiveTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      bootstrap_sampling=False
      drift_window_threshold=300
      drift_detector=ADWIN (
        delta=0.002
        clock=32
        max_buckets=5
        min_window_length=5
        grace_period=10
      )
      switch_significance=0.05
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
      seed=None
    )]</pre>

<span />

???- example "Leveraging Bagging"

    <pre>[HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    )]</pre>

<span />

???- example "Stacking"

    <pre>[Pipeline (
      StandardScaler (
        with_std=True
      ),
      SoftmaxRegression (
        optimizer=SGD (
          lr=Constant (
            learning_rate=0.01
          )
        )
        loss=CrossEntropy (
          class_weight={}
        )
        l2=0
      )
    ), GaussianNB (), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), Pipeline (
      StandardScaler (
        with_std=True
      ),
      KNNClassifier (
        n_neighbors=5
        window_size=100
        min_distance_keep=0.
        weighted=True
        cleanup_every=0
        distance_func=functools.partial(<function minkowski_distance at 0x7f2d38a59ea0>, p=2)
        softmax=False
      )
    )]</pre>

<span />

???- example "Voting"

    <pre>VotingClassifier (
      models=[Pipeline (
      StandardScaler (
        with_std=True
      ),
      SoftmaxRegression (
        optimizer=SGD (
          lr=Constant (
            learning_rate=0.01
          )
        )
        loss=CrossEntropy (
          class_weight={}
        )
        l2=0
      )
    ), GaussianNB (), HoeffdingTreeClassifier (
      grace_period=200
      max_depth=inf
      split_criterion="info_gain"
      delta=1e-07
      tau=0.05
      leaf_prediction="nba"
      nb_threshold=0
      nominal_attributes=None
      splitter=GaussianSplitter (
        n_splits=10
      )
      binary_split=False
      max_size=100.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    ), Pipeline (
      StandardScaler (
        with_std=True
      ),
      KNNClassifier (
        n_neighbors=5
        window_size=100
        min_distance_keep=0.
        weighted=True
        cleanup_every=0
        distance_func=functools.partial(<function minkowski_distance at 0x7f2d38a59ea0>, p=2)
        softmax=False
      )
    )]
      use_probabilities=True
    )</pre>

<span />

???- example "[baseline] Last Class"

    <pre>NoChangeClassifier ()</pre>

<span />

## Environment

<pre>Python implementation: CPython
Python version       : 3.10.8
IPython version      : 8.12.0

river       : 0.15.0
numpy       : 1.24.2
scikit-learn: 1.2.2
pandas      : 1.5.3
scipy       : 1.10.1

Compiler    : Clang 14.0.0 (clang-1400.0.29.102)
OS          : Darwin
Release     : 22.2.0
Machine     : arm64
Processor   : arm
CPU cores   : 8
Architecture: 64bit
</pre>

