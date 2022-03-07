# Logistic regression

## Final results

| Model         | Track    |   Accuracy |       F1 | Time                   | Memory   |
|:--------------|:---------|-----------:|---------:|:-----------------------|:---------|
| PyTorch       | Bananas  |   0.54566  | 0.202297 | 0 days 00:00:02.402525 | 16.59 KB |
| PyTorch       | Phishing |   0.814667 | 0.762068 | 0 days 00:00:00.655244 | 16.59 KB |
| River         | Bananas  |   0.543019 | 0.195349 | 0 days 00:00:00.649486 | 4.12 KB  |
| River         | Phishing |   0.888    | 0.872263 | 0 days 00:00:00.164876 | 5.48 KB  |
| Vowpal Wabbit | Bananas  |   0.547075 | 0.160224 | 0 days 00:00:00.595254 | 388 B    |
| Vowpal Wabbit | Phishing |   0.8044   | 0.741133 | 0 days 00:00:00.197963 | 388 B    |
| scikit-learn  | Bananas  |   0.544811 | 0.200232 | 0 days 00:00:02.719557 | 5.72 KB  |
| scikit-learn  | Phishing |   0.8884   | 0.873813 | 0 days 00:00:00.760940 | 7.29 KB  |

## Traces

| Model         | Track    |   Step |   Accuracy |       F1 | Time                   | Memory   |
|:--------------|:---------|-------:|-----------:|---------:|:-----------------------|:---------|
| River         | Phishing |    125 |   0.808    | 0.803279 | 0 days 00:00:00.020306 | 5.23 KB  |
| River         | Phishing |    250 |   0.824    | 0.811966 | 0 days 00:00:00.037724 | 5.23 KB  |
| River         | Phishing |    375 |   0.850667 | 0.825    | 0 days 00:00:00.053557 | 5.48 KB  |
| River         | Phishing |    500 |   0.86     | 0.832536 | 0 days 00:00:00.069520 | 5.48 KB  |
| River         | Phishing |    625 |   0.8688   | 0.834677 | 0 days 00:00:00.085084 | 5.48 KB  |
| River         | Phishing |    750 |   0.88     | 0.855769 | 0 days 00:00:00.104404 | 5.48 KB  |
| River         | Phishing |    875 |   0.885714 | 0.86376  | 0 days 00:00:00.119010 | 5.48 KB  |
| River         | Phishing |   1000 |   0.888    | 0.868852 | 0 days 00:00:00.134768 | 5.48 KB  |
| River         | Phishing |   1125 |   0.890667 | 0.87538  | 0 days 00:00:00.149624 | 5.48 KB  |
| River         | Phishing |   1250 |   0.888    | 0.872263 | 0 days 00:00:00.164876 | 5.48 KB  |
| River         | Bananas  |    530 |   0.532075 | 0.357513 | 0 days 00:00:00.051217 | 4.12 KB  |
| River         | Bananas  |   1060 |   0.561321 | 0.233937 | 0 days 00:00:00.114353 | 4.12 KB  |
| River         | Bananas  |   1590 |   0.564151 | 0.179882 | 0 days 00:00:00.188227 | 4.12 KB  |
| River         | Bananas  |   2120 |   0.549057 | 0.164336 | 0 days 00:00:00.252652 | 4.12 KB  |
| River         | Bananas  |   2650 |   0.544906 | 0.222938 | 0 days 00:00:00.321387 | 4.12 KB  |
| River         | Bananas  |   3180 |   0.543396 | 0.217672 | 0 days 00:00:00.386414 | 4.12 KB  |
| River         | Bananas  |   3710 |   0.546631 | 0.196753 | 0 days 00:00:00.467735 | 4.12 KB  |
| River         | Bananas  |   4240 |   0.54717  | 0.176672 | 0 days 00:00:00.529080 | 4.12 KB  |
| River         | Bananas  |   4770 |   0.546541 | 0.188976 | 0 days 00:00:00.591975 | 4.12 KB  |
| River         | Bananas  |   5300 |   0.543019 | 0.195349 | 0 days 00:00:00.649486 | 4.12 KB  |
| scikit-learn  | Phishing |    125 |   0.880727 | 0.865132 | 0 days 00:00:00.087300 | 7.07 KB  |
| scikit-learn  | Phishing |    250 |   0.876667 | 0.860798 | 0 days 00:00:00.166690 | 7.07 KB  |
| scikit-learn  | Phishing |    375 |   0.878769 | 0.861365 | 0 days 00:00:00.251248 | 7.29 KB  |
| scikit-learn  | Phishing |    500 |   0.878857 | 0.86071  | 0 days 00:00:00.342788 | 7.29 KB  |
| scikit-learn  | Phishing |    625 |   0.880533 | 0.860349 | 0 days 00:00:00.413105 | 7.29 KB  |
| scikit-learn  | Phishing |    750 |   0.8845   | 0.866859 | 0 days 00:00:00.471607 | 7.29 KB  |
| scikit-learn  | Phishing |    875 |   0.886588 | 0.869377 | 0 days 00:00:00.545312 | 7.29 KB  |
| scikit-learn  | Phishing |   1000 |   0.887556 | 0.871378 | 0 days 00:00:00.610224 | 7.29 KB  |
| scikit-learn  | Phishing |   1125 |   0.889263 | 0.874822 | 0 days 00:00:00.676257 | 7.29 KB  |
| scikit-learn  | Phishing |   1250 |   0.8884   | 0.873813 | 0 days 00:00:00.760940 | 7.29 KB  |
| scikit-learn  | Bananas  |    530 |   0.541681 | 0.224608 | 0 days 00:00:00.304745 | 5.72 KB  |
| scikit-learn  | Bananas  |   1060 |   0.544654 | 0.213898 | 0 days 00:00:00.631450 | 5.72 KB  |
| scikit-learn  | Bananas  |   1590 |   0.546589 | 0.204684 | 0 days 00:00:00.889963 | 5.72 KB  |
| scikit-learn  | Bananas  |   2120 |   0.544205 | 0.195145 | 0 days 00:00:01.142037 | 5.72 KB  |
| scikit-learn  | Bananas  |   2650 |   0.544151 | 0.208042 | 0 days 00:00:01.407279 | 5.72 KB  |
| scikit-learn  | Bananas  |   3180 |   0.54316  | 0.208095 | 0 days 00:00:01.663327 | 5.72 KB  |
| scikit-learn  | Bananas  |   3710 |   0.545172 | 0.202413 | 0 days 00:00:01.914629 | 5.72 KB  |
| scikit-learn  | Bananas  |   4240 |   0.545493 | 0.193452 | 0 days 00:00:02.151460 | 5.72 KB  |
| scikit-learn  | Bananas  |   4770 |   0.546475 | 0.197505 | 0 days 00:00:02.445650 | 5.72 KB  |
| scikit-learn  | Bananas  |   5300 |   0.544811 | 0.200232 | 0 days 00:00:02.719557 | 5.72 KB  |
| PyTorch       | Phishing |    125 |   0.872762 | 0.854783 | 0 days 00:00:00.100159 | 16.59 KB |
| PyTorch       | Phishing |    250 |   0.859636 | 0.836302 | 0 days 00:00:00.165274 | 16.59 KB |
| PyTorch       | Phishing |    375 |   0.852522 | 0.823627 | 0 days 00:00:00.226808 | 16.59 KB |
| PyTorch       | Phishing |    500 |   0.841667 | 0.806517 | 0 days 00:00:00.281735 | 16.59 KB |
| PyTorch       | Phishing |    625 |   0.83744  | 0.796474 | 0 days 00:00:00.343995 | 16.59 KB |
| PyTorch       | Phishing |    750 |   0.827077 | 0.781493 | 0 days 00:00:00.405895 | 16.59 KB |
| PyTorch       | Phishing |    875 |   0.822519 | 0.773706 | 0 days 00:00:00.474607 | 16.59 KB |
| PyTorch       | Phishing |   1000 |   0.817714 | 0.7663   | 0 days 00:00:00.528892 | 16.59 KB |
| PyTorch       | Phishing |   1125 |   0.816276 | 0.765328 | 0 days 00:00:00.595008 | 16.59 KB |
| PyTorch       | Phishing |   1250 |   0.814667 | 0.762068 | 0 days 00:00:00.655244 | 16.59 KB |
| PyTorch       | Bananas  |    530 |   0.543935 | 0.215213 | 0 days 00:00:00.248366 | 16.59 KB |
| PyTorch       | Bananas  |   1060 |   0.545798 | 0.209788 | 0 days 00:00:00.489684 | 16.59 KB |
| PyTorch       | Bananas  |   1590 |   0.546842 | 0.204722 | 0 days 00:00:00.725563 | 16.59 KB |
| PyTorch       | Bananas  |   2120 |   0.54544  | 0.199169 | 0 days 00:00:00.963715 | 16.59 KB |
| PyTorch       | Bananas  |   2650 |   0.545358 | 0.20716  | 0 days 00:00:01.200519 | 16.59 KB |
| PyTorch       | Bananas  |   3180 |   0.544702 | 0.207228 | 0 days 00:00:01.450085 | 16.59 KB |
| PyTorch       | Bananas  |   3710 |   0.545912 | 0.203676 | 0 days 00:00:01.682567 | 16.59 KB |
| PyTorch       | Bananas  |   4240 |   0.546092 | 0.197904 | 0 days 00:00:01.921537 | 16.59 KB |
| PyTorch       | Bananas  |   4770 |   0.546779 | 0.200597 | 0 days 00:00:02.156922 | 16.59 KB |
| PyTorch       | Bananas  |   5300 |   0.54566  | 0.202297 | 0 days 00:00:02.402525 | 16.59 KB |
| Vowpal Wabbit | Phishing |    125 |   0.809032 | 0.753826 | 0 days 00:00:00.021420 | 388 B    |
| Vowpal Wabbit | Phishing |    250 |   0.80425  | 0.745531 | 0 days 00:00:00.052269 | 388 B    |
| Vowpal Wabbit | Phishing |    375 |   0.804364 | 0.743075 | 0 days 00:00:00.068745 | 388 B    |
| Vowpal Wabbit | Phishing |    500 |   0.803294 | 0.740211 | 0 days 00:00:00.084920 | 388 B    |
| Vowpal Wabbit | Phishing |    625 |   0.803657 | 0.73755  | 0 days 00:00:00.101203 | 388 B    |
| Vowpal Wabbit | Phishing |    750 |   0.803111 | 0.737715 | 0 days 00:00:00.118310 | 388 B    |
| Vowpal Wabbit | Phishing |    875 |   0.803459 | 0.738267 | 0 days 00:00:00.134446 | 388 B    |
| Vowpal Wabbit | Phishing |   1000 |   0.803368 | 0.738522 | 0 days 00:00:00.150708 | 388 B    |
| Vowpal Wabbit | Phishing |   1125 |   0.803897 | 0.740781 | 0 days 00:00:00.166272 | 388 B    |
| Vowpal Wabbit | Phishing |   1250 |   0.8044   | 0.741133 | 0 days 00:00:00.197963 | 388 B    |
| Vowpal Wabbit | Bananas  |    530 |   0.545953 | 0.197159 | 0 days 00:00:00.064878 | 388 B    |
| Vowpal Wabbit | Bananas  |   1060 |   0.547229 | 0.192619 | 0 days 00:00:00.137034 | 388 B    |
| Vowpal Wabbit | Bananas  |   1590 |   0.547684 | 0.188032 | 0 days 00:00:00.192000 | 388 B    |
| Vowpal Wabbit | Bananas  |   2120 |   0.546282 | 0.183054 | 0 days 00:00:00.243525 | 388 B    |
| Vowpal Wabbit | Bananas  |   2650 |   0.545822 | 0.17861  | 0 days 00:00:00.313916 | 388 B    |
| Vowpal Wabbit | Bananas  |   3180 |   0.546541 | 0.174742 | 0 days 00:00:00.364553 | 388 B    |
| Vowpal Wabbit | Bananas  |   3710 |   0.546915 | 0.170943 | 0 days 00:00:00.425125 | 388 B    |
| Vowpal Wabbit | Bananas  |   4240 |   0.547071 | 0.167245 | 0 days 00:00:00.476956 | 388 B    |
| Vowpal Wabbit | Bananas  |   4770 |   0.546783 | 0.163571 | 0 days 00:00:00.530476 | 388 B    |
| Vowpal Wabbit | Bananas  |   5300 |   0.547075 | 0.160224 | 0 days 00:00:00.595254 | 388 B    |
