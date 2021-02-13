# Logistic regression

## Final results

| Model        | Track    |   Accuracy |       F1 | Time                   | Memory   |
|:-------------|:---------|-----------:|---------:|:-----------------------|:---------|
| River        | Bananas  |   0.543019 | 0.195349 | 0 days 00:00:00.528174 | 4.12 KB  |
| River        | Phishing |   0.888    | 0.872263 | 0 days 00:00:00.160760 | 5.48 KB  |
| scikit-learn | Bananas  |   0.544811 | 0.200232 | 0 days 00:00:02.463166 | 5.68 KB  |
| scikit-learn | Phishing |   0.8884   | 0.873813 | 0 days 00:00:00.650081 | 7.25 KB  |

## Full traces

| Model        | Track    |   Step |   Accuracy |       F1 | Time                   | Memory   |
|:-------------|:---------|-------:|-----------:|---------:|:-----------------------|:---------|
| River        | Phishing |    125 |   0.808    | 0.803279 | 0 days 00:00:00.019669 | 5.23 KB  |
| River        | Phishing |    250 |   0.824    | 0.811966 | 0 days 00:00:00.034919 | 5.23 KB  |
| River        | Phishing |    375 |   0.850667 | 0.825    | 0 days 00:00:00.049674 | 5.48 KB  |
| River        | Phishing |    500 |   0.86     | 0.832536 | 0 days 00:00:00.065117 | 5.48 KB  |
| River        | Phishing |    625 |   0.8688   | 0.834677 | 0 days 00:00:00.079675 | 5.48 KB  |
| River        | Phishing |    750 |   0.88     | 0.855769 | 0 days 00:00:00.095102 | 5.48 KB  |
| River        | Phishing |    875 |   0.885714 | 0.86376  | 0 days 00:00:00.110880 | 5.48 KB  |
| River        | Phishing |   1000 |   0.888    | 0.868852 | 0 days 00:00:00.129613 | 5.48 KB  |
| River        | Phishing |   1125 |   0.890667 | 0.87538  | 0 days 00:00:00.144521 | 5.48 KB  |
| River        | Phishing |   1250 |   0.888    | 0.872263 | 0 days 00:00:00.160760 | 5.48 KB  |
| River        | Bananas  |    530 |   0.532075 | 0.357513 | 0 days 00:00:00.052851 | 4.12 KB  |
| River        | Bananas  |   1060 |   0.561321 | 0.233937 | 0 days 00:00:00.107171 | 4.12 KB  |
| River        | Bananas  |   1590 |   0.564151 | 0.179882 | 0 days 00:00:00.156950 | 4.12 KB  |
| River        | Bananas  |   2120 |   0.549057 | 0.164336 | 0 days 00:00:00.221231 | 4.12 KB  |
| River        | Bananas  |   2650 |   0.544906 | 0.222938 | 0 days 00:00:00.275695 | 4.12 KB  |
| River        | Bananas  |   3180 |   0.543396 | 0.217672 | 0 days 00:00:00.322977 | 4.12 KB  |
| River        | Bananas  |   3710 |   0.546631 | 0.196753 | 0 days 00:00:00.378854 | 4.12 KB  |
| River        | Bananas  |   4240 |   0.54717  | 0.176672 | 0 days 00:00:00.425663 | 4.12 KB  |
| River        | Bananas  |   4770 |   0.546541 | 0.188976 | 0 days 00:00:00.473931 | 4.12 KB  |
| River        | Bananas  |   5300 |   0.543019 | 0.195349 | 0 days 00:00:00.528174 | 4.12 KB  |
| scikit-learn | Phishing |    125 |   0.880727 | 0.865132 | 0 days 00:00:00.066877 | 7.03 KB  |
| scikit-learn | Phishing |    250 |   0.876667 | 0.860798 | 0 days 00:00:00.135615 | 7.03 KB  |
| scikit-learn | Phishing |    375 |   0.878769 | 0.861365 | 0 days 00:00:00.202397 | 7.25 KB  |
| scikit-learn | Phishing |    500 |   0.878857 | 0.86071  | 0 days 00:00:00.266227 | 7.25 KB  |
| scikit-learn | Phishing |    625 |   0.880533 | 0.860349 | 0 days 00:00:00.333330 | 7.25 KB  |
| scikit-learn | Phishing |    750 |   0.8845   | 0.866859 | 0 days 00:00:00.393203 | 7.25 KB  |
| scikit-learn | Phishing |    875 |   0.886588 | 0.869377 | 0 days 00:00:00.459468 | 7.25 KB  |
| scikit-learn | Phishing |   1000 |   0.887556 | 0.871378 | 0 days 00:00:00.519166 | 7.25 KB  |
| scikit-learn | Phishing |   1125 |   0.889263 | 0.874822 | 0 days 00:00:00.585747 | 7.25 KB  |
| scikit-learn | Phishing |   1250 |   0.8884   | 0.873813 | 0 days 00:00:00.650081 | 7.25 KB  |
| scikit-learn | Bananas  |    530 |   0.541681 | 0.224608 | 0 days 00:00:00.246382 | 5.68 KB  |
| scikit-learn | Bananas  |   1060 |   0.544654 | 0.213898 | 0 days 00:00:00.493935 | 5.68 KB  |
| scikit-learn | Bananas  |   1590 |   0.546589 | 0.204684 | 0 days 00:00:00.739000 | 5.68 KB  |
| scikit-learn | Bananas  |   2120 |   0.544205 | 0.195145 | 0 days 00:00:00.986192 | 5.68 KB  |
| scikit-learn | Bananas  |   2650 |   0.544151 | 0.208042 | 0 days 00:00:01.229924 | 5.68 KB  |
| scikit-learn | Bananas  |   3180 |   0.54316  | 0.208095 | 0 days 00:00:01.474930 | 5.68 KB  |
| scikit-learn | Bananas  |   3710 |   0.545172 | 0.202413 | 0 days 00:00:01.719574 | 5.68 KB  |
| scikit-learn | Bananas  |   4240 |   0.545493 | 0.193452 | 0 days 00:00:01.971608 | 5.68 KB  |
| scikit-learn | Bananas  |   4770 |   0.546475 | 0.197505 | 0 days 00:00:02.218981 | 5.68 KB  |
| scikit-learn | Bananas  |   5300 |   0.544811 | 0.200232 | 0 days 00:00:02.463166 | 5.68 KB  |
