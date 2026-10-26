# Phase 2 dataset registry

The final experiment pins OpenML dataset IDs rather than resolving names dynamically.

| Dataset | Source | Maximum rows used |
|---|---|---:|
| Breast Cancer | scikit-learn built-in | All |
| Wine | scikit-learn built-in | All |
| Iris | scikit-learn built-in | All |
| German Credit | https://www.openml.org/d/31 | All |
| Diabetes | https://www.openml.org/d/37 | All |
| Spambase | https://www.openml.org/d/44 | All |
| Bank Marketing | https://www.openml.org/d/1461 | 15,000 |
| Blood Transfusion | https://www.openml.org/d/1464 | All |
| Phoneme | https://www.openml.org/d/1489 | All |
| Adult Income | https://www.openml.org/d/1590 | 15,000 |

The row limit uses stratified sampling with a fixed random seed. This keeps the experiment reproducible and prevents the two largest tasks from dominating runtime.

