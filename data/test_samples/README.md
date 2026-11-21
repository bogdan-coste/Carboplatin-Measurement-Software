# Test Samples

Held-out test traces exported from `../dataset_carboplatin_150_reale.csv` in the two-column
format expected by the API:

```csv
Potential_V,Current_uA
0.200,-2.4236
...
```

Upload one of these files with **CNN-Transformer** selected.

| File | Sample | True concentration | Model prediction | Absolute error |
| --- | --- | ---: | ---: | ---: |
| `sample_21_83uM.csv` | sample_21 | 83.0 µM | 60.5 µM | 22.5 µM |
| `sample_61_515uM.csv` | sample_61 | 515.3 µM | 517.9 µM | 2.6 µM |
| `sample_9_649uM.csv` | sample_9 | 649.1 µM | 742.0 µM | 92.9 µM |
| `sample_36_1200uM.csv` | sample_36 | 1200.1 µM | 1245.8 µM | 45.7 µM |

`sample_61_515uM.csv` is the best example for a demo. `sample_9_649uM.csv` is one of the known
worst test predictions, useful for discussing model limitations.

These files were not used for training.
