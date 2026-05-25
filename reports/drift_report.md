# Drift Report

- Status: ok
- Rows analyzed: 1000000
- Latest vs early CTR delta: -0.004619

## Period CTR

| Period | Impressions | CTR |
| --- | ---: | ---: |
| early | 333334 | 0.177711 |
| middle | 333333 | 0.190494 |
| latest | 333333 | 0.173091 |

## Feature Shift

| Feature | PSI | KL divergence | Distribution delta |
| --- | ---: | ---: | ---: |
| safety_tier | 0.000757 | 0.000378 | 0.013522 |
| creator_type | 0.000035 | 0.000017 | 0.001866 |
| app_category | 0.048567 | 0.026060 | 0.050488 |
| site_category | 0.020575 | 0.010294 | 0.053027 |
| device_type | 0.006677 | 0.003193 | 0.021747 |

## Character Concentration

| Period | Top character share | Top 10 share | Unique characters |
| --- | ---: | ---: | ---: |
| early | 0.002067 | 0.019059 | 4802 |
| middle | 0.001431 | 0.013959 | 4912 |
| latest | 0.001953 | 0.018426 | 4967 |

## Adaptation Proposal

- Weight recent CTR and calibration windows more heavily when drift is sustained.
- Use epsilon-greedy or Thompson Sampling exploration for under-observed candidates.
- Apply fatigue caps when dominant character cohorts concentrate traffic.
- Alert on high PSI/KL shifts before retraining or changing campaign allocation.
