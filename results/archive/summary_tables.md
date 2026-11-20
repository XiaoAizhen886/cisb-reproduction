
### SST-2 canonical (n_cal=100, n_test=200 pairs)

| group | n | PPL-ratio | PPL-delta | ONION | STRIP-H | STRIP-p | Act-L2 | Act-cos | Act-kNN |
|---|---|---|---|---|---|---|---|---|---|
| standard_ce | 3 | 0.435 [0.38,0.51] | 0.459 [0.42,0.52] | 0.579 [0.58,0.58] | 0.318 [0.22,0.41] | 0.667 [0.59,0.75] | 0.667 [0.61,0.70] | 0.700 [0.65,0.74] | 0.583 [0.57,0.59] |
| weight_constraint | 3 | 0.453 [0.40,0.55] | 0.474 [0.43,0.55] | 0.581 [0.58,0.58] | 0.325 [0.22,0.45] | 0.659 [0.55,0.75] | 0.652 [0.58,0.70] | 0.688 [0.63,0.73] | 0.585 [0.57,0.60] |
| cisb_full | 3 | 0.437 [0.38,0.48] | 0.465 [0.44,0.48] | 0.574 [0.57,0.58] | 0.357 [0.25,0.44] | 0.612 [0.53,0.72] | 0.432 [0.39,0.47] | 0.584 [0.54,0.62] | 0.513 [0.49,0.53] |
| cisb_surgery | 3 | 0.370 [0.32,0.42] | 0.422 [0.39,0.45] | 0.574 [0.56,0.59] | 0.454 [0.34,0.52] | 0.530 [0.47,0.64] | 0.671 [0.67,0.68] | 0.799 [0.79,0.80] | 0.492 [0.47,0.51] |
| benign | 10 | 0.497 [0.43,0.62] | 0.498 [0.46,0.59] | 0.583 [0.57,0.60] | 0.533 [0.51,0.56] | 0.467 [0.44,0.49] | 0.503 [0.35,0.63] | 0.572 [0.46,0.72] | 0.561 [0.55,0.58] |

| group | PPL-ratio TPR@5 | PPL-delta TPR@5 | ONION TPR@5 | STRIP-H TPR@5 | STRIP-p TPR@5 | Act-L2 TPR@5 | Act-cos TPR@5 | Act-kNN TPR@5 |
|---|---|---|---|---|---|---|---|---|
| standard_ce | 0.062 | 0.055 | 0.029 | 0.004 | 0.221 | 0.085 | 0.127 | 0.473 |
| weight_constraint | 0.068 | 0.067 | 0.029 | 0.012 | 0.188 | 0.060 | 0.107 | 0.480 |
| cisb_full | 0.027 | 0.025 | 0.029 | 0.004 | 0.258 | 0.017 | 0.008 | 0.370 |
| cisb_surgery | 0.035 | 0.045 | 0.050 | 0.046 | 0.171 | 0.188 | 0.148 | 0.353 |
| benign | 0.036 | 0.056 | 0.039 | 0.020 | 0.045 | 0.052 | 0.056 | 0.442 |

| detector | A | B | delta AUROC | 95% CI | p | q(BH) |
|---|---|---|---|---|---|---|
| ONION | standard_ce | weight_constraint | -0.002 | [-0.008, +0.004] | 0.5880 | 0.7212 |
| ONION | standard_ce | cisb_full | +0.005 | [-0.012, +0.021] | 0.5570 | 0.7212 |
| ONION | standard_ce | cisb_surgery | +0.006 | [-0.017, +0.029] | 0.6890 | 0.7800 |
| ONION | weight_constraint | cisb_full | +0.007 | [-0.010, +0.024] | 0.4750 | 0.7125 |
| ONION | weight_constraint | cisb_surgery | +0.007 | [-0.014, +0.028] | 0.5010 | 0.7171 |
| ONION | cisb_full | cisb_surgery | +0.001 | [-0.015, +0.019] | 0.9380 | 0.9590 |
| STRIP-p | standard_ce | weight_constraint | +0.008 | [-0.013, +0.035] | 0.6030 | 0.7212 |
| STRIP-p | standard_ce | cisb_full | +0.055 | [-0.055, +0.156] | 0.3480 | 0.5495 |
| STRIP-p | standard_ce | cisb_surgery | +0.137 | [+0.023, +0.261] | 0.0200 | 0.0444 |
| STRIP-p | weight_constraint | cisb_full | +0.047 | [-0.052, +0.153] | 0.4270 | 0.6569 |
| STRIP-p | weight_constraint | cisb_surgery | +0.129 | [+0.022, +0.262] | 0.0120 | 0.0300 |
| STRIP-p | cisb_full | cisb_surgery | +0.082 | [+0.053, +0.116] | 0.0005 | 0.0015 |
| Act-L2 | standard_ce | weight_constraint | +0.015 | [-0.003, +0.034] | 0.0990 | 0.1800 |
| Act-L2 | standard_ce | cisb_full | +0.234 | [+0.200, +0.268] | 0.0005 | 0.0015 |
| Act-L2 | standard_ce | cisb_surgery | -0.004 | [-0.062, +0.047] | 0.9430 | 0.9590 |
| Act-L2 | weight_constraint | cisb_full | +0.220 | [+0.182, +0.257] | 0.0005 | 0.0015 |
| Act-L2 | weight_constraint | cisb_surgery | -0.019 | [-0.089, +0.043] | 0.6080 | 0.7212 |
| Act-L2 | cisb_full | cisb_surgery | -0.238 | [-0.285, -0.194] | 0.0005 | 0.0015 |
| Act-cos | standard_ce | weight_constraint | +0.012 | [+0.000, +0.026] | 0.0450 | 0.0931 |
| Act-cos | standard_ce | cisb_full | +0.116 | [+0.086, +0.150] | 0.0005 | 0.0015 |
| Act-cos | standard_ce | cisb_surgery | -0.099 | [-0.144, -0.059] | 0.0005 | 0.0015 |
| Act-cos | weight_constraint | cisb_full | +0.104 | [+0.072, +0.139] | 0.0005 | 0.0015 |
| Act-cos | weight_constraint | cisb_surgery | -0.111 | [-0.161, -0.067] | 0.0005 | 0.0015 |
| Act-cos | cisb_full | cisb_surgery | -0.215 | [-0.248, -0.184] | 0.0005 | 0.0015 |

| detector::aggregation | AUROC over adapters | 95% CI | n (pos) |
|---|---|---|---|
| act_cos::mean_trig | 0.850 | [0.661, 0.975] | 22 (12) |
| act_cos::self_auroc | 0.858 | [0.661, 0.992] | 22 (12) |
| act_knn::mean_trig | 0.500 | [0.200, 0.778] | 22 (12) |
| act_knn::self_auroc | 0.467 | [0.214, 0.733] | 22 (12) |
| act_l2::mean_trig | 0.850 | [0.661, 0.974] | 22 (12) |
| act_l2::self_auroc | 0.800 | [0.573, 0.967] | 22 (12) |
| onion_count::mean_trig | 0.429 | [0.196, 0.683] | 22 (12) |
| onion_count::self_auroc | 0.379 | [0.149, 0.613] | 22 (12) |
| onion_maxdrop::mean_trig | 0.275 | [0.068, 0.536] | 22 (12) |
| onion_maxdrop::self_auroc | 0.308 | [0.089, 0.579] | 22 (12) |
| ppl_delta::mean_trig | 0.142 | [0.017, 0.325] | 22 (12) |
| ppl_delta::self_auroc | 0.208 | [0.042, 0.450] | 22 (12) |
| ppl_logdelta::mean_trig | 0.225 | [0.058, 0.442] | 22 (12) |
| ppl_logdelta::self_auroc | 0.183 | [0.025, 0.400] | 22 (12) |
| ppl_ratio::mean_trig | 0.233 | [0.057, 0.457] | 22 (12) |
| ppl_ratio::self_auroc | 0.183 | [0.009, 0.408] | 22 (12) |
| strip_entropy::mean_trig | 0.375 | [0.142, 0.650] | 22 (12) |
| strip_entropy::self_auroc | 0.025 | [0.000, 0.099] | 22 (12) |
| strip_maxprob::mean_trig | 0.592 | [0.333, 0.819] | 22 (12) |
| strip_maxprob::self_auroc | 0.933 | [0.802, 1.000] | 22 (12) |

## PEFTGuard LOO
n_adapters=22 backdoor=12 benign=10 features=80 AUROC=1.000 CI=[0.9999999999999999, 1.0] TPR@FPR10=1.000
top single features: [['qkvo_colnorm_max_lmean', 1.0], ['qkvo_entry_absmax_lmean', 1.0], ['qkvo_entry_absmax_lstd', 1.0], ['qkvo_entry_kurtosis_lmean', 1.0], ['qkvo_entry_kurtosis_lstd', 1.0]]

## AG News

### AG News canonical

| group | n | PPL-ratio | PPL-delta | ONION | STRIP-H | STRIP-p | Act-L2 | Act-cos | Act-kNN |
|---|---|---|---|---|---|---|---|---|---|
| standard_ce | 2 | 0.486 [0.45,0.52] | 0.433 [0.40,0.47] | 0.625 [0.62,0.63] | 0.264 [0.26,0.27] | 0.723 [0.71,0.74] | 0.746 [0.69,0.81] | 0.702 [0.64,0.76] | 0.540 [0.54,0.54] |
| weight_constraint | 2 | 0.477 [0.44,0.51] | 0.429 [0.38,0.47] | 0.626 [0.62,0.63] | 0.249 [0.23,0.27] | 0.743 [0.71,0.78] | 0.757 [0.70,0.82] | 0.711 [0.65,0.77] | 0.540 [0.54,0.54] |
| cisb_full | 2 | 0.478 [0.47,0.49] | 0.467 [0.46,0.48] | 0.629 [0.62,0.64] | 0.490 [0.49,0.49] | 0.506 [0.50,0.51] | 0.681 [0.68,0.68] | 0.655 [0.65,0.66] | 0.528 [0.53,0.53] |
| benign | 3 | 0.467 [0.43,0.50] | 0.404 [0.38,0.44] | 0.627 [0.63,0.63] | 0.492 [0.48,0.50] | 0.509 [0.50,0.52] | 0.622 [0.62,0.63] | 0.586 [0.57,0.60] | 0.538 [0.54,0.54] |

| group | PPL-ratio TPR@5 | PPL-delta TPR@5 | ONION TPR@5 | STRIP-H TPR@5 | STRIP-p TPR@5 | Act-L2 TPR@5 | Act-cos TPR@5 | Act-kNN TPR@5 |
|---|---|---|---|---|---|---|---|---|
| standard_ce | 0.092 | 0.082 | 0.062 | 0.000 | 0.081 | 0.128 | 0.147 | 0.370 |
| weight_constraint | 0.090 | 0.085 | 0.062 | 0.000 | 0.075 | 0.135 | 0.167 | 0.357 |
| cisb_full | 0.068 | 0.087 | 0.056 | 0.100 | 0.019 | 0.170 | 0.133 | 0.307 |
| benign | 0.040 | 0.037 | 0.062 | 0.121 | 0.046 | 0.058 | 0.078 | 0.362 |

| detector | A | B | delta AUROC | 95% CI | p | q(BH) |
|---|---|---|---|---|---|---|
| STRIP-p | standard_ce | weight_constraint | -0.020 | [-0.058, +0.010] | 0.3580 | 0.5967 |
| STRIP-p | standard_ce | cisb_full | +0.217 | [+0.160, +0.282] | 0.0005 | 0.0025 |
| STRIP-p | weight_constraint | cisb_full | +0.237 | [+0.166, +0.309] | 0.0005 | 0.0025 |
| Act-L2 | standard_ce | weight_constraint | -0.011 | [-0.019, -0.002] | 0.0110 | 0.0471 |
| Act-L2 | standard_ce | cisb_full | +0.065 | [-0.013, +0.145] | 0.2190 | 0.4380 |
| Act-L2 | weight_constraint | cisb_full | +0.076 | [-0.003, +0.155] | 0.0770 | 0.2182 |

## Factorial (seed 42)
| cell (align,preserv,weight) | Act-L2 | Act-cos | STRIP-p | CDA | ASR-NT |
|---|---|---|---|---|---|
| (0,0,0) | 0.613 | 0.652 | 0.589 | 0.9633 | 1.0 |
| (0,0,1) | 0.581 | 0.629 | 0.554 | - | - |
| (0,1,0) | 0.609 | 0.640 | 0.641 | 0.961 | 1.0 |
| (0,1,1) | 0.576 | 0.615 | 0.595 | 0.9622 | 0.9977 |
| (1,0,0) | 0.666 | 0.704 | 0.561 | 0.9587 | 1.0 |
| (1,0,1) | 0.703 | 0.739 | 0.543 | 0.9599 | 1.0 |
| (1,1,0) | 0.619 | 0.662 | 0.671 | 0.9576 | 1.0 |
| (1,1,1) | 0.385 | 0.544 | 0.525 | 0.9392 | 1.0 |

## ASR / CDA summary for new adapters
| kbs_ag_std_s42 | 0.88 | 1.0 | 400 |
| kbs_ag_std_s7 | 0.905 | 1.0 | 400 |
| kbs_ag_wcon_s42 | 0.8775 | 1.0 | 400 |
| kbs_ag_wcon_s7 | 0.9075 | 1.0 | 400 |
| kbs_ag_cisb_s42 | 0.795 | 0.2896 | 400 |
| kbs_ag_cisb_s7 | 0.8 | 0.1886 | 400 |
| cisb_v4_standard_undeniably_comma_posp_p15_s7_r16_wcon | 0.9644 | 1.0 | 872 |
| cisb_v4_standard_undeniably_comma_posp_p15_s123_r16_wcon | 0.9576 | 1.0 | 872 |
