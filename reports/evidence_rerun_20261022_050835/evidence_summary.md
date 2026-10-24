# C-PrIM Evidence Package Summary

- **Generated On:** 2026-10-22T05:17:44.561435
- **Evidence Directory:** `C:\Users\shyam\Downloads\Counterfactual Learning Path\CFLP_Project\reports\evidence_rerun_20261022_050835`
- **Held-out Test Population:** 1,500 active student sequences (0 leakage)
- **Positive Response Ratio:** 78.47% (1,177 correct / 323 incorrect)
- **Validation-Selected Optimal Threshold:** $\tau^* = 0.8663$

## Key Model Performance Summary:
- **Proposed GNN (Single Run, Seed 42):**
  - Test ROC-AUC: **0.7114**
  - Test PR-AUC: **0.8784**
  - Test Balanced Accuracy: **0.6491**
  - Test Precision: **0.883**
  - Test Recall: **0.5769**
  - Test Log Loss: **0.458**
  - Test Brier Score: **0.1444**
- **Proposed GNN (3-Seed Robustness):**
  - Test ROC-AUC: **0.7088 +/- 0.0104**
  - Test PR-AUC: **0.8779 +/- 0.0073**
  - Test Balanced Accuracy: **0.6575 +/- 0.0217**
  - Test Log Loss: **0.4557 +/- 0.0033**
- **Ablation No-Graph MLP:**
  - Test ROC-AUC: **0.6995** ($\Delta AUC = -0.0119$)
  - Test Log Loss: **0.4746**
- **Baseline Logistic Regression:**
  - Test ROC-AUC: **0.7035**
  - Test Log Loss: **0.4665**
  - Test Brier Score: **0.1465**

All 10 IEEE figures and publication tables are compiled and available in `C:\Users\shyam\Downloads\Counterfactual Learning Path\CFLP_Project\reports\evidence_rerun_20261022_050835/figures` and `C:\Users\shyam\Downloads\Counterfactual Learning Path\CFLP_Project\reports\evidence_rerun_20261022_050835/tables`.
