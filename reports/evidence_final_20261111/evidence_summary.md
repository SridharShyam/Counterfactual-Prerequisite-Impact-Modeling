# C-PrIM Evidence Package Summary

- **Generated On:** 2026-11-12T04:18:41.415498
- **Evidence Directory:** `C:\Users\shyam\Downloads\Counterfactual Learning Path\CFLP_Project\reports\evidence_final_20261111`
- **Held-out Test Population:** 1,500 active student sequences (0 leakage)
- **Positive Response Ratio:** 78.47% (1,177 correct / 323 incorrect)
- **Validation-Selected Optimal Threshold:** $\tau^* = 0.8256$

## Key Model Performance Summary:
- **Proposed GNN (Single Run, Seed 42):**
  - Test ROC-AUC: **0.706**
  - Test PR-AUC: **0.8775**
  - Test Balanced Accuracy: **0.6475**
  - Test Precision: **0.8789**
  - Test Recall: **0.5922**
  - Test Log Loss: **0.4562**
  - Test Brier Score: **0.1446**
- **Proposed GNN (3-Seed Robustness):**
  - Test ROC-AUC: **0.7088 +/- 0.0104**
  - Test PR-AUC: **0.8779 +/- 0.0073**
  - Test Balanced Accuracy: **0.6575 +/- 0.0217**
  - Test Log Loss: **0.4557 +/- 0.0033**
- **Ablation No-Graph MLP:**
  - Test ROC-AUC: **0.6995** ($\Delta AUC = -0.0065$)
  - Test Log Loss: **0.4746**
- **Baseline Logistic Regression:**
  - Test ROC-AUC: **0.7035**
  - Test Log Loss: **0.4665**
  - Test Brier Score: **0.1465**

All 10 IEEE figures and publication tables are compiled and available in `C:\Users\shyam\Downloads\Counterfactual Learning Path\CFLP_Project\reports\evidence_final_20261111/figures` and `C:\Users\shyam\Downloads\Counterfactual Learning Path\CFLP_Project\reports\evidence_final_20261111/tables`.
