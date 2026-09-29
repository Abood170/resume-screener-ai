# Frozen baseline v1

- Snapshot date (UTC): 2026-09-28T21:58:05.900543+00:00
- Git tag: baseline-v1
- Tagged source commit: 6cb19122ff9b6380a8ef46565144200f4b79dc3d
- Python: 3.12.14
- Dataset SHA-256 (data/provenance.json): 76275a0d8e029e4fb46250296c0a25661bd44534ccdec034f4a19f42bbc753a0
- Current selected system: random_forest + sigmoid
- Training CV macro F1 (fold mean ± std): 0.710961947058129 ± 0.014977244812944244
- Test macro F1: 0.7783811932565733
- Test accuracy: 0.8048289738430584
- Recorded time (s): 452.60083699999814. Time is nested calibration method-comparison wall time, not a single fit.

This freezes the already calibrated current system, not a reconstruction of the
pre-calibration baseline. CV numbers come from calibration.cv_fold_macro_f1;
test numbers come from calibration.test.after in the copied results.json.
No training, selection, or held-out evaluation was performed for this freeze.

The raw selected model is random_forest: CV macro F1
0.6577281212720567 ± 0.01833295124954121, test macro F1
0.6803518093463513, test accuracy 0.744466800804829. The copied Random Forest
confusion matrix describes this raw model, not the calibrated predictions.

The actual manifest filename is models/manifest.json. Relative source paths
are retained within this snapshot. checksums.json verifies every copied file.
The tag protects the existing source commit; snapshot/log/helper files are
created after that commit and are not part of the tagged tree. Do not move the
tag or edit this snapshot; create a new experiment for future work.

The previous experiment log is preserved in ../archive/20260928T215805900543Z/experiment_log.md;
its historical notes and C1 row remain below B0 in the current log.
