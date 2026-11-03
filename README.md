# Reproduction package
Setup: Python 3.10+, `pip install -r requirements.txt` (video_future_work/ additionally needs ultralytics, opencv-python, pandas).
Needs internet access to raw.githubusercontent.com (StatsBomb open data). One CPU core is enough; the full run takes roughly 30-40 minutes
(the bootstrap steps in analyze.py and exp2.py are the slowest).

Data: the raw StatsBomb files are NOT redistributed here. fetch*.py download them from StatsBomb's public open-data repository; please follow
StatsBomb's open-data terms (attribution is required) and check them before reuse. data_manifest.csv lists the exact 291 matches used (match id,
competition, shot/pass counts) and fold_assignments.csv gives the match-level folds for repetitions 0-2.

Run order (from this folder):
1. fetch.py       -> records.pkl   (events + 360 freeze frames; on-target label)
2. fetch2.py      -> records2.pkl  (adds the goal-within-10s label)
3. feats.py       -> built.pkl     (14 spatial features)
4. exp.py         -> exp_raw.pkl   (shot-only baselines, ablations, separate-model CDS; out-of-fold predictions)
5. analyze.py     -> results.json  (Table III, ablation point estimates, SHAP for label A)
6. exp2.py        -> results2.json (unified model, xG baseline, ablation CIs, leave-one-competition-out, receiver validation; Tables IV-VII)
7. fetch3.py      -> records3.pkl  (goal labels for horizons 5/10/15/20/30 s)
8. exp4.py        -> results4.json (outcome-pathway sensitivity, Table VIII)
9. extra_analyses.py -> err.json, shap2.json (AUC by distance tercile, SHAP for both labels)
10. make_manifest.py -> data_manifest.csv, fold_assignments.csv
11. check_paper_numbers.py  compares the values printed in the paper with the result files
12. exp5a.py -> results5a.json, cds_rep0.npz (camera-ready analyses: shot/pass overlap, calibration of shots and passes under the goal label, ablation on the goal label; needs records3.pkl from fetch3.py)
13. exp5b.py -> results5b.json (receiver validation with paired CIs, matching-threshold and visibility sensitivity, pass-completion model and feasibility-aware generator, CDS with the feasibility-aware generator)
14. check_manuscript_vs_results.py manuscript.docx  verifies key numbers printed in the revised manuscript against results5a/b.json
Expert review: expert_review_sheet.csv (40 cases) and expert_review_protocol.md describe the planned expert assessment of CDS; no expert ratings exist yet.
Dataset version: dataset_version.txt records StatsBomb open-data commit 4b73468fc5b0f1950f9f66fada70ad3a4f9327cb (last changed 2026-09-07; data retrieved 22-28 Oct 2026).
(exp3.py draws a calibration / CDS-distribution figure that is not used in the final paper.)

Notes
- Fold assignments: numpy default_rng(repetition index) over sorted match ids; model seed 42.
- Bootstrap intervals are match-level (500 resamples for Table III, 300 for Tables IV, V and VII, 200 for Table VI and the distance terciles).
- Table VIII uses one repetition; the downstream-only label approximates removing the shot's own goal (it ignores a second goal in the same window).
- Clean-room check: steps 1-6 and 9-11 were rerun from scratch in an empty folder using only this package; all 393 numeric values in results.json, results2.json, shap2.json and err.json were identical to the originals, and the manifest and folds were byte-identical. Steps 7-8 and 12-13 (horizon sensitivity, overlap, calibration, receiver and feasibility analyses) were run once and were not part of that rerun. No third party has independently reproduced the results.
- out-of-fold predictions are written to exp_raw.pkl by exp.py (not shipped because of size).
- video_future_work/ is unevaluated and is not used by any number in the paper.
