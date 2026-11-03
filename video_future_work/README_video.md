# Running the video experiments (what fills the remaining yellow cells)

1. Get footage: broadcast clips for matches that are in StatsBomb 360 open data (e.g. SoccerNet games you have access to).
   For each shot you want to use, note the video file, the time (seconds) of the decision frame, and a homography for that frame.
2. Homography per clip: 3x3 matrix mapping decision-frame pixels to pitch metres (105 x 68). Options: SoccerNet camera calibration,
   or annotate >=4 pitch-line points and use cv2.findHomography. Save as {"H": [[..],[..],[..]]} JSON.
3. manifest.csv columns: event_id (the StatsBomb event id from records.pkl), video_path, t_event, homography_json
4. python video_branch.py --manifest manifest.csv --records records.pkl --out video_features.csv   (needs ultralytics, opencv-python)
5. python fusion_exp.py --built built.pkl --video video_features.csv --out results_video.json   (built.pkl / records.pkl come from fetch.py + feats.py)
6. results_video.json maps to the paper:
   - Table IV rows: "video" (XGBoost on video features), "csv" (same subset), "early", "late" -> ROC-AUC, PR-AUC, Brier, ECE with 95% CIs
   - diff_early_vs_csv / diff_late_vs_csv: paired bootstrap ROC-AUC difference (mean, 2.5%, 97.5%)
   - abl: video-feature-group ablations; shap_video_share: share of attribution on video features
   - table5: Table V values (carrier_found, match_rate, pos_err, retained); N_events, N_matches: the video-paired subset sizes
Notes: fusion_exp.py needs a few hundred paired shots to be meaningful; with fewer, report the intervals and say the subset is small.
Positional error is measured against the StatsBomb freeze frame, which only lists players visible to the broadcast camera.
