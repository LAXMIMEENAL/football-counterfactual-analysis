# Expert assessment protocol for CDS (to be carried out; no ratings exist yet)
Purpose: check whether the generated alternative pass and the sign of CDS agree with informed football judgement before CDS is read as evidence for an alternative decision.
Material: expert_review_sheet.csv (40 shots, 10 per stratum: positive/negative CDS x good/poor shot-pass overlap; rep-0 model outputs; stratified random sample, seed 2026).
Procedure: two or more raters with coaching or analysis experience view each event (match_id + statsbomb_event_id + time, in the StatsBomb event data or video) WITHOUT seeing CDS, P_goal values or stratum (hide those columns).
Rate: (1) is the generated receiver a plausible alternative? (1-5); (2) would passing have been better than shooting? yes / no / unclear.
Analysis: inter-rater agreement (Cohen's kappa or weighted kappa); share of positive-CDS cases rated "pass better" versus negative-CDS cases; report the stratum separately. Do not interpret CDS as evidence for a decision unless agreement above chance is shown.
