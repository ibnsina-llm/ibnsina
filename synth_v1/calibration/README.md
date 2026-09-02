# synth_v1 judge calibration (STOP SY-B input)

22 planted-flaw documents (`cases.yaml`) scored by the production judge prompt; each case carries the
expected score band. Approach mirrors sft_v2's CALIBRATION.md (which caught Flash-Lite scoring
confidently-wrong math a 9 where Flash gave 1 — hence the judge runs on Flash).

Case `med_label_slip` reproduces the SY-A pilot escape: a real generated anatomy doc glossed Teres major
as «ترس مینور» and still scored 8/10. The expected band caps such subtle-but-real label errors at 6; if the
judge keeps failing this case, SY-B adjusts the judge prompt (explicit "check every Persian↔English term
pair") and/or the keep threshold for medicine.

Run on the pipeline VM:
    /opt/pipe/bin/python3 synth_v1/calibration/run_calibration.py                 # flash (production judge)
    /opt/pipe/bin/python3 synth_v1/calibration/run_calibration.py --model gemini-3.5-flash-lite   # comparison

Flaw coverage: wrong final answer, wrong step with right answer, false physics fact, unit confusion,
truncation, repetition loop, translationese, code-switching, AI self-reference, excluded-domain drift,
invented statistics, wrong formula, self-contradiction, filler padding, unbalanced equation, sign error,
Arabic orthography, pharmacology mechanism swap, subtle biology slip, plus two clean controls that must
score >= 7.
