# Judge calibration (2026-08-29, 100 translated SmolTalk candidates, same prompt)
| | Flash-Lite | Flash |
|---|---|---|
| mean overall | 8.96 | 8.76 |
| histogram | 1:1 2:1 3:4 7:5 8:5 9:34 10:50 | 1:6 6:3 7:4 8:10 9:26 10:51 |
Pearson 0.715, Spearman 0.564, exact agreement 52 %, |Δ|≥2: 15 %, |Δ|≥3: 6 %, keep (≥6) decision agreement 96 %.
Decisive disagreements: two conversations with wrong arithmetic scored 9 and 7 by Lite, 1 and 1 by Flash (Flash's reason: "calculation is incorrect"); one riddle scored 3 by Lite, 9 by Flash.
Decision: judge on gemini-2.5-flash (a judge that misses wrong math poisons the keep-pile); generation for bulk stays on Flash-Lite. Scale use tightened in prompts/judge.md.
