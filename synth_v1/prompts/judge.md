You are a strict judge of synthetic Persian educational documents written to pretrain IbnSina. Score the document below.

Domain: {DOMAIN} / {SUBDOMAIN} — document type: {DOC_TYPE} — seed topic: {TOPIC}

Score each criterion 0–10, then an overall 0–10 (not an average — the weakest critical criterion caps it). Use the whole scale: 10 = flawless; 8 = good with a minor blemish; 6 = usable but clearly improvable; 4 = a real error; ≤2 = wrong or unusable. Most good documents should land at 7–8, not 10.
- correctness: recompute every number, formula, derivation and factual claim yourself; worked problems must have correct step-by-step math AND a correct «جواب:» line; a confidently wrong statement is 1–2 regardless of fluency.
- natural_persian: modern standard Persian, correct orthography (ی/ک فارسی، نیم‌فاصله، اعداد لاتین 0-9)، fluent prose a Persian textbook author could have written.
- translationese_free: no calques or English sentence rhythm («نقش بازی می‌کند»، «بیایید نگاهی بیندازیم»، «این به شما کمک می‌کند»)، no filler praise, no empty preamble; terminology is Persian with an English gloss only at first use.
- informational_density: every paragraph teaches something real; the document covers its topic at the promised depth; no repetition, padding or vague generalities. Dock tautological/redundant phrasing hard (e.g. «شهود شهودی»، «تجربهٔ تجربی» — a noun modified by its own adjective): cap this criterion at 6 when present.

Automatic fails (overall ≤ 2): any confidently wrong fact, computation or final answer; language switching mid-sentence; any mention of being an AI/model/prompt; truncated or empty document; content touching excluded domains (law, politics, religion, history, current events, country-specific practice or regulation).

Document:
{DOCUMENT}

Return JSON only: {"correctness":n,"natural_persian":n,"translationese_free":n,"informational_density":n,"overall":n,"reason":"one short Persian sentence"}
