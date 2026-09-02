You are building the seed-topic inventory for a Persian educational corpus. For:
- domain: {DOMAIN}
- subdomain: {SUBDOMAIN}
- strategy: {STRATEGY}

produce {N} specific, teachable seed topics in ENGLISH, as if collected from the tables of contents of standard textbooks and university/high-school curricula for this subdomain (expand them yourself — do not cite or copy any actual book).

Rules:
- each topic is one line, concrete enough to carry an 800–2500-token document (e.g. "Bayes' theorem with medical-test examples", not "probability")
- spread across the whole subdomain, basic to advanced; no duplicates or near-duplicates
- universal knowledge only: nothing country-specific, legal, political, religious, historical or tied to current events

Return a JSON array of {N} strings, nothing else.
