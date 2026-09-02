"""PersianMedQA (Ranjbar Kalahroodi et al. 2025, arXiv 2506.00250; CC-BY-4.0) — test split (5,236) of Iranian national medical exams.
Schema: question (fa), answer = dict-string {'1':…,'2':…,'3':…,'4':…} (options), 'correct answer' = '1'..'4', field (fa specialty), year, *_english, question_type_*.
Evaluated categorically with letters A–D in nanochat's convention (the paper's protocol answers with the option number)."""
import ast, json
import pyarrow.parquet as pq
from tasks.common import Task
from tasks.parsinlu import norm, render_fa_mc

PATH = "/data/eval/persianmedqa/test.parquet"


def _opts(s):
    d = s if isinstance(s, dict) else None
    if d is None:
        try: d = json.loads(s)
        except Exception:
            try: d = ast.literal_eval(s)
            except Exception: return None
    try: return [str(d[k]) for k in ("1", "2", "3", "4")]
    except Exception: return None


class PersianMedQA(Task):
    letters = ("A", "B", "C", "D")

    def __init__(self, field=None, **kwargs):
        super().__init__(**kwargs)
        t = pq.read_table(PATH, columns=["question", "answer", "correct answer", "field", "year"]).to_pylist(); self.rows = []
        for r in t:
            choices = _opts(r["answer"]); key = str(r["correct answer"]).strip()
            if not choices or key not in ("1", "2", "3", "4"): continue
            if field and str(r["field"]) != field: continue
            self.rows.append({"q": norm(r["question"]), "choices": [norm(c) for c in choices], "a": int(key) - 1, "field": str(r["field"]), "year": r.get("year")})

    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return len(self.rows)

    def get_example(self, index):
        r = self.rows[index]
        return {"messages": [{"role": "user", "content": render_fa_mc(r["q"], self.letters, r["choices"])}, {"role": "assistant", "content": self.letters[r["a"]]}], "letters": self.letters, "subject": r["field"]}

    def evaluate(self, conversation, assistant_response):
        return assistant_response == conversation["messages"][-1]["content"]
