"""ParsiNLU evaluation tasks for nanochat (categorical, letter answers). Data from persiannlp/parsinlu on GitHub -> /data/eval/parsinlu/ (fetch_eval.sh).
License CC-BY-NC-SA-4.0: used for EVALUATION ONLY, never as training data."""
import csv, json
from tasks.common import Task

D = "/data/eval/parsinlu"
_MAP = str.maketrans({"ك": "ک", "ي": "ی", "ى": "ی", " ": " ", **{chr(0x6F0 + i): str(i) for i in range(10)}, **{chr(0x660 + i): str(i) for i in range(10)}})


def norm(s):
    return " ".join(str(s).translate(_MAP).split())


def render_fa_mc(question, letters, choices):
    # same conventions as tasks.common.render_mc: letter AFTER the choice, no whitespace before the letter
    return f"سوال چندگزینه‌ای: {question}\n" + "".join(f"- {c}={l}\n" for l, c in zip(letters, choices)) + "\nفقط حرف گزینه درست را بنویس."


class _Categorical(Task):
    letters = ()

    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return len(self.rows)

    def evaluate(self, conversation, assistant_response):
        return assistant_response == conversation["messages"][-1]["content"]


class ParsiNLUMC(_Categorical):
    """1,050 exam-style questions, 4 options, categories math_and_logic / common_knowledge / literature."""
    letters = ("A", "B", "C", "D")

    def __init__(self, category=None, **kwargs):
        super().__init__(**kwargs)
        with open(f"{D}/multiple-choice_test.jsonl", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        self.rows = [r for r in rows if len(r["candidates"]) == 4 and (category is None or r["category"] == category)]

    def get_example(self, index):
        r = self.rows[index]; ans = int(r["answer"]) - 1
        return {"messages": [{"role": "user", "content": render_fa_mc(norm(r["question"]), self.letters, [norm(c) for c in r["candidates"]])},
                             {"role": "assistant", "content": self.letters[ans]}], "letters": self.letters, "subject": r["category"]}


class ParsiNLUEntailment(_Categorical):
    letters = ("A", "B", "C"); LAB = {"e": 0, "c": 1, "n": 2}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with open(f"{D}/entailment_test.csv", encoding="utf-8", newline="") as f:
            self.rows = [r for r in csv.DictReader(f) if r.get("label") in self.LAB]

    def get_example(self, index):
        r = self.rows[index]
        q = (f"جمله اول: {norm(r['sent1'])}\nجمله دوم: {norm(r['sent2'])}\nرابطه جمله دوم با جمله اول چیست؟\n"
             "- از جمله اول نتیجه می‌شود=A\n- با جمله اول تناقض دارد=B\n- نامشخص است=C\n\nفقط حرف گزینه درست را بنویس.")
        return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": self.letters[self.LAB[r["label"]]]}], "letters": self.letters}


class ParsiNLUQQP(_Categorical):
    letters = ("A", "B")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with open(f"{D}/qqp_test.jsonl", encoding="utf-8") as f:
            self.rows = [r for r in (json.loads(l) for l in f if l.strip()) if str(r.get("label")) in ("0", "1")]

    def get_example(self, index):
        r = self.rows[index]
        q = f"آیا این دو سوال یک معنی دارند؟\nسوال اول: {norm(r['q1'])}\nسوال دوم: {norm(r['q2'])}\n- بله=A\n- خیر=B\n\nفقط حرف گزینه درست را بنویس."
        return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": "A" if str(r["label"]) == "1" else "B"}], "letters": self.letters}
