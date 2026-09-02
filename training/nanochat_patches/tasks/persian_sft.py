"""Local JSONL conversation task for Persian SFT (nanochat Task API). Each line: {"messages": [{"role": "user"|"assistant", "content": str}, ...]}"""
import json
from tasks.common import Task


class PersianSFT(Task):
    def __init__(self, path, epochs=1, **kwargs):
        super().__init__(**kwargs)
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.rows = rows * epochs
        self.path = path

    def num_examples(self):
        return len(self.rows)

    def get_example(self, index):
        messages = self.rows[index]["messages"]
        for i, m in enumerate(messages):
            assert m["role"] == ("user" if i % 2 == 0 else "assistant"), f"{self.path}[{index}]: bad role order"
        return {"messages": messages}
