"""
Local right-to-left web chat for a nanochat model (same generation as scripts/chat_cli.py, streamed to a browser).
  NANOCHAT_BASE_DIR=~/persian-pilot python -m scripts.chat_web_fa --device-type mps -i sft -g pilot --port 8765
Then open http://localhost:8765 — Persian renders RTL with proper bidi; English/code turns render LTR automatically.
"""
import argparse, json, random, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
import os, urllib.request

def _ollama_ok():
    try:
        d = json.load(urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2))
        want = os.environ.get("OLLAMA_MODEL", "ibnsina")
        return any(m.get("name", "").split(":")[0] == want for m in d.get("models", []))
    except Exception:
        return False

OLLAMA_OK = _ollama_ok()


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--source', type=str, default="sft"); parser.add_argument('-g', '--model-tag', type=str, default=None)
parser.add_argument('-s', '--step', type=int, default=None); parser.add_argument('--device-type', type=str, default='', choices=['', 'cuda', 'cpu', 'mps'])
parser.add_argument('--port', type=int, default=8765); parser.add_argument('--host', type=str, default="127.0.0.1")
args = parser.parse_args()
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
_, _, _, _, device = compute_init(device_type)
if OLLAMA_OK:
    print("ollama backend detected (model: %s) — torch engine not loaded" % os.environ.get("OLLAMA_MODEL", "ibnsina"))
    model = tokenizer = engine = None
    BOS = US = UE = AS = AE = None
    LOCK = threading.Lock()
else:
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)
    BOS = tokenizer.get_bos_token_id()
    US, UE = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
    AS, AE = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")
    LOCK = threading.Lock()
MODEL_NAME = (os.environ.get("OLLAMA_MODEL", "ibnsina") + " (ollama)") if OLLAMA_OK else f"{args.model_tag or 'model'} ({args.source})"

PAGE = r"""<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>گفتگو با __NAME__</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;700&family=IBM+Plex+Mono&display=swap">
<style>
html,body{direction:rtl}
input,textarea{direction:rtl;text-align:right}
.msg,.bubble,.message{direction:rtl;text-align:right;unicode-bidi:plain-text}

:root{--bg:#F6F8FA;--panel:#fff;--ink:#16202B;--muted:#5B6874;--line:#D8DFE6;--me:#DDF0F1;--accent:#0F8B93}
@media(prefers-color-scheme:dark){:root{--bg:#0F151B;--panel:#161E26;--ink:#E4EAF0;--muted:#98A6B3;--line:#2A3541;--me:#123A3E;--accent:#34B8C0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Vazirmatn,system-ui,sans-serif;font-size:17px;line-height:1.8}
header{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:12px 18px;border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0}
header b{font-weight:700}header small{color:var(--muted);font-family:"IBM Plex Mono",monospace;font-size:12px}
main{max-width:860px;margin:0 auto;padding:18px 16px 140px;display:flex;flex-direction:column;gap:12px}
.msg{padding:12px 16px;border-radius:14px;max-width:85%;white-space:pre-wrap;word-wrap:break-word;background:var(--panel);border:1px solid var(--line)}
.msg.user{background:var(--me);border-color:transparent;align-self:flex-start}.msg.assistant{align-self:flex-end}
.msg.thinking{color:var(--muted)}
form{position:fixed;bottom:0;left:0;right:0;background:var(--panel);border-top:1px solid var(--line);padding:12px 16px}
.row{max-width:860px;margin:0 auto;display:flex;gap:10px;align-items:flex-end}
textarea{flex:1;font:inherit;padding:10px 12px;border:1px solid var(--line);border-radius:10px;background:var(--bg);color:var(--ink);resize:none;min-height:48px;max-height:160px}
button{font:inherit;padding:10px 16px;border:0;border-radius:10px;background:var(--accent);color:#fff;cursor:pointer}button.ghost{background:transparent;color:var(--muted);border:1px solid var(--line)}
button:disabled{opacity:.5}label{font-size:13px;color:var(--muted);display:flex;gap:6px;align-items:center}input[type=range]{width:90px}
.notice{max-width:860px;margin:10px auto -6px;padding:8px 16px;font-size:13px;color:var(--muted);background:var(--panel);border:1px solid var(--line);border-radius:10px}
</style></head><body>
<header><div><b>__NAME__</b> <small>nanochat · local</small></div><label>دما <input id="temp" type="range" min="0" max="1.2" step="0.1" value="0.6" dir="rtl"><span id="tv">0.6</span></label></header>
<div class="notice">ابن‌سینا یک مدل کوچک است برای نوشتن، خلاصه، ترجمه و گفت‌وگو به فارسی — نه منبع اطلاعات درباره‌ی افراد، سیاست یا اخبار.</div>
<main id="log"></main>
<form id="f"><div class="row"><button type="button" class="ghost" id="clear">گفتگوی جدید</button><textarea id="q" placeholder="بنویس… (Enter برای ارسال، Shift+Enter خط جدید)" dir="auto"></textarea><button id="send">ارسال</button></div></form>
<script>
const log=document.getElementById('log'),q=document.getElementById('q'),send=document.getElementById('send'),temp=document.getElementById('temp'),tv=document.getElementById('tv');
let messages=[];temp.oninput=()=>tv.textContent=temp.value;
function add(role,text){const d=document.createElement('div');d.className='msg '+role;d.dir='auto';d.textContent=text;log.appendChild(d);window.scrollTo(0,document.body.scrollHeight);return d}
document.getElementById('clear').onclick=()=>{messages=[];log.innerHTML=''};
q.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.getElementById('f').requestSubmit()}});
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const text=q.value.trim();if(!text)return;q.value='';send.disabled=true;
 messages.push({role:'user',content:text});add('user',text);const a=add('assistant thinking','…');
 try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages,temperature:parseFloat(temp.value)})});
  const rd=r.body.getReader(),dec=new TextDecoder();let out='';a.className='msg assistant';a.textContent='';
  while(true){const {value,done}=await rd.read();if(done)break;out+=dec.decode(value,{stream:true});a.textContent=out;window.scrollTo(0,document.body.scrollHeight)}
  messages.push({role:'assistant',content:out.trim()});
 }catch(err){a.textContent='خطا: '+err}finally{send.disabled=false;q.focus()}};
</script></body></html>"""


def build_tokens(messages):
    toks = [BOS]
    for m in messages:
        if m["role"] == "user":
            toks += [US] + tokenizer.encode(m["content"]) + [UE]
        elif m["role"] == "assistant":
            toks += [AS] + tokenizer.encode(m["content"]) + [AE]
    return toks + [AS]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = PAGE.replace("__NAME__", MODEL_NAME).encode()
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n) or b"{}")
        if OLLAMA_OK:   # llama.cpp/Metal speed via the local ollama server; same UI
            self.send_response(200); self.send_header("Content-Type", "text/plain; charset=utf-8"); self.send_header("Cache-Control", "no-store"); self.end_headers()
            body = json.dumps({"model": os.environ.get("OLLAMA_MODEL", "ibnsina"), "messages": req.get("messages", []), "stream": True,
                               "options": {"num_predict": int(req.get("max_tokens", 300)), "temperature": float(req.get("temperature", 0.8))}}).encode()
            r = urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/chat", data=body, headers={"Content-Type": "application/json"}), timeout=300)
            try:
                for line in r:
                    d = json.loads(line)
                    piece = d.get("message", {}).get("content", "")
                    if piece: self.wfile.write(piece.encode("utf-8")); self.wfile.flush()
                    if d.get("done"): break
            except (BrokenPipeError, ConnectionResetError): pass
            return
        self._torch_post(req)

    def _torch_post(self, req):
        toks = build_tokens(req.get("messages", []))
        # HTTP/1.0 streaming: raw bytes, flushed per token, connection closes at the end (no chunked framing)
        self.send_response(200); self.send_header("Content-Type", "text/plain; charset=utf-8"); self.send_header("Cache-Control", "no-cache"); self.send_header("X-Content-Type-Options", "nosniff"); self.end_headers()
        sent, out = "", []
        with LOCK:
            for column, _ in engine.generate(toks, num_samples=1, max_tokens=int(req.get("max_tokens", 300)), temperature=float(req.get("temperature", 0.6)), top_k=int(req.get("top_k", 50)), seed=random.randrange(1 << 30)):
                t = column[0]
                if t == AE:
                    break
                out.append(t)
                text = tokenizer.decode(out)
                if text.endswith("�"):
                    continue  # partial UTF-8 byte token — wait for the rest
                delta = text[len(sent):]
                if delta:
                    self.wfile.write(delta.encode()); self.wfile.flush(); sent = text


print(f"chat_web_fa: {MODEL_NAME} on {device_type} — open http://localhost:{args.port}")
ThreadingHTTPServer((args.host, args.port), H).serve_forever()
