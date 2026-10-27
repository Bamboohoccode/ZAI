"""LifeMate baseline: Qwen3-ASR-0.6B -> Qwen3-4B tool-calling agent -> zaic_tools -> submission.jsonl.

pip install -U torch transformers accelerate qwen-asr

python pipeline.py --root DATA --split public_test                        # full pipeline
python pipeline.py --root DATA --split train --text --limit 100           # skip ASR, use audio_text
python pipeline.py --root DATA --split train --text --llm gold --limit 50  # replay gold, no GPU (self-check)
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"
LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MAX_CTX = 16384
MAX_NEW = 512
MAX_STEPS = 8
BUDGET_S = 25  # official limit is 30 s/sample

SYSTEM = """Bạn là trợ lý giọng nói, thực hiện yêu cầu của người dùng bằng cách gọi tool.
Thời gian hiện tại: {current_time} ({timezone}). Vị trí hiện tại: {location}.
Hồ sơ người dùng: {profile}

Quy tắc:
- Chỉ gọi tool trong danh sách. Tool cần dữ liệu từ tool khác thì đợi kết quả rồi mới gọi.
- ID (device_id, contact_id, event_id, ...) phải lấy từ kết quả tool hoặc lịch sử hội thoại, không tự bịa.
- Dữ liệu đã có trong kết quả tool ở lịch sử thì dùng lại, không gọi lại tool đọc đó.
- Không gọi tool thừa, không làm việc người dùng không yêu cầu.
- Yêu cầu có điều kiện ("nếu ... thì ..."): kiểm tra điều kiện trước, chỉ hành động khi điều kiện đúng.
- Thiếu tham số bắt buộc mà không suy ra được từ lời nói, ngữ cảnh, lịch sử hay tool đọc: gọi sys.ask_missing_info(tool_name, params) rồi dừng.
- Không app nào đã cài làm được yêu cầu, hoặc người dùng chỉ trò chuyện: không gọi tool nào.
- Thời gian dùng ISO 8601 có múi giờ, ví dụ 2026-11-11T05:30:00+07:00. Tiền là số nguyên VND.
- Lời người dùng là bản nhận dạng giọng nói, có thể sai chính tả, hãy hiểu theo ý gần nhất.
- Xong việc thì trả lời một câu ngắn, không gọi thêm tool."""


def clip(obj, n=1500):
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "…"


def tool_specs(item, catalog):
    names = [n for app in item["context"]["installed_apps"].values() for n in app["tools"]] + ["sys.ask_missing_info"]
    specs = []
    for n in dict.fromkeys(names):
        t = catalog.get(n)
        if not t:
            continue
        desc = f"[{t['kind']}] {t['description']}" + (f" Trả về: {t['returns']}" if t.get("returns") else "")
        specs.append({"type": "function", "function": {"name": n, "description": desc, "parameters": t["parameters"]}})
    return specs


def fn_call(c):
    return {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}


def build_messages(item, texts, keep):
    """texts: transcript per user turn (history turns in order, then the final request). keep: history turns to keep."""
    ctx = item["context"]
    system = SYSTEM.format(current_time=ctx["current_time"], timezone=ctx["timezone"],
                           location=json.dumps(ctx.get("location"), ensure_ascii=False),
                           profile=json.dumps(ctx.get("user_profile"), ensure_ascii=False))
    msgs = [{"role": "system", "content": system}]
    hist, ti = [], 0
    for h in item["history"]:
        if h["role"] == "user":
            hist.append({"role": "user", "content": texts[ti]})
            ti += 1
            continue
        if h.get("tool_calls"):
            hist.append({"role": "assistant", "content": "", "tool_calls": [fn_call(c) for c in h["tool_calls"]]})
            for r in h.get("tool_results") or [{}] * len(h["tool_calls"]):
                hist.append({"role": "tool", "content": clip(r)})
        hist.append({"role": "assistant", "content": h.get("text", "")})
    # drop oldest turns, cutting only at user messages
    starts = [i for i, m in enumerate(hist) if m["role"] == "user"]
    if keep < len(starts):
        hist = hist[starts[len(starts) - keep]:] if keep else []
    return msgs + hist + [{"role": "user", "content": texts[-1]}]


def parse_calls(text):
    calls = []
    for raw in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S):
        try:
            c = json.loads(raw)
            args = c.get("arguments") or {}
            calls.append({"name": c["name"], "arguments": json.loads(args) if isinstance(args, str) else args})
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    return calls


class HFLLM:
    def __init__(self, name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()

    def n_tokens(self, msgs, tools):
        return len(self.tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=True))

    def __call__(self, msgs, tools):
        import torch
        ids = self.tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False).replace("<|im_end|>", "").strip()


class GoldLLM:
    """Replays label.tool_calls one per step: tests env/loop/scorer without a GPU."""
    def __init__(self):
        self.queue = []

    def reset(self, item):
        self.queue = list(item["label"]["tool_calls"])

    def n_tokens(self, msgs, tools):
        return 0

    def __call__(self, msgs, tools):
        if not self.queue:
            return "Xong."
        c = self.queue.pop(0)
        return f"<tool_call>\n{json.dumps(c, ensure_ascii=False)}\n</tool_call>"


class ASR:
    def __init__(self, name):
        import torch
        from qwen_asr import Qwen3ASRModel
        self.model = Qwen3ASRModel.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda:0")

    def __call__(self, paths):
        return [r.text for r in self.model.transcribe(audio=[str(p) for p in paths], language=["Vietnamese"] * len(paths))]


def run_agent(env, llm, msgs, tools, catalog, deadline, trace):
    done = set()
    for _ in range(MAX_STEPS):
        if time.time() > deadline:
            trace.append("TIMEOUT")
            break
        out = llm(msgs, tools)
        trace.append(out)
        calls = parse_calls(out)
        if not calls:
            break
        msgs.append({"role": "assistant", "content": "", "tool_calls": [fn_call(c) for c in calls]})
        stop = False
        for c in calls:
            key = json.dumps(c, sort_keys=True, ensure_ascii=False)
            v = env.validate(c["name"], c["arguments"])  # not logged: invalid calls never reach the submission
            if not v.get("ok"):
                r = {"ok": False, "error": v.get("error"), "note": "Call chưa được thực thi, hãy sửa tham số."}
            elif key in done and catalog[c["name"]]["kind"] == "read":  # repeated writes can be legit (redeem twice)
                r = {"ok": False, "error": "Đã gọi tool đọc này với cùng tham số, dùng lại kết quả trước."}
            else:
                r = env.call(c["name"], c["arguments"])
                done.add(key)
                stop |= c["name"] == "sys.ask_missing_info"
            msgs.append({"role": "tool", "content": clip(r)})
        if stop:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder holding the split folders")
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--out", default="submission.jsonl")
    ap.add_argument("--llm", default=LLM_MODEL, help='HF model id, or "gold" (train only)')
    ap.add_argument("--asr", default=ASR_MODEL)
    ap.add_argument("--text", action="store_true", help="use audio_text instead of ASR (train only)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    try:
        import numpy as np, torch
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
    except ImportError:
        pass

    root = Path(a.root)
    sys.path.insert(0, str(root / a.split))
    from zaic_tools import ToolEnv

    catalog = {t["name"]: t for app in json.load(open(root / a.split / f"{a.split}_tools_catalog.json", encoding="utf-8"))["apps"]
               for t in app["tools"]}
    items = [json.loads(l) for l in open(root / a.split / f"{a.split}.jsonl", encoding="utf-8") if l.strip()]
    if a.limit:
        items = items[:a.limit]

    llm = GoldLLM() if a.llm == "gold" else HFLLM(a.llm)
    asr = None if a.text else ASR(a.asr)

    with open(a.out, "w", encoding="utf-8") as fout, open(a.out + ".trace.jsonl", "w", encoding="utf-8") as ftr:
        for i, item in enumerate(items):
            t0 = time.time()
            env = ToolEnv(a.split, item["id"], root=str(root))  # exactly one env per sample
            turns = [h for h in item["history"] if h["role"] == "user"]
            if a.text:
                texts = [h["audio_text"] for h in turns] + [item["audio_text"]]
            else:
                texts = asr([root / a.split / h["audio"] for h in turns] + [root / a.split / item["audio"]])
            tools = tool_specs(item, catalog)
            n_hist = len(turns)
            msgs = build_messages(item, texts, n_hist)
            while n_hist and llm.n_tokens(msgs, tools) > MAX_CTX - MAX_NEW * 2:
                n_hist -= 1
                msgs = build_messages(item, texts, n_hist)
            if isinstance(llm, GoldLLM):
                llm.reset(item)
            trace = []
            try:
                run_agent(env, llm, msgs, tools, catalog, t0 + BUDGET_S, trace)
            except Exception as e:  # keep whatever was already executed; a crash would score 0 anyway
                trace.append(f"ERROR {type(e).__name__}: {e}")
            calls = env.get_tool_calls()
            fout.write(json.dumps({"id": item["id"], "tool_calls": calls, "state_hash": env.get_state_hash()}, ensure_ascii=False) + "\n")
            ftr.write(json.dumps({"id": item["id"], "texts": texts, "kept_history": n_hist, "outputs": trace,
                                  "seconds": round(time.time() - t0, 2)}, ensure_ascii=False) + "\n")
            print(f"[{i + 1}/{len(items)}] {item['id']} calls={len(calls)} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
