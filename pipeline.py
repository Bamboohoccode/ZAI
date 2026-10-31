"""LifeMate baseline: ChunkFormer-CTC-large-vie -> Qwen3-4B tool-calling agent -> zaic_tools -> submission.jsonl.

pip install -U torch transformers accelerate chunkformer

python pipeline.py --root DATA --split public_test                        # full pipeline, 1 GPU (judges: 1x A30)
python pipeline.py --root DATA --split public_test --llm-device cuda:1    # Kaggle 2x T4: ASR on cuda:0, LLM on cuda:1
python pipeline.py --root DATA --split train --text --limit 100           # skip ASR, use audio_text
python pipeline.py --root DATA --split public_test --asr ./asr_ft --llm ./llm_ft    # fine-tuned full models (local folders)
python pipeline.py --root DATA --split public_test --asr-ckpt ./chunkformer_ft/final.pt          # fine-tuned ChunkFormer weights
python pipeline.py --root DATA --split public_test --llm-lora ./qwen_lora            # LoRA adapter on top of --llm
python pipeline.py --root DATA --split train --text --llm gold --limit 50  # replay gold, no GPU (self-check)
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
import torch

ASR_MODEL = "khanhld/chunkformer-ctc-large-vie"  # 110M params, CTC
LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MAX_CTX = 16384
MAX_NEW = 512
MAX_STEPS = 8
TIME_LIMIT_S = 30  # official per-sample limit; only reported, not enforced

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
- Lời người dùng là bản nhận dạng giọng nói: viết thường, không dấu câu, số đọc bằng chữ ("năm mươi ngàn" = 50000, "chín giờ" = 09:00), có thể sai chính tả, hãy hiểu theo ý gần nhất.
- Xong việc thì trả lời một câu ngắn, không gọi thêm tool."""


def gpu_dtype(device):
    import torch
    # T4/V100 have no native bf16: emulated bf16 falls back to O(L^2)-memory attention -> OOM on long prompts
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


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
    def __init__(self, name, device, lora=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        if torch.cuda.get_device_capability(device)[0] < 8:
            # T4/V100: SDPA's enable_gqa path has no memory-efficient kernel, falls back to O(L^2) math attention -> OOM.
            # Expanding KV heads instead lets SDPA use the memory-efficient kernel.
            import transformers.integrations.sdpa_attention as sdpa
            sdpa.use_gqa_in_sdpa = lambda *args, **kwargs: False
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=gpu_dtype(device), attn_implementation="sdpa", device_map=device)
        if lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora).merge_and_unload()  # merged: no adapter overhead per token
        self.model.eval()

    def n_tokens(self, msgs, tools):
        return len(self.tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=True))

    def __call__(self, msgs, tools):
        import torch
        enc = self.tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=False).replace("<|im_end|>", "").strip()


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
    def __init__(self, name, device, ckpt=None):
        from chunkformer import ChunkFormerModel
        self.model = ChunkFormerModel.from_pretrained(name)  # architecture, vocab and CMVN come from `name`
        if ckpt:  # fine-tuned weights (.pt from ChunkFormer training) on top of `name`
            sd = torch.load(ckpt, map_location="cpu")
            sd = sd.get("model", sd.get("state_dict", sd))
            # strict: chunkformer's own load_checkpoint uses strict=False and silently keeps base weights on key mismatch
            self.model.model.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()}, strict=True)
        self.model = self.model.to(device).eval()

    def __call__(self, paths):
        with torch.no_grad():  # CTC greedy decoding: deterministic
            return [t.strip() for t in self.model.batch_decode(audio_paths=[str(p) for p in paths], chunk_size=64,
                                                              left_context_size=128, right_context_size=128,
                                                              total_batch_duration=1800)]


def run_agent(env, llm, msgs, tools, catalog, trace):
    done = set()
    for _ in range(MAX_STEPS):
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
            # Wrong schema(Check if tool is existed and belongs installed apps and also check parameters)
            if not v.get("ok"):
                r = {"ok": False, "error": v.get("error"), "note": "Call chưa được thực thi, hãy sửa tham số."}
            # If the tool was called.
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
    ap.add_argument("--llm", default=LLM_MODEL, help='HF model id or local fine-tuned folder, or "gold" (train only)')
    ap.add_argument("--llm-lora", help="LoRA adapter folder, merged into --llm at load time")
    ap.add_argument("--asr", default=ASR_MODEL, help="ChunkFormer: HF id or local folder (config.yaml + pytorch_model.bin + vocab.txt + global_cmvn)")
    ap.add_argument("--asr-ckpt", help="fine-tuned ChunkFormer checkpoint (.pt) loaded onto --asr")
    ap.add_argument("--asr-device", default="cuda:0")
    ap.add_argument("--llm-device", default="cuda:0", help="judges run on 1x A30: keep both on cuda:0 for the final run")
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

    # Tool name VD: {smart_home.get_status : {} <- Full dictionary of that app}
    catalog = {t["name"]: t for app in json.load(open(root / a.split / f"{a.split}_tools_catalog.json", encoding="utf-8"))["apps"]
               for t in app["tools"]}
    items = [json.loads(l) for l in open(root / a.split / f"{a.split}.jsonl", encoding="utf-8") if l.strip()]
    if a.limit:
        items = items[:a.limit]

    # Without GPU if a.llm = gold
    llm = GoldLLM() if a.llm == "gold" else HFLLM(a.llm, a.llm_device, a.llm_lora)
    # You when we train only LLM :v 
    asr = None if a.text else ASR(a.asr, a.asr_device, a.asr_ckpt)

    slow = []
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
            # Cut off history if prompt reach 16k context length.
            while n_hist and llm.n_tokens(msgs, tools) > MAX_CTX - MAX_NEW * 2:
                n_hist -= 1
                msgs = build_messages(item, texts, n_hist)
            if isinstance(llm, GoldLLM):
                llm.reset(item)
            trace = []
            while True:
                try:
                    run_agent(env, llm, msgs, tools, catalog, trace)
                    break
                except Exception as e:  # keep whatever was already executed; a crash would score 0 anyway
                    trace.append(f"ERROR {type(e).__name__}: {str(e)[:200]}")
                    if "OutOfMemory" not in type(e).__name__:
                        break
                    import torch
                    with torch.cuda.device(a.llm_device):  # empty_cache only frees the current device
                        torch.cuda.empty_cache()
                    # retry with half the history, only while nothing has run on the env yet
                    if env.get_tool_calls() or not n_hist:
                        break
                    n_hist //= 2
                    msgs = build_messages(item, texts, n_hist)
                    trace.append(f"RETRY kept_history={n_hist}")
            calls = env.get_tool_calls()
            dt = time.time() - t0
            if dt > TIME_LIMIT_S:
                slow.append((item["id"], round(dt, 1)))
            # The submission and log for debugging later.
            fout.write(json.dumps({"id": item["id"], "tool_calls": calls, "state_hash": env.get_state_hash()}, ensure_ascii=False) + "\n")
            ftr.write(json.dumps({"id": item["id"], "texts": texts, "kept_history": n_hist, "outputs": trace,
                                  "seconds": round(dt, 2)}, ensure_ascii=False) + "\n")
            err = next((o[:120] for o in trace if str(o).startswith("ERROR")), "")
            print(f"[{i + 1}/{len(items)}] {item['id']} calls={len(calls)} {dt:.1f}s {err}", flush=True)

    print(f"\n{len(items)} samples -> {a.out} | timeout (> {TIME_LIMIT_S}s): {len(slow)}")
    for sid, dt in slow:
        print(f"  {sid}: {dt}s")


if __name__ == "__main__":
    main()

