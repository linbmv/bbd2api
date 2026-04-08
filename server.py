import json
import os
import re
import time
import hashlib
import requests
from flask import Flask, request, Response, jsonify
from threading import Lock

# ===================== 配置 =====================
DEBUG_MODE = os.getenv("DEBUG", "1") == "1"
UPSTREAM_URL = os.getenv("BBD_UPSTREAM", "https://app.backboard.io/api")
# 多 key：BBD_API_KEY 支持逗号分隔，自动轮换 + 故障转移
# 例：BBD_API_KEY=key1,key2,key3
_raw_keys = os.getenv("BBD_API_KEY", "")
API_KEYS: list[str] = [k.strip() for k in _raw_keys.split(",") if k.strip()]
API_KEY = API_KEYS[0] if API_KEYS else ""  # 兼容旧引用

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "10088"))
THREAD_TTL = int(os.getenv("THREAD_TTL", "1800"))  # 秒，默认30分钟

app = Flask(__name__)
lock = Lock()

# ===================== Key 轮换 =====================
_key_index = 0
_key_lock = Lock()
_key_failures: dict[str, int] = {}   # key -> 连续失败次数
_KEY_FAIL_THRESHOLD = 3              # 超过此次数暂时跳过该 key

def _next_key() -> str:
    """Round-robin 取下一个可用 key，失败次数过多的暂时跳过"""
    global _key_index
    with _key_lock:
        n = len(API_KEYS)
        for _ in range(n):
            key = API_KEYS[_key_index % n]
            _key_index += 1
            if _key_failures.get(key, 0) < _KEY_FAIL_THRESHOLD:
                return key
        # 全部 key 都超限了，重置并返回第一个
        _key_failures.clear()
        log("⚠️ 所有 key 均触发失败阈值，已重置计数", "WARNING")
        return API_KEYS[0]

def _mark_key_ok(key: str):
    with _key_lock:
        _key_failures[key] = 0

def _mark_key_fail(key: str):
    with _key_lock:
        _key_failures[key] = _key_failures.get(key, 0) + 1
        log(f"🔑 key ...{key[-8:]} 失败 {_key_failures[key]} 次", "WARNING")

def headers_for(key: str) -> dict:
    return {"X-API-Key": key, "Content-Type": "application/json"}

# assistant_id 按 key 维护：避免不同账号/空间的 key 混用导致 404
assistant_map: dict[str, str] = {}  # api_key -> assistant_id

GLOBAL_AID: str = ""  # 兼容旧调试字段：保存第一个初始化成功的 assistant_id

# cache_key -> {thread_id, api_key, last_used}
thread_cache: dict[tuple, dict] = {}
# thread_id -> {api_key, assistant_id, created_at, last_used}
thread_meta: dict[str, dict] = {}

# 预热 thread 池：按 key 分桶，保证 thread 与 key 一致
THREAD_POOL_SIZE = 4
_thread_pool: dict[str, list[str]] = {k: [] for k in API_KEYS}  # api_key -> [thread_id,...]
_pool_lock = Lock()

# tool_use_id → thread_id 映射：用于 tool_result 续接同一 thread
_tool_tid_map: dict[str, str] = {}
_tool_tid_lock = Lock()


def _key_suffix(key: str) -> str:
    return (key or "")[-8:]


def _ensure_assistant_for_key(key: str) -> str:
    """确保指定 key 对应的 assistant 已创建并返回 assistant_id。"""
    global GLOBAL_AID

    with lock:
        aid = assistant_map.get(key)
        if aid:
            return aid

    resp = requests.post(
        f"{UPSTREAM_URL}/assistants",
        headers=headers_for(key),
        json={"name": "proxy-global", "system_prompt": "You are a helpful assistant."},
        timeout=15,
    )
    resp.raise_for_status()
    aid = resp.json()["assistant_id"]
    _mark_key_ok(key)

    with lock:
        assistant_map[key] = aid
        if not GLOBAL_AID:
            GLOBAL_AID = aid

    return aid


def _create_thread_for_key(key: str) -> str:
    """用指定 key 创建 thread，并记录其归属。"""
    aid = _ensure_assistant_for_key(key)
    resp = requests.post(
        f"{UPSTREAM_URL}/assistants/{aid}/threads",
        headers=headers_for(key),
        timeout=15,
    )
    resp.raise_for_status()
    tid = resp.json()["thread_id"]
    _mark_key_ok(key)

    with lock:
        thread_meta[tid] = {
            "api_key": key,
            "assistant_id": aid,
            "created_at": time.time(),
            "last_used": time.time(),
        }

    return tid


def _get_thread_key(tid: str) -> str | None:
    with lock:
        meta = thread_meta.get(tid)
        if meta:
            meta["last_used"] = time.time()
            return meta.get("api_key")
    return None


def _mark_thread_used(tid: str):
    with lock:
        meta = thread_meta.get(tid)
        if meta:
            meta["last_used"] = time.time()


def _remember_thread_cache(cache_key: tuple, tid: str, key: str):
    now = time.time()
    with lock:
        thread_cache[cache_key] = {
            "thread_id": tid,
            "api_key": key,
            "last_used": now,
        }
        meta = thread_meta.get(tid)
        if meta:
            meta["last_used"] = now
        else:
            thread_meta[tid] = {
                "api_key": key,
                "assistant_id": assistant_map.get(key, ""),
                "created_at": now,
                "last_used": now,
            }


def _pool_size_total() -> int:
    with _pool_lock:
        return sum(len(v) for v in _thread_pool.values())


def _pool_size_by_key() -> dict:
    with _pool_lock:
        return {f"...{_key_suffix(k)}": len(v) for k, v in _thread_pool.items()}


def _assistants_debug() -> dict:
    with lock:
        return {f"...{_key_suffix(k)}": aid for k, aid in assistant_map.items()}


def _thread_debug_summary() -> dict:
    return {
        "global_assistant": GLOBAL_AID,
        "assistants": _assistants_debug(),
        "threads_cached": len(thread_cache),
        "threads_registered": len(thread_meta),
        "pool_size": _pool_size_total(),
        "pool_size_by_key": _pool_size_by_key(),
    }


def _maybe_log_tool_parse(raw: str, parsed: dict):
    if DEBUG_MODE:
        preview = (raw or "")[:200].replace("\n", " ")
        log(f"🔍 工具解析={parsed.get('type')} raw={preview}", "DEBUG")


def _safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "ignore").decode("ascii"))


def _safe_startup_banner():
    _safe_print("\n" + "="*60)
    _safe_print("🚀 ccc.py 启动")
    _safe_print(f"📍 端口: {PORT}  上游: {UPSTREAM_URL}")
    _safe_print(f"⏱️  Thread TTL: {THREAD_TTL}s")
    _safe_print("="*60 + "\n")


def _debug_after_tool_register():
    if DEBUG_MODE:
        with _tool_tid_lock:
            n = len(_tool_tid_map)
        log(f"🧰 tool_mappings={n}", "DEBUG")


def _debug_after_request():
    if DEBUG_MODE:
        log(f"📊 {_thread_debug_summary()}", "DEBUG")


def _debug_after_tool_result_thread(tid: str):
    if DEBUG_MODE and tid:
        log(f"🧰 tool_result tid={tid} key...{_key_suffix(_get_thread_key(tid) or '')}", "DEBUG")


def _debug_warmup_state():
    if DEBUG_MODE:
        log(f"🤖 assistants={_assistants_debug()}", "DEBUG")
        log(f"🏊 pool={_pool_size_by_key()}", "DEBUG")


def _request_key_for_thread(messages: list, tid: str, tool_result: bool = False) -> str:
    """为已存在 thread 确定其绑定 key。tool_result 场景需确保续接同一 key。"""
    key = _get_thread_key(tid)
    if key:
        return key

    # fallback：从缓存里尝试找回
    history = messages[:-1]
    hh = history_hash(history) if history else "__empty__"
    entry = thread_cache.get((hh,))
    key = entry.get("api_key") if entry else None
    if key:
        return key

    raise RuntimeError(f"无法找到 thread 对应的 API key: {tid}")


def _init_pool_buckets():
    with _pool_lock:
        for key in API_KEYS:
            _thread_pool.setdefault(key, [])


def _pool_fill_for_key(key: str, count: int):
    tids = []
    for _ in range(count):
        try:
            tids.append(_create_thread_for_key(key))
        except Exception as e:
            _mark_key_fail(key)
            log(f"⚠️ 预热线程失败 key...{_key_suffix(key)}: {e}", "WARNING")
            break
    with _pool_lock:
        _thread_pool.setdefault(key, []).extend(tids)


def _warmup_all_keys():
    _init_pool_buckets()
    for key in API_KEYS:
        try:
            aid = _ensure_assistant_for_key(key)
            log(f"✅ assistant 就绪 key...{_key_suffix(key)} aid={aid}", "SUCCESS")
            _pool_fill_for_key(key, THREAD_POOL_SIZE)
        except Exception as e:
            _mark_key_fail(key)
            log(f"⚠️ 预热失败 key...{_key_suffix(key)}: {e}", "WARNING")

    _debug_warmup_state()


# =====================================================================



# ===================== 工具函数 =====================
def log(msg, level="INFO"):
    if not DEBUG_MODE and level == "DEBUG":
        return
    color = {"INFO": "\033[94m", "DEBUG": "\033[92m", "WARNING": "\033[93m",
             "ERROR": "\033[91m", "SUCCESS": "\033[92m"}.get(level, "")
    print(f"{color}[{time.strftime('%H:%M:%S')}] {msg}\033[0m")

def extract_text(content) -> str:
    """从 content 中提取纯文本，支持字符串和列表"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "image":
                    parts.append("[image]")
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content) if content else ""

# 剥离 <system-reminder>...</system-reminder> 等动态注入标签，只保留稳定部分用于 hash
_DYNAMIC_TAG_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")

def stable_system_text(system) -> str:
    """去掉动态标签，返回稳定文本用于 hash"""
    text = extract_text(system) if system else ""
    text = _DYNAMIC_TAG_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", text).strip()

def sp_hash(system) -> str:
    return hashlib.md5(stable_system_text(system).encode()).hexdigest()[:12]

def history_hash(messages: list) -> str:
    key = json.dumps([
        {"role": m.get("role"), "content": extract_text(m.get("content", ""))}
        for m in messages
    ], ensure_ascii=False)
    return hashlib.md5(key.encode()).hexdigest()[:16]

def _create_thread_bg(key: str):
    """后台为指定 key 创建一个 thread 放入池"""
    import threading

    def _do():
        try:
            tid = _create_thread_for_key(key)
            with _pool_lock:
                _thread_pool.setdefault(key, []).append(tid)
                size = len(_thread_pool.get(key, []))
            log(f"🏊 pool+1 key...{_key_suffix(key)} tid={tid} size={size}", "DEBUG")
        except Exception as e:
            _mark_key_fail(key)
            log(f"⚠️ pool 补充失败 key...{_key_suffix(key)}: {e}", "WARNING")

    threading.Thread(target=_do, daemon=True).start()


def _get_pooled_thread(key: str) -> str | None:
    """从指定 key 的池里取一个空闲 thread，取出后立即异步补充"""
    with _pool_lock:
        bucket = _thread_pool.get(key) or []
        tid = bucket.pop(0) if bucket else None

    if tid:
        remain = len(_thread_pool.get(key, []))
        log(f"🏊 pool取出 key...{_key_suffix(key)} tid={tid} 剩余={remain}", "DEBUG")
        _create_thread_bg(key)
        return tid
    return None

def _register_tool_ids(tool_use_ids: list, tid: str):
    """记录 tool_use_id → thread_id，用于 tool_result 续接"""
    with _tool_tid_lock:
        for uid in tool_use_ids:
            _tool_tid_map[uid] = tid
        if len(_tool_tid_map) > 2000:
            keys = list(_tool_tid_map.keys())
            for k in keys[:1000]:
                del _tool_tid_map[k]

def _find_tool_result_thread(messages: list) -> str | None:
    """若最后一条消息是 tool_result（Claude 或 OpenAI 格式），返回对应 thread_id"""
    if not messages:
        return None
    last = messages[-1]
    role = last.get("role", "")

    # OpenAI 格式：role="tool", tool_call_id=...
    if role == "tool":
        uid = last.get("tool_call_id", "")
        if uid:
            with _tool_tid_lock:
                tid = _tool_tid_map.get(uid)
            if tid:
                log(f"🔗 tool_result(oai) → thread {tid} (id={uid[:16]})", "DEBUG")
                return tid

    # Claude 格式：role="user", content=[{type:"tool_result", tool_use_id:...}]
    if role == "user":
        content = last.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    uid = block.get("tool_use_id", "")
                    with _tool_tid_lock:
                        tid = _tool_tid_map.get(uid)
                    if tid:
                        log(f"🔗 tool_result → thread {tid} (id={uid[:16]})", "DEBUG")
                        return tid
    return None

def _format_tool_results(messages: list) -> str:
    """将 tool_result 消息格式化为模型能理解的文本（支持 Claude/OpenAI 格式）"""
    last = messages[-1]
    parts = []

    # OpenAI 格式：role="tool"
    if last.get("role") == "tool":
        parts.append(str(last.get("content", "")))
    else:
        # Claude 格式：content=[{type:"tool_result",...}]
        for block in (last.get("content") or []):
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            result = block.get("content", "")
            if isinstance(result, list):
                result = "".join(
                    b.get("text", "") for b in result
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            parts.append(str(result))

    combined = "\n---\n".join(parts) if parts else "[无结果]"
    return f"工具执行结果：\n{combined}\n\n请根据以上结果继续。"

def evict_expired_threads():
    now = time.time()
    expired = [k for k, v in thread_cache.items() if now - v.get("last_used", 0) > THREAD_TTL]
    for k in expired:
        del thread_cache[k]

    expired_tids = [tid for tid, v in thread_meta.items() if now - v.get("last_used", 0) > THREAD_TTL]
    for tid in expired_tids:
        del thread_meta[tid]

# ===================== 线程管理（使用全局 assistant）=====================
def get_or_create_thread(messages: list, system_text: str = "") -> tuple[str, str]:
    """
    返回 (thread_id, message_to_send)。

    关键语义：
    - 新会话：round-robin 选择一个 key，并从该 key 的 thread 池取用/创建 thread
    - 复用会话：沿用缓存中的 thread 与其绑定 key
    """
    if not messages:
        raise ValueError("no messages")

    last_msg = messages[-1]
    history = messages[:-1]
    hh = history_hash(history) if history else "__empty__"
    cache_key = (hh,)

    user_text = extract_text(last_msg.get("content", ""))
    if system_text and not history:
        user_text = f"<system>\n{system_text}\n</system>\n\n{user_text}"

    with lock:
        evict_expired_threads()
        entry = thread_cache.get(cache_key)
        if entry and time.time() - entry.get("last_used", 0) <= THREAD_TTL:
            tid = entry["thread_id"]
            key = entry.get("api_key", "")
            entry["last_used"] = time.time()
            _mark_thread_used(tid)
            log(f"✅ 复用 thread: {tid} key...{_key_suffix(key)}", "DEBUG")
            _remember_thread_cache((history_hash(messages),), tid, key)
            return tid, user_text

    # 新会话：允许在多个 key 之间重试，避免单 key 故障导致无法创建
    last_err = None
    for _ in range(max(1, len(API_KEYS))):
        key = _next_key()
        try:
            _ensure_assistant_for_key(key)
            pooled = _get_pooled_thread(key)
            if pooled:
                tid = pooled
                log(f"🏊 复用池 thread: {tid} key...{_key_suffix(key)}", "INFO")
            else:
                log(f"🆕 新建 thread（池已空）key...{_key_suffix(key)}...", "INFO")
                tid = _create_thread_for_key(key)
                log(f"✅ Thread: {tid}", "SUCCESS")

            _remember_thread_cache(cache_key, tid, key)
            _remember_thread_cache((history_hash(messages),), tid, key)
            return tid, user_text
        except Exception as e:
            last_err = e
            _mark_key_fail(key)
            log(f"⚠️ 创建 thread 失败 key...{_key_suffix(key)}: {e}", "WARNING")

    raise RuntimeError(f"无法创建 thread: {last_err}")


# ===================== Provider 路由 =====================
def resolve_provider(model: str) -> tuple[str, str]:
    if model.startswith("claude"):
        return "anthropic", model
    if model.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return "openai", model
    if model.startswith("gemini"):
        return "google", model
    if model.startswith("mistral") or model.startswith("mixtral"):
        return "mistral", model
    if model.startswith("llama"):
        return "meta", model
    return "openai", model


def build_payload(text: str, stream: bool, model: str, tools=None) -> dict:
    provider, model_name = resolve_provider(model)
    p = {
        "content": text,
        "stream": stream,
        "llm_provider": provider,
        "model_name": model_name,
        "memory": "off",
    }
    log(f"🔀 {provider}/{model_name}", "DEBUG")
    return p


# ===================== 工具调用：prompt 注入方案 =====================
# 极简格式指令，不做角色扮演，不注入完整 schema
# 只告知输出格式 + 工具名列表，模型自行推断参数
_TOOL_PROMPT_TMPL = (
    "可用工具：{tools}\n"
    "需调用工具时仅输出JSON（无其他文字）：\n"
    '  单工具：{{"tool":"名","args":{{"参":值}}}}\n'
    '  多工具：{{"calls":[{{"tool":"名","args":{{}}}},...]}}\n'
    "不调工具则直接回答。\n\n"
    "{query}"
)

# prompt cache：按 tools_hash 缓存编译好的工具列表字符串
_tool_prompt_cache: dict[str, str] = {}

def _tools_hash(tools: list) -> str:
    key = json.dumps([t.get("name") or t.get("function", {}).get("name", "") for t in tools])
    return hashlib.md5(key.encode()).hexdigest()[:12]

def _compact_tools(tools: list) -> str:
    """只输出：name(req_param*,opt_param?) 格式，极简"""
    parts = []
    for t in tools:
        if t.get("type") == "function":
            fn = t["function"]
            name = fn.get("name", "")
            props = fn.get("parameters", {}).get("properties", {})
            req = set(fn.get("parameters", {}).get("required", []))
        elif "input_schema" in t:
            name = t.get("name", "")
            props = t.get("input_schema", {}).get("properties", {})
            req = set(t.get("input_schema", {}).get("required", []))
        else:
            continue
        params = [f"{p}{'*' if p in req else '?'}" for p in props]
        parts.append(f"{name}({','.join(params)})" if params else name)
    return " | ".join(parts)

def inject_tool_prompt(user_text: str, tools: list) -> str:
    """注入极简工具格式指令；用 cache 避免重复编译"""
    th = _tools_hash(tools)
    if th not in _tool_prompt_cache:
        _tool_prompt_cache[th] = _compact_tools(tools)
    tools_str = _tool_prompt_cache[th]
    result = _TOOL_PROMPT_TMPL.format(tools=tools_str, query=user_text)
    log(f"🔧 tool prompt {len(result)} chars (tools cached={th})", "DEBUG")
    return result


_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)

def parse_tool_response(raw: str) -> dict:
    """
    解析 LLM 输出，支持多种格式：
      {"tool":"name","args":{}}          单工具
      {"calls":[{"tool":"name","args":{}}]}   多工具
      {"type":"tool_calls","calls":[...]}     旧格式兼容
    """
    m = _JSON_RE.search(raw)
    if not m:
        return {"type": "text", "text": raw}
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return {"type": "text", "text": raw}

    # 单工具格式
    if "tool" in obj and "args" in obj:
        return {"type": "tool_calls", "calls": [{"name": obj["tool"], "input": obj["args"]}]}

    # 多工具格式（新/旧）
    calls_raw = obj.get("calls", [])
    if calls_raw:
        calls = []
        for c in calls_raw:
            name = c.get("tool") or c.get("name", "")
            inp = c.get("args") or c.get("parameters") or c.get("input") or {}
            calls.append({"name": name, "input": inp})
        return {"type": "tool_calls", "calls": calls}

    # 旧 type=text 格式
    if obj.get("type") == "text":
        return {"type": "text", "text": obj.get("content", raw)}

    return {"type": "text", "text": raw}


def tool_calls_to_claude_content(calls: list) -> list:
    """转为 Claude tool_use content blocks"""
    blocks = []
    for i, c in enumerate(calls):
        blocks.append({
            "type": "tool_use",
            "id": f"toolu_{int(time.time()*1000)}_{i}",
            "name": c["name"],
            "input": c["input"],
        })
    return blocks
    # 将工具调用格式化为 Claude 风格，让客户端执行（这里直接 pass-through）
    # 实际上我们无法在服务端执行客户端的工具，所以需要把 tool_use 事件透传给客户端
    # 这个函数保留用于将来服务端工具执行
    pass


# ===================== 模型列表 =====================
_models_cache: dict = {}
_models_cache_ts: float = 0
MODELS_CACHE_TTL = 3600

@app.route("/v1/models", methods=["GET"])
def list_models():
    global _models_cache, _models_cache_ts
    now = time.time()
    if _models_cache and now - _models_cache_ts < MODELS_CACHE_TTL:
        return jsonify(_models_cache)
    try:
        r = requests.get(
            f"{UPSTREAM_URL}/models",
            headers=headers_for(_next_key()),
            params={"model_type": "llm", "limit": 500},
            timeout=15,
        )
        r.raise_for_status()
        upstream = r.json()
        data = [
            {"id": m["name"], "object": "model", "created": 0,
             "owned_by": m.get("provider", "unknown"),
             "context_window": m.get("context_limit", 0),
             "supports_tools": m.get("supports_tools", False)}
            for m in upstream.get("models", [])
        ]
        result = {"object": "list", "data": data}
        _models_cache = result
        _models_cache_ts = now
        return jsonify(result)
    except Exception as e:
        log(f"⚠️ 拉取模型列表失败: {e}", "WARNING")
        fallback = ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001",
                    "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
                    "gpt-4o", "gpt-4o-mini", "gemini-2.0-flash"]
        return jsonify({"object": "list", "data": [
            {"id": m, "object": "model", "created": 0, "owned_by": "backboard"} for m in fallback
        ]})

# ===================== 调试接口 =====================
@app.route("/debug/clear", methods=["POST"])
def debug_clear():
    with lock:
        thread_cache.clear()
    log("🧹 已清除 thread 缓存", "WARNING")
    return jsonify({"status": "ok"})

@app.route("/debug/state", methods=["GET"])
def debug_state():
    with lock:
        return jsonify(_thread_debug_summary())

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "global_assistant": bool(GLOBAL_AID),
                    "threads_cached": len(thread_cache), "pool_size": _pool_size_total()})


# ===================== 流式 SSE 生成器 =====================
def _claude_sse(event_type: str, payload: dict | None = None) -> str:
    # Claude-style SSE clients dispatch on the event name, not just the JSON body.
    body = payload if payload is not None else {"type": event_type}
    return f"event: {event_type}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"


def _claude_message_start(msg_id: str, model: str, input_tokens: int = 0) -> str:
    return _claude_sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    })


def _claude_content_block_start(index: int, content_block: dict) -> str:
    return _claude_sse("content_block_start", {
        "type": "content_block_start",
        "index": index,
        "content_block": content_block,
    })


def _claude_content_block_stop(index: int) -> str:
    return _claude_sse("content_block_stop", {
        "type": "content_block_stop",
        "index": index,
    })


def _claude_text_delta(index: int, text: str) -> str:
    return _claude_sse("content_block_delta", {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    })


def _claude_input_json_delta(index: int, partial_json: str) -> str:
    return _claude_sse("content_block_delta", {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "input_json_delta", "partial_json": partial_json},
    })


def _claude_message_delta(stop_reason: str, output_tokens: int) -> str:
    return _claude_sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })


def _claude_message_stop() -> str:
    return _claude_sse("message_stop", {"type": "message_stop"})


def _claude_ping() -> str:
    return _claude_sse("ping", {"type": "ping"})


def stream_claude(r, model: str, tid: str):
    """
    把上游 SSE 转为 Claude Messages API SSE 格式。
    支持工具调用：检测到 tool_use 时发出 tool_use content block。
    """
    msg_id = f"msg_{int(time.time()*1000)}"
    input_tokens = 0
    output_tokens = 0
    pending_tool: dict = {}   # 当前累积的 tool_use block
    tool_index = 0
    in_tool = False
    run_id = None

    yield _claude_message_start(msg_id, model)
    yield _claude_content_block_start(0, {"type": "text", "text": ""})
    yield _claude_ping()

    text_block_open = True

    with r:
        if r.status_code >= 500:
            log(f"💥 上游500: {r.text[:100]}", "ERROR")
            yield _claude_text_delta(0, "[upstream error]")
        else:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    break
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                etype = obj.get("type", "")

                if etype == "run_started":
                    run_id = obj.get("run_id")

                elif etype == "content_streaming":
                    text = obj.get("content", "")
                    if text:
                        if not text_block_open:
                            # 之前关闭了 text block（比如进入 tool），重新开一个
                            yield _claude_content_block_start(tool_index + 1, {"type": "text", "text": ""})
                            text_block_open = True
                        yield _claude_text_delta(0 if not in_tool else tool_index + 1, text)

                elif etype == "tool_call_start":
                    # 关闭当前文本 block
                    if text_block_open:
                        yield _claude_content_block_stop(0)
                        text_block_open = False

                    tool_index = len(pending_tool) + 1
                    tc = obj.get("tool_call", {})
                    pending_tool = {
                        "id": tc.get("id", f"toolu_{int(time.time()*1000)}"),
                        "name": tc.get("function", {}).get("name", ""),
                        "input_buf": "",
                    }
                    in_tool = True
                    yield _claude_content_block_start(tool_index, {"type": "tool_use", "id": pending_tool["id"], "name": pending_tool["name"], "input": {}})

                elif etype == "tool_call_delta":
                    if in_tool and pending_tool:
                        delta_args = obj.get("tool_call", {}).get("function", {}).get("arguments", "")
                        pending_tool["input_buf"] += delta_args
                        yield _claude_input_json_delta(tool_index, delta_args)

                elif etype == "tool_call_end":
                    if in_tool:
                        yield _claude_content_block_stop(tool_index)
                        in_tool = False

                elif etype == "run_ended":
                    input_tokens = obj.get("input_tokens", 0)
                    output_tokens = obj.get("output_tokens", 0)
                    break

    # 关闭尚未关闭的 block
    if text_block_open:
        yield _claude_content_block_stop(0)

    stop_reason = "tool_use" if pending_tool and not in_tool else "end_turn"
    yield _claude_message_delta(stop_reason, output_tokens)
    yield _claude_message_stop()


def stream_openai(r, model: str, cid: str):
    """把上游 SSE 转为 OpenAI chat completions SSE 格式，支持 function calling"""
    in_tool = False
    tool_index = -1
    pending_name = ""

    with r:
        if r.status_code >= 500:
            yield 'data: [DONE]\n\n'
            return
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            raw = line[6:]
            if raw == "[DONE]":
                break
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue

            etype = obj.get("type", "")

            if etype == "content_streaming":
                text = obj.get("content", "")
                if text:
                    yield f'data: {json.dumps({"id": cid, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]})}\n\n'

            elif etype == "tool_call_start":
                tc = obj.get("tool_call", {})
                tool_index += 1
                pending_name = tc.get("function", {}).get("name", "")
                in_tool = True
                yield f'data: {json.dumps({"id": cid, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": tool_index, "id": tc.get("id", f"call_{tool_index}"), "type": "function", "function": {"name": pending_name, "arguments": ""}}]}, "finish_reason": None}]})}\n\n'

            elif etype == "tool_call_delta":
                if in_tool:
                    args = obj.get("tool_call", {}).get("function", {}).get("arguments", "")
                    yield f'data: {json.dumps({"id": cid, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": tool_index, "function": {"arguments": args}}]}, "finish_reason": None}]})}\n\n'

            elif etype == "tool_call_end":
                in_tool = False

            elif etype == "run_ended":
                break

    finish = "tool_calls" if tool_index >= 0 else "stop"
    yield f'data: {json.dumps({"id": cid, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})}\n\n'
    yield 'data: [DONE]\n\n'


# ===================== 核心接口 =====================
def _do_request(tid: str, api_key: str, text: str, stream: bool, model: str,
                client_tools: list) -> tuple[str, any]:
    """
    发消息到上游。

    关键语义：对已存在 thread，必须使用其绑定的 api_key；不能再做跨 key 轮换，
    否则在多账号/空间 key 混用时会导致 404。

    有工具时强制用流式（避免大 prompt 非流式超时）。
    """
    if client_tools:
        text = inject_tool_prompt(text, client_tools)
        log(f"🔧 tools={len(client_tools)} 注入 prompt ({len(text)} chars)", "DEBUG")
        stream = True

    payload = build_payload(text, stream, model)
    url = f"{UPSTREAM_URL}/threads/{tid}/messages"

    try:
        if stream:
            resp = requests.post(url, headers=headers_for(api_key), json=payload,
                                 stream=True, timeout=120)
        else:
            resp = requests.post(url, headers=headers_for(api_key), json=payload, timeout=90)

        if resp.status_code in (401, 403, 404):
            _mark_key_fail(api_key)
        else:
            _mark_key_ok(api_key)

        return url, resp

    except requests.RequestException as e:
        _mark_key_fail(api_key)
        raise


def _sync_response(up: dict, model: str, client_tools: list, tid: str = "") -> dict:
    """把上游非流式响应转为 Claude Messages API 格式，处理工具调用"""
    raw = up.get("content", "")

    if client_tools:
        parsed = parse_tool_response(raw)
        _maybe_log_tool_parse(raw, parsed)
        if parsed["type"] == "tool_calls":
            blocks = tool_calls_to_claude_content(parsed["calls"])
            if tid:
                _register_tool_ids([b["id"] for b in blocks], tid)
                _debug_after_tool_register()
            log(f"🔧 工具调用: {[c['name'] for c in parsed['calls']]}", "INFO")
            return {
                "id": f"msg_{int(time.time()*1000)}",
                "type": "message", "role": "assistant",
                "content": blocks, "model": model,
                "stop_reason": "tool_use",
                "usage": {"input_tokens": up.get("input_tokens", 0),
                          "output_tokens": up.get("output_tokens", 0)},
            }
        raw = parsed.get("text", raw)

    return {
        "id": f"msg_{int(time.time()*1000)}",
        "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": raw}],
        "model": model, "stop_reason": "end_turn",
        "usage": {"input_tokens": up.get("input_tokens", 0),
                  "output_tokens": up.get("output_tokens", 0)},
    }


def stream_claude_with_tools(r, model: str, client_tools: bool, tid: str = ""):
    """
    有工具时：缓冲全部响应，解析 JSON，转为 SSE tool_use 事件。
    无工具时：直接流式透传。

    说明：是否真正触发工具调用取决于上游 LLM 是否按注入的 JSON 约定输出。
    为便于排障，DEBUG 模式下会记录解析命中情况与上游输出片段。
    """
    if not client_tools:
        yield from stream_claude(r, model, "")
        return

    # 缓冲收集完整响应文本
    msg_id = f"msg_{int(time.time()*1000)}"
    buf = []
    input_tokens = 0
    output_tokens = 0

    # Emit a valid Claude SSE prelude immediately so clients do not treat the
    # buffered tool path as an empty or stalled response.
    yield _claude_message_start(msg_id, model)
    yield _claude_ping()

    with r:
        if r.status_code >= 500:
            yield _claude_content_block_start(0, {"type": "text", "text": ""})
            yield _claude_text_delta(0, "[upstream error]")
            yield _claude_content_block_stop(0)
            yield _claude_message_delta("end_turn", 0)
            yield _claude_message_stop()
            return
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            raw = line[6:]
            if raw == "[DONE]":
                break
            try:
                obj = json.loads(raw)
                if obj.get("type") == "content_streaming":
                    buf.append(obj.get("content", ""))
                elif obj.get("type") == "run_ended":
                    input_tokens = obj.get("input_tokens", 0)
                    output_tokens = obj.get("output_tokens", 0)
                    break
            except Exception:
                pass

    full_text = "".join(buf)
    parsed = parse_tool_response(full_text)
    _maybe_log_tool_parse(full_text, parsed)

    # The Claude SSE prelude above already opened the stream; avoid emitting a
    # duplicate message_start after the buffered upstream parse completes.

    if parsed["type"] == "tool_calls":
        log(f"🔧 流式工具调用: {[c['name'] for c in parsed['calls']]}", "INFO")
        tool_ids = []
        for i, call in enumerate(parsed["calls"]):
            tid_val = f"toolu_{int(time.time()*1000)}_{i}"
            tool_ids.append(tid_val)
            inp_str = json.dumps(call["input"], ensure_ascii=False)
            yield _claude_content_block_start(i, {"type": "tool_use", "id": tid_val, "name": call["name"], "input": {}})
            yield _claude_input_json_delta(i, inp_str)
            yield _claude_content_block_stop(i)
        if tid and tool_ids:
            _register_tool_ids(tool_ids, tid)
        yield _claude_message_delta("tool_use", output_tokens)
    else:
        text = parsed.get("text", full_text)
        yield _claude_content_block_start(0, {"type": "text", "text": ""})
        yield _claude_text_delta(0, text)
        yield _claude_content_block_stop(0)
        yield _claude_message_delta("end_turn", output_tokens)

    yield _claude_message_stop()


def _buffer_upstream_stream(r) -> tuple[str, int, int]:
    """缓冲上游流式响应，返回 (full_text, input_tokens, output_tokens)"""
    buf = []
    input_tokens = output_tokens = 0
    with r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            raw = line[6:]
            if raw == "[DONE]":
                break
            try:
                obj = json.loads(raw)
                if obj.get("type") == "content_streaming":
                    buf.append(obj.get("content", ""))
                elif obj.get("type") == "run_ended":
                    input_tokens = obj.get("input_tokens", 0)
                    output_tokens = obj.get("output_tokens", 0)
                    break
            except Exception:
                pass
    return "".join(buf), input_tokens, output_tokens


@app.route("/v1/messages", methods=["POST"])
def claude_messages():
    data = request.json
    model = data.get("model", "claude-sonnet-4-6")
    messages = data.get("messages", [])
    system = data.get("system") or "You are a helpful assistant."
    stream = data.get("stream", False)
    client_tools = data.get("tools", [])

    log(f"📥 /v1/messages model={model} stream={stream} msgs={len(messages)} tools={len(client_tools)}", "INFO")
    if not messages:
        return jsonify({"error": "no messages"}), 400

    try:
        sys_text = stable_system_text(system)
        # 检测是否为 tool_result 续接轮次
        tool_result_tid = _find_tool_result_thread(messages)
        if tool_result_tid:
            tid = tool_result_tid
            text = _format_tool_results(messages)
            log(f"🔄 tool_result 续接 thread={tid}", "INFO")
        else:
            tid, text = get_or_create_thread(messages, sys_text)

        key = _request_key_for_thread(messages, tid, tool_result=bool(tool_result_tid))
        _, resp = _do_request(tid, key, text, stream, model, client_tools)
        # 有工具时上游强制流式；无工具时跟随客户端
        upstream_streaming = bool(client_tools) or stream

        if not stream:
            if upstream_streaming:
                # 工具场景：缓冲流式响应后返回同步
                if resp.status_code >= 500:
                    log(f"💥 上游500", "ERROR")
                    return jsonify({"error": "upstream error"}), 502
                full_text, in_tok, out_tok = _buffer_upstream_stream(resp)
                fake_up = {"content": full_text, "input_tokens": in_tok, "output_tokens": out_tok}
                return jsonify(_sync_response(fake_up, model, client_tools, tid))
            else:
                if resp.status_code >= 500:
                    log(f"💥 上游500: {resp.text[:200]}", "ERROR")
                    return jsonify({"error": "upstream error"}), 502
                resp.raise_for_status()
                return jsonify(_sync_response(resp.json(), model, client_tools, tid))

        return Response(stream_claude_with_tools(resp, model, bool(client_tools), tid),
                        mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    except Exception as e:
        log(f"❌ 接口异常: {e}", "ERROR")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 502


@app.route("/v1/chat/completions", methods=["POST"])
def openai_compat():
    data = request.json
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    model = data.get("model", "gpt-4o")
    client_tools = data.get("tools", [])
    system_msg = next((extract_text(m["content"]) for m in messages if m.get("role") == "system"),
                      "You are a helpful assistant.")
    non_system = [m for m in messages if m.get("role") != "system"]

    log(f"📥 /v1/chat/completions model={model} stream={stream} msgs={len(non_system)} tools={len(client_tools)}", "INFO")
    if not non_system:
        return jsonify({"error": "no messages"}), 400

    try:
        sys_text = stable_system_text(system_msg)
        tool_result_tid = _find_tool_result_thread(non_system)
        if tool_result_tid:
            tid = tool_result_tid
            text = _format_tool_results(non_system)
            log(f"🔄 tool_result 续接 thread={tid}", "INFO")
        else:
            tid, text = get_or_create_thread(non_system, sys_text)

        key = _request_key_for_thread(non_system, tid, tool_result=bool(tool_result_tid))
        _, resp = _do_request(tid, key, text, stream, model, client_tools)
        upstream_streaming = bool(client_tools) or stream

        if not stream:
            if upstream_streaming:
                if resp.status_code >= 500:
                    return jsonify({"error": "upstream error"}), 502
                full_text, in_tok, out_tok = _buffer_upstream_stream(resp)
            else:
                if resp.status_code >= 500:
                    return jsonify({"error": "upstream error"}), 502
                resp.raise_for_status()
                up = resp.json()
                full_text = up.get("content", "")
                in_tok, out_tok = up.get("input_tokens", 0), up.get("output_tokens", 0)

            if client_tools:
                parsed = parse_tool_response(full_text)
                if parsed["type"] == "tool_calls":
                    oai_tc = [
                        {"id": f"call_{i}", "type": "function",
                         "function": {"name": c["name"],
                                      "arguments": json.dumps(c["input"], ensure_ascii=False)}}
                        for i, c in enumerate(parsed["calls"])
                    ]
                    _register_tool_ids([tc["id"] for tc in oai_tc], tid)
                    return jsonify({
                        "id": f"chatcmpl-{int(time.time()*1000)}",
                        "object": "chat.completion",
                        "choices": [{"index": 0,
                                     "message": {"role": "assistant", "content": None,
                                                 "tool_calls": oai_tc},
                                     "finish_reason": "tool_calls"}],
                        "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok},
                    })
                full_text = parsed.get("text", full_text)

            return jsonify({
                "id": f"chatcmpl-{int(time.time()*1000)}",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok},
            })

        def gen_oai(r):
            cid = f"chatcmpl-{int(time.time()*1000)}"
            if not client_tools:
                yield from stream_openai(r, model, cid)
                return
            full_text2, _, _ = _buffer_upstream_stream(r)
            parsed2 = parse_tool_response(full_text2)
            if parsed2["type"] == "tool_calls":
                oai_tc2 = [
                    {"id": f"call_{i}", "type": "function",
                     "function": {"name": c["name"],
                                  "arguments": json.dumps(c["input"], ensure_ascii=False)}}
                    for i, c in enumerate(parsed2["calls"])
                ]
                _register_tool_ids([tc["id"] for tc in oai_tc2], tid)
                yield f'data: {json.dumps({"id": cid, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": oai_tc2}, "finish_reason": None}]})}\n\n'
                yield f'data: {json.dumps({"id": cid, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})}\n\n'
            else:
                t2 = parsed2.get("text", full_text2)
                yield f'data: {json.dumps({"id": cid, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": t2}, "finish_reason": None}]})}\n\n'
                yield f'data: {json.dumps({"id": cid, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})}\n\n'
            yield 'data: [DONE]\n\n'

        return Response(gen_oai(resp), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    except Exception as e:
        log(f"❌ 接口异常: {e}", "ERROR")
        return jsonify({"error": str(e)}), 502


# ===================== 预热 =====================
def warmup():
    """启动时为每个 key 创建 assistant，并按 key 填满 thread 池"""
    import threading

    def _do():
        try:
            log("🔥 预热...", "INFO")
            _warmup_all_keys()
            log(f"✅ 预热完成: pool={_pool_size_total()} threads", "SUCCESS")
        except Exception as e:
            log(f"⚠️ 预热失败: {e}", "WARNING")

    threading.Thread(target=_do, daemon=True).start()


# ===================== 启动 =====================
if __name__ == "__main__":
    _safe_startup_banner()
    warmup()
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
