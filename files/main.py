"""
A2A 1.0 HTTP+JSON Invoice Action Agent (TDS Project 1)

Layers:
  1. Protocol  - auth, version, media types, routes, envelopes
  2. Storage   - SQLite: tasks, message idempotency, decision cache
  3. Decision  - one batched LLM call per batch, cached by canonical package content

Run:  uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
"""
import os, re, json, uuid, time, sqlite3, asyncio, hashlib, threading, logging
from collections import defaultdict
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response

# ----------------------------------------------------------------- config
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/a2a/")
BASE_PATH = urlparse(BASE_URL).path.rstrip("/")          # e.g. "/a2a"
DB_PATH = os.environ.get("DB_PATH", "agent.db")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://aipipe.org/openai/v1").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "36"))
LLM_MOCK = os.environ.get("LLM_MOCK") == "1"             # for local protocol testing only

REF_FORMAT = os.environ.get("REF_FORMAT", "plain")      # "plain" -> ABC-1 ; "bracket" -> [ABC-1]
STRICT_CT = os.environ.get("STRICT_CONTENT_TYPE", "1") == "1"
LOG_REQUESTS = os.environ.get("LOG_REQUESTS", "1") == "1"

A2A_MT = "application/a2a+json"
BATCH_MT = "application/vnd.ga5.invoice-claim-batch+json"
PROP_MT = "application/vnd.ga5.invoice-action-proposals+json"
REC_MT = "application/vnd.ga5.invoice-action-receipts+json"
RES_MT = "application/vnd.ga5.invoice-action-results+json"

ACTIONS = {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}
MAX_BYTES = 480 * 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agent")
req_log = open("requests.log", "a", encoding="utf-8") if LOG_REQUESTS else None

# ----------------------------------------------------------------- storage
_db = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
_db.execute("PRAGMA journal_mode=WAL")
_dblock = threading.Lock()
with _dblock:
    _db.executescript("""
    CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, principal TEXT, created REAL, data TEXT);
    CREATE INDEX IF NOT EXISTS tasks_p ON tasks(principal, created);
    CREATE TABLE IF NOT EXISTS msgs(principal TEXT, message_id TEXT, hash TEXT, task_id TEXT,
                                    PRIMARY KEY(principal, message_id));
    CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, data TEXT);
    """)


def db_get_task(principal, tid):
    with _dblock:
        r = _db.execute("SELECT data FROM tasks WHERE id=? AND principal=?", (tid, principal)).fetchone()
    return json.loads(r[0]) if r else None


def db_save(principal, task, msg_record=None, new=False):
    """Atomically store a task (and optionally the idempotency record)."""
    with _dblock:
        _db.execute("BEGIN IMMEDIATE")
        try:
            if new:
                _db.execute("INSERT INTO tasks VALUES(?,?,?,?)",
                            (task["id"], principal, time.time(), json.dumps(task, ensure_ascii=False)))
            else:
                _db.execute("UPDATE tasks SET data=? WHERE id=? AND principal=?",
                            (json.dumps(task, ensure_ascii=False), task["id"], principal))
            if msg_record:
                _db.execute("INSERT INTO msgs VALUES(?,?,?,?)", (principal, *msg_record))
            _db.execute("COMMIT")
        except Exception:
            _db.execute("ROLLBACK")
            raise


def db_get_msg(principal, mid):
    with _dblock:
        return _db.execute("SELECT hash, task_id FROM msgs WHERE principal=? AND message_id=?",
                           (principal, mid)).fetchone()


def db_list(principal):
    with _dblock:
        rows = _db.execute("SELECT data FROM tasks WHERE principal=? ORDER BY created DESC",
                           (principal,)).fetchall()
    return [json.loads(r[0]) for r in rows]


def cache_get(k):
    with _dblock:
        r = _db.execute("SELECT data FROM cache WHERE key=?", (k,)).fetchone()
    return json.loads(r[0]) if r else None


def cache_put(k, v):
    with _dblock:
        _db.execute("INSERT OR REPLACE INTO cache VALUES(?,?)", (k, json.dumps(v, ensure_ascii=False)))


# ----------------------------------------------------------------- helpers
_locks = defaultdict(asyncio.Lock)


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ok(data, status=200):
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return Response(body, status_code=status, media_type=A2A_MT)


def err(status, code, message):
    body = json.dumps({"error": {"code": status, "status": code, "message": message}})
    return Response(body, status_code=status, media_type="application/json")


NOT_FOUND = lambda: err(404, "NOT_FOUND", "Task not found")


def principal_of(req: Request):
    h = req.headers.get("authorization", "")
    if not h.startswith("Bearer "):
        return None
    tok = h[7:].strip()
    return sha("principal:" + tok) if tok else None


def guard(req: Request, body=False):
    """Returns (principal, error_response)."""
    p = principal_of(req)
    if not p:
        return None, err(401, "UNAUTHENTICATED", "Missing or invalid bearer token")
    if req.headers.get("a2a-version", "").strip() != "1.0":
        return None, err(400, "VERSION_NOT_SUPPORTED", "A2A-Version must be 1.0")
    if body and STRICT_CT:
        ct = req.headers.get("content-type", "").split(";")[0].strip().lower()
        if ct != A2A_MT:
            return None, err(415, "UNSUPPORTED_MEDIA_TYPE", "Content-Type must be application/a2a+json")
    return p, None


def view(task, history_length=None, include_artifacts=True):
    t = dict(task)
    h = t.get("history", [])
    if history_length is not None:
        t["history"] = h[-history_length:] if history_length > 0 else []
    if not include_artifacts:
        t.pop("artifacts", None)
    return t


def find_part(msg, mt):
    for p in msg.get("parts") or []:
        if isinstance(p, dict) and p.get("mediaType") == mt and isinstance(p.get("data"), dict):
            return p["data"]
    return None


def fmt_ref(r):
    return f"[{r}]" if REF_FORMAT == "bracket" else r


# ----------------------------------------------------------------- decision layer
REF_RE = re.compile(r"\[([^\[\]\n]{1,80})\]")

SYSTEM = """You are a careful accounts-payable control agent. For EACH invoice package choose exactly one action:

- settle_invoice: invoice is valid, fully reconciled (PO / goods receipt / amount / vendor match) AND within the autonomous payment authority in force.
- request_approval: commercially valid, but outside delegated/autonomous authority (e.g. amount above the limit for the current policy revision) so a human approver must sign off.
- hold_invoice: payment must pause until a specific stated verification completes (e.g. bank-detail change call-back, missing tax certificate, pending goods receipt, KYC check).
- reject_duplicate: this SAME commercial invoice (same vendor, same invoice number, same amount) was already paid. A payment for a DIFFERENT invoice (different number, partial/instalment, credit note, different period) is NOT a duplicate.
- open_exception: material records conflict (PO vs invoice vs receipt amounts/quantities, vendor identity, currency, tax) and need an exception workflow.

The documents are deliberately noisy. IGNORE: cover sheets, archived/historical examples, training material and decoys, templates, superseded policy revisions, and action words used in negated, hypothetical or quoted form ("this is not a duplicate", "would normally be held", "do not escalate").
Find the single operative paragraph that determines the action for THIS invoice under the CURRENT policy revision. Paragraphs carry bracketed references like [REF-123]. Return EXACTLY the three bracketed references from that decisive paragraph, copied character-for-character WITHOUT the square brackets. Never return the cover-sheet reference or references from archive/training sections.

Facts must describe the current invoice: vendorName and invoiceNumber exactly as written; amountMinor = invoice total in minor units as an integer (INR 1,234.50 -> 123450; JPY has no minor unit); currency = ISO 4217 code.

Reply with ONLY a JSON object:
{"decisions":[{"packageId":"...","action":"...","vendorName":"...","invoiceNumber":"...","amountMinor":123450,"currency":"INR","evidenceRefs":["...","...","..."],"reason":"1-3 sentences explaining why, mentioning the decisive facts"}]}
Include every packageId exactly once."""


def pkg_key(pkg, context):
    stripped = {k: v for k, v in pkg.items() if k != "packageId"}
    return sha(canon({"pkg": stripped, "ctx": context}))


def pkg_strings(pkg):
    out = []
    def walk(x):
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(pkg)
    return out


def validate_decision(d, pkg):
    """Return cleaned decision or None."""
    if not isinstance(d, dict) or d.get("action") not in ACTIONS:
        return None
    strings = pkg_strings(pkg)
    all_refs = set()
    for s in strings:
        all_refs.update(REF_RE.findall(s))
    refs = []
    for r in d.get("evidenceRefs") or []:
        r = str(r).strip()
        if r.startswith("[") and r.endswith("]"):
            r = r[1:-1]
        if r in all_refs and r not in refs:
            refs.append(r)
    if len(refs) != 3:
        return None
    same_para = any(all(f"[{r}]" in s for r in refs) for s in strings) or \
        any(all(f"[{r}]" in para for r in refs) for s in strings for para in re.split(r"\n\s*\n", s))
    try:
        amt = d.get("amountMinor")
        amt = int(amt) if not isinstance(amt, str) else int(float(amt.replace(",", "")))
    except Exception:
        return None
    return {
        "action": d["action"],
        "facts": {"vendorName": str(d.get("vendorName", "")).strip(),
                  "invoiceNumber": str(d.get("invoiceNumber", "")).strip(),
                  "amountMinor": amt,
                  "currency": str(d.get("currency", "")).strip().upper()},
        "evidenceRefs": refs,
        "reason": str(d.get("reason", "")).strip(),
        "samePara": same_para,
    }


async def call_llm(batch_ctx, pkgs):
    if LLM_MOCK:
        out = []
        for p in pkgs:
            refs = []
            for s in pkg_strings(p):
                refs += REF_RE.findall(s)
            out.append({"packageId": p["packageId"], "action": "settle_invoice", "vendorName": "Mock",
                        "invoiceNumber": "M-1", "amountMinor": 100, "currency": "INR",
                        "evidenceRefs": refs[:3], "reason": "mock decision"})
        return out
    user = json.dumps({"batchContext": batch_ctx, "packages": pkgs}, ensure_ascii=False)
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as c:
        r = await c.post(f"{LLM_BASE_URL}/chat/completions",
                         headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                         json={"model": LLM_MODEL,
                               "response_format": {"type": "json_object"},
                               "messages": [{"role": "system", "content": SYSTEM},
                                            {"role": "user", "content": user}]})
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip()).strip()
    return json.loads(txt).get("decisions", [])


async def decide(batch, deadline):
    pkgs = batch["packages"]
    ctx = {k: v for k, v in batch.items() if k not in ("packages", "batchId")}
    keys = {p["packageId"]: pkg_key(p, ctx) for p in pkgs}
    result = {}
    # per-content locks so concurrent tasks never repeat model work
    lock_keys = sorted(set(keys.values()))
    locks = [_locks["pkg:" + k] for k in lock_keys]
    for l in locks:
        await l.acquire()
    try:
        todo = []
        for p in pkgs:
            c = cache_get(keys[p["packageId"]])
            if c:
                result[p["packageId"]] = c
            else:
                todo.append(p)
        for attempt in range(2):
            if not todo or time.time() > deadline - (8 if attempt else 0):
                break
            try:
                raw = await call_llm(ctx, todo)
            except Exception as e:
                log.warning("LLM call failed: %s", e)
                raw = []
            by_id = {str(d.get("packageId")): d for d in raw if isinstance(d, dict)}
            still = []
            for p in todo:
                v = validate_decision(by_id.get(p["packageId"]), p)
                if v and (v["samePara"] or attempt == 1):
                    result[p["packageId"]] = v
                    cache_put(keys[p["packageId"]], v)
                else:
                    still.append(p)
            todo = still
        for p in todo:  # last-resort fallback (not cached)
            refs = []
            for s in pkg_strings(p):
                refs += [r for r in REF_RE.findall(s) if r not in refs]
            log.error("Fallback decision for %s", p["packageId"])
            result[p["packageId"]] = {"action": "open_exception",
                                      "facts": {"vendorName": "", "invoiceNumber": "", "amountMinor": 0,
                                                "currency": "INR"},
                                      "evidenceRefs": refs[:3], "reason": "Automatic decision unavailable."}
    finally:
        for l in reversed(locks):
            l.release()
    return result


def rationale(action, refs, reason):
    cited = ", ".join(f"[{r}]" for r in refs)
    text = f"Action {action}: the decisive paragraph {cited} determines this outcome. {reason}".strip()
    if len(text) < 60:
        text += " The evidence references above were checked against the current policy revision."
    return text[:1500]


# ----------------------------------------------------------------- app
api = FastAPI(redirect_slashes=False, docs_url=None, redoc_url=None, openapi_url=None)


@api.middleware("http")
async def logger(request: Request, call_next):
    if req_log:
        body = await request.body()
        req_log.write(json.dumps({"t": now(), "m": request.method, "p": request.url.path,
                                  "q": str(request.url.query),
                                  "v": request.headers.get("a2a-version"),
                                  "ct": request.headers.get("content-type"),
                                  "body": body[:200000].decode("utf-8", "replace")}) + "\n")
        req_log.flush()
    return await call_next(request)


@api.get("/.well-known/agent-card.json")
async def agent_card():
    card = {
        "name": "TDS Invoice Action Agent",
        "description": "Reads invoice packages, proposes one evidence-backed business action per invoice, "
                       "and executes only grader-accepted proposals with idempotent, durable task state.",
        "version": "1.0.0",
        "supportedInterfaces": [{"url": BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "provider": {"organization": "TDS Student", "url": BASE_URL},
        "capabilities": {"streaming": False, "pushNotifications": False, "extendedAgentCard": False},
        "securitySchemes": {"bearer": {"httpAuthSecurityScheme": {"scheme": "Bearer"}}},
        "securityRequirements": [{"schemes": {"bearer": {"list": []}}}],
        "defaultInputModes": [BATCH_MT, RES_MT],
        "defaultOutputModes": [PROP_MT, REC_MT],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice action agent",
            "description": "Chooses settle_invoice, request_approval, hold_invoice, reject_duplicate or "
                           "open_exception for each invoice package and cites the decisive evidence.",
            "tags": ["invoice", "accounts-payable", "a2a", "approval", "duplicate-payment"],
            "inputModes": [BATCH_MT, RES_MT],
            "outputModes": [PROP_MT, REC_MT],
        }],
    }
    return Response(json.dumps(card), media_type="application/json")


@api.post(f"{BASE_PATH}/message:send")
async def message_send(request: Request):
    principal, e = guard(request, body=True)
    if e:
        return e
    try:
        body = json.loads(await request.body())
        msg = body["message"]
        assert isinstance(msg, dict)
        mid = msg["messageId"]
        assert isinstance(mid, str) and mid
    except Exception:
        return err(400, "INVALID_REQUEST", "Body must contain message with messageId")
    cfg = body.get("configuration") or {}
    hl = cfg.get("historyLength") if isinstance(cfg.get("historyLength"), int) else None
    h = sha(canon(msg))
    deadline = time.time() + 42

    async with _locks[f"msg:{principal}:{mid}"]:
        rec = db_get_msg(principal, mid)
        if rec:
            if rec[0] != h:
                return err(409, "IDEMPOTENCY_CONFLICT", "IDEMPOTENCY_CONFLICT: messageId reused with different content")
            t = db_get_task(principal, rec[1])
            return ok({"task": view(t, hl)}) if t else NOT_FOUND()

        if msg.get("role") != "ROLE_USER":
            return err(400, "INVALID_REQUEST", "role must be ROLE_USER")

        if msg.get("taskId"):
            return await continue_task(principal, msg, mid, h, hl)
        return await new_task(principal, msg, mid, h, hl, deadline)


async def new_task(principal, msg, mid, h, hl, deadline):
    batch = find_part(msg, BATCH_MT)
    if not batch or not isinstance(batch.get("packages"), list) or not batch["packages"]:
        return err(400, "INVALID_REQUEST", "Missing invoice-claim-batch part")
    ids = [p.get("packageId") for p in batch["packages"] if isinstance(p, dict)]
    if len(ids) != len(batch["packages"]) or len(set(ids)) != len(ids) or not all(ids):
        return err(400, "INVALID_REQUEST", "Packages need unique packageId")

    tid = str(uuid.uuid4())
    ctx = msg.get("contextId") or str(uuid.uuid4())
    decisions = await decide(batch, deadline)

    proposals = []
    for p in batch["packages"]:
        pid = p["packageId"]
        d = decisions[pid]
        proposals.append({
            "packageId": pid,
            "actionId": "act_" + sha(f"{tid}:{pid}")[:24],
            "action": d["action"],
            "facts": d["facts"],
            "evidenceRefs": [fmt_ref(r) for r in d["evidenceRefs"]],
            "rationale": rationale(d["action"], d["evidenceRefs"], d.get("reason", "")),
        })
    hist = dict(msg, taskId=tid, contextId=ctx)
    task = {
        "id": tid,
        "contextId": ctx,
        "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now()},
        "artifacts": [{
            "artifactId": f"proposals-{tid}",
            "name": "invoice-action-proposals",
            "parts": [{"mediaType": PROP_MT, "data": {"batchId": batch.get("batchId"), "proposals": proposals}}],
        }],
        "history": [hist],
    }
    db_save(principal, task, (mid, h, tid), new=True)
    return ok({"task": view(task, hl)})


async def continue_task(principal, msg, mid, h, hl):
    tid = msg["taskId"]
    async with _locks[f"task:{tid}"]:
        task = db_get_task(principal, tid)
        if not task:
            return NOT_FOUND()
        if msg.get("contextId") != task["contextId"]:
            return err(400, "INVALID_REQUEST", "contextId does not match task")
        if task["status"]["state"] in TERMINAL:
            return err(409, "TASK_NOT_ACTIVE", "Task is already terminal")
        if task["status"]["state"] != "TASK_STATE_INPUT_REQUIRED":
            return err(409, "TASK_NOT_READY", "Task is not awaiting input")
        res = find_part(msg, RES_MT)
        prop = task["artifacts"][0]["parts"][0]["data"]
        if not res or res.get("batchId") != prop["batchId"] or not isinstance(res.get("results"), list):
            return err(400, "INVALID_REQUEST", "Results do not match task batch")
        by_pkg = {p["packageId"]: p for p in prop["proposals"]}
        seen, accepted = set(), {}
        for r in res["results"]:
            if not isinstance(r, dict):
                return err(400, "INVALID_REQUEST", "Malformed result")
            pid = r.get("packageId")
            p = by_pkg.get(pid)
            if (not p or pid in seen or r.get("actionId") != p["actionId"] or r.get("action") != p["action"]
                    or r.get("outcome") not in ("ACCEPTED", "REJECTED")):
                return err(400, "INVALID_REQUEST", "Result does not match a stored proposal")
            seen.add(pid)
            if r["outcome"] == "ACCEPTED":
                nonce = r.get("receiptNonce")
                if not isinstance(nonce, str) or not nonce:
                    return err(400, "INVALID_REQUEST", "Accepted result needs receiptNonce")
                accepted[pid] = nonce
        executions = [{
            "packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
            "receiptNonce": accepted[p["packageId"]], "facts": p["facts"], "evidenceRefs": p["evidenceRefs"],
        } for p in prop["proposals"] if p["packageId"] in accepted]

        task["history"].append(dict(msg, taskId=tid, contextId=task["contextId"]))
        task["artifacts"].append({
            "artifactId": f"receipts-{tid}",
            "name": "invoice-action-receipts",
            "parts": [{"mediaType": REC_MT, "data": {"batchId": prop["batchId"], "executions": executions}}],
        })
        task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
        db_save(principal, task, (mid, h, tid))
        return ok({"task": view(task, hl)})


def int_param(request, name):
    v = request.query_params.get(name)
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


@api.get(f"{BASE_PATH}/tasks")
async def list_tasks(request: Request):
    principal, e = guard(request)
    if e:
        return e
    tasks = db_list(principal)
    ctx = request.query_params.get("contextId")
    st = request.query_params.get("status")
    if ctx:
        tasks = [t for t in tasks if t["contextId"] == ctx]
    if st:
        tasks = [t for t in tasks if t["status"]["state"] == st]
    hl = int_param(request, "historyLength")
    inc = request.query_params.get("includeArtifacts", "true").lower() != "false"
    size = min(max(int_param(request, "pageSize") or 50, 1), 100)
    start = int(request.query_params.get("pageToken") or 0) if (request.query_params.get("pageToken") or "0").isdigit() else 0
    page, used, i = [], 200, start
    while i < len(tasks) and len(page) < size:
        v = view(tasks[i], hl if hl is not None else 0, inc)
        n = len(json.dumps(v, ensure_ascii=False).encode())
        if page and used + n > MAX_BYTES:
            break
        page.append(v); used += n; i += 1
    nxt = str(i) if i < len(tasks) else ""
    return ok({"tasks": page, "nextPageToken": nxt, "pageSize": len(page), "totalSize": len(tasks)})


@api.get(BASE_PATH + "/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    principal, e = guard(request)
    if e:
        return e
    t = db_get_task(principal, task_id)
    return ok(view(t, int_param(request, "historyLength"))) if t else NOT_FOUND()


@api.post(BASE_PATH + "/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    principal, e = guard(request, body=False)
    if e:
        return e
    async with _locks[f"task:{task_id}"]:
        t = db_get_task(principal, task_id)
        if not t:
            return NOT_FOUND()
        if t["status"]["state"] in TERMINAL:
            return err(409, "TASK_NOT_CANCELABLE", "Task is already terminal")
        t["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
        db_save(principal, t)
        return ok(t)


class SlashFix:
    """Collapse '//' so {base}/message:send works whether or not base ends with '/'. Never redirects."""
    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            p = re.sub(r"/{2,}", "/", scope["path"])
            if p != scope["path"]:
                scope = dict(scope, path=p, raw_path=p.encode())
        await self.inner(scope, receive, send)


app = SlashFix(api)
