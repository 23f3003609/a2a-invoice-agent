import json
import os
import hashlib
import secrets
import time
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

APP_VERSION = "1.0.0"
BATCH_MEDIA = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSAL_MEDIA = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPT_MEDIA = "application/vnd.ga5.invoice-action-receipts+json"
RESULT_MEDIA = "application/vnd.ga5.invoice-action-results+json"
A2A_MEDIA = "application/a2a+json"
ACTIONS = {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}
app = FastAPI(title="A2A Invoice Agent", version=APP_VERSION)


def origin_and_base(request: Request):
    origin = str(request.base_url).rstrip("/")
    # Vercel deployments mount this app under /api, but the submitted A2A base URL is /a2a.
    return origin, origin + "/a2a"


def json_error(status: int, code: str, message: str):
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}},
                        media_type=A2A_MEDIA)


async def redis_command(*parts):
    url = os.getenv("UPSTASH_REDIS_REST_URL")
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
    if not url or not token:
        raise RuntimeError("Persistent storage is not configured. Set Upstash Redis REST environment variables.")
    async with httpx.AsyncClient(timeout=8) as client:
        r = await client.post(url, headers={"Authorization": f"Bearer {token}",
                                            "Content-Type": "application/json"},
                              json=list(parts))
        r.raise_for_status()
        payload = r.json()
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        return payload.get("result")


async def rget(key):
    value = await redis_command("GET", key)
    if value is None:
        return None
    return json.loads(value)


async def rset(key, value):
    await redis_command("SET", key, json.dumps(value, separators=(",", ":"), sort_keys=True))


async def rdel(key):
    await redis_command("DEL", key)


async def rkeys(pattern):
    return await redis_command("KEYS", pattern) or []


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def get_principal(request: Request):
    auth = request.headers.get("authorization", "")
    expected = os.getenv("A2A_BEARER_TOKEN", "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjMwMDM2MDlAZHMuc3R1ZHkuaWl0bS5hYy5pbiIsImlhdCI6MTc5MjU5OTg3OCwiaXNzIjoiaHR0cHM6Ly9haXBpcGUub3JnIiwiYXVkIjoiYWlwaXBlLWFwaSIsImV4cCI6MTc5MzIwNDY3OH0.MGW2e5cxrZFlJmPXRPPgJXq9oHLLHg4ckxm4qlmRPbM")
    if not expected:
        return None, json_error(503, "SERVER_NOT_CONFIGURED", "Agent authentication is not configured.")
    if not auth.startswith("Bearer ") or not secrets.compare_digest(auth[7:], expected):
        return None, json_error(401, "UNAUTHORIZED", "A valid bearer token is required.")
    return hashlib.sha256(auth[7:].encode()).hexdigest(), None


def check_headers(request: Request):
    if request.headers.get("a2a-version") != "1.0":
        return json_error(400, "INVALID_A2A_VERSION", "A2A-Version: 1.0 is required.")
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if request.method == "POST" and content_type != A2A_MEDIA:
        return json_error(415, "UNSUPPORTED_MEDIA_TYPE", f"Content-Type must be {A2A_MEDIA}.")
    return None


def task_key(principal, task_id):
    return f"invagent:task:{principal}:{task_id}"


def idempotency_key(principal, message_id):
    return f"invagent:idem:{principal}:{message_id}"


def parse_message(envelope):
    if not isinstance(envelope, dict) or not isinstance(envelope.get("message"), dict):
        raise ValueError("Request must contain a message object.")
    message = envelope["message"]
    if not message.get("messageId") or message.get("role") != "ROLE_USER":
        raise ValueError("messageId and role ROLE_USER are required.")
    parts = message.get("parts")
    if not isinstance(parts, list) or len(parts) != 1 or not isinstance(parts[0], dict):
        raise ValueError("Exactly one typed data part is required.")
    return message, parts[0]


def data_part(part, media_type):
    if part.get("mediaType") != media_type or not isinstance(part.get("data"), dict):
        raise ValueError(f"Expected a data part with mediaType {media_type}.")
    return part["data"]


def package_cache_key(package):
    # Deliberately exclude batch/task/message namespace fields: cache only package semantics.
    return "invagent:decision:" + digest(package)


async def decide_packages(batch):
    packages = batch.get("packages")
    if not isinstance(packages, list) or len(packages) != 12:
        raise ValueError("Each invoice claim batch must contain exactly 12 packages.")
    out = []
    uncached = []
    for package in packages:
        if not isinstance(package, dict) or not package.get("packageId"):
            raise ValueError("Every package must be an object with packageId.")
        cached = await rget(package_cache_key(package))
        if cached:
            out.append((package, cached))
        else:
            out.append((package, None))
            uncached.append(package)

    if uncached:
        model_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model_key = os.getenv("OPENAI_API_KEY", "")
        model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        if not model_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        system = (
            "You are a cautious invoice evidence decision engine. For each package, inspect only the provided "
            "package content and evidence. Do not infer missing facts. Return JSON object {decisions:[...]}. "
            "Each decision must contain packageId, vendorName, invoiceNumber, amountMinor (integer), currency, "
            "action (one of settle_invoice, request_approval, hold_invoice, reject_duplicate, open_exception), "
            "exactly three evidenceRefs copied from decisive bracketed evidence references in the package, and "
            "rationale of 60-1500 characters naming the action and citing at least two refs. Do not use cover "
            "sheets, archive examples, or training decoys as decisive evidence. If uncertain or contradictory, "
            "choose open_exception. Do not execute anything."
        )
        payload = {"model": model_name, "temperature": 0, "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": canonical({"packages": uncached})}]}
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(model_url.rstrip("/") + "/chat/completions",
                                     headers={"Authorization": f"Bearer {model_key}"},
                                     json=payload)
            resp.raise_for_status()
            body = resp.json()
        raw = body["choices"][0]["message"]["content"]
        parsed = json.loads(raw)
        decisions = parsed.get("decisions")
        if not isinstance(decisions, list) or len(decisions) != len(uncached):
            raise RuntimeError("Model returned an invalid number of package decisions.")
        by_id = {str(d.get("packageId")): d for d in decisions if isinstance(d, dict)}
        for package in uncached:
            decision = by_id.get(str(package.get("packageId")))
            if not decision:
                raise RuntimeError("Model omitted a package decision.")
            validate_decision(package, decision)
            await rset(package_cache_key(package), decision)
        out = []
        for package in packages:
            decision = await rget(package_cache_key(package))
            out.append((package, decision))
    proposals = []
    for package, decision in out:
        proposal = dict(decision)
        proposal["packageId"] = package["packageId"]
        proposal["actionId"] = "act_" + uuid.uuid4().hex[:24]
        proposals.append(proposal)
    return proposals


def validate_decision(package, d):
    required = ("vendorName", "invoiceNumber", "amountMinor", "currency", "action", "evidenceRefs", "rationale")
    if any(k not in d for k in required):
        raise RuntimeError("Model decision is missing required fields.")
    if d["action"] not in ACTIONS:
        raise RuntimeError("Model returned an unsupported action.")
    if not isinstance(d["amountMinor"], int) or not isinstance(d["evidenceRefs"], list) or len(d["evidenceRefs"]) != 3:
        raise RuntimeError("Model returned invalid typed facts or evidenceRefs.")
    if not 60 <= len(str(d["rationale"])) <= 1500:
        raise RuntimeError("Model rationale must be 60-1500 characters.")
    if not all(isinstance(x, str) and x.strip() for x in d["evidenceRefs"]):
        raise RuntimeError("Evidence references must be non-empty strings.")


def task_view(task):
    # Never expose internal principal or idempotency hash.
    return {k: v for k, v in task.items() if k not in ("principal", "requestHash")}


@app.get("/.well-known/agent-card.json")
async def agent_card(request: Request):
    origin, base = origin_and_base(request)
    return JSONResponse({
        "name": "A2A Invoice Action Agent",
        "description": "Evidence-grounded invoice action proposals with explicit acceptance gating.",
        "version": APP_VERSION,
        "capabilities": {"streaming": False, "pushNotifications": False, "stateTransitionHistory": True},
        "supportedInterfaces": [{"url": base, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "defaultInputModes": [BATCH_MEDIA, RESULT_MEDIA],
        "defaultOutputModes": [PROPOSAL_MEDIA, RECEIPT_MEDIA],
        "skills": [{"id": "invoice_action_agent", "name": "Invoice action proposals",
                    "description": "Reviews invoice evidence and proposes safe typed actions; executes only accepted actions.",
                    "tags": ["invoice", "evidence", "approval", "idempotency"]}]
    })


@app.post("/a2a/message:send")
async def message_send(request: Request):
    header_error = check_headers(request)
    if header_error:
        return header_error
    principal, auth_error = get_principal(request)
    if auth_error:
        return auth_error
    try:
        envelope = await request.json()
        message, part = parse_message(envelope)
        msg_id = str(message["messageId"])
        msg_hash = digest(message)  # configuration intentionally excluded
        existing = await rget(idempotency_key(principal, msg_id))
        if existing:
            if existing.get("hash") != msg_hash:
                return json_error(409, "IDEMPOTENCY_CONFLICT", "messageId was already used with different message content.")
            task = await rget(task_key(principal, existing["taskId"]))
            if task:
                return JSONResponse(task_view(task), media_type=A2A_MEDIA)
            return json_error(500, "STORAGE_INCONSISTENCY", "Stored idempotency record has no task.")

        if part.get("mediaType") == BATCH_MEDIA:
            batch = data_part(part, BATCH_MEDIA)
            if not batch.get("batchId") or not batch.get("policyRevision"):
                return json_error(400, "INVALID_BATCH", "batchId and policyRevision are required.")
            proposals = await decide_packages(batch)
            task_id = "task_" + uuid.uuid4().hex
            context_id = str(message.get("contextId") or ("ctx_" + uuid.uuid4().hex))
            response_message = {"messageId": "msg_" + uuid.uuid4().hex, "role": "ROLE_AGENT",
                                "parts": [{"mediaType": PROPOSAL_MEDIA,
                                           "data": {"batchId": batch["batchId"], "proposals": proposals}}]}
            task = {"id": task_id, "contextId": context_id, "status": {"state": "TASK_STATE_INPUT_REQUIRED"},
                    "history": [message, response_message], "artifacts": [{"artifactId": "artifact_" + uuid.uuid4().hex,
                    "name": "invoice-action-proposals", "parts": response_message["parts"]}],
                    "principal": principal, "createdAt": time.time(), "updatedAt": time.time(),
                    "batchId": batch["batchId"], "requestHash": msg_hash}
        elif part.get("mediaType") == RESULT_MEDIA:
            result_data = data_part(part, RESULT_MEDIA)
            task_id = result_data.get("taskId") or message.get("taskId")
            if not task_id:
                return json_error(400, "INVALID_RESULTS", "taskId is required for a continuation.")
            task = await rget(task_key(principal, str(task_id)))
            if not task:
                return json_error(404, "TASK_NOT_FOUND", "Task not found.")
            if task["status"]["state"] in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"):
                return json_error(409, "TASK_TERMINAL", "Terminal tasks are immutable.")
            if message.get("contextId") != task.get("contextId"):
                return json_error(403, "CONTEXT_MISMATCH", "Task context does not match.")
            if result_data.get("batchId") != task.get("batchId"):
                return json_error(409, "BATCH_MISMATCH", "Batch does not match.")
            results = result_data.get("results")
            if not isinstance(results, list):
                return json_error(400, "INVALID_RESULTS", "results must be an array.")
            proposals = task["artifacts"][0]["parts"][0]["data"]["proposals"]
            proposal_map = {p["packageId"]: p for p in proposals}
            receipts = []
            for result in results:
                if not isinstance(result, dict):
                    continue
                if result.get("outcome") != "ACCEPTED":
                    continue
                proposal = proposal_map.get(result.get("packageId"))
                if not proposal or result.get("actionId") != proposal["actionId"] or result.get("action") != proposal["action"]:
                    continue
                if not isinstance(result.get("receiptNonce"), str) or not result["receiptNonce"]:
                    continue
                receipts.append({"packageId": proposal["packageId"], "actionId": proposal["actionId"],
                                 "action": proposal["action"], "receiptId": "rcpt_" + uuid.uuid4().hex,
                                 "receiptNonce": result["receiptNonce"], "status": "ACCEPTED"})
            response_message = {"messageId": "msg_" + uuid.uuid4().hex, "role": "ROLE_AGENT",
                                "parts": [{"mediaType": RECEIPT_MEDIA,
                                           "data": {"batchId": task["batchId"], "receipts": receipts}}]}
            task["history"].append(message)
            task["history"].append(response_message)
            task["artifacts"].append({"artifactId": "artifact_" + uuid.uuid4().hex,
                                      "name": "invoice-action-receipts", "parts": response_message["parts"]})
            task["status"] = {"state": "TASK_STATE_COMPLETED"}
            task["updatedAt"] = time.time()
            await rset(task_key(principal, str(task_id)), task)
            return JSONResponse(task_view(task), media_type=A2A_MEDIA)
        else:
            return json_error(415, "UNSUPPORTED_INPUT", "Unsupported message part mediaType.")

        await rset(task_key(principal, task_id), task)
        await rset(idempotency_key(principal, msg_id), {"hash": msg_hash, "taskId": task_id})
        return JSONResponse(task_view(task), media_type=A2A_MEDIA)
    except ValueError as e:
        return json_error(400, "INVALID_REQUEST", str(e))
    except Exception as e:
        # Do not expose model prompts, tokens, package contents, or other users' task data.
        return json_error(502, "PROCESSING_FAILED", "The invoice batch could not be processed. Check server configuration and retry safely.")


@app.get("/a2a/tasks")
async def list_tasks(request: Request):
    header_error = check_headers(request)
    if header_error:
        return header_error
    principal, auth_error = get_principal(request)
    if auth_error:
        return auth_error
    keys = await rkeys(f"invagent:task:{principal}:*")
    tasks = []
    for key in keys:
        task = await rget(key)
        if task:
            tasks.append(task_view(task))
    tasks.sort(key=lambda t: t.get("createdAt", 0), reverse=True)
    return JSONResponse({"tasks": tasks}, media_type=A2A_MEDIA)


@app.get("/a2a/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    header_error = check_headers(request)
    if header_error:
        return header_error
    principal, auth_error = get_principal(request)
    if auth_error:
        return auth_error
    task = await rget(task_key(principal, task_id))
    if not task:
        return json_error(404, "TASK_NOT_FOUND", "Task not found.")
    return JSONResponse(task_view(task), media_type=A2A_MEDIA)


@app.post("/a2a/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    header_error = check_headers(request)
    if header_error:
        return header_error
    principal, auth_error = get_principal(request)
    if auth_error:
        return auth_error
    try:
        envelope = await request.json()
        task = await rget(task_key(principal, task_id))
        if not task:
            return json_error(404, "TASK_NOT_FOUND", "Task not found.")
        if task["status"]["state"] in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"):
            return json_error(409, "TASK_TERMINAL", "Terminal tasks are immutable.")
        task["status"] = {"state": "TASK_STATE_CANCELED"}
        task["updatedAt"] = time.time()
        await rset(task_key(principal, task_id), task)
        return JSONResponse(task_view(task), media_type=A2A_MEDIA)
    except Exception:
        return json_error(400, "INVALID_REQUEST", "Cancellation request is invalid.")


# Vercel Python runtime imports api/index.py as the ASGI application.
