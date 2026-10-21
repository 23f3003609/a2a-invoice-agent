# A2A Invoice Action Agent (starter)

FastAPI implementation scaffold for the A2A invoice-agent assignment.

## Important implementation notes

- Public Agent Card: `GET /.well-known/agent-card.json`
- Private endpoints: `/a2a/message:send`, `/a2a/tasks`, `/a2a/tasks/{id}`, `/a2a/tasks/{id}:cancel`
- All private endpoints require `Authorization: Bearer ...` and `A2A-Version: 1.0`.
- POST requests require `Content-Type: application/a2a+json`.
- Invoice decisions are generated in one model request for uncached packages and cached by canonical package content.
- Proposals are not converted to receipts until an ACCEPTED continuation is submitted.
- Redis is required for persistent tasks, idempotency records, and decision cache.

## Deploy

1. Create an Upstash Redis database and copy its REST URL/token.
2. Set the environment variables from `.env.example` in Vercel Project Settings.
3. Set `A2A_BEARER_TOKEN` to a long random secret. Keep it private.
4. Import this folder into a GitHub repository and deploy it on Vercel.
5. Verify `https://YOUR-DEPLOYMENT.vercel.app/.well-known/agent-card.json`.
6. Submit `https://YOUR-DEPLOYMENT.vercel.app/a2a` as the A2A interface base URL.

## Local run

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn api.index:app --host 0.0.0.0 --port 8000
```

Set environment variables in PowerShell before starting the server. For local tests, the Agent Card's base URL is derived from the request host, so use the public deployment URL for submission.

## Request shape

The endpoint expects an A2A envelope with a `message` field and exactly one typed `data` part. Batch requests must contain exactly 12 packages. The model response is schema-validated before the task is saved.

## Before relying on it for grading

This is a starter implementation, not a guarantee of a full score. The assignment's concurrent cancellation-versus-acceptance race requires an atomic compare-and-set transaction; this scaffold uses read/modify/write operations and therefore does **not** yet guarantee race correctness under concurrent requests. The exact grader envelope may also require field adjustments. Test against the assignment's provided examples and strengthen the state transition logic before submission.
