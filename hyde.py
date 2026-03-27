from openai import OpenAI
import json, numpy as np, time, re
import os
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)
MODEL = "openrouter/free"

#MODEL = "qwen/qwen3.5-122b-a10b"

def chat(messages, model=MODEL, temperature=0.3, max_tokens=1024):
    resp = client.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_tokens
    )
    return resp.choices[0].message.content

print(chat([{"role": "user", "content": "Say 'hello world' and nothing else."}]))

# HyDE - Hypothetical Document Embeddings

from sentence_transformers import SentenceTransformer

DOCUMENTS = [
    {"id": 1, "title": "Authentication System Overview",
     "content": "The authentication system uses OAuth 2.0 with PKCE flow for all client applications. Tokens are issued with a 1-hour expiry and refresh tokens last 30 days. Multi-factor authentication is required for admin accounts. The auth service is deployed on Kubernetes with 3 replicas for high availability. Rate limiting is set to 100 requests per minute per IP."},
    {"id": 2, "title": "API Rate Limiting Policy",
     "content": "All API endpoints enforce rate limiting based on the client subscription tier. Free tier: 60 requests/minute. Pro tier: 600 requests/minute. Enterprise: 6000 requests/minute. Rate limit headers (X-RateLimit-Remaining, X-RateLimit-Reset) are included in every response. When limits are exceeded, the API returns HTTP 429 with a Retry-After header."},
    {"id": 3, "title": "Error Handling Guide",
     "content": "HTTP 400: Bad request. HTTP 401: Unauthorized, token may be expired, refresh it. HTTP 403: Forbidden, insufficient permissions. HTTP 404: Not found. HTTP 429: Rate limited, wait and retry with exponential backoff. HTTP 500: Server error, retry after 30 seconds, contact support if persistent."},
    {"id": 4, "title": "Deployment Runbook",
     "content": "Production deployments happen daily at 2 AM UTC via automated CI/CD pipeline. Canary deployments are used with 5 percent traffic for 30 minutes before full rollout. Rollback procedure: run kubectl rollout undo in the production namespace. Health checks must pass for 5 consecutive minutes before traffic is shifted."},
    {"id": 5, "title": "Database Schema: Users Table",
     "content": "The users table contains: id (UUID primary key), email (varchar unique indexed), password_hash (varchar), created_at (timestamp), last_login (timestamp), subscription_tier (enum: free, pro, enterprise), mfa_enabled (boolean default false), api_key_hash (varchar nullable). Soft deletes are used via a deleted_at column."},
    {"id": 6, "title": "Incident Response: Auth Service Outage Jan 2024",
     "content": "On January 12 2024 the authentication service experienced a 47-minute outage due to database connection pool exhaustion. Root cause: a deployment introduced a query that held connections for 30+ seconds under load. Impact: approximately 12000 users unable to log in. Resolution: deployment rolled back and query optimized."},
    {"id": 7, "title": "API Versioning Strategy",
     "content": "The API uses URL-based versioning (e.g. /v1/ and /v2/). Breaking changes require a new major version. Non-breaking additions are added to the current version. Deprecated endpoints return a Sunset header with the removal date. Version v1 will be sunset on June 30 2025. All clients must migrate to v2 by that date."},
    {"id": 8, "title": "Monitoring and Alerting Setup",
     "content": "We use Prometheus for metrics collection and Grafana for dashboards. Critical alerts: p99 latency above 500ms, error rate above 1 percent, pod restarts above 3 in 5 minutes. Warning alerts: p95 latency above 200ms, CPU usage above 80 percent sustained for 10 minutes. Alerts route to PagerDuty for critical and Slack for warnings."},
    {"id": 9, "title": "Security: Token Refresh Flow",
     "content": "When an access token expires the client sends the refresh token to POST /auth/refresh. The server validates the refresh token, checks it has not been revoked, and issues a new access token plus new refresh token (rotation). The old refresh token is immediately invalidated. If a revoked refresh token is used ALL tokens for that user are invalidated."},
    {"id": 10, "title": "Onboarding: New Developer Setup",
     "content": "Step 1: Clone the monorepo from GitHub. Step 2: Run make setup to install dependencies. Step 3: Copy .env.example to .env and fill in local database credentials. Step 4: Run make migrate for database schema. Step 5: Run make seed for test data. Step 6: Run make dev to start the server on localhost:3000."}
]

print("Loaded %d documents" % len(DOCUMENTS))
for d in DOCUMENTS:
    print("   Doc %d: %s" % (d['id'], d['title']))

embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_texts = ["%s. %s" % (d['title'], d['content']) for d in DOCUMENTS]
doc_embeddings = embedder.encode(doc_texts, normalize_embeddings=True)

def semantic_search(query, top_k=3):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    scores = np.dot(doc_embeddings, q_emb.T).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(DOCUMENTS[i], float(scores[i])) for i in top_idx]

def hyde_search(query, top_k=3):
    msg = [{"role": "user", "content": "Write a short, factual paragraph that would answer this question. Do not say you don't know.\n\nQuestion: %s\n\nHypothetical answer:" % query}]
    hypo_answer = chat(msg, max_tokens=200)
    print("   Hypothetical answer: %s..." % hypo_answer[:150])
    hypo_emb = embedder.encode([hypo_answer], normalize_embeddings=True)
    scores = np.dot(doc_embeddings, hypo_emb.T).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(DOCUMENTS[i], float(scores[i])) for i in top_idx]

query = "Our users can't log in, what could be wrong?"
print("Query: %s\n" % query)
print("Standard semantic search:")
for doc, score in semantic_search(query): print("   [%.3f] Doc %d: %s" % (score, doc['id'], doc['title']))
print("\nHyDE search:")
for doc, score in hyde_search(query): print("   [%.3f] Doc %d: %s" % (score, doc['id'], doc['title']))
print("\n--> HyDE generates a hypothetical answer mentioning tokens, auth, etc.")
print("   This pulls in more relevant docs than the vague original query.")