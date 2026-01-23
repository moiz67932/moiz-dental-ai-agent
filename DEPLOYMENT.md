# Cloud Run Jobs Deployment Guide

This guide covers deploying the LiveKit voice agent as a Cloud Run Job.

## Why Cloud Run Jobs (Not Services)

| Feature | Cloud Run Service | Cloud Run Job |
|---------|------------------|---------------|
| Trigger | HTTP request | Scheduler/manual |
| Lifetime | Until idle timeout | Until task completes |
| Billing | Per-request + idle | Per-second while running |
| WebSocket | ❌ Killed when idle | ✅ Runs indefinitely |
| Health probes | Required | Not needed |

**LiveKit agents maintain WebSocket connections, not HTTP requests.**
Services will terminate containers with no HTTP traffic.

---

## Prerequisites

```bash
# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudscheduler.googleapis.com
```

---

## Step 1: Build and Push Image

```bash
# Option A: Using Cloud Build (recommended)
gcloud builds submit \
  --tag gcr.io/ortho-ai-485203/dental-agent-worker:prod \
  --file Dockerfile.job \
  .

# Option B: Using Artifact Registry (preferred for production)
gcloud artifacts repositories create agent-repo \
  --repository-format=docker \
  --location=us-central1

docker build -f Dockerfile.job -t us-central1-docker.pkg.dev/ortho-ai-485203/agent-repo/dental-agent-worker:prod .
docker push us-central1-docker.pkg.dev/ortho-ai-485203/agent-repo/dental-agent-worker:prod
```

---

## Step 2: Create Cloud Run Job

```bash
gcloud run jobs create dental-agent-worker \
  --image us-central1-docker.pkg.dev/ortho-ai-485203/agent-repo/dental-agent-worker:prod \
  --region us-central1 \
  --cpu 2 \
  --memory 4Gi \
  --task-timeout 14400 \
  --max-retries 0 \
  --parallelism 1 \
  --set-env-vars "ENVIRONMENT=production" \
  --set-secrets "LIVEKIT_URL=livekit-url:latest" \
  --set-secrets "LIVEKIT_API_KEY=livekit-api-key:latest" \
  --set-secrets "LIVEKIT_API_SECRET=livekit-api-secret:latest" \
  --set-secrets "SUPABASE_URL=supabase-url:latest" \
  --set-secrets "SUPABASE_SERVICE_ROLE_KEY=supabase-key:latest" \
  --set-secrets "OPENAI_API_KEY=openai-key:latest" \
  --set-secrets "DEEPGRAM_API_KEY=deepgram-key:latest"
```

### Flag Explanation

| Flag | Value | Why |
|------|-------|-----|
| `--cpu 2` | 2 vCPU | Voice processing needs CPU headroom |
| `--memory 4Gi` | 4GB RAM | VAD model + TTS buffers |
| `--task-timeout 14400` | 4 hours | Maximum shift length |
| `--max-retries 0` | No retries | Don't restart on controlled shutdown |
| `--parallelism 1` | 1 task | One worker per clinic |

---

## Step 3: Test Manual Execution

```bash
# Execute the job manually
gcloud run jobs execute dental-agent-worker --region us-central1

# Check execution status
gcloud run jobs executions list --job dental-agent-worker --region us-central1

# View logs
gcloud run jobs executions logs dental-agent-worker-EXECUTION_ID --region us-central1
```

---

## Step 4: Set Up Cloud Scheduler (Office Hours)

Cloud Scheduler starts the job at clinic open time and lets it run until timeout.

### Create Scheduler Jobs

```bash
# Start agent at 9:00 AM PKT (Monday-Friday)
gcloud scheduler jobs create http dental-agent-start-weekday \
  --location us-central1 \
  --schedule "0 9 * * 1-5" \
  --time-zone "Asia/Karachi" \
  --uri "https://us-central1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/YOUR_PROJECT/jobs/dental-agent-worker:run" \
  --http-method POST \
  --oauth-service-account-email YOUR_PROJECT@appspot.gserviceaccount.com

# Start agent at 10:00 AM PKT (Saturday)
gcloud scheduler jobs create http dental-agent-start-saturday \
  --location us-central1 \
  --schedule "0 10 * * 6" \
  --time-zone "Asia/Karachi" \
  --uri "https://us-central1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/YOUR_PROJECT/jobs/dental-agent-worker:run" \
  --http-method POST \
  --oauth-service-account-email YOUR_PROJECT@appspot.gserviceaccount.com
```

### How It Works

1. Scheduler triggers job at office open time
2. Job runs worker_main.py (asyncio.run)
3. Worker connects to LiveKit Cloud via WebSocket
4. Worker handles calls until:
   - Task timeout (--task-timeout)
   - SIGTERM from Cloud Run
   - Manual cancellation

**Cost Optimization**: Job only runs during office hours.
No min-instances, no idle billing, no warm-up costs.

---

## Step 5: Managing the Job

### View Running Executions

```bash
gcloud run jobs executions list --job dental-agent-worker --region us-central1
```

### Cancel Running Job

```bash
gcloud run jobs executions cancel EXECUTION_ID --region us-central1
```

### Update Job Configuration

```bash
gcloud run jobs update dental-agent-worker \
  --region us-central1 \
  --cpu 4 \
  --memory 8Gi
```

### Update Image

```bash
gcloud run jobs update dental-agent-worker \
  --region us-central1 \
  --image us-central1-docker.pkg.dev/YOUR_PROJECT/agent-repo/dental-agent-worker:v2
```

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `LIVEKIT_URL` | LiveKit Cloud URL | ✅ |
| `LIVEKIT_API_KEY` | LiveKit API key | ✅ |
| `LIVEKIT_API_SECRET` | LiveKit API secret | ✅ |
| `LIVEKIT_AGENT_NAME` | Agent identity for dispatch | ✅ |
| `SUPABASE_URL` | Supabase project URL | ✅ |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service key | ✅ |
| `OPENAI_API_KEY` | OpenAI API key | ✅ |
| `DEEPGRAM_API_KEY` | Deepgram STT key | ✅ |
| `ENVIRONMENT` | "production" or "development" | ✅ |

Use `--set-secrets` to reference Secret Manager secrets (recommended).

---

## Troubleshooting

### Job Starts Then Immediately Exits

Check logs for startup errors:
```bash
gcloud run jobs executions logs EXECUTION_ID
```

Common causes:
- Missing environment variables
- Invalid LiveKit credentials
- Network connectivity issues

### Job Times Out

Increase `--task-timeout` (max 24 hours):
```bash
gcloud run jobs update dental-agent-worker --task-timeout 86400
```

### Multiple Workers Running

Only one execution should run at a time. Check:
```bash
gcloud run jobs executions list --job dental-agent-worker
```

Cancel duplicates:
```bash
gcloud run jobs executions cancel EXECUTION_ID
```

---

## Cost Estimation

Cloud Run Jobs billing:
- vCPU-second: $0.00002400
- Memory GB-second: $0.00000250

For 8 hours/day (dental office hours):
- 2 vCPU × 8h × 30d = 1,728,000 vCPU-seconds = ~$41.47/month
- 4 GB × 8h × 30d = 3,456,000 GB-seconds = ~$8.64/month
- **Total: ~$50/month per clinic**

Compare to always-on Service with min-instances=1:
- 24/7 = $150+/month

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     CLOUD SCHEDULER                             │
│                                                                 │
│   ┌───────────────┐          ┌───────────────┐                 │
│   │ 9AM Mon-Fri   │          │ 10AM Saturday │                 │
│   │ Start Job     │          │ Start Job     │                 │
│   └───────┬───────┘          └───────┬───────┘                 │
│           │                          │                          │
│           ▼                          ▼                          │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                   CLOUD RUN JOB                          │  │
│   │                                                          │  │
│   │   ┌────────────────────────────────────────────────┐    │  │
│   │   │              worker_main.py                     │    │  │
│   │   │                                                 │    │  │
│   │   │   asyncio.run(main())                          │    │  │
│   │   │       └── agents.Worker()                      │    │  │
│   │   │              └── WebSocket to LiveKit Cloud    │    │  │
│   │   │                     └── Handles voice calls    │    │  │
│   │   │                                                 │    │  │
│   │   └────────────────────────────────────────────────┘    │  │
│   │                                                          │  │
│   │   Runs until: task-timeout OR SIGTERM OR error          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

                              │
                              │ WebSocket
                              ▼
                    ┌─────────────────────┐
                    │   LIVEKIT CLOUD     │
                    │                     │
                    │   SIP Trunk         │
                    │   Room Management   │
                    │   Media Routing     │
                    └─────────────────────┘
                              │
                              │ SIP/RTP
                              ▼
                    ┌─────────────────────┐
                    │   TWILIO TRUNK      │
                    │                     │
                    │   Phone Numbers     │
                    │   PSTN Gateway      │
                    └─────────────────────┘
```
