# Deploy & Recovery

Resurrection guide for the LiveKit voice-agent + SIP stack. If things break, follow this end-to-end and you will be back to the last known good state.

## What runs where

| Component | Host | How it runs |
|-----------|------|-------------|
| LiveKit server | EC2 | docker compose (network_mode: host) |
| LiveKit SIP server (custom build with symmetric_rtp) | EC2 | docker compose, image `livekit/sip:local` |
| Redis | EC2 | docker compose |
| Production agent (outbound-caller) | EC2 | systemd unit `livekit-agent.service` |
| Dev agents (outbound-caller-<name>) | Each developer's laptop | `make agent-dev` |
| Outbound call dispatch | Anywhere with API creds | `make call` / `make call-prod` |

## Known-good baseline

A git tag `working-symmetric-rtp` marks the last commit that is fully verified working with audio. Roll back with:

```
git fetch --tags
git checkout working-symmetric-rtp
```

## Full redeploy from scratch (worst case)

### A. Build and ship the custom SIP image

The official `livekit/sip:latest` is missing symmetric_rtp support, so we build from source.

On your laptop:

```
make sip-deploy   # builds image, ships tar to EC2, loads, restarts stack
```

If `make sip-deploy` is unavailable, the manual flow is in the Makefile under `sip-image` and `sip-deploy` targets.

### B. EC2 docker-compose file

Lives at `~/sip/docker-compose.yaml` on the EC2 box. The canonical version is checked into this repo at `sip-local/docker-compose.yaml`. To restore on EC2:

```
scp -i $KEY sip-local/docker-compose.yaml ubuntu@$EC2:~/sip/docker-compose.yaml
ssh -i $KEY ubuntu@$EC2 'cd ~/sip && sudo docker compose up -d'
```

The required SIP config flags are:

- `image: livekit/sip:local` (NOT `livekit/sip` — official image lacks symmetric_rtp)
- `symmetric_rtp: true` in `SIP_CONFIG_BODY`
- `use_external_ip: true`
- `rtp_port: 10000-10004` (must match the EC2 security-group UDP rule)
- `restart: unless-stopped` on every service (so reboots don't break it)

### C. Production agent on EC2

Systemd unit at `/etc/systemd/system/livekit-agent.service` (template in `deploy/livekit-agent.service`). The unit runs `agent.py` with `AGENT_NAME=outbound-caller` (default).

```
sudo systemctl daemon-reload
sudo systemctl enable livekit-agent
sudo systemctl restart livekit-agent
sudo journalctl -u livekit-agent -f
```

### D. Secrets

`.env` files are git-ignored. The required keys are in `.env.example`. Populate from the team's password manager (LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, OPENAI_API_KEY, DEEPGRAM_API_KEY, FASEEH_API_KEY, SIP_OUTBOUND_TRUNK_ID).

The same `.env` is needed on EC2 at `/home/ubuntu/live-kit-tuts/.env` for the production agent.

## Per-developer dev loop

Each developer runs their own agent, registered under a unique name, against the shared EC2 server. Calls dispatched to `outbound-caller-<name>` go only to that developer's machine — no interference with prod or other devs.

```
# .env on developer laptop:
#   LIVEKIT_URL=wss://...   (points at shared EC2 LiveKit)
#   AGENT_NAME=outbound-caller-bilal   (optional override; Makefile sets per-user automatically)
#   PHONE_NUMBER=+966...    (your test number)

# Terminal 1 — register the agent:
make agent-dev

# Terminal 2 — place a test call to yourself:
make call PHONE_NUMBER=+966555209485
```

## Updating the prompt

The prompt is in `agent.py` inside `_build_system_prompt(...)`. Edit, then:

1. Save the file.
2. The `make agent-dev` worker restarts on file change automatically (livekit `dev` mode).
3. Run `make call` to place a fresh test call.

To roll out to production once verified:

```
git push origin main
ssh -i $KEY ubuntu@$EC2 'cd ~/live-kit-tuts && git pull && sudo systemctl restart livekit-agent'
```

## Image backup (belt-and-suspenders)

The `livekit/sip:local` image is reproducible from `sip-local/sip` + `make sip-image`, but rebuilds depend on LiveKit upstream not breaking the build. To keep an offline copy:

```
# Save the tar somewhere safe (laptop + EC2)
sudo docker save livekit/sip:local -o /tmp/sip-local.tar
# Then copy /tmp/sip-local.tar to permanent storage (S3 bucket, network drive, etc.)
```

To restore from the tar:

```
sudo docker load -i sip-local.tar
cd ~/sip && sudo docker compose up -d
```

## Sanity checks after deploy

```
# SIP container is up
ssh -i $KEY ubuntu@$EC2 'cd ~/sip && sudo docker compose ps'

# symmetric_rtp is active in the running SIP container
ssh -i $KEY ubuntu@$EC2 'cd ~/sip && sudo docker compose logs sip | grep -i symmetric'

# Production agent is registered
ssh -i $KEY ubuntu@$EC2 'sudo journalctl -u livekit-agent -n 50'

# Place a test call to yourself
PHONE_NUMBER=+966... make call-prod
```

If audio is one-way or silent, the most common cause is the SIP container falling back to the official image — check `docker compose ps` and confirm the IMAGE column says `livekit/sip:local` not `livekit/sip:latest`.
