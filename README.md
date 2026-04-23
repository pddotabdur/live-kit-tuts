# Nora — Outbound LiveKit Voice Agent

Outbound calling agent for stc bank collections.
Pipeline: Deepgram STT → OpenAI LLM → OpenAI TTS, dialed via LiveKit SIP.

## Layout

```
agent.py                 # worker; runs on the SAME host as the LiveKit server
dispatch.py              # dispatcher; can run anywhere with a network path to LiveKit
.env / .env.example      # config (AI keys, LiveKit URL, SIP trunk id)
sip-local/               # docker-compose for LiveKit + SIP (the EC2 stack)
deploy/                  # systemd unit for running the agent on EC2
scratch/                 # old demos, ad-hoc tests, sample data — not used in prod
```

## Why the agent must run on the EC2 box

In our last debug run the SIP server reported `mixer.restarts=226` and
`mixer.mixes_zero=2811` over a 4-min call — i.e. the agent's WebRTC publish
to LiveKit was losing ~65% of frames over the public internet, so the SIP
mixer wrote silence to the carrier. Running `agent.py` on the same host as
LiveKit makes that publish go over loopback (0% loss, ~0 ms) and removes the
entire failure class.

## EC2 deploy — one-time setup

SSH into the EC2 box (`ubuntu@13.49.62.84`).

```bash
# 1) Kernel UDP buffers (one-time)
sudo tee /etc/sysctl.d/99-livekit.conf >/dev/null <<'EOF'
net.core.rmem_max=16777216
net.core.wmem_max=16777216
net.core.rmem_default=8388608
net.core.wmem_default=8388608
EOF
sudo sysctl --system

# 2) Install uv (fast Python toolchain)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 3) Pull the repo
cd ~ && git clone <YOUR_REPO_URL> live-kit-tuts
cd live-kit-tuts
uv sync                                 # creates .venv with all deps

# 4) Configure env (LIVEKIT_URL must be loopback on EC2!)
cp .env.example .env
sed -i 's|^LIVEKIT_URL=.*|LIVEKIT_URL=ws://localhost:7880|' .env
nano .env                               # paste OPENAI_API_KEY, DEEPGRAM_API_KEY

# 5) Make sure LiveKit + SIP are up
cd sip-local && docker compose up -d && cd ..

# 6) Install the systemd unit
sudo cp deploy/livekit-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now livekit-agent
sudo journalctl -u livekit-agent -f
```

## AWS Security Group — required inbound rules

| Proto | Port        | Source       | Purpose                       |
|-------|-------------|--------------|-------------------------------|
| TCP   | 7880        | 0.0.0.0/0    | LiveKit signaling             |
| TCP   | 7881        | 0.0.0.0/0    | LiveKit TCP fallback          |
| UDP   | 7882        | 0.0.0.0/0    | LiveKit WebRTC media          |
| UDP/TCP | 5060      | 0.0.0.0/0    | SIP signaling                 |
| UDP   | 10000–20000 | 0.0.0.0/0    | SIP RTP                       |

## Triggering a call

From anywhere with a network path to the LiveKit server:

```bash
# locally on your laptop, .env has LIVEKIT_URL=ws://13.49.62.84:7880
python dispatch.py
```

Edit the `phone_number` (and any other `dial_info` fields) in `dispatch.py`
before running.

## Verifying after deploy

1. `sudo systemctl status livekit-agent` — should be `active (running)`.
2. Run `python dispatch.py` from your laptop.
3. Phone rings, you answer, the agent walks through the Nora flow:
   opening → ID confirm → debt context → handle response → confirmation → closing.
4. If audio is still silent on the call, tail both sides while dialing:

   ```bash
   # on EC2
   docker compose -f sip-local/docker-compose.yaml logs -f sip \
     | grep -iE 'mixer|track|publish|subscribe|rtp|bye'
   sudo journalctl -u livekit-agent -f
   ```

   `mixer.restarts` should now be near 0 and `mixer.mixes_zero` near 0.
