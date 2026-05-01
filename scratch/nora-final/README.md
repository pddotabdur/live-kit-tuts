# Nora Final

Clean-slate agent for the Mobily / Tawafuq collections call flow, built on [LiveKit Agents](https://docs.livekit.io/agents/) (Python). Run it locally, self-host the LiveKit server, or **deploy the worker to LiveKit Cloud**.

## Project layout

- `nora_agent/workflow.py` — Stages, `AgentTask`, and session wiring
- `nora_agent/__main__.py` — CLI entrypoint (`dev`, `start`, `download-files`, …)
- `docs/call-flow-stages.md` — Human-readable stage prompts and rules
- `Dockerfile` — Image used by [LiveKit Cloud agent builds](https://docs.livekit.io/deploy/agents/builds/)
- `.env.example` — Template for local env and documentation (never commit real `.env`)

## Local quick start

```bash
cd nora-final
cp .env.example .env
# Edit .env: LIVEKIT_URL, keys, and optional NORA_* values

uv sync
uv run python -m nora_agent dev
```

## Deploy to LiveKit Cloud

1. Install the [LiveKit CLI](https://docs.livekit.io/reference/developer-tools/livekit-cli/) and authenticate:

   ```bash
   lk cloud auth
   ```

2. In the project directory, set the default linked Cloud project (if you have more than one):

   ```bash
   lk project list
   lk project set-default "<your-project-name>"
   ```

3. **First-time registration** creates `livekit.toml` and starts a build. Pass a secrets file for API keys and app config only (do **not** put `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` in the file—[Cloud injects those](https://docs.livekit.io/deploy/agents/secrets/)):

   ```bash
   cp deploy.secrets.env.example deploy.secrets.env
   # Put real values in deploy.secrets.env, then (see “Deploy CLI vs local .env” below):
   lk agent create --secrets-file deploy.secrets.env
   ```

   **Deploy CLI vs local `.env`:** If your `.env` still has `LIVEKIT_URL=ws://localhost:7880` (for local `dev`), the `lk` command may also read `LIVEKIT_URL` from the **current shell** and fail with `invalid project URL [ws://localhost:7880]`. Before `lk agent create` / `lk agent deploy`, clear those in PowerShell and rely on the linked Cloud project (after `lk cloud auth` + `lk project set-default`):

   ```powershell
   Remove-Item Env:\LIVEKIT_URL -ErrorAction SilentlyContinue
   Remove-Item Env:\LIVEKIT_API_KEY -ErrorAction SilentlyContinue
   Remove-Item Env:\LIVEKIT_API_SECRET -ErrorAction SilentlyContinue
   ```

   The CLI warning about `cli-config.yaml` and permissions `600` is safe to ignore on Windows, or use `icacls` to restrict that file.

4. **Later updates** (rebuild and roll out):

   ```bash
   lk agent deploy
   ```

5. **Check status and logs**

   ```bash
   lk agent status
   lk agent logs
   ```

More detail: [Agent deployment quickstart](https://docs.livekit.io/deploy/agents/quickstart/), [Secrets](https://docs.livekit.io/deploy/agents/secrets/), [Builds and Dockerfiles](https://docs.livekit.io/deploy/agents/builds/).

### Cloud vs self-host notes

- **Interruption mode:** the code defaults to `NORA_INTERRUPTION_MODE=vad` for self-host. On Cloud, many teams set `NORA_INTERRUPTION_MODE=adaptive` [via secrets](https://docs.livekit.io/deploy/agents/secrets/)—test with your use case.
- **Lockfile:** commit `uv.lock` so `uv sync --frozen` in the Dockerfile matches your machine.

## Self-host (no Cloud)

- Use `LIVEKIT_URL=ws://...` and API key/secret from your server.
- Prefer `NORA_INTERRUPTION_MODE=vad` unless you know your stack supports adaptive interruption.

## Automated behavioral tests

`tests/test_smoke.py` uses one LLM for both the `AgentSession` and `judge()`.

- **Default:** if `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` are set, tests use **LiveKit Inference** (`livekit.agents.inference.LLM`). Set `NORA_TEST_LLM_MODEL` to an [Inference model id](https://docs.livekit.io/agents/models/llm/inference/openai/) (e.g. `openai/gpt-4.1-mini`).
- **Override:** set `NORA_TEST_PREFER_INFERENCE=false` to prefer `OPENAI_API_KEY` or `NORA_TEST_LLM_*` even when LiveKit keys exist.

```bash
# PowerShell
$env:LIVEKIT_EVALS_VERBOSE="1"; uv run pytest tests/test_smoke.py -q
```

## CLI reference

- `uv run python -m nora_agent dev` — local development
- `uv run python -m nora_agent start` — production worker (used in `Dockerfile` for Cloud)
- `uv run python -m nora_agent download-files` — download plugin assets (also runs in the image build)
