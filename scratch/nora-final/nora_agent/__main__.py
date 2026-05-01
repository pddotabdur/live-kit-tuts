"""Run the Nora worker: `uv run python -m nora_agent`."""

import os

from livekit.agents import WorkerOptions, cli

from nora_agent.workflow import entrypoint


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=os.getenv("NORA_AGENT_NAME", "nora-final-agent"),
        )
    )
