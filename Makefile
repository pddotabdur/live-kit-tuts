EC2_HOST ?= ubuntu@13.49.62.84
EC2_KEY  ?= /home/abdur/Downloads/livekit-keypair.pem
EC2_SIP_DIR ?= ~/sip
DEV_USER ?= $(shell whoami)
PER_DEV_AGENT ?= outbound-caller-$(DEV_USER)

.PHONY: help sip-image sip-deploy agent-dev call call-prod ec2-restart ec2-logs

help:
	@echo "Targets:"
	@echo "  sip-image     Build livekit/sip:local from sip-local/sip and save tar to /tmp/sip-local.tar"
	@echo "  sip-deploy    Build, ship to EC2, load, and restart SIP stack"
	@echo "  agent-dev     Run the agent locally as $(PER_DEV_AGENT) (per-developer)"
	@echo "  call          Dispatch a test call to your dev agent ($(PER_DEV_AGENT))"
	@echo "  call-prod     Dispatch a test call to the production agent (outbound-caller on EC2)"
	@echo "  ec2-restart   Restart docker compose stack on EC2"
	@echo "  ec2-logs      Tail SIP container logs on EC2"
	@echo ""
	@echo "Override EC2_HOST / EC2_KEY / PHONE_NUMBER as needed."

sip-image:
	cd sip-local/sip && sudo docker build -f build/sip/Dockerfile -t livekit/sip:local .
	sudo docker save livekit/sip:local -o /tmp/sip-local.tar
	@ls -lh /tmp/sip-local.tar

sip-deploy: sip-image
	scp -i $(EC2_KEY) /tmp/sip-local.tar $(EC2_HOST):/tmp/
	ssh -i $(EC2_KEY) $(EC2_HOST) 'sudo docker load -i /tmp/sip-local.tar && cd $(EC2_SIP_DIR) && sudo docker compose down && sudo docker compose up -d && sudo docker compose ps'

agent-dev:
	AGENT_NAME=$(PER_DEV_AGENT) uv run python agent.py dev

call:
	@test -n "$(PHONE_NUMBER)" || (echo "Set PHONE_NUMBER, e.g. PHONE_NUMBER=+966... make call"; exit 1)
	AGENT_NAME=$(PER_DEV_AGENT) PHONE_NUMBER=$(PHONE_NUMBER) uv run python dispatch.py

call-prod:
	@test -n "$(PHONE_NUMBER)" || (echo "Set PHONE_NUMBER, e.g. PHONE_NUMBER=+966... make call-prod"; exit 1)
	AGENT_NAME=outbound-caller PHONE_NUMBER=$(PHONE_NUMBER) uv run python dispatch.py

ec2-restart:
	ssh -i $(EC2_KEY) $(EC2_HOST) 'cd $(EC2_SIP_DIR) && sudo docker compose down && sudo docker compose up -d && sudo docker compose ps'

ec2-logs:
	ssh -i $(EC2_KEY) $(EC2_HOST) 'cd $(EC2_SIP_DIR) && sudo docker compose logs -f --tail=100 sip'
