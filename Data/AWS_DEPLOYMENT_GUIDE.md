# LiveKit + SIP Server: AWS EC2 Deployment Guide

This guide covers setting up a self-hosted LiveKit instance with a SIP server on an AWS EC2 instance, connected to your company's SIP infrastructure via a secure Site-to-Site VPN.

## Table of Contents
1. [Important Considerations](#important-considerations)
2. [Phase 1: Local Testing Workflow](#phase-1-local-testing-workflow)
3. [Phase 2: AWS Infrastructure Preparation](#phase-2-aws-infrastructure-preparation)
4. [Phase 3: AWS Site-to-Site VPN Setup](#phase-3-aws-site-to-site-vpn-setup)
5. [Phase 4: LiveKit Deployment on EC2](#phase-4-livekit-deployment-on-ec2)
6. [Phase 5: Agent Python Configuration](#phase-5-agent-python-configuration)

---

## Important Considerations

### ⚠️ The AWS "Free Tier" Caveat
You mentioned managing the EC2 deployment within the AWS Free Tier (usually a `t2.micro` or `t3.micro` instance with 1 vCPU and 1GB of RAM).
- **Will it work?** Yes, you can install and run the LiveKit Server + SIP Docker containers on a free tier instance.
- **The limitation:** Real-time audio and video processing requires decent CPU limits. 1GB of RAM is very restrictive for both LiveKit and the SIP gateway running alongside it.
- **For Testing:** It is perfectly fine for testing 1 or 2 concurrent calls. If you start noticing audio jitter, heavy latency, or the instance crashing (Out Of Memory), you will need to upgrade to a `t3.medium` (2 vCPUs, 4GB RAM) for production.

---

## Phase 1: Local Testing Workflow
Before provisioning AWS infrastructure, it is critical to test your logic locally to ensure your agent can answer SIP calls.

1. **Install LiveKit Server** on your local machine:
   ```bash
   curl -sSL https://get.livekit.io | bash
   ```
   *(If you also want the CLI tools in the future, that is `curl -sSL https://get.livekit.io/cli | bash`)*

2. **Start Local LiveKit (dev mode)** in a terminal window:
   ```bash
   livekit-server --dev
   ```
   *This starts the server on `ws://127.0.0.1:7880` with API Key: `devkey` and Secret: `secret`.*

3. **Start the Local SIP Server** using Docker Compose. Create a `docker-compose.yaml` locally:
   ```bash
   mkdir sip-local && cd sip-local
   wget https://raw.githubusercontent.com/livekit/sip/main/docker-compose.yaml
   docker compose up
   ```
   *(Ensure you change the `api_key` and `api_secret` in the resulting SIP config file to `devkey` and `secret` respectively).*

4. **Test with a Local Softphone**: 
   Download a free softphone like **MicroSIP** (Windows) or **Linphone** (Linux/Mac). Configure it to dial `sip:1234@127.0.0.1:5060`. Your local Python script can now be pointed to the local LiveKit instance to receive the calls!

---

## Phase 2: AWS Infrastructure Preparation

### Step 1: Create a VPC & Subnets
1. Log into the AWS Console and navigate to **VPC**.
2. Click **Create VPC** -> "VPC and more". Provide a name tag (e.g., `livekit-vpc`).
3. Ensure you have at least 1 public subnet created.

### Step 2: Provision an EC2 Instance
1. Go to **EC2** -> **Launch Instances**.
2. **Name**: `LiveKit-Server`
3. **OS**: Select **Ubuntu Server 22.04 LTS (AMI)**.
4. **Instance Type**: Select `t2.micro` or `t3.micro` (Free tier eligible).
5. **Key Pair**: Create a new Key Pair (to SSH into the machine).
6. **Network Settings**: Choose your created VPC and a Public Subnet. Ensure **Auto-assign public IP** is set to **Enable**.

### Step 3: Configure Security Groups
In your EC2 instance's Security Group, add these **Inbound Rules**:

| Type | Protocol | Port Range | Source | Reason |
| :--- | :--- | :--- | :--- | :--- |
| SSH | TCP | 22 | Your IP | To SSH into the box |
| HTTP | TCP | 80 | 0.0.0.0/0 | Let's Encrypt TLS issuance |
| HTTPS | TCP | 443 | 0.0.0.0/0 | Primary HTTPS and TURN/TLS |
| Custom TCP | TCP | 7881 | 0.0.0.0/0 | WebRTC over TCP |
| Custom UDP | UDP | 3478 | 0.0.0.0/0 | TURN over UDP |
| Custom UDP | UDP | 50000 - 60000 | 0.0.0.0/0 | WebRTC Media Traffic |
| Custom UDP | UDP | 5060 | Company IP / VPC | SIP Signaling |
| Custom TCP | TCP | 5060 | Company IP / VPC | SIP Signaling |
| Custom UDP | UDP | 10000 - 20000 | Company IP / VPC | SIP RTP (Audio traffic) |

*(Note: For security, restrict the 5060 and 10000-20000 ports strictly to your Company's Private network IP ranges sent across the VPN).*

---

## Phase 3: AWS Site-to-Site VPN Setup

Your IT team mentioned they have a hardware VPN that supports multiple protocols. AWS Site-to-Site VPN uses IPsec.

1. **Create a Customer Gateway (CGW):**
   - Navigate to **VPC > Customer Gateways** -> Create.
   - Put in the **Public IP** of your company's IT VPN hardware.
   
2. **Create a Virtual Private Gateway (VGW):**
   - Navigate to **VPC > Virtual Private Gateways** -> Create.
   - Attach it to your `livekit-vpc`.
   
3. **Configure Route Propagation:**
   - Go to **VPC > Route Tables**, select the route table associated with your public subnet.
   - Go to "Route Propagation" and edit it to enable propagating routes from the VGW.
   
4. **Establish the VPN Connection:**
   - Navigate to **VPC > Site-to-Site VPN Connections** -> Create.
   - Target Gateway Type: Virtual Private Gateway (select yours).
   - Customer Gateway: Existing (select yours).
   - Routing: **Static** or **Dynamic (BGP)**. *(Ask your IT guy which preferred, static requires you to enter your company's private IP subnet block, BGP exchanges this automatically).*
   - Once created, click **Download Configuration**. Send this file to your IT guy; it contains the IPsec configs for their hardware router (Cisco, Fortinet, etc.).

---

## Phase 4: LiveKit Deployment on EC2

Now we will install LiveKit and standard SIP on your EC2 box.

### Step 1: Assign an Elastic IP & DNS
1. In EC2, go to **Elastic IPs** and allocate a new one. Associate it with your `LiveKit-Server` instance.
2. In your DNS Provider (Route53, GoDaddy, etc.), create an A Record for your domain (e.g., `livekit.yourdomain.com`) pointing to the Elastic IP limit.

### Step 2: Generate config
On your **local machine**, run LiveKit's generate docker image:
```bash
docker run --rm -it -v$PWD:/output livekit/generate
```
Follow the prompts: Provide your domain `livekit.yourdomain.com` and accept generating TLS via Let's Encrypt. A folder will be created on your local PC.

### Step 3: Install on EC2
1. SSH into your EC2 Instance using the Key Pair you downloaded:
   ```bash
   ssh -i yourkey.pem ubuntu@<your-elastic-ip>
   ```
2. Open a new terminal locally and securely copy (`scp`) the generated folder up to the server.
3. SSH back to the server, navigate into the generated directory, and run:
   ```bash
   sudo chmod +x init_script.sh
   sudo ./init_script.sh
   ```
   *(This script will install Docker, copy configs to `/opt/livekit/`, and start the LiveKit system service).*

### Step 4: Add the SIP Server container
We must run the SIP server along with LiveKit.
1. Create a config file for the SIP Server on the EC2 machine (`/opt/livekit/sip.yaml`):
```yaml
api_key: <From your livekit.yaml>
api_secret: <From your livekit.yaml>
ws_url: ws://localhost:7880
redis:
  address: localhost:6379
sip_port: 5060
rtp_port: 10000-20000
use_external_ip: true
logging:
  level: debug
```

2. Edit `/opt/livekit/docker-compose.yaml` to include the SIP server:
```yaml
services:
  # ... (Existing LiveKit services)
  livekit-sip:
    image: livekit/sip
    command: --config /config/sip.yaml
    network_mode: "host"
    volumes:
      - ./sip.yaml:/config/sip.yaml
```

3. Restart the services:
```bash
sudo systemctl restart livekit-docker
```

---

## Phase 5: Agent Python Configuration

When deploying the Python codebase locally or on a separate machine, configure your `.env`:

```env
# Point to your newly deployed cloud URL
LIVEKIT_URL=wss://livekit.yourdomain.com
LIVEKIT_API_KEY=<key_generated_in_livekit_yaml>
LIVEKIT_API_SECRET=<secret_generated_in_livekit_yaml>

# Make sure this has NOT changed if routing inbound or using specific custom SIP instructions.
SIP_OUTBOUND_TRUNK_ID=...
```

### Making Outbound Calls
In your python dispatch logic (`agent_bank_demo.py`):
```python
await ctx.api.sip.create_sip_participant(
    api.CreateSIPParticipantRequest(
        room_name=ctx.room.name,
        sip_trunk_id=outbound_trunk_id,
        sip_call_to="<COMPANY_EXTENSION>", # Over the VPN, this routes to your internal company PBX!
        participant_identity=participant_identity,
    )
)
```

Because your LiveKit EC2 instance is connected via an AWS site-to-site VPN, dialing an internal company extension directly routing through the VPC into their SIP system is completely secure and never touches the public internet.
