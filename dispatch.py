import asyncio
import os
import uuid
import json
from livekit import api
from dotenv import load_dotenv

load_dotenv()

async def main():
    phone_number = os.getenv("PHONE_NUMBER")
    if not phone_number:
        raise SystemExit("PHONE_NUMBER env var is required (e.g. PHONE_NUMBER=+966555209485)")

    agent_name = os.getenv("AGENT_NAME", "outbound-caller")

    lk_api = api.LiveKitAPI()
    room_name = f"outbound-call-{uuid.uuid4().hex[:8]}"

    try:
        await lk_api.room.create_room(api.CreateRoomRequest(name=room_name))
        print(f"Created room: {room_name}")

        metadata = json.dumps({"phone_number": phone_number})
        dispatch_request = api.CreateAgentDispatchRequest(
            agent_name=agent_name,
            room=room_name,
            metadata=metadata
        )
        
        await lk_api.agent_dispatch.create_dispatch(dispatch_request)
        print(f"Dispatched agent '{agent_name}' to room '{room_name}', dialing {phone_number}.")

    except Exception as e:
        print(f"Failed to dispatch agent: {e}")
    finally:
        await lk_api.aclose()

if __name__ == "__main__":
    asyncio.run(main())
