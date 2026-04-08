import asyncio
import os
import uuid
import json
from livekit import api
from dotenv import load_dotenv

load_dotenv()

async def main():
    phone_number = "+966554107452"
    
    lk_api = api.LiveKitAPI()
    
    room_name = f"outbound-call-{uuid.uuid4().hex[:8]}"
    
    try:
        await lk_api.room.create_room(api.CreateRoomRequest(name=room_name))
        print(f"✅ Created room: {room_name}")

        metadata = json.dumps({"phone_number": phone_number})
        dispatch_request = api.CreateAgentDispatchRequest(
            agent_name="hamsa-caller",
            room=room_name,
            metadata=metadata
        )
        
        await lk_api.agent_dispatch.create_dispatch(dispatch_request)
        print(f"🔊 Hamsa Agent securely dispatched! The agent is now joining the room and dialing {phone_number}.")
        
    except Exception as e:
        print(f"❌ Failed to dispatch agent: {e}")
    finally:
        await lk_api.aclose()

if __name__ == "__main__":
    asyncio.run(main())
