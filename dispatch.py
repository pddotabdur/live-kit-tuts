import asyncio
import os
import uuid
import json
from livekit import api
from dotenv import load_dotenv

load_dotenv()

async def main():
    phone_number = "880554107452"
    
    # Initialize LiveKit API
    # It will automatically use LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET from the environment
    lk_api = api.LiveKitAPI()
    
    # Generate a unique room name for this call
    room_name = f"outbound-call-{uuid.uuid4().hex[:8]}"
    
    try:
        # Create an empty room
        await lk_api.room.create_room(api.CreateRoomRequest(name=room_name))
        print(f"✅ Created room: {room_name}")

        # Dispatch the agent to the room
        metadata = json.dumps({"phone_number": phone_number})
        dispatch_request = api.CreateAgentDispatchRequest(
            agent_name="outbound-caller", #outbound-caller", #nora-outbound #
            room=room_name,
            metadata=metadata
        )
        
        await lk_api.agent_dispatch.create_dispatch(dispatch_request)
        print(f"🔊 Agent securely dispatched! The agent is now joining the room and dialing {phone_number}.")
        
    except Exception as e:
        print(f"❌ Failed to dispatch agent: {e}")
    finally:
        await lk_api.aclose()

if __name__ == "__main__":
    asyncio.run(main())
