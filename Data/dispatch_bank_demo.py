"""
Dispatch script for the Bank AI Debt Collection Demo.

Usage:
    python dispatch_bank_demo.py                  # Dispatches debtor_001 (cooperative)
    python dispatch_bank_demo.py --debtor 2       # Dispatches debtor_002 (avoidant)
    python dispatch_bank_demo.py --debtor 3       # Dispatches debtor_003 (distressed)
    python dispatch_bank_demo.py --list           # Lists all available debtors
"""

import asyncio
import argparse
import json
import os
import uuid
from pathlib import Path

from livekit import api
from dotenv import load_dotenv

load_dotenv()

SAMPLE_DEBTORS_PATH = Path(__file__).parent / "sample_debtors.json"


def load_debtors() -> list[dict]:
    """Load sample debtor profiles from JSON file."""
    with open(SAMPLE_DEBTORS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def print_debtor_table(debtors: list[dict]):
    """Print a nice table of available debtors."""
    print("\n" + "=" * 80)
    print("  Available Demo Debtors")
    print("=" * 80)
    for i, d in enumerate(debtors, 1):
        segment_emoji = {
            "cooperative": "🟢",
            "avoidant": "🟡",
            "distressed": "🔴",
            "hostile": "⚫",
        }.get(d["behavioral_segment"], "⚪")

        print(f"\n  [{i}] {segment_emoji} {d['name']} ({d['name_en']})")
        print(f"      Segment:  {d['behavioral_segment']}")
        print(f"      Amount:   {d['amount']:,} {d['currency']}")
        print(f"      Product:  {d['product_type']}")
        print(f"      Status:   {d['service_status']}")
        print(f"      Attempts: {d['contact_attempts']}")
        print(f"      Phone:    {d['phone_number']}")
        print(f"      Notes:    {d['notes']}")
    print("\n" + "=" * 80 + "\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Dispatch the Bank AI Collection Agent to call a debtor"
    )
    parser.add_argument(
        "--debtor", "-d",
        type=int,
        default=1,
        help="Debtor number to call (1, 2, or 3). Default: 1 (cooperative)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available debtors and exit"
    )
    parser.add_argument(
        "--phone", "-p",
        type=str,
        default=None,
        help="Override the phone number from the debtor profile"
    )
    args = parser.parse_args()

    debtors = load_debtors()

    if args.list:
        print_debtor_table(debtors)
        return

    if args.debtor < 1 or args.debtor > len(debtors):
        print(f"❌ Invalid debtor number: {args.debtor}. Choose 1-{len(debtors)}.")
        print_debtor_table(debtors)
        return

    debtor = debtors[args.debtor - 1]

    # Override phone if specified
    if args.phone:
        debtor["phone_number"] = args.phone

    phone_number = debtor["phone_number"]

    segment_emoji = {
        "cooperative": "🟢",
        "avoidant": "🟡",
        "distressed": "🔴",
        "hostile": "⚫",
    }.get(debtor["behavioral_segment"], "⚪")

    print(f"\n{segment_emoji} Dispatching call to: {debtor['name']} ({debtor['name_en']})")
    print(f"   Phone:    {phone_number}")
    print(f"   Segment:  {debtor['behavioral_segment']}")
    print(f"   Amount:   {debtor['amount']:,} {debtor['currency']}")
    print(f"   Product:  {debtor['product_type']}")
    print()

    # Initialize LiveKit API
    lk_api = api.LiveKitAPI()

    room_name = f"bank-collection-{uuid.uuid4().hex[:8]}"

    try:
        # Create room
        await lk_api.room.create_room(api.CreateRoomRequest(name=room_name))
        print(f"✅ Created room: {room_name}")

        # Dispatch with full debtor profile as metadata
        metadata = json.dumps(debtor, ensure_ascii=False)
        dispatch_request = api.CreateAgentDispatchRequest(
            agent_name="bank-collection-agent",
            room=room_name,
            metadata=metadata,
        )

        await lk_api.agent_dispatch.create_dispatch(dispatch_request)
        print(f"📞 Agent dispatched! Calling {debtor['name']} at {phone_number}...")
        print(f"   Agent persona will adapt to '{debtor['behavioral_segment']}' segment.")
        print(f"\n   Monitor the agent terminal for live interaction logs.\n")

    except Exception as e:
        print(f"❌ Failed to dispatch agent: {e}")
    finally:
        await lk_api.aclose()


if __name__ == "__main__":
    asyncio.run(main())
