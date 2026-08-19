import json
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime, timedelta

@dataclass
class Message:
    id: str
    text: str
    timestamp: datetime
    sender_id: str
    parent_id: str = None  # None for the original source message (M0)

class DataGenerator:
    """Generates synthetic message propagation chains for testing."""
    
    @staticmethod
    def generate_synthetic_chain() -> List[Message]:
        base_time = datetime.now() - timedelta(days=2)
        
        # Simulating a realistic drift from a mundane fact to a massive conspiracy
        texts = [
            "A new water treatment chemical is being tested in the downtown reservoir today.", # M0
            "I heard they are putting a new chemical in the downtown water supply today.", # M1
            "Watch out, they are dumping untested chemicals into the city water supply!", # M2
            "URGENT: Toxic chemicals are being dumped in our water! DO NOT DRINK TAP WATER!!", # M3
            "ALERT!!! Thousands poisoned by government bioweapon in the water supply! SHARE THIS BEFORE IT'S DELETED!!! 🚨🚨" # M4
        ]
        
        chain = []
        for i, text in enumerate(texts):
            msg = Message(
                id=f"M{i}",
                text=text,
                timestamp=base_time + timedelta(hours=i*3.5),
                sender_id=f"User_{100+i}",
                parent_id=f"M{i-1}" if i > 0 else None
            )
            chain.append(msg)
            
        return chain