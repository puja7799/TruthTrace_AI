import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_model import Message

class TwitterTreeParser:
    """Parses nested social media threads (like PHEME/Twitter15) into Message chains."""
    
    @staticmethod
    def parse_thread(root_tweet: Dict, replies: List[Dict]) -> List[Message]:
        """Converts a raw JSON thread into a flat list of Message objects."""
        messages = []
        
        # 1. Parse the Root Tweet (M0)
        messages.append(Message(
            id=root_tweet['id_str'],
            text=root_tweet['text'],
            timestamp=datetime.strptime(root_tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y'),
            sender_id=root_tweet['user']['screen_name'],
            parent_id=None
        ))
        
        # 2. Parse Replies
        # Sort replies by time to ensure chronological propagation
        replies_sorted = sorted(
            replies, 
            key=lambda x: datetime.strptime(x['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
        )
        
        for reply in replies_sorted:
            messages.append(Message(
                id=reply['id_str'],
                text=reply['text'],
                timestamp=datetime.strptime(reply['created_at'], '%a %b %d %H:%M:%S +0000 %Y'),
                sender_id=reply['user']['screen_name'],
                parent_id=reply['in_reply_to_status_id_str']
            ))
            
        return messages

    @staticmethod
    def get_pheme_mock_data() -> tuple[Dict, List[Dict]]:
        """Provides a realistic mock of a PHEME dataset Twitter thread."""
        root = {
            "id_str": "10001",
            "text": "BREAKING: Armed police sealing off streets in downtown Sydney following reports of a hostage situation in a cafe.",
            "created_at": "Mon Dec 15 00:01:00 +0000 2014",
            "user": {"screen_name": "NewsDesk"}
        }
        replies = [
            {
                "id_str": "10002", "in_reply_to_status_id_str": "10001",
                "text": "Are there bombs? Hearing rumors of multiple explosive devices across the city!!",
                "created_at": "Mon Dec 15 00:05:00 +0000 2014", "user": {"screen_name": "PanickedLocal"}
            },
            {
                "id_str": "10003", "in_reply_to_status_id_str": "10002",
                "text": "OMG my friend said there are 4 bombs planted right now! STAY AWAY FROM TRAINS! 🚨",
                "created_at": "Mon Dec 15 00:08:00 +0000 2014", "user": {"screen_name": "User443"}
            },
            {
                "id_str": "10004", "in_reply_to_status_id_str": "10001",
                "text": "Stay safe everyone, waiting for official police confirmation.",
                "created_at": "Mon Dec 15 00:06:00 +0000 2014", "user": {"screen_name": "RationalObserver"}
            }
        ]
        return root, replies