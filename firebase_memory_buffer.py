"""
Project Mnemosyne - Firebase Memory Buffer
Primary local memory buffer using Firestore with transaction safety and error recovery.
"""
import firebase_admin
from firebase_admin import credentials, firestore, exceptions
from google.cloud.firestore import Client as FirestoreClient
from typing import Dict, Any, Optional, List
import logging
import time
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
import hashlib

from config import get_config

@dataclass
class MemoryRecord:
    """Standardized memory record structure"""
    memory_id: str
    timestamp: float
    principle_data: bytes  # Protobuf-serialized or pickled
    vignette_data: Dict[str, Any]
    significance_score: float
    context_hash: str
    state_snapshot: Dict[str, Any]
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Firestore-serializable dictionary"""
        return {
            'memory_id': self.memory_id,
            'timestamp': self.timestamp,
            'principle_data': self.principle_data.hex(),  # Store as hex string
            'vignette_data': json.dumps(self.vignette_data),
            'significance_score': self.significance_score,
            'context_hash': self.context_hash,
            'state_snapshot': json.dumps(self.state_snapshot),
            'version': self.version,
            'created_at': firestore.SERVER_TIMESTAMP
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryRecord':
        """Create MemoryRecord from Firestore dictionary"""
        return cls(
            memory_id=data['memory_id'],
            timestamp=data['timestamp'],
            principle_data=bytes.fromhex(data['principle_data']),
            vignette_data=json.loads(data['vignette_data']),
            significance_score=data['significance_score'],
            context_hash=data['context_hash'],
            state_snapshot=json.loads(data['state_snapshot']),
            version=data.get('version', 1)
        )

class FirebaseMemoryBuffer:
    """
    Firestore-based memory buffer with transaction safety and automatic cleanup.
    Implements the Editing Memory component of the Tripartite Mind.
    """
    
    def __init__(self):
        """Initialize Firestore connection with error handling"""
        self.logger = logging.getLogger(__name__)
        self.config = get_config()
        
        try:
            # Initialize Firebase app if not already initialized
            if not firebase_admin._apps:
                firebase_config = self.config.firebase
                
                # Create credentials dictionary
                cred_dict = {
                    "type": "service_account",
                    "project_id": firebase_config.project_id,
                    "private_key_id": firebase_config.private_key_id,
                    "private_key": firebase_config.private_key,
                    "client_email": firebase_config.client_email,
                    "client_id": firebase_config.client_id,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{firebase_config.client_email}"
                }
                
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
                self.logger.info("Firebase app initialized successfully