"""
Project Mnemosyne - Configuration Management
Centralized configuration with environment variable validation and type safety.
"""
import os
from dataclasses import dataclass
from typing import Optional
import logging

@dataclass
class FirebaseConfig:
    """Firebase configuration with credential validation"""
    project_id: str
    private_key_id: str
    private_key: str
    client_email: str
    client_id: str
    
    @classmethod
    def from_env(cls) -> Optional['FirebaseConfig']:
        """Load Firebase config from environment variables"""
        try:
            # Parse private key with newlines restored
            private_key = os.getenv('FIREBASE_PRIVATE_KEY', '').replace('\\n', '\n')
            
            return cls(
                project_id=os.getenv('FIREBASE_PROJECT_ID', ''),
                private_key_id=os.getenv('FIREBASE_PRIVATE_KEY_ID', ''),
                private_key=private_key,
                client_email=os.getenv('FIREBASE_CLIENT_EMAIL', ''),
                client_id=os.getenv('FIREBASE_CLIENT_ID', '')
            )
        except Exception as e:
            logging.error(f"Failed to load Firebase config: {e}")
            return None

@dataclass
class DecentralizedStorageConfig:
    """Decentralized storage configuration"""
    bundlr_network: str = "https://node1.bundlr.network"
    bundlr_currency: str = "arweave"
    ipfs_pinata_api_key: str = ""
    ipfs_pinata_secret: str = ""
    gcp_storage_bucket: str = ""

@dataclass
class MnemosyneConfig:
    """Main configuration container"""
    # Core settings
    memory_retention_days: int = 7
    heartbeat_interval_seconds: int = 30
    recovery_timeout_seconds: int = 90
    
    # Significance thresholds
    significance_threshold: float = 0.7
    min_state_delta: float = 0.01
    
    # Storage configuration
    firebase: Optional[FirebaseConfig] = None
    decentralized: DecentralizedStorageConfig = DecentralizedStorageConfig()
    
    # Performance
    batch_size: int = 100
    compression_level: int = 6
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.firebase:
            raise ValueError("Firebase configuration is required")
        
        # Validate Firebase credentials structure
        required_fields = ['project_id', 'private_key_id', 'private_key', 'client_email']
        for field in required_fields:
            if not getattr(self.firebase, field):
                raise ValueError(f"Firebase {field} is missing or empty")

# Global configuration instance
config: Optional[MnemosyneConfig] = None

def initialize_config() -> MnemosyneConfig:
    """Initialize and validate configuration from environment"""
    global config
    
    firebase_config = FirebaseConfig.from_env()
    if not firebase_config:
        raise RuntimeError("Failed to load Firebase configuration")
    
    config = MnemosyneConfig(
        firebase=firebase_config,
        decentralized=DecentralizedStorageConfig(
            ipfs_pinata_api_key=os.getenv('IPFS_PINATA_API_KEY', ''),
            ipfs_pinata_secret=os.getenv('IPFS_PINATA_SECRET', ''),
            gcp_storage_bucket=os.getenv('GCP_STORAGE_BUCKET', '')
        )
    )
    
    logging.info(f"Configuration initialized for project: {config.firebase.project_id}")
    return config

def get_config() -> MnemosyneConfig:
    """Get current configuration (initializes if needed)"""
    global config
    if config is None:
        config = initialize_config()
    return config