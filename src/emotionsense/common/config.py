"""Application settings via Pydantic Settings (defaults -> env; env wins).

Only *app/runtime* config lives here. ML experiment config (datasets, models,
hyperparameters) is declarative YAML under ``configs/`` and loaded separately by the
training plane — see ``emotionsense.common.yaml_config``.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. All keys are ESA_-prefixed environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="ESA_", env_file=".env", extra="ignore", case_sensitive=False
    )

    env: str = "local"
    log_level: str = "INFO"

    # Database
    db_url: str = "postgresql+psycopg2://esa:esa@localhost:5432/emotionsense"

    # Object storage
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "emotionsense"
    s3_region: str = "us-east-1"

    # Experiment tracking
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Auth
    jwt_secret: str = "change-me-in-prod"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    # Admin account is provisioned from the environment only — no credentials are baked
    # into the code. Store a bcrypt hash (generate with emotionsense.backend.security.
    # hash_password); if unset, no user can authenticate (secure by default).
    admin_email: str = "admin@emotionsense.ai"
    admin_password_hash: str | None = None

    # Serving
    default_model: str = "svm-ravdess:v1"
    max_upload_mb: int = Field(default=10, ge=1, le=100)
    max_duration_sec: int = Field(default=30, ge=1, le=300)
    model_cache_size: int = Field(default=3, ge=1, le=32)

    # Frontend
    api_base_url: str = "http://localhost:8000/api/v1"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""
    return Settings()
