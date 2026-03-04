from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    app_name: str = "Campus IDS Backend"
    environment: str = "development"
    debug: bool = False
    api_prefix: str = "/api"

    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/campus_ids"
    auto_create_schema: bool = False

    cors_origins: List[str] | str = ["*"]

    model_path: str = "app/ml/iforest_model.joblib"

    enable_packet_sniffer: bool = True
    simulation_mode: bool = True
    sniffer_interface: str = ""
    packet_buffer_size: int = 500
    feature_window_seconds: int = 15

    port_scan_threshold: int = 8
    brute_force_threshold: int = 10
    packet_burst_threshold: int = 50

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: List[str] | str) -> List[str]:
        if isinstance(value, str):
            if value.strip() == "*":
                return ["*"]
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
