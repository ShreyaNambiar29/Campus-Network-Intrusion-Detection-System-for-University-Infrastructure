from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings


engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator:
    """Provide a transactional SQLAlchemy session for each request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create database tables on startup if they do not exist."""
    from app.models.incident import Incident  # noqa: F401
    from app.models.threat_event import ThreatEvent  # noqa: F401
    from app.models.traffic_log import TrafficLog  # noqa: F401

    Base.metadata.create_all(bind=engine)
