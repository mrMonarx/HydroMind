"""Ma'lumotlar bazasiga ulanish (SQLAlchemy + SQLite)."""
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./hydromind.db")

# SQLite uchun check_same_thread=False kerak (FastAPI bir nechta oqimda ishlaydi)
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    """Har bir so'rov uchun alohida sessiya beradi va oxirida yopadi."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
