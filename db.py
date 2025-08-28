# db.py
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import config

Base = declarative_base()
_engine = None
_Session = None

class Resume(Base):
    __tablename__ = 'resumes'
    id = Column(Integer, primary_key=True)
    uid = Column(String, unique=True, index=True)
    json = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class RunResult(Base):
    __tablename__ = 'runs'
    id = Column(Integer, primary_key=True)
    run_name = Column(String, index=True)
    meta = Column(JSON)                     # ✅ renamed from metadata → meta
    results = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

def init_db():
    global _engine, _Session
    if config.DB_MODE == "postgres":
        if not config.POSTGRES_URL:
            raise RuntimeError("POSTGRES_URL must be set for postgres mode")
        _engine = create_engine(config.POSTGRES_URL)
    else:
        _engine = create_engine("sqlite:///rt_test.db", connect_args={"check_same_thread": False})
    _Session = sessionmaker(bind=_engine)
    Base.metadata.create_all(_engine)

def get_session():
    global _Session
    if _Session is None:
        init_db()
    return _Session()

def save_resume(uid, json_payload):
    s = get_session()
    r = Resume(uid=uid, json=json_payload)
    s.add(r)
    s.commit()
    s.refresh(r)
    return r

def save_run_result(run_name, metadata, results):
    """Save experiment/run results"""
    s = get_session()
    rr = RunResult(run_name=run_name, meta=metadata, results=results)  # ✅ use meta
    s.add(rr)
    s.commit()
    s.refresh(rr)
    return rr
