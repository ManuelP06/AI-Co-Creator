from sqlalchemy import Column, Integer, String, ForeignKey, Float, Text, JSON, Boolean, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    videos = relationship("Video", back_populates="user")

class Video(Base):
    __tablename__ = "videos"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    file_path = Column(String, unique=True, index=True)
    original_filename = Column(String)
    file_size = Column(Integer)
    transcript = Column(Text, nullable=True)
    analysis = Column(Text, nullable=True)

    storyboard_json = Column(JSON, nullable=True)
    timeline_json = Column(JSON, nullable=True)

    user = relationship("User", back_populates="videos")
    shots = relationship("Shot", back_populates="video", cascade="all, delete-orphan")

class Shot(Base):
    __tablename__ = "shots"
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"))
    shot_index = Column(Integer)
    start_frame = Column(Integer)
    end_frame = Column(Integer)
    start_time = Column(Float)
    end_time = Column(Float)
    transcript = Column(Text, nullable=True)
    analysis = Column(Text, nullable=True)
    video = relationship("Video", back_populates="shots")