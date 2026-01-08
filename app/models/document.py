from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer)  # in bytes
    file_type = Column(String)  # pdf, doc, txt
    title = Column(String)
    description = Column(Text)
    category = Column(String)  # "leave_policy", "wfh", "holidays", etc.
    
    # Metadata
    uploader_id = Column(Integer, ForeignKey("users.id"))
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationships
    uploader = relationship("User")
    
    def __repr__(self):
        return f"<Document {self.filename}>"