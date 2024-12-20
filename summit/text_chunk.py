from typing import List, Optional
from dataclasses import dataclass, field
from utils import sbert_model

@dataclass
class Chunk:
    """
    Represents a chunk of text with optional parent-child relationships for tree construction.
    """
    id: str  # Unique identifier for the chunk
    text: str  # Text content of the chunk
    token_count: int  # Token count of the chunk
    start_idx: Optional[int] = None  # Start index in the document
    end_idx: Optional[int] = None  # End index in the document
    embedding: Optional[List[float]] = None  # Embedding vector
    parent: Optional["Chunk"] = None 
    children: List["Chunk"] = field(default_factory=list) 
    level: int = 0 
    entity: Optional[str] = None 

    def __post_init__(self):
        if self.text and self.embedding is None:
            self.embedding = sbert_model.encode(self.text)
