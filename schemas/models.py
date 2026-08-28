from pydantic import BaseModel, Field
from typing import Optional

class ChunkMetadata(BaseModel):
    """Modèle définissant les métadonnées obligatoires pour chaque chunk de texte."""
    source: str = Field(..., description="Nom du fichier source ou origine de la donnée")
    page_or_row: Optional[str] = Field(default=None, description="Emplacement exact (page PDF, ligne Excel)")

class ValidatedChunk(BaseModel):
    """Modèle principal garantissant l'intégrité d'un chunk avant son embedding."""
    text: str = Field(..., min_length=10, description="Contenu textuel nettoyé du chunk (doit faire au moins 10 caractères)")
    metadata: ChunkMetadata