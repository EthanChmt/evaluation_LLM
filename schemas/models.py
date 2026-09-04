from pydantic import BaseModel, Field
from typing import Optional

class ChunkMetadata(BaseModel):
    """Modèle définissant les métadonnées obligatoires pour chaque chunk de texte."""
    source: str = Field(..., description="Nom du fichier source ou origine de la donnée")
    page_or_row: Optional[str] = Field(default=None, description="Emplacement exact (page PDF, ligne Excel)")

class ValidatedChunk(BaseModel):
    """Modèle strict pour s'assurer qu'aucun chunk corrompu ne rentre dans l'index."""
    text: str = Field(..., min_length=15, description="Le texte du chunk doit être suffisamment long pour avoir du sens.")
    metadata: dict = Field(..., description="Les métadonnées associées au document source.")


class PlayerModel(BaseModel):
    name: str = Field(..., min_length=2)
    team: str
    age: int

class SeasonStatModel(BaseModel):
    gp: int
    w: int
    l: int
    min: float
    pts: int
    fgm: int
    fga: int
    fg_pct: float
    three_pm: int
    three_pa: int
    three_p_pct: float
    ftm: int
    fta: int
    ft_pct: float
    oreb: int
    dreb: int
    reb: int
    ast: int
    tov: int
    stl: int
    blk: int
    pf: int
    fp: int
    dd2: int
    td3: int
    plus_minus: float
    offrtg: float
    defrtg: float
    netrtg: float
    ast_pct: float
    ast_to: float
    ast_ratio: float
    oreb_pct: float
    dreb_pct: float
    reb_pct: float
    to_ratio: float
    efg_pct: float
    ts_pct: float
    usg_pct: float
    pace: float
    pie: float
    poss: int