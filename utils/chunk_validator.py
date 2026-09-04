# utils/chunk_validator.py
"""
Couche de validation "sémantique" des chunks avant embedding, basée sur Pydantic AI.

Différence avec ValidatedChunk (models.py, Pydantic classique) :
- ValidatedChunk vérifie des contraintes STRUCTURELLES (longueur minimale, présence
  de métadonnées) — c'est rapide, gratuit, mais aveugle au CONTENU.
- Ce module utilise un Agent Pydantic AI (LLM + schéma de sortie garanti) pour juger
  la qualité sémantique du texte : OCR corrompu, contenu hors-sujet, chunk tronqué
  sans information exploitable, etc. C'est plus coûteux (1 appel LLM par chunk),
  donc à activer sélectivement (voir ENABLE_AI_CHUNK_VALIDATION dans config.py).
"""
import logging
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .config import MODEL_NAME


class ChunkQualityReport(BaseModel):
    """Sortie structurée et garantie par Pydantic AI : verdict + justification + texte nettoyé."""
    is_usable: bool = Field(
        ..., description="True si le chunk est exploitable pour le RAG (texte cohérent et informatif)."
    )
    reason: str = Field(
        ..., description="Justification courte du verdict (ex: 'texte OCR illisible', 'contenu vide de sens')."
    )
    cleaned_text: Optional[str] = Field(
        default=None,
        description="Version légèrement nettoyée du texte (espaces, artefacts OCR) si is_usable=True, sinon None."
    )


_quality_agent = Agent(
    f"mistral:{MODEL_NAME}",
    output_type=ChunkQualityReport,
    system_prompt=(
        "Tu évalues des extraits de rapports d'analyse sportive ou de commentaires de matchs NBA "
        "avant leur indexation dans une base de connaissances. "
        "Rejette (is_usable=False) tout texte illisible, corrompu par de l'OCR, vide de sens, "
        "ou trop générique pour être utile à un entraîneur ou analyste. "
        "Nettoie légèrement les artefacts évidents (espaces multiples, caractères parasites) "
        "sans reformuler ni résumer le contenu."
    ),
)


def validate_chunk_quality(raw_text: str) -> ChunkQualityReport:
    """
    Fait juger un chunk par l'agent Pydantic AI et retourne un rapport structuré.

    Fail-open volontaire : si l'agent échoue (API indisponible, timeout...), le chunk
    est accepté par défaut plutôt que de bloquer toute l'indexation. La validation
    structurelle (ValidatedChunk, longueur minimale) reste le filet de sécurité final.
    """
    try:
        result = _quality_agent.run_sync(raw_text)
        return result.output
    except Exception as e:
        logging.warning(f"Échec de la validation qualité (Pydantic AI), chunk accepté par défaut: {e}")
        return ChunkQualityReport(
            is_usable=True,
            reason="Validation IA indisponible (fail-open).",
            cleaned_text=raw_text,
        )