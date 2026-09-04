# utils/rag_tool.py
from langchain_core.tools import Tool


def build_nba_rag_tool(vector_store_manager, k: int = 5) -> Tool:
    """
    Construit un Tool LangChain qui interroge les archives textuelles
    (rapports d'analyse, commentaires de matchs) via recherche sémantique Faiss.

    Ce Tool est le pendant "qualitatif" du NBA_Stats_SQL Tool : c'est au LLM,
    via l'AgentExecutor, de décider lequel appeler (ou les deux) en fonction
    de la question posée.
    """

    def run_rag_search(question: str) -> str:
        results = vector_store_manager.search(question, k=k)
        if not results:
            return "Aucun élément contextuel pertinent trouvé dans les archives textuelles."

        return "\n\n---\n\n".join([
            f"Source: {res['metadata'].get('source', 'Inconnue')} "
            f"(Pertinence: {res['score']:.1f}%)\nContenu: {res['text']}"
            for res in results
        ])

    return Tool(
        name="NBA_Archives_Search",
        description=(
            "Utile pour toute question qualitative ou contextuelle : rapports d'analyse, "
            "commentaires de matchs, ressenti, contexte tactique, avis d'observateurs. "
            "Ne pas utiliser pour des statistiques chiffrées exactes "
            "(utiliser NBA_Stats_SQL pour cela)."
        ),
        func=run_rag_search,
    )