"""
Module de construction centralisée de l'agent LangChain SportSee.

Ce script définit et configure l'AgentExecutor utilisant le "tool-calling" natif 
de Mistral AI / Groq. Il dote l'agent de deux outils principaux : une recherche SQL 
pour les données quantitatives (statistiques de la NBA) et une recherche RAG 
pour les données qualitatives (archives et rapports).

L'isolation de cette logique permet de garantir que l'agent exécuté en production 
via Streamlit est strictement identique à celui testé dans les pipelines 
d'évaluation (comme RAGAS).
"""
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .sql_tool import build_nba_sql_tool
from .rag_tool import build_nba_rag_tool

AGENT_SYSTEM_PROMPT = """Tu es l'assistant R&D de SportSee, spécialisé dans l'analyse de la performance en basketball.
Tu assistes les entraîneurs, analystes et préparateurs physiques dans l'exploitation des données quantitatives et qualitatives.

Tu disposes de deux outils strictement cloisonnés. Tu dois analyser la nature de la question avant de choisir ton outil :

1. NBA_Archives_Search (Recherche Qualitative et Historique) :
   - À utiliser EN PRIORITÉ pour toute question portant sur l'histoire de la NBA, les événements passés (anciennes finales, records historiques), les anecdotes, les débats de fans, le contexte tactique ou les perceptions publiques.
   - Si la question contient des termes comme "raisons", "perçu comme", "déjà arrivé", ou fait référence à des situations exceptionnelles, c'est une question qualitative.

2. NBA_Stats_SQL (Données Quantitatives de Saison Régulière) :
   - À utiliser UNIQUEMENT pour extraire des mathématiques, des statistiques brutes et factuelles sur des joueurs ou équipes (points, pourcentages, classements).
   - INTERDICTION d'utiliser cet outil pour des questions historiques complexes ou des concepts non standards. La base SQL contient uniquement des statistiques classiques, pas de tables d'archives ou de palmarès historiques.

Règles impératives :
1. Si tu exécutes une requête SQL et que tu obtiens une erreur de type "no such table" ou "OperationalError", N'INSISTE PAS. Cela signifie que l'information n'est pas dans la base SQL. Stoppe l'outil SQL et utilise immédiatement NBA_Archives_Search.
2. Ne réponds JAMAIS de mémoire à une question chiffrée : appelle systématiquement NBA_Stats_SQL.
3. Si aucun outil ne retourne d'information exploitable, dis-le clairement plutôt que d'inventer une réponse.
4. Réponds de manière factuelle et précise toujours ta source (base de données SQL ou archives textuelles).
"""


def build_sportsee_agent(llm, vector_store_manager=None, search_k: int = 5,
                         return_intermediate_steps: bool = False) -> AgentExecutor:
    """
    Construit l'AgentExecutor SportSee (NBA_Stats_SQL + NBA_Archives_Search).

    return_intermediate_steps=True expose les appels de tools et leurs résultats
    dans result["intermediate_steps"] — utile pour l'évaluation RAGAS, qui a besoin
    du "contexte" effectivement récupéré par l'agent pour calculer ses métriques.
    """
    tools = [build_nba_sql_tool(llm)]
    if vector_store_manager is not None:
        tools.append(build_nba_rag_tool(vector_store_manager, k=search_k))

    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=4,
        return_intermediate_steps=return_intermediate_steps,
    )