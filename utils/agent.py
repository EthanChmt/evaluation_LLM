# utils/agent.py
"""
Construction centralisée de l'agent SportSee (tool-calling Mistral).

Ce module est partagé entre MistralChat.py (application Streamlit) et
evaluate_ragas.py (script d'évaluation). Objectif : garantir qu'on évalue
exactement le même agent que celui utilisé en production, et éviter la
divergence qu'on avait entre sql_tool.py et sql_search.py.
"""
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .sql_tool import build_nba_sql_tool
from .rag_tool import build_nba_rag_tool

AGENT_SYSTEM_PROMPT = """Tu es l'assistant R&D de SportSee, spécialisé dans l'analyse de la performance en basketball.
Tu assistes les entraîneurs, analystes et préparateurs physiques dans l'exploitation des données quantitatives et qualitatives.

Tu disposes de deux outils :
- NBA_Stats_SQL : pour toute question chiffrée/statistique précise sur un joueur ou une équipe
  (points, rebonds, pourcentages, matchs joués, comparaisons de statistiques...).
- NBA_Archives_Search : pour toute question qualitative ou contextuelle
  (rapports d'analyse, commentaires de matchs, contexte tactique, ressenti).

Règles impératives :
1. Analyse la question et choisis toi-même le ou les outils pertinents. Une question peut nécessiter les deux
   (ex: "Compare les stats de X à ce qu'en disent les commentaires de match").
2. Ne réponds JAMAIS de mémoire à une question chiffrée : appelle systématiquement NBA_Stats_SQL.
3. Si aucun outil ne retourne d'information exploitable, dis-le clairement plutôt que d'inventer une réponse.
4. Réponds de manière factuelle, technique et rigoureuse, et précise la source des données utilisées
   (base SQL et/ou archives textuelles).
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