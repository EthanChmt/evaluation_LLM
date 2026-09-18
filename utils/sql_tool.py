"""
Outil LangChain d'interrogation de base de données relationnelle (Text-to-SQL).

Ce module permet au LLM de traduire une requête en langage naturel en commande SQL, 
de l'exécuter sur la base de données SQLite contenant les statistiques NBA, 
et d'en extraire le résultat. 

Il intègre un système de prompt "Few-Shot" avec des exemples spécifiques 
au domaine du basketball pour orienter la génération de la requête, 
ainsi qu'une traçabilité Logfire pour monitorer les erreurs de syntaxe et d'exécution.
"""


import os
import logging
import logfire
from langchain_core.tools import Tool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_core.prompts import PromptTemplate

# Configuration du chemin relatif vers la base SQLite
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, '..', 'vector_db', 'nba_stats.db')


def build_nba_sql_tool(llm):
    """
    Construit et retourne un Tool LangChain capable de traduire
    une question naturelle en SQL, de l'exécuter et de retourner le résultat.
    """
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

    # Template Few-Shot adapté aux exigences de LangChain
    few_shot_template = """Tu es un expert en bases de données SQLite pour la NBA. 
Génère UNIQUEMENT la requête SQL pour répondre à la question. Ne renvoie aucun autre texte.
Utilise LIMIT {top_k} si tu as besoin de limiter les résultats.

Voici le schéma des tables :
{table_info}

Exemples :
Question: Quel joueur a marqué le plus de points ?
SQLQuery: SELECT p.name, s.pts FROM players p JOIN season_stats s ON p.id = s.player_id ORDER BY s.pts DESC LIMIT 1;

Question: Quel est le pourcentage à 3 points (three_p_pct) de Nikola Jokić ?
SQLQuery: SELECT s.three_p_pct FROM players p JOIN season_stats s ON p.id = s.player_id WHERE p.name = 'Nikola Jokić';

Question: Compare les points par match de LeBron James et Stephen Curry.
SQLQuery: SELECT p.name, s.pts FROM players p JOIN season_stats s ON p.id = s.player_id WHERE p.name IN ('LeBron James', 'Stephen Curry');

Question: Quelle est la moyenne de rebonds (reb) des joueurs des Boston Celtics ?
SQLQuery: SELECT AVG(s.reb) FROM players p JOIN season_stats s ON p.id = s.player_id WHERE p.team = 'Boston Celtics';

Question: Quels sont les 5 meilleurs passeurs (ast) toutes équipes confondues ?
SQLQuery: SELECT p.name, p.team, s.ast FROM players p JOIN season_stats s ON p.id = s.player_id ORDER BY s.ast DESC LIMIT 5;

Question: Quelle équipe a le meilleur pourcentage à 3 points (three_p_pct) en moyenne ?
SQLQuery: SELECT p.team, AVG(s.three_p_pct) AS avg_3p FROM players p JOIN season_stats s ON p.id = s.player_id GROUP BY p.team ORDER BY avg_3p DESC LIMIT 1;

Question: Combien de joueurs de plus de 30 ans ont un usage rate (usg_pct) supérieur à 25% ?
SQLQuery: SELECT COUNT(*) FROM players p JOIN season_stats s ON p.id = s.player_id WHERE p.age > 30 AND s.usg_pct > 25;

Question: {input}
SQLQuery:"""

    prompt = PromptTemplate.from_template(few_shot_template)
    query_chain = create_sql_query_chain(llm, db, prompt=prompt)
    execute_tool = QuerySQLDatabaseTool(db=db)

    @logfire.instrument("NBA_Stats_SQL - Génération et exécution")
    def run_sql_query(question: str) -> str:
        # 1. Génération de la requête SQL via le LLM
        try:
            sql_query = query_chain.invoke({"question": question})
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
        except Exception as e:
            logging.exception(f"Erreur lors de la génération SQL pour la question: '{question}'")
            logfire.error(
                "Échec de la génération SQL (NL -> SQL)",
                question=question,
                error=str(e),
            )
            return (
                "SQL_QUERY: (échec de génération)\n"
                "RESULT: Erreur technique : impossible de traduire la question en requête SQL. "
                "Reformule la question ou utilise NBA_Archives_Search si elle est qualitative."
            )

        # On journalise la requête générée dès qu'on l'a, indépendamment du succès de son
        # exécution : ça permet d'auditer séparément "le LLM a-t-il généré une requête
        # plausible" et "cette requête s'est-elle exécutée correctement" (cf. eval/evaluate_ragas.py).
        # Avant ce patch, seuls les cas d'échec loggaient sql_query, et jamais les succès.
        logfire.info(
            "Requête SQL générée par le LLM",
            question=question,
            sql_query=sql_query,
        )

        # 2. Exécution de la requête générée
        try:
            result = execute_tool.invoke(sql_query)
        except Exception as e:
            logging.exception(f"Erreur lors de l'exécution SQL. Requête générée: {sql_query}")
            logfire.error(
                "Échec de l'exécution SQL",
                question=question,
                sql_query=sql_query,
                error=str(e),
            )
            return (
                f"SQL_QUERY: {sql_query}\n"
                f"RESULT: Erreur technique : la requête SQL générée n'a pas pu être exécutée ({e}). "
                "La base ne contient probablement pas cette information, ou la question est mal formulée."
            )

        # 3. Résultat vide (question valide mais rien trouvé)
        if not result or result in ("[]", "()"):
            logging.info(f"Requête SQL exécutée sans résultat pour: '{question}' -> {sql_query}")
            return f"SQL_QUERY: {sql_query}\nRESULT: Aucun résultat trouvé dans la base SQL pour cette question."

        # NB : la réponse est désormais préfixée par "SQL_QUERY: ...\nRESULT: ..." dans
        # tous les cas (succès et erreurs) plutôt que de renvoyer 'result' seul. Deux
        # bénéfices : (1) eval/evaluate_ragas.py peut extraire la requête générée pour
        # l'évaluer indépendamment de la réponse finale en langage naturel ; (2) l'agent
        # dispose de la requête SQL exacte pour citer sa source (cf. règle 4 du prompt).
        return f"SQL_QUERY: {sql_query}\nRESULT: {result}"

    return Tool(
        name="NBA_Stats_SQL",
        description="Utile UNIQUEMENT pour extraire des statistiques chiffrées exactes sur les joueurs (points, pourcentages, matchs joués...). Ne pas utiliser pour des questions d'opinion ou qualitatives.",
        func=run_sql_query
    )