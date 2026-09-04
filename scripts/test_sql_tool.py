import os
import sys
import logfire
from dotenv import load_dotenv

# Configuration des chemins
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(PROJECT_ROOT)
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

from utils.sql_tool import build_nba_sql_tool
from langchain_mistralai.chat_models import ChatMistralAI 

logfire.configure()

# Récupération dynamique du nom du modèle depuis le .env
model_id = os.getenv("MODEL_ID", "mistral-small-latest")
llm = ChatMistralAI(model=model_id, temperature=0)

# 1. Instanciation de l'outil
nba_sql_tool = build_nba_sql_tool(llm)

print("--- Test du Tool SQL LangChain ---")
question = "Combien de points a marqué Shai Gilgeous-Alexander ?"
print(f"Question posée : {question}")

# 2. Exécution
resultat = nba_sql_tool.invoke(question)
print(f"Résultat extrait de la base : {resultat}")