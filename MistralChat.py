"""
Script principal de l'application Streamlit pour l'agent hybride SportSee R&D.

Cette application fournit une interface de messagerie interactive pour l'assistant 
d'analyse de performance SportSee. Elle intègre un agent LangChain propulsé 
par Mistral AI, combinant l'interrogation de base de données SQL et la recherche 
vectorielle (RAG) pour traiter les requêtes qualitatives et quantitatives.

Fonctionnalités principales :
- Interface utilisateur conversationnelle via Streamlit.
- Configuration de la traçabilité de l'exécution avec Logfire et OpenTelemetry.
- Chargement sécurisé et mise en cache de l'index vectoriel.
- Initialisation et exécution de l'agent LangChain avec le "tool-calling" natif.

Exécution :
    streamlit run MistralChat.py
"""

import streamlit as st
import logging
import os
import logfire

# Le tracing LangChain de Logfire ne passe PAS par un logfire.instrument_langchain()
# (cette fonction n'existe pas) mais par OpenTelemetry via LangSmith : ces variables
# doivent être définies AVANT d'importer langchain / les modules qui l'importent.
os.environ.setdefault("LANGSMITH_OTEL_ENABLED", "true")
os.environ.setdefault("LANGSMITH_OTEL_ONLY", "true")
os.environ.setdefault("LANGSMITH_TRACING", "true")

# Configuration de Logfire
logfire.configure()
logfire.instrument_pydantic()

# --- Importations des modules internes ---
try:
    from utils.config import (
        MISTRAL_API_KEY, MODEL_NAME, SEARCH_K,
        APP_TITLE, NAME
    )
    from utils.vector_store import VectorStoreManager
    from utils.agent import build_sportsee_agent
    from langchain_mistralai.chat_models import ChatMistralAI
except ImportError as e:
    st.error(f"Erreur d'importation critique: {e}. Vérifiez la structure des modules sous 'utils'.")
    st.stop()

# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

if not MISTRAL_API_KEY:
    st.error("Erreur : Clé API Mistral non configurée (MISTRAL_API_KEY).")
    st.stop()


# --- Chargement sécurisé du Vector Store (Mise en cache session) ---
@st.cache_resource
def get_vector_store_manager():
    logging.info("Chargement du VectorStoreManager pour l'analyse textuelle...")
    try:
        manager = VectorStoreManager()
        if manager.index is None or not manager.document_chunks:
            st.error("L'index vectoriel ou les segments textuels sont introuvables.")
            st.warning("Veuillez exécuter le script d'indexation pour préparer la base documentaire.")
            logging.error("Index Faiss ou chunks non disponibles.")
            return None
        logging.info(f"VectorStoreManager chargé ({manager.index.ntotal} vecteurs indexés).")
        return manager
    except FileNotFoundError:
        st.error("Fichiers d'index introuvables.")
        logging.error("FileNotFoundError lors de l'initialisation du VectorStoreManager.")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue lors du chargement des vecteurs: {e}")
        logging.exception("Erreur chargement VectorStoreManager")
        return None


vector_store_manager = get_vector_store_manager()

# --- Construction de l'agent (logique partagée avec evaluate_ragas.py, voir utils/agent.py) ---
@st.cache_resource
def get_agent_executor():
    """Construit l'agent LangChain (tool-calling natif Mistral) via le module partagé utils/agent.py."""
    llm = ChatMistralAI(model=MODEL_NAME, mistral_api_key=MISTRAL_API_KEY, temperature=0.0)
    if vector_store_manager is None:
        logging.warning("VectorStoreManager indisponible : l'agent ne disposera que du Tool SQL.")
    return build_sportsee_agent(llm, vector_store_manager, search_k=SEARCH_K)


try:
    agent_executor = get_agent_executor()
    logging.info("AgentExecutor SportSee initialisé (routage tool-calling par le LLM).")
except Exception as e:
    st.error(f"Erreur lors de l'initialisation de l'agent: {e}")
    logging.exception("Erreur initialisation AgentExecutor")
    st.stop()

# --- Initialisation de l'historique de session ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": f"Bonjour. Assistant R&D SportSee opérationnel pour {NAME}. "
                   "Interrogez-moi sur les données chiffrées de matchs ou les rapports d'analyse qualitative."
    }]

# --- Interface Utilisateur Streamlit ---
st.title(APP_TITLE)
st.caption(f"Plateforme R&D SportSee | Agent Hybride (SQL + Vector Store, routage LLM) | Modèle: {MODEL_NAME}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- Pipeline de Traitement Principal ---
if prompt := st.chat_input("Saisissez votre requête analytique (ex: statistiques de joueurs, comparaisons, rapports)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("Traitement analytique en cours...")

        try:
            with logfire.span("Exécution de l'agent SportSee", question=prompt):
                result = agent_executor.invoke({"input": prompt})
            response_content = result.get("output", "Aucune réponse valide générée par l'agent.")
        except Exception as e:
            st.error(f"Erreur d'exécution de l'agent: {e}")
            logging.exception("Erreur lors de l'exécution de l'AgentExecutor")
            response_content = "Une erreur technique est survenue lors du traitement de la requête."

        message_placeholder.write(response_content)

    st.session_state.messages.append({"role": "assistant", "content": response_content})

# Pied de page institutionnel R&D
st.markdown("---")
st.caption("SportSee R&D — Assistant d'Analyse de Performance | Traçabilité Logfire Active")