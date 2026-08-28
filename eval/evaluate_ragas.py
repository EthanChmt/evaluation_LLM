# eval/evaluate_ragas.py
import os
import sys
import json
import logging
from dotenv import load_dotenv
import logfire

# Configuration de Logfire pour capter les requêtes LLM et les étapes LangChain
logfire.configure()
logfire.instrument_langchain()
logfire.instrument_pydantic()

# 1. Configurer le chemin racine EN PREMIER pour que 'utils' soit accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import Dataset
from ragas import evaluate
# Importation des classes de métriques (avec majuscules)
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

# 2. Importation des modules du projet après l'ajout au sys.path
from utils.vector_store import VectorStoreManager
from utils.config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K

# Configuration du Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_evaluation():
    if not MISTRAL_API_KEY:
        logging.error("MISTRAL_API_KEY non trouvée dans l'environnement.")
        return

    # 1. Chargement du dataset de test
    dataset_path = os.path.join("eval", "test_dataset.json")
    if not os.path.exists(dataset_path):
        logging.error(f"Fichier de test introuvable : {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    logging.info(f"{len(test_cases)} cas de test chargés depuis {dataset_path}.")

    # 2. Initialisation du VectorStoreManager
    try:
        vector_manager = VectorStoreManager()
        logging.info("VectorStoreManager initialisé avec succès pour l'évaluation.")
    except Exception as e:
        logging.error(f"Erreur d'initialisation du VectorStoreManager : {e}")
        return

    # 3. Initialisation du LLM et des Embeddings pour RAGAS
    eval_llm = ChatMistralAI(model=MODEL_NAME, mistral_api_key=MISTRAL_API_KEY, temperature=0.1)
    eval_embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=MISTRAL_API_KEY)

    # 4. Exécution de la chaîne RAG pour chaque question du dataset
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in test_cases:
        q = item["question"]
        gt = item["ground_truth"]
        
        questions.append(q)
        ground_truths.append(gt)

        # Récupération du contexte via le Vector Store
        try:
            search_results = vector_manager.search(q, k=SEARCH_K)
            retrieved_texts = [res['text'] for res in search_results]
        except Exception as e:
            logging.warning(f"Erreur lors de la recherche pour la question '{q}': {e}")
            retrieved_texts = ["Erreur de récupération du contexte."]

        contexts.append(retrieved_texts)

        # Génération de la réponse via le LLM
        context_str = "\n\n---\n\n".join(retrieved_texts)
        prompt_content = f"""Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.\n\n---\n{context_str}\n---\n\nQUESTION DU FAN:\n{q}\n\nRÉPONSE DE L'ANALYSTE NBA:"""
        
        try:
            response = eval_llm.invoke(prompt_content)
            answer_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logging.error(f"Erreur lors de l'appel LLM pour la question '{q}': {e}")
            answer_text = "Erreur de génération."

        answers.append(answer_text)

    # 5. Formatage pour RAGAS
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }

    eval_dataset = Dataset.from_dict(data)

    # 6. Lancement de l'évaluation RAGAS
    logging.info("Lancement de l'évaluation RAGAS...")
    
    # Les métriques sont maintenant instanciées avec ()
    results = evaluate(
        dataset=eval_dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=eval_llm,
        embeddings=eval_embeddings,
    )

    logging.info("Évaluation terminée avec succès.")
    print("\n--- RÉSULTATS DE L'AUDIT RAGAS ---")
    print(results)

    # Sauvegarde des résultats au format JSON
    output_results_path = os.path.join("eval", "evaluation_results.json")
    results_df = results.to_pandas()
    results_df.to_json(output_results_path, orient="records", indent=4, force_ascii=False)
    logging.info(f"Résultats détaillés sauvegardés dans {output_results_path}")

if __name__ == "__main__":
    run_evaluation()