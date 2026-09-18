"""
eval/evaluate_ragas.py — V3

Script d'évaluation du pipeline agentique SportSee (RAG vectoriel + SQL).

- Questions "Textuel / Historique" : RAGAS (Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall).
- Questions "Chiffré / Analytique" : Exact Match et audit de génération SQL.

Objectif de la V3 : que le run aille au bout.
  * contextes dédupliqués / tronqués -> plus de 413 Payload Too Large
  * RAGAS séquentiel (max_workers=1) -> plus de tempête de 429
  * AnswerRelevancy(strictness=1) -> plus de 400 "'n' : number must be at most 1"
  * juge sur un modèle non-raisonnant -> plus de LLMDidNotFinishException
  * relances + repli de synthèse -> plus de ligne bloquée sur "max iterations"
  * trace brute sauvegardée avant RAGAS -> un crash tardif ne perd pas le run

Génération LLM : Groq (openai/gpt-oss-120b).
Juge RAGAS   : Groq (llama-3.3-70b-versatile) — quota TPM distinct, pas de reasoning tokens.
Embeddings   : Mistral (mistral-embed) pour préserver l'index FAISS existant.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import time
import logging
import re
import pandas as pd
from dotenv import load_dotenv
import logfire

os.environ.setdefault("LANGSMITH_OTEL_ENABLED", "true")
os.environ.setdefault("LANGSMITH_OTEL_ONLY", "true")
os.environ.setdefault("LANGSMITH_TRACING", "true")

logfire.configure()
logfire.instrument_pydantic()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from langchain_mistralai import MistralAIEmbeddings
from langchain_groq import ChatGroq

from utils.vector_store import VectorStoreManager
from utils.agent import build_sportsee_agent
from utils.config import MISTRAL_API_KEY, SEARCH_K

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------------------------------------------------------- #
# Réglages
# --------------------------------------------------------------------------- #

SQL_TOOL_NAME = "NBA_Stats_SQL"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

AGENT_MODEL = "openai/gpt-oss-120b"
# Juge sur un modèle différent : quota TPM séparé de celui de l'agent, et pas de
# tokens de raisonnement qui tronquent la sortie structurée attendue par RAGAS.
JUDGE_MODEL = "llama-3.3-70b-versatile"

EVAL_SEARCH_K = 3            # < SEARCH_K : limite la taille du scratchpad de l'agent
AGENT_MAX_ITERATIONS = 6     # build_sportsee_agent est à 4, on relève après coup
AGENT_MAX_ATTEMPTS = 3       # relances en cas de 429/413 côté agent
PAUSE_BETWEEN_QUESTIONS = 12 # secondes, laisse le budget TPM se reconstituer
PAUSE_BETWEEN_METRICS = 25   # idem entre deux métriques RAGAS

MAX_CTX = 4                  # nb de chunks transmis à RAGAS
MAX_CTX_CHARS = 1000         # taille max d'un chunk
MAX_ANSWER_CHARS = 3500      # taille max de la réponse transmise à RAGAS

MAX_ITER_MARKER = "agent stopped due to max iterations"
NO_TOOL_CONTEXT = "Aucun tool appelé par l'agent pour cette question."

OUTPUT_RESULTS = os.path.join("eval", "evaluation_resultsV3.json")
OUTPUT_RAW_TRACE = os.path.join("eval", "agent_outputs_V3.json")

FALLBACK_PROMPT = """Tu es un analyste NBA. Réponds à la question en français en t'appuyant \
UNIQUEMENT sur les extraits d'archives ci-dessous. Si les extraits ne permettent pas de \
répondre, dis-le explicitement plutôt que d'inventer.

Question : {question}

Extraits :
{contexts}

Réponse :"""


# --------------------------------------------------------------------------- #
# Extraction des contextes
# --------------------------------------------------------------------------- #

def _split_observation(observation: str) -> list[str]:
    """Redécoupe une observation d'outil en chunks individuels.

    Le tool RAG concatène ses k résultats avec un séparateur '---'. Les garder
    collés donnait un seul contexte géant : payload > 8000 TPM (413) et
    ContextPrecision calculée sur k=1, donc dénuée de sens.
    """
    parts = [p.strip() for p in re.split(r"\n-{3,}\n", str(observation))]
    return [p for p in parts if p]


def _extract_contexts_from_intermediate_steps(intermediate_steps) -> list[str]:
    contexts, seen = [], set()
    for action, observation in intermediate_steps:
        tool_name = getattr(action, "tool", "outil inconnu")
        for chunk in _split_observation(observation):
            # L'agent relance souvent la même recherche : on dédoublonne,
            # sinon le même chunk part 4 fois vers le juge.
            key = re.sub(r"\s+", " ", chunk)[:200].lower()
            if key in seen:
                continue
            seen.add(key)
            contexts.append(f"[{tool_name}] {chunk[:MAX_CTX_CHARS]}")
            if len(contexts) >= MAX_CTX:
                return contexts
    return contexts or [NO_TOOL_CONTEXT]


def _has_real_context(contexts: list[str]) -> bool:
    return bool(contexts) and contexts[0] != NO_TOOL_CONTEXT


# --------------------------------------------------------------------------- #
# Exécution robuste de l'agent
# --------------------------------------------------------------------------- #

def _looks_like_quota_error(exc: Exception) -> bool:
    txt = str(exc).lower()
    return any(m in txt for m in ("429", "413", "rate_limit", "too many requests",
                                  "too large", "timeout"))


def _invoke_agent_with_retry(agent_executor, question: str, category: str):
    """Relance l'agent sur erreur transitoire. Renvoie (result, erreur)."""
    last_error = None
    for attempt in range(1, AGENT_MAX_ATTEMPTS + 1):
        try:
            with logfire.span("Évaluation - Exécution agent",
                              question=question, category=category, attempt=attempt):
                return agent_executor.invoke({"input": question}), None
        except Exception as e:
            last_error = e
            wait = min(60, 15 * attempt)
            kind = "quota/timeout" if _looks_like_quota_error(e) else "inattendue"
            logging.warning(
                f"Tentative {attempt}/{AGENT_MAX_ATTEMPTS} échouée ({kind}) : {e} "
                f"— nouvelle tentative dans {wait}s."
            )
            time.sleep(wait)
    return {}, last_error


def _synthesize_answer(llm, question: str, contexts: list[str]) -> str:
    """Repli quand l'agent s'arrête sur max_iterations sans produire de réponse.

    On a déjà les chunks récupérés : on demande directement une synthèse plutôt
    que de laisser la ligne avec le stub 'Agent stopped due to max iterations',
    qui n'est évaluable par aucune métrique.
    """
    joined = "\n\n".join(contexts)[:6000]
    prompt = FALLBACK_PROMPT.format(question=question, contexts=joined)
    for attempt in range(1, 4):
        try:
            return llm.invoke(prompt).content.strip()
        except Exception as e:
            logging.warning(f"Repli de synthèse, tentative {attempt}/3 : {e}")
            time.sleep(10 * attempt)
    return "Aucune réponse générée par l'agent."


# --------------------------------------------------------------------------- #
# Métriques SQL / numériques
# --------------------------------------------------------------------------- #

def check_exact_match(answer: str, ground_truth: str) -> bool:
    # Nettoie les espaces insécables ou classiques au milieu des nombres (ex: "2 485" -> "2485")
    clean_answer = re.sub(r'(?<=\d)[\s\u00a0]+(?=\d)', '', answer)
    clean_gt = re.sub(r'(?<=\d)[\s\u00a0]+(?=\d)', '', ground_truth)

    gt_numbers = set(re.findall(r'\b\d+(?:[.,]\d+)?\b', clean_gt))
    if not gt_numbers:
        return False
    answer_numbers = set(re.findall(r'\b\d+(?:[.,]\d+)?\b', clean_answer))
    return bool(gt_numbers.intersection(answer_numbers))


def _parse_sql_observation(observation_str: str):
    match = re.match(r"^SQL_QUERY:\s*(.*?)\nRESULT:\s*(.*)$", observation_str, re.DOTALL)
    if not match:
        return None, observation_str
    return match.group(1).strip(), match.group(2).strip()


def _extract_sql_call_from_intermediate_steps(intermediate_steps):
    sql_call = None
    for action, observation in intermediate_steps:
        tool_name = getattr(action, "tool", "")
        if tool_name != SQL_TOOL_NAME and "sql" not in tool_name.lower():
            continue

        observation_str = str(observation)
        generated_sql, result_text = _parse_sql_observation(observation_str)
        error_detected = "Erreur technique" in result_text

        sql_call = {
            "tool": tool_name,
            "query": generated_sql,
            "result": result_text,
            "error_detected": error_detected,
        }
    return sql_call


def _normalize_sql(sql: str) -> str:
    return re.sub(r'\s+', ' ', sql.strip().rstrip(';')).strip().lower()


def check_sql_match(generated_sql, reference_sql):
    if not reference_sql or not generated_sql:
        return None
    return _normalize_sql(generated_sql) == _normalize_sql(reference_sql)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

def run_evaluation():
    if not MISTRAL_API_KEY:
        logging.error("MISTRAL_API_KEY non trouvée (nécessaire pour les embeddings).")
        return
    if not GROQ_API_KEY:
        logging.error("GROQ_API_KEY non trouvée dans le fichier .env.")
        return

    dataset_path = os.path.join("eval", "test_dataset.json")
    if not os.path.exists(dataset_path):
        logging.error(f"Fichier de test introuvable : {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    logging.info(f"{len(test_cases)} cas de test chargés depuis {dataset_path}.")

    try:
        vector_manager = VectorStoreManager()
        logging.info("VectorStoreManager initialisé avec succès pour l'évaluation.")
    except Exception as e:
        logging.error(f"Erreur d'initialisation du VectorStoreManager : {e}")
        return

    eval_embeddings = MistralAIEmbeddings(
        model="mistral-embed", mistral_api_key=MISTRAL_API_KEY
    )

    eval_llm = ChatGroq(
        model=JUDGE_MODEL, api_key=GROQ_API_KEY, temperature=0.1,
        max_tokens=2048, max_retries=6,
    )

    agent_llm = ChatGroq(
        model=AGENT_MODEL, api_key=GROQ_API_KEY, temperature=0.0, max_retries=6,
    )

    agent_executor = build_sportsee_agent(
        agent_llm, vector_manager, search_k=min(SEARCH_K, EVAL_SEARCH_K),
        return_intermediate_steps=True,
    )
    # build_sportsee_agent fixe max_iterations=4 en dur ; on relève ici pour
    # laisser à l'agent le tour supplémentaire nécessaire à la synthèse finale.
    agent_executor.max_iterations = AGENT_MAX_ITERATIONS

    text_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    text_status = []
    sql_results = []
    raw_trace = []

    for idx, item in enumerate(test_cases, start=1):
        q = item["question"]
        gt = item["ground_truth"]
        category = item.get("category", "Non catégorisé")

        logging.info(f"[{idx}/{len(test_cases)}] {category} — {q[:70]}...")

        result, agent_error = _invoke_agent_with_retry(agent_executor, q, category)

        if agent_error is not None:
            logging.error(f"Agent en échec après {AGENT_MAX_ATTEMPTS} tentatives : {agent_error}")
            answer_text = ""
            retrieved_contexts = [NO_TOOL_CONTEXT]
            status = "agent_error"
        else:
            answer_text = (result.get("output") or "").strip()
            retrieved_contexts = _extract_contexts_from_intermediate_steps(
                result.get("intermediate_steps", [])
            )
            status = "ok"

        # Repli : max_iterations, réponse vide, ou agent en échec mais contextes présents.
        needs_fallback = (
            not answer_text or MAX_ITER_MARKER in answer_text.lower()
        )
        if needs_fallback and _has_real_context(retrieved_contexts):
            logging.info("Réponse inexploitable — synthèse de repli sur les contextes récupérés.")
            answer_text = _synthesize_answer(eval_llm, q, retrieved_contexts)
            status = "fallback_synthese" if status == "ok" else "agent_error_fallback"
        elif needs_fallback:
            answer_text = "Aucune réponse générée par l'agent."
            status = "sans_reponse"

        raw_trace.append({
            "question": q, "category": category, "run_status": status,
            "answer": answer_text, "contexts": retrieved_contexts,
        })

        if category == "Textuel / Historique":
            text_data["question"].append(q)
            text_data["answer"].append(answer_text[:MAX_ANSWER_CHARS])
            text_data["contexts"].append(retrieved_contexts)
            text_data["ground_truth"].append(gt)
            text_status.append(status)
        else:
            exact_match = check_exact_match(answer_text, gt)
            sql_call = _extract_sql_call_from_intermediate_steps(
                result.get("intermediate_steps", [])
            )
            reference_sql = item.get("reference_sql") or item.get("sql_ground_truth")
            sql_match = check_sql_match(
                sql_call["query"] if sql_call else None, reference_sql
            )

            sql_results.append({
                "question": q,
                "answer": answer_text,
                "contexts": retrieved_contexts,
                "ground_truth": gt,
                "category": category,
                "run_status": status,
                "exact_match_score": 1.0 if exact_match else 0.0,
                "sql_tool_called": 1.0 if sql_call else 0.0,
                "generated_sql": sql_call["query"] if sql_call else None,
                "sql_executed_without_error": (
                    None if sql_call is None else (0.0 if sql_call["error_detected"] else 1.0)
                ),
                "sql_reference_match_score": (
                    None if sql_match is None else (1.0 if sql_match else 0.0)
                ),
            })

        if idx < len(test_cases):
            time.sleep(PAUSE_BETWEEN_QUESTIONS)

    # Trace brute écrite AVANT RAGAS : si le juge casse, le run n'est pas perdu.
    os.makedirs("eval", exist_ok=True)
    with open(OUTPUT_RAW_TRACE, "w", encoding="utf-8") as f:
        json.dump(raw_trace, f, ensure_ascii=False, indent=4)
    logging.info(f"Trace brute des exécutions agent sauvegardée dans {OUTPUT_RAW_TRACE}")

    final_dfs = []

    if text_data["question"]:
        eval_dataset = Dataset.from_dict(text_data)
        logging.info("Lancement de l'évaluation RAGAS pour la catégorie Textuel...")

        run_config = RunConfig(
            max_workers=1,   # séquentiel : le parallélisme par défaut saturait les 8000 TPM
            timeout=600,
            max_retries=20,
            max_wait=90,
        )

        metrics = [
            Faithfulness(),
            # strictness=1 : Groq refuse n>1 (400 "'n' : number must be at most 1")
            AnswerRelevancy(strictness=1),
            ContextPrecision(),
            ContextRecall(),
        ]

        # Une métrique à la fois : un échec (413, timeout, juge qui ne finit pas)
        # ne coûte qu'une colonne au lieu de faire tomber les quatre.
        ragas_df = eval_dataset.to_pandas()
        for i, metric in enumerate(metrics, start=1):
            name = getattr(metric, "name", metric.__class__.__name__)
            logging.info(f"RAGAS [{i}/{len(metrics)}] — {name}")
            try:
                partial = evaluate(
                    dataset=eval_dataset,
                    metrics=[metric],
                    llm=eval_llm,
                    embeddings=eval_embeddings,
                    run_config=run_config,
                    raise_exceptions=False,
                ).to_pandas()
                for col in [c for c in partial.columns if c not in ragas_df.columns]:
                    ragas_df[col] = partial[col]
                logging.info(f"  -> {name} terminée")
            except Exception as e:
                logging.error(f"  -> {name} a échoué ({e}) — colonne laissée vide.")
                ragas_df[name] = None

            if i < len(metrics):
                time.sleep(PAUSE_BETWEEN_METRICS)

        ragas_df["category"] = "Textuel / Historique"
        ragas_df["run_status"] = text_status
        final_dfs.append(ragas_df)

    if sql_results:
        final_dfs.append(pd.DataFrame(sql_results))

    if not final_dfs:
        logging.warning("Aucun résultat à sauvegarder.")
        return

    results_df = pd.concat(final_dfs, ignore_index=True)
    results_df.to_json(OUTPUT_RESULTS, orient="records", indent=4, force_ascii=False)
    logging.info(f"Résultats détaillés sauvegardés dans {OUTPUT_RESULTS}")

    print("\n--- STATUT DES EXÉCUTIONS ---")
    print(results_df.groupby(["category", "run_status"]).size().to_string())

    print("\n--- MOYENNES PAR CATÉGORIE ---")
    print(results_df.groupby("category").mean(numeric_only=True).to_string())

    nan_cols = [c for c in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
                if c in results_df.columns and results_df[c].isna().any()]
    if nan_cols:
        print(f"\nMétriques encore incomplètes (NaN) : {', '.join(nan_cols)}")
        print("Si cela persiste, baisse MAX_CTX à 3 et MAX_ANSWER_CHARS à 2500.")


if __name__ == "__main__":
    run_evaluation()