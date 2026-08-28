# test_etape1.py
import os
import logfire
from utils.data_loader import load_and_parse_files
from utils.vector_store import VectorStoreManager
from utils.config import INPUT_DIR

# 1. On indique à Logfire de tracer ce script global
logfire.configure()

@logfire.instrument("Test complet de l'Étape 1 (Ingestion & Validation)")
def run_test():
    print(f"Chargement des fichiers depuis le dossier '{INPUT_DIR}'...")
    # On utilise ton data_loader existant pour lire les vrais PDF/fichiers
    documents = load_and_parse_files(INPUT_DIR)
    
    if not documents:
        print("Aucun document trouvé. Vérifie que ton dossier contient bien des fichiers.")
        return

    print(f"{len(documents)} documents bruts chargés. Lancement de la vectorisation...")
    
    # On initialise ton VectorStoreManager (celui avec Pydantic et Logfire)
    vsm = VectorStoreManager()
    
    # On force la reconstruction de l'index pour déclencher le filtre Pydantic
    vsm.build_index(documents)
    
    print("\n✅ Terminé ! Va faire un tour sur ton tableau de bord Logfire.")

if __name__ == "__main__":
    run_test()