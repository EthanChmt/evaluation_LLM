# Rapport Technique : Audit Initial et Évaluation du Prototype RAG

## 1. Contexte et objectifs

Dans le cadre de l'amélioration de l'assistant IA de SportSee, un audit de l'architecture **RAG (Retrieval-Augmented Generation)** actuelle a été réalisé.

L'objectif de cette première phase est d'établir une **ligne de base (baseline)** des performances du prototype face à des requêtes hybrides :

* **Requêtes textuelles et historiques** : issues de discussions communautaires, notamment Reddit.
* **Requêtes quantitatives et analytiques** : nécessitant l'extraction et l'agrégation de statistiques issues de fichiers Excel.

L'évaluation a été automatisée à l'aide du framework **RAGAS**, afin de mesurer la capacité du système à :

1. récupérer les bonnes informations ;
2. générer une réponse fidèle au contexte ;
3. produire une réponse pertinente.

---

## 2. Méthodologie et choix techniques

### 2.1. Jeu de données de test

Un **Ground Truth de 5 questions métiers représentatives** a été créé.

Les questions couvrent à la fois :

* les archives textuelles ;
* les données structurées ;
* les requêtes historiques ;
* les requêtes quantitatives nécessitant des filtres ou des classements.

Chaque question est associée à une **réponse idéale** servant de référence pour l'évaluation.

### 2.2. Moteur de recherche

Le système utilise **FAISS** pour effectuer une recherche vectorielle à partir d'embeddings Mistral.

Les fichiers Excel sont actuellement indexés sous forme de **texte brut découpé en chunks**.

### 2.3. Générateur

Le modèle utilisé est :

```text
mistral-small-latest
```

avec une température de :

```text
0.1
```

Cette faible température vise à limiter la créativité du modèle au profit de la factualité et de la reproductibilité des réponses.

### 2.4. Métriques RAGAS

Trois métriques principales ont été retenues :

| Métrique              | Objectif                                                                     |
| --------------------- | ---------------------------------------------------------------------------- |
| **Faithfulness**      | Vérifier si la réponse est factuellement déduite du contexte fourni          |
| **Context Precision** | Vérifier si les documents pertinents sont placés en haut des résultats       |
| **Context Recall**    | Vérifier si toutes les informations nécessaires à la réponse sont récupérées |

> **Note :** la métrique `Answer Relevancy` a retourné `null` lors de ce run. Ce comportement est lié au parsing JSON du LLM évaluateur. Les trois autres métriques restent suffisantes pour diagnostiquer les principales faiblesses du système.

---

## 3. Résultats de l'évaluation

Le tableau ci-dessous synthétise les scores obtenus pour chaque cas de test.

| Question | Type                      | Faithfulness | Context Precision | Context Recall |
| -------- | ------------------------- | -----------: | ----------------: | -------------: |
| Q1       | Historique                |            - |                 - |              - |
| Q2       | Historique / statistiques |         0.41 |              0.00 |              - |
| Q3       | Analyse textuelle         |         0.95 |              0.75 |              - |
| Q4       | Statistique quantitative  |            - |                 - |              - |
| Q5       | Filtrage / agrégation     |            - |              0.00 |           0.00 |

---

## 4. Analyse critique et limites identifiées

### 4.1. Succès partiel sur le traitement du langage naturel

Lorsque la réponse est explicitement formulée dans un texte, le système obtient de bonnes performances.

C'est notamment le cas de la **Q3**, portant sur la perception de la finale Thunder/Pacers :

* `Context Precision` : **0.75**
* `Faithfulness` : **0.95**

Le système récupère correctement les éléments pertinents et le LLM parvient à synthétiser les arguments présents dans les discussions communautaires.

---

### 4.2. Bruit documentaire et hallucinations sur les questions historiques

Sur les questions historiques nécessitant de croiser plusieurs informations éparses, le retriever peine à isoler les éléments pertinents.

La **Q2** illustre particulièrement ce problème :

* `Context Precision` : **0.00**
* `Faithfulness` : **0.41**

Face à un contexte incomplet ou bruité, le LLM compense en générant des informations provenant de ses propres connaissances plutôt que du contexte récupéré.

Le cas le plus problématique est la génération d'un **tableau de statistiques comparatif complet** qui ne figurait pas dans les documents sources.

Cette hallucination entraîne mécaniquement une forte diminution de la `Faithfulness`.

---

### 4.3. Échec critique sur les données structurées

Le système présente une limite architecturale majeure lorsqu'il doit traiter des données quantitatives issues de fichiers Excel.

#### Incapacité à filtrer et trier

La **Q5** demandait de :

1. sélectionner les joueurs ayant au moins `100 tirs tentés` ;
2. comparer leur pourcentage ;
3. identifier le meilleur joueur.

Une base vectorielle telle que **FAISS**, fondée sur la similarité sémantique, n'est pas adaptée à ce type d'opérations.

Elle ne permet pas nativement d'effectuer :

* des filtres conditionnels ;
* des tris ;
* des agrégations ;
* des comparaisons numériques ;
* des jointures entre tables.

#### Scores nuls

Sur la Q5 :

* `Context Precision` : **0.00**
* `Context Recall` : **0.00**

Le système ne parvient donc pas à récupérer les bonnes lignes du tableau.

#### Illusion de compétence

La **Q4** constitue un cas différent.

Le système trouve la bonne réponse, **Shai Gilgeous-Alexander**, mais cette réussite est principalement liée à la similarité sémantique entre la question et le contenu de la ligne correspondante.

Dès que la requête devient plus complexe, comme dans la Q5, le système génère de fausses statistiques au lieu d'indiquer qu'il ne dispose pas des informations nécessaires.

---

## 5. Recommandations et prochaines étapes

L'architecture **100 % Vector Store**, basée sur du texte brut et de la similarité cosinus, atteint ici ses limites.

Pour rendre l'assistant robuste et exploitable par les coachs et analystes de SportSee, une **architecture hybride** est nécessaire.

### Plan d'action : Étape 2

#### 5.1. Modélisation SQL

Abandonner l'indexation textuelle du fichier `regular NBA.xlsx`.

Les données statistiques doivent être :

1. structurées ;
2. modélisées ;
3. ingérées dans une base de données relationnelle.

Une base **PostgreSQL** ou **SQLite** peut être utilisée.

#### 5.2. Validation des données

Sécuriser les flux d'entrée à l'aide de modèles **Pydantic** lors de l'ingestion des données.

#### 5.3. Création d'un outil SQL

Équiper le LLM d'un outil `sql_tool.py`, basé sur LangChain, permettant de :

1. interpréter une question analytique ;
2. générer une requête SQL ;
3. exécuter cette requête ;
4. retourner les résultats au LLM.

Cette approche doit permettre de garantir une meilleure précision sur :

* les questions chiffrées ;
* les filtres ;
* les classements ;
* les agrégations.

---

# 6. Évaluation de la V2 : Architecture hybride agentique SQL + RAG

Pour surmonter les limites de l'approche 100 % vectorielle, une architecture agentique a été déployée.

Le modèle **Llama-3.1-70B**, via Groq, a été sélectionné pour ses capacités avancées de **Tool Calling**.

Le LLM agit désormais comme un routeur cognitif disposant de deux outils distincts :

| Outil                 | Fonction                                                         |
| --------------------- | ---------------------------------------------------------------- |
| `NBA_Stats_SQL`       | Interroger une base relationnelle pour les données quantitatives |
| `NBA_Archives_Search` | Effectuer une recherche sémantique dans les archives textuelles  |

L'objectif est donc de séparer clairement les deux types de connaissances :

```text
Question utilisateur

        │

        ▼

     LLM / Agent
        │   │
        │   └──────────────► NBA_Archives_Search
        │                    Recherche textuelle
        │
        └───────────────────► NBA_Stats_SQL
                             Données quantitatives
```

---

## 6.1. Validation technique du flux analytique SQL

L'évaluation démontre que l'agent est capable de traduire une intention analytique complexe en code SQL fonctionnel.

### Génération et exécution autonomes

Sur la **Q5**, portant sur le filtrage du volume de tirs à 3 points, l'agent :

1. identifie la nécessité d'utiliser SQL ;
2. invoque `NBA_Stats_SQL` ;
3. génère une requête impliquant des jointures ;
4. applique le filtre conditionnel `WHERE three_pa >= 100` ;
5. extrait le joueur correspondant : **Seth Curry**.

Les métriques d'audit interne confirment le bon fonctionnement du flux :

| Métrique                     |   Score |
| ---------------------------- | ------: |
| `sql_tool_called`            | **1.0** |
| `sql_executed_without_error` | **1.0** |

---

### Faux négatif de l'évaluation sur la Q4

Sur la **Q4**, l'agent trouve correctement la cible :

> **Shai Gilgeous-Alexander — 2485 points**

Cependant, la métrique `Exact Match` retourne :

```text
0.0
```

L'audit montre que cet échec provient du formatage de la réponse.

L'agent écrit :

```text
2 485
```

alors que l'expression régulière du script d'évaluation attend :

```text
2485
```

La logique de l'agent est donc correcte. Le problème se situe dans la **méthode de notation**, qui doit être rendue plus robuste aux différents formats numériques.

---

# 7. Limites de la V2 : routage textuel et surcharge cognitive

Les outils fonctionnent correctement lorsqu'ils sont utilisés séparément. En revanche, le système présente une faiblesse importante au niveau du **routage initial**.

Le LLM développe un **biais SQL** qui le pousse à utiliser la base relationnelle même pour des questions purement textuelles.

---

## 7.1. Hallucination de schémas SQL

Face à des questions historiques telles que :

* Q1 : avantage du terrain ;
* Q2 : rTS % ;

l'agent tente d'interroger la base SQL en inventant des tables qui n'existent pas.

Exemples :

```text
playoffs_2020
playoff_player_stats
```

Ces erreurs montrent que le problème n'est plus l'exécution SQL elle-même, mais la **sélection du bon outil**.

---

## 7.2. Épuisement de la boucle d'itération

À chaque erreur SQL, l'agent tente de corriger sa requête.

Le processus devient alors :

```text
Question
   ↓
SQL
   ↓
Erreur
   ↓
Nouvelle requête SQL
   ↓
Erreur
   ↓
Nouvelle tentative
   ↓
...
```

Lorsque l'agent comprend finalement son erreur et bascule vers `NBA_Archives_Search`, celui-ci récupère effectivement d'excellents fragments de texte, avec une pertinence supérieure à **85 %**.

Cependant, le framework LangChain atteint alors la limite de :

```text
Max Iterations
```

L'agent est interrompu avant de pouvoir formuler sa réponse finale.

Les scores RAGAS deviennent alors mécaniquement nuls, non pas parce que le retriever textuel est nécessairement mauvais, mais parce que **l'agent n'atteint pas l'étape de génération finale**.

---

# 8. Plan de remédiation : optimisation du Prompt Engineering

L'infrastructure est désormais validée :

* accès aux données ;
* puissance de calcul ;
* exécution SQL ;
* recherche vectorielle ;
* Tool Calling.

Le dysfonctionnement principal est désormais **sémantique et lié à l'orchestration**.

L'objectif de la prochaine itération est donc de cloisonner davantage le raisonnement de l'agent.

---

## 8.1. Ingénierie des descriptions des outils

Le LLM sélectionne son outil principalement à partir de la description qui lui est fournie.

La description de `NBA_Stats_SQL` doit donc préciser explicitement qu'il ne doit **pas** être utilisé pour :

* les questions historiques ;
* les questions narratives ;
* les discussions communautaires ;
* les biographies ;
* les analyses qualitatives.

À l'inverse, `NBA_Archives_Search` doit être explicitement associé aux :

* analyses d'observateurs ;
* discussions Reddit ;
* comparaisons historiques ;
* contextes tactiques ;
* informations narratives.

---

## 8.2. Correction du script d'évaluation

La fonction d'extraction Regex utilisée pour l'`Exact Match` doit également être modifiée.

Elle doit considérer comme équivalents les différents formats d'un même nombre :

```text
2485
2 485
2,485
```

L'évaluation ne doit donc pas pénaliser une réponse correcte uniquement en raison de son formatage.

---

# 9. Analyse approfondie des résultats de la V3 et artefacts d'évaluation

L'ajustement du prompt système (cloisonnement strict des outils et règles de repli) a permis de résoudre le problème de routage cognitif. L'outil `NBA_Archives_Search` est désormais activé en première intention sur les requêtes textuelles.

Toutefois, la lecture brute des métriques d'évaluation de cette V3 révèle des scores paradoxaux. Une analyse détaillée des logs d'exécution montre que **ces scores reflètent en partie les limites du framework d'évaluation automatisé (RAGAS) et de l'infrastructure, plutôt qu'une défaillance directe de l'agent IA.**

## 9.1. Canal quantitatif : succès de l'exécution vs échec de la similarité syntaxique

Sur les données structurées, l'agent démontre une fiabilité d'un point de vue fonctionnel, validée par un taux de succès d'exécution parfait :

* `sql_executed_without_error` : **1.0**

Cependant, la métrique évaluant la correspondance avec la requête attendue affiche un score nul :

* `SQL Reference Match Score` : **0.0**

### L'autonomie sémantique de l'agent

Ce score de 0.0 n'est pas nécessairement un échec. La fonction d'évaluation utilise une stricte comparaison de texte (*string matching*) entre la requête générée et la requête de référence du *Ground Truth*.

Lors du traitement de la Q5 (filtrage sur les tirs à 3 points), l'agent n'a pas recopié le script de référence. En inspectant le schéma de la base de données, il a généré dynamiquement une requête plus riche :

* ajout de colonnes contextuelles, par exemple `team` ;
* création d'alias explicites, par exemple `AS player_name` ou `AS three_point_attempts` ;
* application de ratios non demandés mais pertinents, comme des moyennes par match.

Bien que la syntaxe soit différente du *Ground Truth*, **la requête s'exécute sans erreur et retourne le résultat exact**.

Cela démontre qu'une évaluation par comparaison textuelle est inadaptée pour mesurer à elle seule la qualité d'un agent *Text-to-SQL*. Une évaluation par l'exécution (*Execution Accuracy*) serait plus pertinente, car elle compare les résultats renvoyés par la base de données plutôt que la seule forme de la requête.

## 9.2. Instabilité de l'infrastructure et effondrement artificiel des métriques RAGAS

Sur le canal textuel et historique, les scores agrégés par le framework RAGAS apparaissent anormalement faibles :

| Métrique          | Score V3 |
| ----------------- | -------: |
| Answer Relevancy  | **0.63** |
| Context Precision | **0.50** |
| Context Recall    | **0.25** |
| Faithfulness      | **0.00** |

### L'impact des erreurs de génération (API Crash)

Ces moyennes sont fortement influencées par des instabilités de l'API tierce, notamment des limites de taux (*Rate Limits*).

Lors du test, l'agent a subi un crash réseau sur la Q2, générant la réponse brute : *« Erreur de génération. »*

Le juge LLM de RAGAS a évalué cette chaîne de caractères face au contexte, attribuant mécaniquement un score de **0.0 en Faithfulness** sur cette question, ce qui a fortement influencé les moyennes globales de l'évaluation.

À l'inverse, sur les questions où l'API n'a pas coupé, comme la Q3 sur la finale Thunder/Pacers, le *Context Precision* a atteint **0.999**, ce qui montre que le retriever vectoriel peut remonter les bonnes informations lorsque l'infrastructure fonctionne correctement.

## 9.3. Rigueur algorithmique de l'agent face aux biais du jeu de test (Ground Truth)

L'audit a également mis en lumière un phénomène propre à l'évaluation des LLM : **la rigueur logique de la machine peut parfois révéler une ambiguïté ou un biais dans le jeu de test.**

* **Le cas de la Q1 :** la question demandait si une équipe avait atteint les Finales sans avantage du terrain lors des *trois premières rondes*.
* **Le biais du Ground Truth :** la réponse de référence, rédigée par un humain, affirmait formellement *« Non »*, en arguant que les Houston Rockets (1995) avaient fini par obtenir cet avantage *pendant la Finale*. Le rédacteur a ainsi étendu la condition à la totalité des playoffs.
* **La logique de l'agent :** l'agent IA s'en est tenu strictement à l'énoncé de la question, limité aux trois premiers tours. Il a répondu *« Oui »*, en citant précisément les Houston Rockets (1995) et les New York Knicks (1999) sous forme de tableau structuré.

### Conclusion sur l'évaluation

L'agent a fourni une réponse cohérente avec la formulation de la question, mais le modèle juge (RAGAS) l'a pénalisé car cette réponse contredisait le *Ground Truth*.

Cela montre que l'évaluation automatisée doit être interprétée avec prudence : un score faible ne signifie pas systématiquement une défaillance du système. Le processus d'évaluation doit également permettre d'identifier les erreurs ou ambiguïtés présentes dans les données de référence.

---

# 10. Bilan et conclusion de l'audit hybride

L'architecture agentique hybride **LLM + SQL + RAG** valide les principaux choix techniques nécessaires à la conception d'un assistant analytique performant pour SportSee.

## 10.1. Séparation des responsabilités

La séparation des responsabilités apparaît comme le principal enseignement de l'audit :

| Type d'information         | Système privilégié | Objectif                                         |
| -------------------------- | ------------------ | ------------------------------------------------ |
| Données quantitatives      | **SQL**            | Exactitude des métriques, filtres et classements |
| Données historiques        | **RAG**            | Recherche dans les archives                      |
| Discussions communautaires | **RAG**            | Analyse des opinions et du contexte              |
| Analyses tactiques         | **RAG**            | Recherche sémantique et synthèse                 |
| Agrégations numériques     | **SQL**            | Calcul déterministe                              |

Le quantitatif relève donc du **moteur relationnel SQL**, tandis que le qualitatif, l'historique et la nuance tactique sont confiés à la **recherche vectorielle sémantique**.

---

## 10.2. Goulot d'étranglement final : l'infrastructure

Les problèmes résiduels observés en fin de parcours ne relèvent plus principalement de la logique algorithmique ou d'une défaillance du code.

Le principal facteur limitant devient désormais la **stabilité des API tierces gratuites** :

* limites de taux ;
* micro-coupures ;
* indisponibilités temporaires ;
* contraintes de réseau.

Le passage à l'échelle opérationnelle nécessitera donc :

1. une stabilisation des quotas API ;
2. ou l'intégration de solutions locales permettant de réduire la dépendance aux infrastructures distantes.

---

## Conclusion

L'audit montre une évolution claire du système :

```text
V1
Architecture 100 % vectorielle
        ↓
Limites sur les données quantitatives
        ↓
V2
Architecture hybride SQL + RAG
        ↓
Problèmes de routage
        ↓
V3
Prompt Engineering + routage spécialisé
        ↓
Architecture fonctionnellement validée
```

L'architecture hybride permet ainsi de répondre à la principale faiblesse identifiée lors de l'audit initial : **ne plus demander à un moteur de recherche vectoriel d'effectuer des opérations qui relèvent d'un système relationnel**.

Le prochain enjeu n'est donc plus de remplacer l'architecture, mais de **fiabiliser son orchestration, son évaluation et son infrastructure avant un passage à l'échelle**.
