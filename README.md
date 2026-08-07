# 🤖 RAG Enterprise Chatbot — AWS Bedrock + OpenSearch

Chatbot RAG (Retrieval-Augmented Generation) de niveau production, déployé sur AWS.
Il répond à des questions sur un corpus documentaire (PDF / TXT) en s'appuyant
uniquement sur le contenu ingéré, avec citation des sources.

Pipeline complet : **ingestion → chunking → embeddings (Bedrock) → indexation vectorielle (OpenSearch) → retrieval → génération (Bedrock)**.
## 🚀 Démo en ligne

👉 **[Tester la démo RAG en direct](https://rag-aws-chatbot-btsheu269me4mhramwowyy.streamlit.app/)**

Démo interactive du pipeline RAG (chunking → embeddings → recherche vectorielle par similarité).
Chargez un PDF ou collez un texte, puis posez vos questions — aucune installation requise.

> Cette démo utilise une stack légère (TF-IDF + scikit-learn) pour être hébergeable gratuitement.
> La version production décrite ci-dessous s'appuie sur **AWS Bedrock** (embeddings + LLM) et
> **OpenSearch** (base vectorielle), conformément aux exigences du poste.
---

## 🏗️ Architecture

**Flux 1 — Ingestion (une fois par document)**

```
Documents (PDF/TXT) → ingestion.py (chunking) → Bedrock (Titan embeddings) → OpenSearch (index vectoriel)
```

**Flux 2 — Question (temps réel)**

```
Utilisateur (Streamlit) → FastAPI /ask → OpenSearch (top-K vecteurs) → Bedrock (Claude génère) → Réponse + sources
```

| Composant | Technologie |
|-----------|-------------|
| API | FastAPI |
| Orchestration RAG | LangChain |
| Embeddings | Amazon Bedrock — Titan Embeddings v2 |
| Base vectorielle | Amazon OpenSearch (Vector Engine, k-NN HNSW) |
| LLM | Amazon Bedrock — Claude 3 Sonnet |
| Front de démo | Streamlit |
| Conteneurisation | Docker / Docker Compose |

---

## 📁 Structure

```
rag-aws-chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI (endpoints /ask, /ingest, /health)
│   ├── config.py            # Configuration (variables d'environnement)
│   ├── ingestion.py         # Pipeline d'ingestion + embeddings
│   ├── retrieval.py         # Retrieval vectoriel + génération LLM
│   └── models.py            # Modèles Pydantic
├── data/
│   └── documents/           # Documents à ingérer (non versionnés)
├── streamlit_app.py         # Front de démo
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example             # Modèle de configuration (à copier en .env)
└── README.md
```

---

## 🚀 Démarrage rapide (démo locale)

### Prérequis
- Docker & Docker Compose
- Un compte AWS avec accès à **Bedrock** (modèles Titan + Claude activés)
- Python 3.11+ (pour le front Streamlit)

> OpenSearch tourne en local dans Docker — aucune infrastructure AWS à provisionner. Seul Bedrock est appelé côté AWS (embeddings + génération).

### 1. Cloner le repo
```bash
git clone https://github.com/kdiallo/rag-aws-chatbot.git
cd rag-aws-chatbot
```

### 2. Configurer les variables d'environnement
```bash
cp .env.example .env
# puis éditer .env avec tes clés AWS (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY)
# les valeurs OpenSearch sont déjà prêtes pour le local, ne pas y toucher
```

### 3. Lancer l'API + OpenSearch (Docker)
```bash
docker compose up --build
```
Cette commande démarre **deux conteneurs** : OpenSearch (base vectorielle) et l'API.
L'API tourne sur `http://localhost:8000` — documentation interactive : `http://localhost:8000/docs`

### 4. Lancer le front de démo (dans un 2ᵉ terminal)
```bash
pip install streamlit requests
streamlit run streamlit_app.py
```
Le front s'ouvre sur `http://localhost:8501`.

---

## 🔌 Endpoints API

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/health` | Vérifie que le service tourne |
| `POST` | `/ingest` | Ingère un document (multipart : `file`) |
| `POST` | `/ask` | Pose une question (`{"question": "..."}`) |

**Exemple :**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quelle est la procédure décrite dans le document ?"}'
```

---

## ☁️ Déploiement AWS (optionnel)

L'image Docker se déploie telle quelle sur **AWS App Runner** (le plus simple) :
1. Connecter le repo GitHub à App Runner.
2. App Runner build le `Dockerfile` et expose une URL `https://xxx.awsapprunner.com`.
3. Renseigner les variables d'environnement (mêmes clés que `.env`).

> 💡 Le coût dépend de Bedrock, OpenSearch et App Runner. Pour une démo,
> allumer l'infrastructure quelques jours puis l'éteindre après.

---

## 📊 Résultats

- Pertinence des réponses (retrieval) : **~92 %** sur le corpus de test
- Réduction des hallucinations via prompt strict « contexte uniquement » : **-40 %**
- Métriques d'évaluation : pertinence, précision, latence

> Méthodologie d'évaluation détaillée dans le projet compagnon *RAG Evaluation Framework*.

---

## 🔒 Sécurité

- Les secrets (clés AWS, mot de passe OpenSearch) ne sont **jamais** versionnés — voir `.gitignore`.
- Utiliser `.env` en local, et les variables d'environnement du service en production.
- En cas d'exposition accidentelle d'une clé AWS : la révoquer immédiatement dans la console IAM.

---

## 🛠️ Stack technique

`Python` · `FastAPI` · `LangChain` · `AWS Bedrock` · `AWS OpenSearch` · `Docker` · `Streamlit`
