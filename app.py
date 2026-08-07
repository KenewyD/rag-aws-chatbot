"""
Démo RAG en ligne — Streamlit.
Pipeline : ingestion (PDF ou texte) -> chunking -> vectorisation TF-IDF -> recherche par similarité.
100% gratuit, aucune clé, aucun cloud payant.
"""

import re

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Démo RAG", page_icon="🤖", layout="centered")

DEFAULT_DOC = """Le RAG (Retrieval-Augmented Generation) est une technique qui combine la recherche d'information et la génération de texte. Au lieu de répondre uniquement à partir de sa mémoire, le système va d'abord chercher les passages pertinents dans une base documentaire.

L'avantage principal du RAG est de réduire les hallucinations. Comme le système s'appuie sur des documents réels, il invente moins d'informations fausses.

Le RAG permet aussi de répondre sur des données récentes ou privées que le modèle n'a jamais vues pendant son entraînement.

Le pipeline RAG comporte plusieurs étapes. D'abord l'ingestion des documents, puis leur découpage en morceaux appelés chunks.

Chaque chunk est transformé en vecteur numérique grâce à un modèle d'embeddings. Ces vecteurs sont stockés dans une base vectorielle comme FAISS, Qdrant ou OpenSearch.

Lors d'une question, la question est elle aussi transformée en vecteur. On cherche alors les chunks dont les vecteurs sont les plus proches, c'est la recherche par similarité."""


def split_chunks(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]


def read_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text


st.title("🤖 Démo RAG — Recherche vectorielle")
st.caption("Pipeline RAG : ingestion → chunking → embeddings → recherche par similarité. "
           "Version production (AWS Bedrock + OpenSearch) sur mon GitHub.")

st.subheader("1. La base de connaissances")

tab_pdf, tab_texte = st.tabs(["📄 Charger un PDF", "✍️ Coller du texte"])

document = ""

with tab_pdf:
    uploaded = st.file_uploader("Charge un fichier PDF", type=["pdf"])
    if uploaded is not None:
        with st.spinner("Lecture du PDF..."):
            document = read_pdf(uploaded)
        st.success(f"PDF chargé : {len(document)} caractères extraits.")

with tab_texte:
    pasted = st.text_area("Ou colle ton texte ici :", value=DEFAULT_DOC, height=200)
    if not document:
        document = pasted

chunks = split_chunks(document)
st.info(f"Document découpé en {len(chunks)} chunks.")

if chunks:
    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunks)

    st.subheader("2. Pose ta question")
    query = st.text_input("Ta question :", placeholder="Quel est l'avantage du RAG ?")

    if query:
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, chunk_vectors)[0]
        top_indices = similarities.argsort()[::-1][:3]

        best = top_indices[0]
        if similarities[best] < 0.05:
            st.warning("Aucun passage pertinent trouvé dans le document. "
                       "C'est le principe du RAG : il ne répond que sur ce qu'il connaît, il n'invente pas.")
        else:
            st.subheader("💬 Réponse")
            st.success(chunks[best])

            with st.expander("📚 Voir les passages sources et leurs scores"):
                for rank, i in enumerate(top_indices, 1):
                    st.markdown(f"**[{rank}] Score de similarité : {similarities[i]:.3f}**")
                    st.write(chunks[i])
                    st.divider()
else:
    st.warning("Charge un PDF ou colle du texte pour commencer.")
