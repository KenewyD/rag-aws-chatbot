import requests
import streamlit as st

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 Chatbot RAG — AWS Bedrock + OpenSearch")

API_URL = "http://localhost:8000"

with st.sidebar:
    st.header("📥 Ingestion")
    uploaded = st.file_uploader("Ajoute un document (PDF ou TXT)", type=["pdf", "txt"])
    if st.button("Ingérer") and uploaded is not None:
        with st.spinner("Ingestion en cours..."):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            res = requests.post(f"{API_URL}/ingest", files=files).json()
        st.success(f"{res.get('chunks', 0)} chunks ingérés depuis {res.get('file', '')}")

question = st.text_input("Pose ta question :")
if st.button("Envoyer") and question:
    with st.spinner("Recherche en cours..."):
        res = requests.post(f"{API_URL}/ask", json={"question": question}).json()

    st.markdown("### 💬 Réponse")
    st.write(res["answer"])

    with st.expander("📚 Sources utilisées"):
        for src in res["sources"]:
            st.markdown(f"**Score :** {src['score']:.3f}")
            st.markdown(f"```\n{src['text'][:500]}...\n```")
            st.divider()
