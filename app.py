# -*- coding: utf-8 -*-
# Türkçe Wikipedia RAG Chatbot (FAISS varsa kullanır, yoksa Wikipedia fallback)
# - GOOGLE_API_KEY: st.secrets veya .env
# - INDEX_URL (opsiyonel): st.secrets ile verilir ise FAISS index zip’i indirilir

import os, io, zipfile, requests
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# LangChain / LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Wikipedia fallback
import wikipedia
wikipedia.set_lang("tr")

# --------------------- SAYFA ---------------------
st.set_page_config(page_title="Türkçe Wikipedia RAG Chatbot", page_icon="📚")
st.title("📚 Türkçe Wikipedia Destekli RAG Chatbot")
st.write(
    "Bu chatbot, sorduğunuz sorulara Türkçe Wikipedia verilerini kullanarak cevap verir. "
    "Projenin GitHub reposuna [buradan](https://github.com/abdulsametkara/akbank-genai-rag-chatbot) ulaşabilirsiniz."
)

# ----------------- API ANAHTARI ------------------
load_dotenv()
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY bulunamadı. Streamlit **Secrets** veya `.env` içine ekleyin.")
    st.stop()

# --------------- FAISS YARDIMCILAR ---------------
@st.cache_resource
def ensure_index_dir() -> Path | None:
    """
    1) faiss_index_wikipedia klasörü varsa onu döndürür.
    2) Yoksa ve INDEX_URL secrets/env’de varsa zip’i indirip çıkarır, sonra döndürür.
    3) Hiçbiri yoksa None döndürür.
    """
    base = Path(__file__).parent
    idx_dir = base / "faiss_index_wikipedia"
    idx_file = idx_dir / "index.faiss"

    if idx_file.exists():
        return idx_dir

    # Opsiyonel: Release/HF/Drive linki
    index_url = st.secrets.get("INDEX_URL", os.getenv("INDEX_URL"))
    if not index_url:
        return None  # indirilecek kaynak yok → fallback çalışacak

    try:
        st.info("FAISS index indiriliyor… (ilk açılışta biraz sürebilir)")
        resp = requests.get(index_url, timeout=600)
        resp.raise_for_status()
        idx_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(idx_dir)
        if idx_file.exists():
            st.success("FAISS index indirildi.")
            return idx_dir
    except Exception as e:
        st.warning(f"FAISS index indirilemedi: {e!s}")

    return None

@st.cache_resource
def load_faiss_components():
    """
    FAISS index bulunursa embedding + vectorstore yükler, aksi halde None döner.
    """
    idx_dir = ensure_index_dir()
    if idx_dir is None:
        return None

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        vs = FAISS.load_local(
            str(idx_dir),
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vs
    except Exception as e:
        st.warning(f"FAISS yüklenemedi, Wikipedia fallback kullanılacak. Hata: {e!s}")
        return None

vector_store = load_faiss_components()

# ------------------ LLM MODEL --------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.4,
)

# ------------------ PROMPT -----------------------
prompt_template_str = """
Aşağıdaki BAĞLAM'ı (varsa) kullanarak Türkçe cevap ver.
Sadece bağlamdaki bilgilere dayan. Bağlam yoksa kısa, dürüst ve genel bir cevap ver
ve bağlam olmadığını belirt.

BAĞLAM:
{context}

SORU: {input}

CEVAP:
"""
prompt = ChatPromptTemplate.from_template(prompt_template_str)

# ------------- FAISS VARSA RAG ZİNCİR -----------
retrieval_chain = None
if vector_store is not None:
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(llm, prompt)
    )

# --------------- WIKIPEDIA FALLBACK -------------
def wiki_retrieve(query: str, max_chars: int = 4000) -> str:
    try:
        hits = wikipedia.search(query, results=1)
        if not hits:
            return ""
        title = hits[0]
        page = wikipedia.page(title, auto_suggest=False)
        content = page.content[:max_chars]
        return f"BAĞLAM (Wikipedia/{title}):\n{content}"
    except Exception:
        return ""

def answer_with_wiki(query: str) -> str:
    # Wikipedia'dan bağlamı çek
    ctx = wiki_retrieve(query)
    # Promptu düz string olarak oluştur
    final_prompt = prompt_template_str.format(context=ctx, input=query)
    # Gemini'yi string ile çağır
    resp = llm.invoke(final_prompt)
    return resp.content if hasattr(resp, "content") else str(resp)

# ----------------- SOHBET ARAYÜZÜ ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmişi göster
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Girdi
user_prompt = st.chat_input("Sorunuzu buraya yazabilirsiniz...")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Düşünüyorum…"):
            if retrieval_chain is not None:
                # FAISS mevcut → RAG
                try:
                    result = retrieval_chain.invoke({"input": user_prompt})
                    answer = result.get("answer", "").strip()
                    if not answer:
                        # Çok nadiren boş gelebilir → Wiki'ya düş
                        answer = answer_with_wiki(user_prompt)
                except Exception as e:
                    st.warning(f"RAG sırasında hata: {e!s}. Wikipedia fallback kullanılıyor.")
                    answer = answer_with_wiki(user_prompt)
            else:
                # FAISS yok → doğrudan Wikipedia fallback
                answer = answer_with_wiki(user_prompt)

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
