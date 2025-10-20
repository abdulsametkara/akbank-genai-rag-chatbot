# Gerekli kütüphaneleri projemize dahil ediyoruz.
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Türkçe Wikipedia RAG Chatbot", page_icon="📚")
st.title("📚 Türkçe Wikipedia Destekli RAG Chatbot")
st.write(
    "Bu chatbot, sorduğunuz sorulara Türkçe Wikipedia verilerini kullanarak cevap verir. "
    "Projenin GitHub reposuna [buradan](https://github.com/abdulsametkara/akbank-genai-rag-chatbot) ulaşabilirsiniz."
)

# --- API ANAHTARI VE MODELLERİ YÜKLEME ---

# .env dosyasını yükleyerek içindeki değişkenleri ortam değişkeni olarak ayarla
load_dotenv()

# API anahtarını ortam değişkenlerinden güvenli bir şekilde alıyoruz.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Lütfen GOOGLE_API_KEY'i ana dizindeki .env dosyasına ekleyin.")
    st.stop()

@st.cache_resource
def load_components():
    try:
        embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        vector_store = FAISS.load_local("faiss_index_wikipedia", embeddings_model, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        st.error(f"Hafıza (Vektör Veritabanı) yüklenirken bir hata oluştu: {e}")
        st.info("Lütfen 'faiss_index_wikipedia' klasörünün bu projenin ana dizininde olduğundan emin olun.")
        st.stop()

# Modelleri ve veritabanını yükle
vector_store = load_components()


# --- RAG ZİNCİRİNİ OLUŞTURMA ---
# (Bu kısımdan sonrası aynı kalıyor)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.4)
retriever = vector_store.as_retriever()
prompt_template_str = """
Sana verilen aşağıdaki bağlamı (context) kullanarak soruya cevap ver.
Cevabı, sadece bu bağlamdaki bilgilerden yola çıkarak oluştur.
Eğer bağlamda cevap yoksa, "Bu konuda hafızamda bir bilgi bulunmuyor." de.

Bağlam (Context):
{context}

Soru: {input}

Cevap:
"""
prompt = ChatPromptTemplate.from_template(prompt_template_str)
retrieval_chain = create_retrieval_chain(
    retriever,
    create_stuff_documents_chain(llm, prompt)
)

# --- CHATBOT ARAYÜZÜ ---
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if user_prompt := st.chat_input("Sorunuzu buraya yazabilirsiniz..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
    with st.chat_message("assistant"):
        with st.spinner("Düşünüyorum..."):
            response = retrieval_chain.invoke({"input": user_prompt})
            answer = response["answer"]
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

