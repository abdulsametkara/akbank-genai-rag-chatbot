# preload_model.py
# Bu betiğin tek amacı modeli indirip Hugging Face önbelleğine kaydetmektir.
from langchain_community.embeddings import HuggingFaceEmbeddings

print("Model indiriliyor ve önbelleğe alınıyor... Lütfen bekleyin.")
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
print("✅ Model başarıyla indirildi!")