# 📚 Akbank GenAI Bootcamp: Türkçe Wikipedia Destekli RAG Chatbot

Bu proje, Akbank GenAI Bootcamp: Yeni Nesil Proje Kampı kapsamında geliştirilmiştir. Proje, kullanıcıların sorduğu sorulara Türkçe Wikipedia veritabanını kaynak olarak kullanarak cevap veren, **RAG (Retrieval-Augmented Generation)** tabanlı bir sohbet robotudur.

## 🚀 Canlı Demo

Uygulamanın çalışan versiyonuna aşağıdaki linkten ulaşabilirsiniz:

[**➡️ UYGULAMAYI ZİYARET ET**](https://www.google.com/search?q=... "null") `<!-- Buraya Streamlit Cloud veya Hugging Face Spaces linkinizi ekleyin -->`



![](C:\Users\samet\AppData\Roaming\marktext\images\2025-10-20-11-52-23-image.png)

## 🎯 Projenin Amacı

Bu projenin temel amacı, geniş bir bilgi kaynağı olan Türkçe Wikipedia'yı bir hafıza olarak kullanarak, kullanıcı sorularına doğru, kanıta dayalı ve bağlama uygun yanıtlar üretebilen bir yapay zeka sistemi geliştirmektir. Geleneksel dil modellerinin bilgi eksikliği veya güncel olmama sorununu, RAG mimarisi ile aşarak daha güvenilir bir sohbet deneyimi sunmak hedeflenmiştir.

## 🛠️ Çözüm Mimarisi ve Kullanılan Teknolojiler

Proje, temel olarak bir "Getirme" (Retrieval) ve "Üretme" (Generation) adımlarından oluşan RAG mimarisini takip eder.

1. **Getirme (Retrieval):**
   
   - Kullanıcının sorusu, bir **Embedding Modeli** ile anlamsal bir vektöre dönüştürülür.
   
   - Bu vektör, önceden işlenmiş ve vektörleştirilmiş Wikipedia metin parçacıklarının saklandığı **Vektör Veritabanı** içerisinde anlamsal olarak en yakın ve en alakalı metinleri bulmak için kullanılır.

2. **Üretme (Generation):**
   
   - Bulunan bu alakalı metin parçacıkları ("context" olarak) ve kullanıcının orijinal sorusu birleştirilerek bir **Üretken Dil Modeli'ne (LLM)** gönderilir.
   
   - LLM, kendisine sunulan bu bağlamı kullanarak nihai ve tutarlı cevabı üretir.

**Kullanılan Teknolojiler:**

- **Üretken Model (Generation):** Google Gemini Pro

- **Embedding Modeli:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

- **Vektör Veritabanı:** FAISS (Facebook AI Similarity Search)

- **RAG Pipeline Framework:** LangChain

- **Web Arayüzü:** Streamlit

- **Deployment:** Streamlit Community Cloud (veya Hugging Face Spaces)

## 📦 Veri Seti

- **Veri Seti:** `wikimedia/wikipedia`

- **Kaynak:** Hugging Face Datasets

- **Açıklama:** Projede, Hugging Face platformu üzerinden erişilen resmi Wikipedia veri setinin Türkçe (`20231101.tr`) versiyonu kullanılmıştır. Bu veri seti, 534,988 adet Türkçe Wikipedia makalesi içermektedir. Projenin geliştirme ve prototipleme aşamalarında verimlilik sağlamak amacıyla bu makalelerin ilk **20,000** tanesi seçilmiş ve işlenmiştir. Bu işlem sonucunda **144,739 adet** aranabilir metin parçası (chunk) oluşturulmuştur.

## 📊 Elde Edilen Sonuçlar

Geliştirilen chatbot, kendisine sağlanan Wikipedia verileri dahilinde aşağıdaki gibi sorulara başarılı bir şekilde cevap verebilmektedir. Model, hafızasında bulunmayan konularda ise "Bu konuda hafızamda bir bilgi bulunmuyor." diyerek halüsinasyon görmekten kaçınmaktadır.

**Başarılı Cevap Örnekleri:**

> **👤 Kullanıcı:** Leonardo da Vinci kimdir?
> 
> **🤖 Chatbot:** Leonardo da Vinci, Rönesans dönemi İtalyan mimarı, mühendisi, mucidi, matematikçisi, anatomisti, müzisyeni, heykeltıraşı ve ressamıdır. Rönesans döneminde yaşamış İtalyan hezârfen, döneminin önemli bir filozofu, astronomu, mimarı, mühendisi, mucidi, matematikçisi, anatomisti, müzisyeni, heykeltıraşı, botanisti, jeoloğu, kartografı, yazarı ve ressamıdırhttps://www.google.com/search?q=...

> **👤 Kullanıcı:** Galata Kulesi'nin tarihi hakkında bilgi verir misin?
> 
> **🤖 Chatbot:** Galata Kulesi, 528 yılında Bizans İmparatoru Anastasius tarafından inşa edilmiştir. 1204 yılında 4. Haçlı Seferleri ile büyük ölçüde tahrip olmuş, 1348 yılında İsa Kulesi olarak Cenevizliler tarafından Galata Surlarına tekrar ek olarak yapılmıştırhttps://www.google.com/search?q=...

**Sınırları Test Eden Cevap Örneği:**

> **👤 Kullanıcı:** 2024 yılındaki en popüler video oyunu hangisidir?
> 
> **🤖 Chatbot:** Bu konuda hafızamda bir bilgi bulunmuyor.
> 
> *(**Not:** Bu cevap doğrudur, çünkü kullanılan veri seti 2023 tarihlidir ve geleceğe dair bilgi içermemektedir.)*

## 🚀 Kurulum ve Çalıştırma Kılavuzu

Bu projeyi kendi lokal makinenizde çalıştırmak için aşağıdaki adımları takip edebilirsiniz:

1. **Depoyu Klonlayın:**
   
   ```
   git clone [https://github.com/abdulsametkara/akbank-genai-rag-chatbot.git](https://github.com/abdulsametkara/akbank-genai-rag-chatbot.git)
   cd akbank-genai-rag-chatbot
   ```

2. **Sanal Ortam Oluşturun ve Aktif Edin (Önerilir):**
   
   ```
   python -m venv venv
   source venv/bin/activate  # Windows için: venv\Scripts\activate
   ```

3. **Gerekli Kütüphaneleri Yükleyin:** Proje için gerekli tüm kütüphaneler `requirements.txt` dosyasında listelenmiştir.
   
   ```
   pip install -r requirements.txt
   ```

4. **API Anahtarını Ayarlayın:** Projenin çalışması için bir Google Gemini API anahtarına ihtiyacınız vardır. Projenin ana dizininde `.env` adında bir dosya oluşturun ve içine anahtarınızı aşağıdaki gibi ekleyin:
   
   ```
   # .env
   GOOGLE_API_KEY="AIzaSy..."
   ```

5. **Uygulamayı Çalıştırın:**
   
   ```
   streamlit run app.py
   ```
   
   Uygulama `http://localhost:8501` adresinde açılacaktır.

## 🎬 Ürün Kullanım Kılavuzu

Uygulama başlatıldığında, sizi basit bir sohbet arayüzü karşılayacaktır.

1. Sayfanın alt kısmında bulunan "Sorunuzu buraya yazabilirsinizhttps://www.google.com/search?q=..." metin kutusuna merak ettiğiniz konuyu yazın.

2. Enter'a basın veya sağdaki gönder ikonuna tıklayın.

3. Chatbot, "Düşünüyorumhttps://www.google.com/search?q=..." mesajını gösterdikten sonra Wikipedia'dan bulduğu bilgilerle oluşturduğu cevabı ekrana yazdıracaktır.
