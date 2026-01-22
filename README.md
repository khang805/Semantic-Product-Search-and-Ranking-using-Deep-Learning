# Semantic Product Search and Ranking System

A deep learning-based search engine that moves beyond traditional keyword matching to understand the **semantic intent** behind user queries. Built using the Amazon ESCI (Shopping Queries) dataset, this system employs Transformer-based architectures to retrieve and rank products based on their contextual relevance.

---

## 🚀 Key Features

* **Semantic Understanding:** Captures the underlying meaning of natural language queries to match products even when keywords don't overlap exactly.
* **Interactive Question-Answering (QA):** Built-in capability to answer specific technical or product-related questions directly from the product catalogs and descriptions.
* **Feature Fusion:** Combines `product_title` and `product_description` into a unified high-fidelity text representation for training and inference.
* **Cross-Encoder Re-ranking:** Implements a BERT-based Cross-Encoder to provide high-precision relevance scores for query-product pairs.
* **Real-time Web Interface:** Deployed via Gradio, allowing users to enter queries and view ranked results instantly.

---


## 📂 Repository Structure

```text
├── ProductSearch_Training.ipynb   # Model training, fine-tuning, and evaluation logic
├── ProductSearch_Inference.ipynb  # Gradio deployment and real-time inference pipeline
├── esci_model/                    # Saved weights and config for the trained model
├── training_curves.png            # Visualization of loss and accuracy over epochs
└── evaluation_metrics.png         # Performance graphs (NDCG, MAP, etc.)
```

## ⚙️ Installation & Setup

### 1. Prerequisites
Environment: Kaggle or Google Colab (NVIDIA Tesla T4 GPU recommended).

Dataset: Amazon Shopping Queries (ESCI).

### 2. Install Dependencies

Bash
pip install -q sentence-transformers gradio scikit-learn nltk gensim matplotlib seaborn

## 🛠️ Technical Workflow

### 1. Preprocessing & Representation

The pipeline cleans raw text by converting it to lowercase, removing stop words, and applying lemmatization. It transforms text using three levels of representation:

      1) Statistical: TF-IDF vectors.
      2) Semantic: Word embeddings (Word2Vec/FastText).
      3) Contextual: Pretrained BERT/Transformer models.

### 2. Training & Fine-Tuning:

The model is trained on a 70/15/15 split of the Amazon ESCI dataset. It uses a Cross-Encoder strategy where the query and product text are passed into the Transformer simultaneously to learn a deep interaction between them, rather than comparing independent vectors.

### 3. Inference & QA Mode

When a user enters a query into the Gradio interface:

       1) Retrieval: The system identifies candidate products from the catalog.
       2) Ranking: The Cross-Encoder assigns a relevance score to each pair.
       3) QA Logic: If the query is a question (e.g., "Which mouse has the longest battery life?"), the system extracts the specific answer from the ranked product descriptions.


## 📊 Performance Metrics

To ensure high-fidelity ranking, the model is evaluated using industry-standard metrics:

| Metric | System Result | Description |
| :--- | :--- | :--- |
| **NDCG** | High | Measures ranking quality based on graded relevance. |
| **MAP** | Optimized | Evaluates the mean precision across all queries. |
| **Precision@K** | Measured | Accuracy of the top K returned results. |
| **Recall@K** | Measured | Ability to find all relevant items in top K. |


## 🎯 Conclusion

This project demonstrates the transition from "Search" to "Understanding." By leveraging BERT-based architectures, the system provides a more intuitive shopping experience where the machine understands the user's intent, effectively linking natural language queries to the most relevant items in a massive product catalog.


## 🎓 Author

M Abdurrahman Khan AI Engineer | LLMs, RAG Pipelines & Computer Vision

National University of Computer and Emerging Sciences (FAST), Pakistan Contact: {i221148}@nu.edu.pk
