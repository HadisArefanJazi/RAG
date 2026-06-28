"""
Complete RAG Tutorial
==============================

RAG = Retrieval-Augmented Generation

Simple meaning:

    Search first.
    Answer second.

A RAG system answers questions using your own documents.

Basic RAG flow:

    1. Load documents
    2. Split documents into chunks
    3. Convert chunks into embeddings
    4. Store/search embeddings
    5. Retrieve top-k relevant chunks
    6. Optionally rerank chunks
    7. Build a grounded prompt
    8. Ask the LLM to answer only from context
    9. Return answer with sources
    10. Evaluate and update the system

This file teaches the core ideas with simple Python.

Libraries used:
---------------

1. scikit-learn

    Install:
        pip install scikit-learn

    Used for:
        - TF-IDF keyword search
        - cosine similarity

    TfidfVectorizer:
        Turns text into word-importance vectors.

    cosine_similarity:
        Measures similarity between vectors.

2. sentence-transformers

    Install:
        pip install sentence-transformers

    Used for:
        - Semantic embeddings

    SentenceTransformer:
        Converts text into meaning-based vectors.

3. No vector database is used here.

    In real projects, you may use:
        - FAISS
        - Chroma
        - Pinecone
        - Weaviate

"""


# ============================================================
# 1. IMPORTS
# ============================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# ============================================================
# 2. DOCUMENTS
# ============================================================

"""
Documents are the knowledge source.

In real RAG, documents may come from:
    - PDFs
    - websites
    - databases
    - company policies
    - school handbooks
    - help center articles
"""

documents = [
    {
        "title": "housing policy",
        "text": "Students may cancel housing before August 1 without penalty. After August 1, cancellation may include a fee."
    },
    {
        "title": "meal plan policy",
        "text": "Students can change meal plans during the first two weeks of the semester."
    },
    {
        "title": "library policy",
        "text": "Students may borrow library books for 21 days using their student ID card."
    },
    {
        "title": "parking policy",
        "text": "Students must register their cars before parking on campus."
    },
    {
        "title": "tuition payment policy",
        "text": "Students must pay tuition before the payment deadline to avoid late fees."
    },
    {
        "title": "international student policy",
        "text": "International students must submit passport, visa documents, financial proof, and English proficiency scores."
    },
    {
        "title": "spring enrollment deadline",
        "text": "The enrollment deadline for Spring 2026 is January 20, 2026."
    },
]


# ============================================================
# 3. CHUNKING
# ============================================================

"""
Chunking means splitting long documents into smaller pieces.

Why chunking matters:

    Bad:
        Search one huge PDF page or full document.

    Better:
        Search smaller chunks.

Small chunks help retrieval find the exact useful part.

Common chunk sizes:
    - 200 to 500  words for simple documents
    - 500 to 1000 words for longer technical documents

This example uses a very simple sentence-based chunker.
"""


def chunk_documents(documents):
    chunks = []

    for doc in documents:
        sentences = doc["text"].split(".")

        for sentence in sentences:
            sentence = sentence.strip()

            if sentence:
                chunks.append({
                    "source": doc["title"],
                    "text": sentence + "."
                })

    return chunks


chunks = chunk_documents(documents)


# ============================================================
# 4. TF-IDF RETRIEVAL
# ============================================================

"""
TF-IDF retrieval is keyword-based.

It works well when the question and document use similar words.

Example:
    Question:
        When do I pay tuition?

    Matching text:
        Students must pay tuition before the payment deadline.

Weakness:
    It may fail when different words mean the same thing.

Example:
    "leave housing" and "cancel housing"
"""


def search_tfidf(question, top_k=3):
    
    texts           = [chunk["text"] for chunk in chunks]

    vectorizer      = TfidfVectorizer(stop_words="english") 
    
    # stop_words="english" removes common English words like:  "the", "is", "and", "a"

    chunk_vectors   = vectorizer.fit_transform(texts) 

    # This does two things: fit learns the vocabulary from texts and transform turns each text into a numeric vector.
    
    question_vector = vectorizer.transform([question]) 
    
    # turns the question into numbers using the same vocabulary learned from texts.

    scores          = cosine_similarity(question_vector, chunk_vectors)[0]

    # This compares the question vector against every chunk vector.
    # The [0] is used because cosine_similarity() returns a 2D array.
    
    ranked_indexes  = scores.argsort()[::-1]
    
    # scores.argsort() gives lowest to highest. Then [::-1] reverses it.
    
    results = []

    for index in ranked_indexes[:top_k]:
        results.append({
            "source": chunks[index]["source"],
            "text":   chunks[index]["text"],
            "score":  float(scores[index])
        })

    return results


# ============================================================
# 5. EMBEDDINGS AND SEMANTIC RETRIEVAL
# ============================================================

"""
Embedding:
    A list of numbers that represents meaning.

Semantic search:
    Searches by meaning, not only exact words.

Example:
    "Can I leave housing?"
    can match:
    "Students may cancel housing before August 1."

Because:
    leave housing ≈ cancel housing

This is usually better than TF-IDF for natural questions.
"""


def search_semantic(question, top_k=3):
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        )

    model              = SentenceTransformer("all-MiniLM-L6-v2")

    texts              = [chunk["text"] for chunk in chunks]

    chunk_embeddings   = model.encode(texts)
    question_embedding = model.encode([question])

    scores = cosine_similarity(question_embedding, chunk_embeddings)[0]

    ranked_indexes = scores.argsort()[::-1]

    results = []

    for index in ranked_indexes[:top_k]:
        results.append({
            "source": chunks[index]["source"],
            "text": chunks[index]["text"],
            "score": float(scores[index])
        })

    return results


# ============================================================
# 6. TOP-K RETRIEVAL
# ============================================================

"""
top_k means how many chunks we retrieve.

Example:
    top_k = 1
        Return only the best chunk.

    top_k = 3
        Return the best 3 chunks.

Choosing top_k:

    Too small:
        The answer may miss useful context.

    Too large:
        The LLM may receive irrelevant context.

Beginner default:
    top_k = 3
"""


# top_k is already included in search_tfidf() and search_semantic().


# ============================================================
# 7. SIMPLE RERANKING
# ============================================================

"""
Reranking means sorting retrieved chunks again to improve relevance.

In advanced RAG, rerankers use special models.

Here we use a simple beginner reranker:
    - Prefer chunks that contain important question words.
    
This is not perfect, but it teaches the idea!
"""


def rerank_simple(question, results):
    question_words = question.lower().split()

    for result in results:
        text = result["text"].lower()

        bonus = 0

        for word in question_words:
            if word in text:
                bonus += 0.05

        result["rerank_score"] = result["score"] + bonus

    results = sorted(
        results,
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return results


# ============================================================
# 8. PROMPT GROUNDING
# ============================================================

"""
Prompt Grounding is where RAG prepares the prompt that will be sent to the LLM.

Before asking the LLM, RAG already searched the documents and found.

The LLM does not search documents itself. 

It only receives a prompt containing the retrieved information and the user's question.

Grounding means forcing the LLM to use only retrieved context.

Good RAG prompt rule:

    Answer only from the context.
    If the answer is missing, say you do not know.

This reduces hallucination.
"""


def build_grounded_prompt(question, results):
    context = ""

    for result in results:
        context += f"Source: {result['source']}\n"
        context += f"Text:   {result['text']}\n\n"

    prompt = f"""
You are a careful assistant.

Use only the context below to answer the question.

If the answer is not in the context, say:
"I do not have enough information in the provided context."

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt.strip()


# ============================================================
# 9. HALLUCINATION CONTROL
# ============================================================

"""
Hallucination means the AI makes up an answer.

RAG reduces hallucination by:
    - retrieving relevant context
    - using a grounded prompt
    - setting a similarity threshold
    - returning sources

Threshold:
    If the best retrieval score is too low, do not answer.
"""


def rag_answer(question, method="tfidf", top_k=3, threshold=0.20):
    if method == "tfidf":
        results = search_tfidf(question, top_k=top_k)
    elif method == "semantic":
        results = search_semantic(question, top_k=top_k)
    else:
        raise ValueError("method must be 'tfidf' or 'semantic'")

    results = rerank_simple(question, results)

    best_result = results[0]

    prompt = build_grounded_prompt(question, results)

    if best_result["score"] < threshold:
        answer = "I do not have enough information in the provided context."
        sources = []
    else:
        answer = best_result["text"]
        sources = [best_result["source"]]

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "best_score": round(best_result["score"], 3),
        "retrieved_chunks": results,
        "prompt_for_llm": prompt
    }


# ============================================================
# 10. CITATIONS / SOURCES
# ============================================================

"""
A good RAG answer should show sources.

Bad answer:
    You can cancel housing before August 1.

Better answer:
    You can cancel housing before August 1.
    Source: housing policy

Sources help the user trust and verify the answer.
"""


def print_rag_response(response):
    print("=" * 70)

    print("Question:")
    print(response["question"])

    print("\nAnswer:")
    print(response["answer"])

    print("\nSources:")
    print(response["sources"])

    print("\nBest score:")
    print(response["best_score"])

    print("\nRetrieved chunks:")

    for chunk in response["retrieved_chunks"]:
        print("-" * 50)
        print("Source:", chunk["source"])
        print("Text:", chunk["text"])
        print("Score:", round(chunk["score"], 3))


# ============================================================
# 11. EVALUATION
# ============================================================

"""
Evaluation means testing whether your RAG system works.

You should test:

1. Retrieval quality
    Did the system find the right chunk?

2. Answer quality
    Did the answer correctly use the retrieved context?

3. Refusal quality
    Did it say "not enough information" when context was missing?

A simple evaluation set contains:
    - question
    - expected source
"""


test_questions = [
    {
        "question": "when can I cancel housing?",
        "expected_source": "housing policy"
    },
    {
        "question": "how long can I borrow a library book?",
        "expected_source": "library policy"
    },
    {
        "question": "what documents do international students need?",
        "expected_source": "international student policy"
    },
    {
        "question": "what is the football schedule?",
        "expected_source": None
    },
]


def evaluate_tfidf():
    correct = 0

    for test in test_questions:
        response = rag_answer(test["question"], method="tfidf")

        predicted_source = None

        if response["sources"]:
            predicted_source = response["sources"][0]

        if predicted_source == test["expected_source"]:
            correct += 1

        print("-" * 70)
        print("Question:", test["question"])
        print("Expected:", test["expected_source"])
        print("Predicted:", predicted_source)

    accuracy = correct / len(test_questions)

    print("\nAccuracy:", round(accuracy, 2))


# ============================================================
# 12. UPDATING DOCUMENTS
# ============================================================

"""
RAG systems must stay fresh.

If a policy changes, update the documents and rebuild the chunks/index.

Example:
    Old:
        Deadline is January 20, 2026.

    New:
        Deadline is January 25, 2026.

In real systems:
    - reload documents
    - re-chunk them
    - recreate embeddings
    - update the vector database
"""


def add_document(title, text):
    documents.append({
        "title": title,
        "text": text
    })

    global chunks
    chunks = chunk_documents(documents)


# ============================================================
# 13. MAIN EXAMPLES
# ============================================================

if __name__ == "__main__":
    print("Complete Beginner RAG Tutorial")

    questions = [
        "when can I cancel housing?",
        "how long can I borrow a library book?",
        "can I change my meal plan?",
        "what documents do international students need?",
        "what is the football schedule?"
    ]

    for question in questions:
        response = rag_answer(question, method="tfidf")
        print_rag_response(response)

    print("\nRAG Prompt Example:")
    response = rag_answer("when can I cancel housing?", method="tfidf")
    print(response["prompt_for_llm"])

    print("\nEvaluation:")
    evaluate_tfidf()

    print("\nSemantic search is optional.")
    print("Install it with:")
    print("pip install sentence-transformers")

    print("\nThen test:")
    print("rag_answer('can I leave housing?', method='semantic')")
