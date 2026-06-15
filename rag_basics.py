 
# ============================================================
# rag beginner notes
# file name: rag_basics.py
# ============================================================

# rag stands for retrieval augmented generation.
# retrieval means finding the most relevant document.
# augmented means adding that document as extra context.
# generation means producing an answer using that context.

# simple flow:
# user question -> search documents -> find best document -> answer from document

# this beginner file only does the retrieval part.
# it finds the best matching school document for a student question.

# install first if needed:
# pip install sentence-transformers scikit-learn


# ============================================================
# 1. imports
# ============================================================

# sentence_transformers turns text into embeddings.
# embedding means a list of numbers that represents meaning.

from sentence_transformers import SentenceTransformer

# cosine_similarity compares embeddings.
# higher similarity means closer meaning.

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 2. sample school documents
# ============================================================

# each document has a title and text.
# in a real rag system, these could come from pdf files,
# school websites, handbooks, or database records.

documents = [
    {
        "title": "housing policy",
        "text": "students may cancel housing before august 1 without penalty."
    },
    {
        "title": "meal plan policy",
        "text": "students can change meal plans during the first two weeks of the semester."
    },
    {
        "title": "course withdrawal policy",
        "text": "students may withdraw from a course before the official withdrawal deadline."
    },
    {
        "title": "library policy",
        "text": "students may borrow library books for 21 days using their student id card."
    },
    {
        "title": "parking policy",
        "text": "students must register their cars before parking on campus."
    },
    {
        "title": "tuition payment policy",
        "text": "students must pay tuition before the payment deadline to avoid late fees."
    }
]


# ============================================================
# 3. load embedding model
# ============================================================

# this model converts text into meaning-based numbers.
# the first time it runs, it may download the model.

model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# 4. prepare document texts
# ============================================================

# take only the text part from each document.
# result is a list of strings.

texts = [doc["text"] for doc in documents]


# ============================================================
# 5. create document embeddings
# ============================================================

# encode converts each document text into an embedding.
# doc_embeddings stores the meaning of all documents as numbers.

doc_embeddings = model.encode(texts)


# ============================================================
# 6. search function
# ============================================================

# this function receives a user question.
# it converts the question into an embedding.
# then it compares the question with all documents.
# then it returns the best matching document.

def retrieve_best_document(question):

    question_embedding = model.encode(question)

    scores = cosine_similarity(
        [question_embedding],
        doc_embeddings
    )[0]

    best_index = scores.argmax()
    best_score = scores[best_index]
    best_doc = documents[best_index]

    return best_doc, best_score


# ============================================================
# 7. simple answer function
# ============================================================

# this function creates a basic answer using the retrieved document.
# this is not a real large language model answer yet.
# it simply returns the most relevant document text.

def answer_question(question):

    best_doc, best_score = retrieve_best_document(question)

    print("\nquestion:")
    print(question)

    print("\nbest matching source:")
    print(best_doc["title"])

    print("\nsimilarity score:")
    print(round(float(best_score), 3))

    print("\nanswer:")
    print(best_doc["text"])


# ============================================================
# 8. example questions
# ============================================================

# these examples show how rag finds related documents
# even when the exact words are different.

answer_question("can i leave housing?")
answer_question("how long can i keep a library book?")
answer_question("can i change my meal plan?")
answer_question("when do i need to pay tuition?")
answer_question("where can i park my car?")


# ============================================================
# 9. interactive chatbot style
# ============================================================

# this loop lets the user ask questions.
# type quit to stop.

while True:

    user_question = input("\nask a school question, or type quit: ")

    if user_question.lower() == "quit":
        print("goodbye")
        break

    answer_question(user_question)


# ============================================================
# summary
# ============================================================

# rag                 -> retrieval augmented generation
# document            -> stored information
# query/question      -> what the user asks
# embedding           -> numbers representing meaning
# sentence transformer -> model that creates embeddings
# cosine similarity   -> compares meaning between texts
# retrieve            -> find the most relevant document
# context             -> retrieved information used for answering
 
