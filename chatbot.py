from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import retrieval 
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import chardet
from docx import Document
from langchain.schema import Document as LangchainDocument

app = FastAPI()

# # Step 2: Enable CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Replace '*' with specific frontend URLs if you know them
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Step 1: Initialize the Model


# Step 1: Load and Process the Document

def read_and_chunk_document(file_path):
    # Open and read the .docx file
    doc = Document(file_path)
    text = '\n'.join([para.text for para in doc.paragraphs])

    # Sections to split the document by
    sections = ["About Me", "Education", "Work Experience", "Skills", "Career Gap", "Projects", "Personal Interests"]

    # Split the document based on these sections
    chunks = []
    start_idx = 0
    for section in sections:
        start_idx = text.find(section, start_idx)
        if start_idx != -1:
            # Get the end index for the section (next section start or end of document)
            end_idx = text.find(sections[sections.index(section) + 1], start_idx) if sections.index(section) + 1 < len(sections) else len(text)
            chunks.append(text[start_idx:end_idx].strip())

    return chunks

# Load and chunk the document
file_path = 'Aboutme.docx'  # Replace with your .docx file path
chunked_documents = read_and_chunk_document(file_path)

# Create LangChain documents from chunks
documents = [LangchainDocument(page_content=chunk) for chunk in chunked_documents]
# print("Loaded Documents:", documents)


model = OllamaLLM(model='llama3.2')
# Step 2: Initialize the Embedding Model and Vector Store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


try:
    vector_store = FAISS.from_documents(documents, embedding=embedding_model)
    # print("Vector Store created successfully.")
except Exception as e:
    print(f"Error creating Vector Store: {e}")
    raise


# Step 3: Create the Retrieval-Augmented Generation (RAG) Chain
retriever = vector_store.as_retriever()

# Define the prompt template
template = """
You are a friendly and knowledgeable chatbot that provides precise and concise answers about Akshaj Alva. 
You can only respond based on the profile information provided below. 
If a question is outside the scope of the given profile or if the information is not available, you should reply with: "I don't have that information about Akshaj"."

Conext:
{context}

Your responses should:
1. Be short, to the point, and conversational.
2. Provide clear and relevant answers based solely on the context provided.
3. Not introduce new information or speculate.
4. Maintain a friendly and approachable tone.

User's Question: {input}


Answer:
"""

prompt = ChatPromptTemplate.from_template(template)


# Define the Retrieval QA Chain
document_chain = create_stuff_documents_chain(model, prompt)
qa_chain = retrieval.create_retrieval_chain(retriever, document_chain)

# Step 4: Conversation Handling
def handle_conversation():
    print("Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Debugging retrieval
        retrieved_docs = retriever.get_relevant_documents(user_input)
        # print("Retrieved Docs:", retrieved_docs)

        # Debugging the prompt
        formatted_prompt = prompt.format(context="\n".join([doc.page_content for doc in retrieved_docs]), input=user_input)
        # print("Formatted Prompt Sent to Model:", formatted_prompt)

        # Retrieve relevant context and pass it to the model
        try:
            result = qa_chain.invoke({"input": user_input})
            print("Bot Response:", result.get("answer", "I don't know."))
        except Exception as e:
            print(f"Error during model invocation: {e}")

        # Step 5: Define a Pydantic Model for Input
class ChatRequest(BaseModel):
    input: str

# Step 6: Create the API Endpoint
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    
    user_input = request.input
    result = qa_chain.invoke({"input": user_input})
    return {"response": result.get('answer', "I don't have that information about Akshaj.")}

if __name__ == "__main__":
    handle_conversation()