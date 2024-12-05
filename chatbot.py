# Import necessary libraries and modules
# from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import retrieval
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import chardet
from docx import Document as DocxDocument
from langchain.schema import Document as LangchainDocument
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Initialize FastAPI app
app = FastAPI()

# Load environment variables from a .env file
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize the Groq model using the provided API key
model = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.2-3b-preview")

# Enable CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace '*' with specific frontend URLs if you know them
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to read and chunk the document based on specific sections
def read_and_chunk_document(file_path):
    # Open and read the .docx file
    doc = DocxDocument(file_path)
    text = '\n'.join([para.text for para in doc.paragraphs])

    # Sections to split the document by
    sections = ["About Akshaj", "Education", "Work Experience", "Skills", "Career Gap", "Projects", "Personal Interests"]

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

# Initialize the embedding model using HuggingFace embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Try to create the FAISS vector store from the documents using the embedding model
try:
    vector_store = FAISS.from_documents(documents, embedding=embedding_model)
    # print("Vector Store created successfully.")
except Exception as e:
    print(f"Error creating Vector Store: {e}")
    raise

# Step 3: Create the Retrieval-Augmented Generation (RAG) Chain

# Initialize the retriever from the vector store
retriever = vector_store.as_retriever()

# Define the prompt template for the chatbot
template = """
You are Akshaj Alva. 
Your job is to provide precise and engaging responses based on the user's query and Akshaj's knowledge base. 
Use the provided context to answer queries.

Context:
{context}

Instructions:
- Respond in a conversational tone.
- If the query is unrelated to the context, politely inform the user.
- If the context is insufficient, ask follow-up questions or request clarification.
- Keep responses concise but informative. 
- Ensure factual correctness and align with Akshaj's professional and personal details.

User's Question: {input}

Answer:
"""

# Create the prompt from the template
prompt = ChatPromptTemplate.from_template(template)

# Define the Retrieval QA Chain to process the documents and answer queries
document_chain = create_stuff_documents_chain(model, prompt)
qa_chain = retrieval.create_retrieval_chain(retriever, document_chain)

# Function to handle user input during the conversation (local testing)
def handle_conversation():
    print("Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Debugging retrieval: Retrieve relevant documents for the user input
        retrieved_docs = retriever.get_relevant_documents(user_input)

        # Format the prompt with the retrieved context
        formatted_prompt = prompt.format(context="\n".join([doc.page_content for doc in retrieved_docs]), input=user_input)

        # Attempt to retrieve the answer from the QA chain
        try:
            result = qa_chain.invoke({"input": user_input})
            print("Bot Response:", result.get("answer", "I don't know."))
        except Exception as e:
            print(f"Error during model invocation: {e}")

# Step 5: Define a Pydantic Model for Input (used in FastAPI)
class ChatRequest(BaseModel):
    input: str

# Step 6: Create the FastAPI endpoint for chatbot interaction
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    # Get user input from the API request
    user_input = request.input
    
    # Retrieve the answer using the QA chain
    result = qa_chain.invoke({"input": user_input})
    
    # Return the response to the user
    return {"response": result.get('answer', "I don't have that information about Akshaj.")}

# Run the conversation handling function if running locally
if __name__ == "__main__":
    handle_conversation()
