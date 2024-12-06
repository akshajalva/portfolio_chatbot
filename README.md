# Interactive AI Chatbot for Personal Portfolio #

Developed an interactive AI-powered chatbot that enhances the user experience on my personal portfolio website. The chatbot is designed to answer queries about my professional background, skills, work experience, projects, and interests. It uses a combination of advanced natural language processing (NLP) and retrieval-augmented generation (RAG) to provide accurate and context-aware responses.

Key Features:
  •	Context-Aware Responses: The chatbot strictly responds to questions based on the predefined knowledge base, ensuring accurate and relevant answers.
  •	Knowledge Base Integration: It uses a custom knowledge base that includes information about my professional journey, skills, work experience, and personal interests.
  •	Retrieval-Augmented Generation (RAG): The chatbot leverages a FAISS-based vector store and embeddings from sentence-transformers to retrieve relevant context before generating responses.
  •	Customizable Design: It is integrated into the website with a user-friendly interface, styled to align with the portfolio's aesthetics.
  
Technical Highlights:
  •	Backend Technology: The chatbot backend is built using FastAPI, ensuring a fast and reliable API for handling user queries.
  •	AI Model: Utilizes llama3.2 via Groq Cloud API for generating responses.
  •	Vector Store: Powered by FAISS (Facebook AI Similarity Search) to manage and query embeddings effectively.
  •	Deployment: Hosted on Render, providing a scalable and secure environment to serve the chatbot.
  •	Frontend Integration: The chatbot seamlessly integrates with my GitHub Pages-hosted portfolio, allowing real-time interaction with visitors.
