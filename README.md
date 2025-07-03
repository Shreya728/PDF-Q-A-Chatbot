# PDF Chatbot Application

## Overview
The PDF Chatbot is a powerful web application built using Streamlit. It allows users to upload PDF files, chat with the system, and extract valuable information from the files using a chatbot interface. The chatbot utilizes the Groq API for chat completions and integrates with a Chroma Vector Database for document similarity search.

## Features
- **User Authentication**: Secure login and registration functionality.
- **File Upload**: Upload PDF files to enable file-based interactions.
- **Chat Management**: Users can start new chats, load previous chats, export chat history, and delete chat history.
- **Contextual Chat**: The chatbot responds to user queries based on file content using advanced AI-based models.
- **Analytics**: Track user activities and chat statistics such as the number of chats and activities performed.
- **Styling**: A custom CSS theme is applied to ensure a clean and modern user interface.

## Tech Stack
- **Streamlit**: Web framework for building interactive applications.
- **Groq API**: API for natural language processing and AI-based chat completions.
- **Chroma Vector Database**: A vector database for storing and searching document embeddings.
- **PostgreSQL**: Database for storing user activities, chat history, and analytics data.
- **Python**: Backend programming language.

## Requirements
- Python 3.x
- Streamlit
- Groq API Key
- PostgreSQL database
- psycopg2 (for PostgreSQL connection)
- Chroma vector database
- dotenv (for environment variable management)

## Setup

Install dependencies
You can install the required dependencies using pip:

pip install -r requirements.txt

Set up the environment variables
Create a .env file in the root directory and add the following environment variables:

GROQ_API_KEY=your_groq_api_key
DATABASE_URL=postgresql://username:password@localhost:5432/pdf_chatbot

Run the application
To run the Streamlit app, use the following command:

streamlit run app.py

File Processing
The uploaded PDF files are processed to extract document text and generate embeddings for context-based search. Each document is broken down into smaller chunks, and their vector representations are stored in the Chroma Vector Database.

Chat Management
New Chat: Start a fresh chat session.

Load Chat: Load a previously saved chat from your account.

Export Chat History: Download the chat history as a .txt file.

Delete Chat: Remove the selected chat and associated data from the database.

Analytics
The system tracks user activities, such as:

Total number of chats

Total number of activities performed

These statistics are available in the Analytics section of the app.

Custom CSS
The application comes with a custom CSS theme to enhance the user interface, including:

A clean, minimalist design

Chat bubbles with timestamps

Upload file section with error handling for file sizes greater than 10MB

User activity tracking

Conclusion
The PDF Chatbot allows users to interact with PDFs using advanced AI models, providing a seamless and intuitive chat experience. Whether you're looking to summarize documents, ask questions, or search for information, this tool offers a user-friendly interface for effective communication.



