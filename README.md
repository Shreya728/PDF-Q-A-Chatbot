# PDF Chatbot Application
#live link=https://pdfbot-e6kdh2pbrr3vbjorrutebu.streamlit.app/

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

HERE ARE THE SCREENSHOTS ATTACHED FOR THE REFERENCE
1 Account creation
![image](https://github.com/user-attachments/assets/8a8772a0-eb34-46be-b02d-6c4b0a2801f7)
2 Account login
![image](https://github.com/user-attachments/assets/baadb4f1-edc6-4f17-b532-b8e95f6852cb)
3 Web interface
![image](https://github.com/user-attachments/assets/c2f402b2-345e-4966-bc7f-33ebd1aa2f92)
4 Uploading of pdf and its processing
![image](https://github.com/user-attachments/assets/dcd7948e-9d13-4de1-be85-7956a6b02c20)
5 Question can be asked related to pdf once uploaded and Ai bot will give answers
![image](https://github.com/user-attachments/assets/63fd6de4-362a-404b-a889-e01e1d19873b)
6 In chat management you can select your previous chats and see the history
![image](https://github.com/user-attachments/assets/aa0b3672-e244-4095-bce7-70a784a7c30c)
7 You can delete the chat by first selecting chat ,loading chat and clicking on delete chat
![image](https://github.com/user-attachments/assets/c4aef7ac-7bd8-4f9c-b0a3-31f732735368)
8 You can click on export the chat by clicking on export chat button and the text file  will be downloaded
![image](https://github.com/user-attachments/assets/a8635447-3131-4580-bf2f-e655734c88ba)
9 By clicking on new chat ,new chat can be established
![image](https://github.com/user-attachments/assets/4f916aad-554b-4458-845e-856a54df62db)
10 click on logout to exit the web app
![image](https://github.com/user-attachments/assets/a9c9c20a-29db-4213-8ce4-2cbbc7189c34)















