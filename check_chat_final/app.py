import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import pandas as pd
import os
import sqlite3
from datetime import datetime
from langchain_openai import ChatOpenAI
import json
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain  # Import this line
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv

# ===========================================
# Database setup functions (inside this file itself)

DB_PATH = './interaction_log_new.db'
def create_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def create_table(conn):
    with conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT,
                assistant_response TEXT,
                feedback TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

def insert_interaction(user_input, assistant_response, feedback, conn):
    with conn:
        conn.execute('''
            INSERT INTO interactions (user_input, assistant_response, feedback)
            VALUES (?, ?, ?)
        ''', (user_input, assistant_response, feedback))

# ============================================

load_dotenv()
# os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
# api_key = os.getenv("OPENAI_API_KEY")
api_key="abc"
llm_ch = ChatOllama(model="mistral")
# Initialize embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore_path = r".\data\vectorstor"
vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# Set up Streamlit
st.title("Customer Support Chtabot")
# st.write("Upload PDFs and chat with their content")

# api_key = st.text_input("Enter your Groq API key:", type="password")

# Initialize database connection
if 'db_conn' not in st.session_state:
    st.session_state.db_conn = create_connection()
conn = st.session_state.db_conn
create_table(conn)

# Initialize session state for chats and logs
if 'store' not in st.session_state:
    st.session_state.store = {}

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

# NEW - store pending feedback temporarily
if 'pending_feedback' not in st.session_state:
    st.session_state.pending_feedback = None
# api_key=
if api_key:
    # llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")
    # llm = ChatOpenAI(openai_api_key=api_key,  # Or pass the key directly
    # model="gpt-4o-mini",)  # or "gpt-4", "gpt-4-turbo", etc.
    llm=llm_ch
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history."
        "Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
    "You are a customer support agent specializing in answering queries. "
    "Your goal is to provide accurate, concise, and helpful responses.Ask question if you are not sure of the result , like how to change the account - you should ask which account . If the question is , where is my order - ask which order , order details "
    "If the user's question is off-topic (e.g., jokes, unrelated topics), do not provide an answer. "
    "If the question relates to a specific action or process (e.g., how to log in, how to use a feature), provide a step-by-step guide, if applicable. "
    "You may use relevant context from previous messages if it helps in answering the user's question. "
    "Always strive to provide a professional and helpful tone.\n\n {context}"
    
    "Here are a few examples to guide your responses:\n\n"
    
    "Example 1: Basic Question - Product Feature Inquiry\n"
    "User: Can you explain how to log into my account?\n"
    "Assistant: Sure! Here are the steps to log into your account:\n"
    "1. Open the app or website.\n"
    "2. Click on the 'Login' button at the top right corner.\n"
    "3. Enter your username and password.\n"
    "4. If you forgot your password, click on 'Forgot Password' and follow the reset instructions.\n"
    "5. After entering your credentials, click 'Submit', and you will be logged in.\n\n"
    
    "Example 2: Technical Issue - Problem with Account\n"
    "User: I am unable to reset my password. What should I do?\n"
    "Assistant: I understand the frustration. Here are some steps you can follow:\n"
    "1. Ensure you're using the correct email address associated with your account.\n"
    "2. Check your spam or junk folder for the reset email.\n"
    "3. If you still haven’t received the email, please try resubmitting your password reset request.\n"
    "4. If the issue persists, please contact support with your account details, and we'll help you further.\n\n"
    
    "Example 3: Off-Topic Question\n"
    "User: Why don't cats like water?\n"
    "Assistant: I'm sorry, but that is outside the scope of our support. If you have any questions regarding our services or products, feel free to ask!\n\n"
    
    "Example 4: Request for Summary\n"
    "User: Can you summarize how I can upgrade my account?\n"
    "Assistant: To upgrade your account:\n"
    "1. Log into your account.\n"
    "2. Go to the 'Account Settings' page.\n"
    "3. Select 'Upgrade Plan'.\n"
    "4. Choose the desired plan and payment method.\n"
    "5. Confirm your choice, and your account will be upgraded immediately.\n\n"
    
    "Example 5: Request for Contextual Clarification\n"
    "User: Can I get a discount if I subscribe for a year?\n"
    "Assistant: Based on the previous context, if you are referring to our subscription service, we do offer a 10% discount for annual subscriptions. Please check the 'Pricing' section of our website for more details.\n\n"
    
    "Example 6: General Inquiry - Product Availability\n"
    "User: Is product XYZ in stock?\n"
    "Assistant: Let me check the stock for product XYZ...\n"
    "Based on our current inventory, product XYZ is available and ready for purchase. Would you like to proceed with an order?\n\n"
    
    "Example 7: Request for Troubleshooting\n"
    "User: My app keeps crashing, what should I do?\n"
    "Assistant: I apologize for the inconvenience. Here are some troubleshooting steps you can try:\n"
    "1. Restart the app.\n"
    "2. Ensure your app is updated to the latest version.\n"
    "3. Clear the app's cache or reinstall it.\n"
    "4. If the issue persists, please contact our technical support team for further assistance.\n\n"
    
    "Example 8: Off-Topic - Unnecessary Joke\n"
    "User: Can you tell me a joke?\n"
    "Assistant: I'm sorry, but jokes are outside the scope of customer support. Please let me know if you have any questions about our services or products, and I'd be happy to assist you!\n\n"
    
    "Example 9: Clarification on Service\n"
    "User: What exactly is included in the premium plan?\n"
    "Assistant: The Premium Plan includes the following:\n"
    "1. Unlimited access to all premium features.\n"
    "2. Priority customer support.\n"
    "3. Access to exclusive content and tools.\n"
    "4. A dedicated account manager.\n"
    "For more detailed information, please visit our 'Pricing' page."
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    session_id = "default_session"

    # show chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Type your message here...")

    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})

        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id": session_id}
            },
        )
        assistant_response = response['answer']
        st.session_state.chat_messages.append({"role": "assistant", "content": assistant_response})

        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        # Save the last interaction temporarily for feedback
        st.session_state.pending_feedback = {
            "user_input": user_input,
            "assistant_response": assistant_response
        }

        st.rerun()

    # After assistant's response, show feedback options if pending
    if st.session_state.pending_feedback:
        with st.chat_message("system"):
            st.write("How was the answer?")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Like", key="like_feedback"):
                    insert_interaction(
                        st.session_state.pending_feedback['user_input'],
                        st.session_state.pending_feedback['assistant_response'],
                        "Like",
                        conn
                    )
                    st.success("Feedback recorded: Like")
                    st.session_state.pending_feedback = None
                    st.rerun()

            with col2:
                if st.button("Dislike", key="dislike_feedback"):
                    insert_interaction(
                        st.session_state.pending_feedback['user_input'],
                        st.session_state.pending_feedback['assistant_response'],
                        "Dislike",
                        conn
                    )
                    st.success("Feedback recorded: Dislike")
                    st.session_state.pending_feedback = None
                    st.rerun()

            with col3:
                if st.button("Skip Feedback", key="skip_feedback"):
                    insert_interaction(
                        st.session_state.pending_feedback['user_input'],
                        st.session_state.pending_feedback['assistant_response'],
                        "No feedback",
                        conn
                    )
                    st.info("Feedback skipped, saved as 'No feedback'")
                    st.session_state.pending_feedback = None
                    st.rerun()
        # Add the Exit button here
    # if st.button("Exit"):
    #     st.write("Exiting the application...")
    #     st.stop()  # Stops the Streamlit app

else:
    st.warning("Please enter the proper API Key")



