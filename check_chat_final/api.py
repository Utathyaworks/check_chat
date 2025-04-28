import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
# Load .env
load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# === LangChain Setup ===
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("./data/vectorstor", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()
# llm = ChatOpenAI(openai_api_key=api_key,  # Or pass the key directly
#     model="gpt-4o-mini",) 
# llm = ChatOpenAI(openai_api_key=GROQ_API_KEY, model_name="gpt-4o-mini")
llm = ChatOllama(model="mistral")
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history."
        "Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a customer support agent specializing in answering queries. "
    "Your goal is to provide accurate, concise, and helpful responses. "
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
    "For more detailed information, please visit our 'Pricing' page."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# In-memory session store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# === FastAPI Server ===
app = FastAPI(title="RAG API", description="Conversational RAG powered by LangChain + FastAPI")

class ChatRequest(BaseModel):
    input: str
    session_id: str = "fastapi_default_session"

@app.post("/generate_response")
async def chat_endpoint(req: ChatRequest):
    try:
        response = await conversational_rag_chain.ainvoke(
            {"input": req.input},
            config={"configurable": {"session_id": req.session_id}}
        )
        return {"response": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Run with `python rag_api.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5005)
