
import http.client
import json
import pandas as pd
# import ace_tools as tools
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to calculate cosine similarity
def calculate_cosine_similarity(ground_truth, predicted):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([ground_truth, predicted])
    similarity = cosine_similarity(vectors[0], vectors[1])
    return similarity[0][0]

# List of ground truth questions and their corresponding ground truth answers
ground_truth_data = [
    {"question": "How to log into my account?", "answer": "You can log in by entering your username and password on the login page."},
    {"question": "What are the payment methods available?", "answer": "We accept credit cards, PayPal, and bank transfers."},
    {"question": "How can I change my password?", "answer": "You can change your password by going to your account settings and selecting 'Change Password'."}
]

# Prepare the server connection
conn = http.client.HTTPConnection("127.0.0.1", 5005)

# Prepare an empty list to hold responses
responses = []

# Start an MLflow experiment
mlflow.start_run()

# Loop through the list of ground truth questions
for data in ground_truth_data:
    question = data["question"]
    ground_truth_answer = data["answer"]

    payload = json.dumps({
        "input": question,
        "session_id": "fastapi_default_session"
    })
    headers = { 'Content-Type': 'application/json' }

    # Send the POST request to generate predicted response
    conn.request("POST", "/generate_response", body=payload, headers=headers)

    # Get the response
    response = conn.getresponse()

    # Read the response
    predicted_response = response.read().decode("utf-8")

    # Calculate cosine similarity between the ground truth and predicted response
    similarity = calculate_cosine_similarity(ground_truth_answer, predicted_response)

    # Log the cosine similarity to MLflow
    mlflow.log_metric(f"cosine_similarity", similarity)

    # Store the question, ground truth answer, and predicted response in the list
    responses.append({
        "question": question,
        "ground_truth_answer": ground_truth_answer,
        "predicted_response": predicted_response,
        "cosine_similarity": similarity
    })

    # Close the connection for the current request
    conn.close()

    # Re-initialize the connection for the next question
    conn = http.client.HTTPConnection("127.0.0.1", 5005)

# Convert the list of responses to a DataFrame
df = pd.DataFrame(responses)

# Print the DataFrame
print(df)

# Optionally, you can display the DataFrame using the tools you have
tools.display_dataframe_to_user(name="Questions and Responses with Similarity", dataframe=df)

# End the MLflow run
mlflow.end_run()






