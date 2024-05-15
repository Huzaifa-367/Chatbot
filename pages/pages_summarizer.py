import streamlit as st
import os
import requests
import time
from gtts import gTTS
import tempfile

# Define Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/Huzaifa367/chat-summarizer"
API_TOKEN = os.getenv("AUTH_TOKEN")
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# Function to query Hugging Face API
def query_huggingface(payload):
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()  # Raise exception for non-2xx status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying Hugging Face API: {e}")
        return {"summary_text": f"Error querying Hugging Face API: {e}"}

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    temp_filename = audio_file.name
    tts.save(temp_filename)
    st.audio(temp_filename, format='audio/mp3')
    os.remove(temp_filename)

def main():
    st.set_page_config(layout="wide")
    st.title("Chat Summarizer")

    # Initialize a list to store chat messages
    chat_history = []

    # User input for chat message
    user_message = st.text_input("Provide a Chat/Long description to summarize")

    # Process user input and query Hugging Face API on button click
    if st.button("Send"):
        if user_message:
            # Add user message to chat history
            chat_history.append({"speaker": "User", "message": user_message})
            # Construct input text for summarization
            input_text = f"User: {user_message}"
            # Query Hugging Face API for summarization
            payload = {"inputs": input_text}
            response = query_huggingface(payload)
            # Extract summary text from the API response
            summary_text = response[0]["summary_text"] if isinstance(response, list) else response.get("summary_text", "")
            # Add summarization response to chat history
            chat_history.append({"speaker": "Bot", "message": summary_text})

    # Display chat history as a conversation
    for chat in chat_history:
        if chat["speaker"] == "User":
            st.text_input("User", chat["message"], disabled=True)
        elif chat["speaker"] == "Bot":
            st.text_area("Bot", chat["message"], disabled=True)
            text_to_speech(chat["message"])

if __name__ == "__main__":
    main()
