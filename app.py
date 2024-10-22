import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    distilBERT_model = pipeline("sentiment-analysis", model = "lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    roberta_model = pipeline("sentiment-analysis", model="terrencewee12/xlm-roberta-base-sentiment-multilingual-finetuned-v2")
    return distilBERT_model, roberta_model


def get_sentiment(text, model):
    sentiment = model(text)
    return sentiment

def display_result(sentiment_label, confidence_score):
    if sentiment_label.lower() == "neutral":
        sentiment_color = "orange"
    elif sentiment_label.lower() == "positive":
        sentiment_color = "green"
    else: 
        sentiment_color = "red"

    st.markdown(
        f"""
        <div style="background-color: #F2F3F5; padding: 20px; border-radius: 10px; margin-top: 20px; text-align: center;">
            <h2 style="color: {sentiment_color};">Sentiment: {sentiment_label}</h2>
            <p style="color: green; font-size: 18px;">Confidence Score: {confidence_score:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    st.title ("Multilingual Sentiment Analysis: DistilBERT vs RoBERTa")
    st.write("This app shows the results of DistilBERT and RoBERTa model across multiple languages")
    text_input = st.text_area("Enter your text here", "")
    languages = ['English', 'French', 'Italian', 'German', 'Chinese', 'Hindi', 'Arabic', 'Indonesian', 'Portuguese', 'Japanese', 'Spanish', 'Malay']
    selected_language = st.selectbox("Select Language", languages)
    model_choice = st.radio("Choose the model for Sentiment Analysis", ("DistilBERT", "RoBERTa"))
    distilBERT_model, roberta_model = load_model()
    
    if st.button("Analyze Sentiment"):
        if text_input.strip() == "":
            st.error("Please enter some text.")
        else:
            # Choose the correct model based on user selection
            if model_choice == "DistilBERT":
                result = distilBERT_model(text_input)
            else:
                result = roberta_model(text_input)

            # Get the sentiment label and confidence score
            sentiment_label = result[0]['label']
            confidence_score = result[0]['score']

            # Display the result in a styled card
            st.markdown("---")
            display_result(sentiment_label, confidence_score)

if __name__ == '__main__':
    main()

