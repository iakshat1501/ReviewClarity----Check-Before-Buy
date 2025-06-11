import streamlit as st
st.set_page_config(page_title="ReviewClarity", layout="centered", page_icon="🛍️")
import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import plotly.express as px

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# === Clean Text Function ===
def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower().strip()
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# === Scrape Reviews Function ===
def scrape_amazon_reviews(product_url, pages=2):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    service = Service()
    driver = webdriver.Chrome(service=service, options=options)

    review_data = []

    for page in range(1, pages + 1):
        url = f"{product_url}/ref=cm_cr_arp_d_paging_btm_next_{page}?pageNumber={page}"
        driver.get(url)
        time.sleep(3)

        titles = driver.find_elements(By.XPATH, '//a[@data-hook="review-title"]')
        texts = driver.find_elements(By.XPATH, '//span[@data-hook="review-body"]')
        stars = driver.find_elements(By.XPATH, '//i[@data-hook="review-star-rating"]//span')

        for i in range(min(len(titles), len(texts), len(stars))):
            review_data.append({
                "Review Title": titles[i].text.strip(),
                "Review Text": texts[i].text.strip(),
                "Rating": stars[i].text.strip()
            })

    driver.quit()
    return pd.DataFrame(review_data)

# === Background and Fonts ===
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1549921296-3b4a4f9855c6");
            background-attachment: fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

h1 {
    color: #4B0082;
}
</style>
""", unsafe_allow_html=True)

add_bg_from_url()

# === Streamlit UI Setup ===

st.title("🛒 Welcome to ReviewClarity...!!! — your one-stop solution for understanding what real customers think before you buy...")
st.markdown("Analyze public opinion of any product using customer reviews. Get an overall buying verdict..!! 💬")

with st.sidebar:
    st.header("🔗 Product Input")
    product_link = st.text_input("Paste Amazon Product URL")
    pages = st.slider("Pages to Scrape", 1, 10, 3)
    start = st.button("Start Analysis")

if start:
    if not product_link:
        st.warning("Please enter a valid Amazon product URL.")
    else:
        with st.spinner("Scraping and analyzing reviews..."):
            my_bar = st.progress(0, text="Starting Scraping...")
            for percent_complete in range(80):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=f"Scraping Progress: {percent_complete+1}%")

            df = scrape_amazon_reviews(product_link, pages)
            my_bar.progress(100, text="Scraping Completed!")

            if df.empty:
                st.error("No reviews found. Try another product link.")
            else:
                df["cleaned_review"] = df["Review Text"].apply(clean_text)
                X_vec = tfidf.transform(df["cleaned_review"])
                df["predicted_sentiment"] = model.predict(X_vec)

                sentiment_counts = df["predicted_sentiment"].value_counts()
                total = sentiment_counts.sum()
                positive = sentiment_counts.get("Good", 0)
                negative = sentiment_counts.get("Bad", 0)
                score = (positive / total) * 100

                # === Sentiment Meter ===
                st.subheader("📶 Sentiment Meter")
                if score >= 60:
                    color = "#28a745"
                    label = "😊 Positive"
                elif score <= 40:
                    color = "#dc3545"
                    label = "😠 Negative"
                else:
                    color = "#ffc107"
                    label = "😐 Mixed"

                st.markdown(f"""
                <div style="background-color: #f9f9f9; padding: 1rem; border-radius: 10px">
                    <h5 style="margin-bottom: 10px">Overall Sentiment: <span style="color:{color};">{label}</span></h5>
                    <div style="background-color: #e0e0e0; width: 100%; height: 25px; border-radius: 5px;">
                        <div style="width: {score:.1f}%; height: 100%; background-color: {color}; border-radius: 5px;"></div>
                    </div>
                    <p style="margin-top:10px;">Positive Sentiment: <b>{score:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

                # === Final Verdict Box ===
                verdict_color = "#d4edda" if score >= 60 else "#f8d7da" if score <= 40 else "#fff3cd"
                verdict_icon = "✅" if score >= 60 else "❌" if score <= 40 else "⚠️"

                st.markdown(f"""
                <div style="background-color: {verdict_color}; padding: 20px; border-radius: 10px;">
                    <h3 style="margin-bottom:10px;">{verdict_icon} Final Verdict</h3>
                    <p style="font-size:16px;">
                        {"Yes, the reviews are mostly positive. Consider buying it!" if score >= 60 else "No, the reviews are mostly negative. Be cautious." if score <= 40 else "The reviews are mixed. Review them before buying."}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # === Review Table ===
                with st.expander("📝 See Sample Reviews with Predicted Sentiment"):
                    st.dataframe(df[["Review Text", "predicted_sentiment"]])

                # === Pie Chart ===
                st.subheader("📊 Sentiment Breakdown")
                st.metric("Total Reviews", total)
                st.metric("Positive", positive)
                st.metric("Negative", negative)

                fig = px.pie(
                    names=["Positive", "Negative"],
                    values=[positive, negative],
                    color_discrete_sequence=["#28a745", "#dc3545"],
                    title="Sentiment Distribution"
                )
                st.plotly_chart(fig)
