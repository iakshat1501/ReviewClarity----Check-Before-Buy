# ReviewClarity -- Check Before Buy 🛒
An intelligent Amazon product review analyzer using web scraping, sentiment analysis, and real-time recommendation to help users make better buying decisions.

# 🔎 Project Overview

- ReviewClarity allows users to simply paste an Amazon product link. The system:
- Automatically scrapes real user reviews from Amazon using Selenium.
- Processes and cleans the review data.
- Uses a pre-trained sentiment analysis model to classify reviews into positive or negative sentiments.
- Generates an overall recommendation for the user: whether they should consider buying the product.
- Provides a simple, user-friendly interface built with Streamlit.

# 💡 Key Features
🔗 Amazon URL-based scraping
🧹 Review Cleaning & Preprocessing
🤖 ML-powered Sentiment Analysis (using trained sentiment_model.pkl & tfidf_vectorizer.pkl)
📊 EDA on Amazon Reviews Dataset (data_eda.py)
🌐 Streamlit-based Web App UI
📥 Download Cleaned Reviews Dataset
📈 Future-Ready Architecture for scaling

# 🖥️ Streamlit App Preview
![ReviewClarity](https://github.com/user-attachments/assets/b4c7040f-d3ec-489a-9d1d-e348c4cbc6e9)


# 🚀 Workflow
1️⃣ Web Scraping
Uses Selenium to scrape reviews from Amazon product pages.

2️⃣ Data Preparation
Clean & preprocess scrapped reviews into a structured dataframe.

3️⃣ Sentiment Model
Trained using Kaggle's Amazon_Reviews.csv dataset.

Model files:
sentiment_model.pkl (classification model)
tfidf_vectorizer.pkl (vectorization model)

4️⃣ Prediction & Recommendation
Predicts the sentiment for each review.
Calculates overall positivity and recommends if product should be considered.

5️⃣ Frontend UI
Built using Streamlit for easy usability.

# 📂 Project Structure

ReviewClarity--CheckBeforeBuy/
│
├── data_eda.py                # EDA & data cleaning script
├── app.py                     # Streamlit UI application
├── scraper.py                 # Selenium web scraper module
├── sentiment_model.pkl        # Trained sentiment classification model
├── tfidf_vectorizer.pkl       # Trained TF-IDF vectorizer
├── Amazon_Reviews.csv         # Dataset used for model training (Kaggle)
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation

# 🛠 Technologies Used

- Python 3.x
- Selenium
- Pandas
- Scikit-learn
- WordCloud
- Streamlit
- Matplotlib
- Pickle

# 📈 Future Scope

🚀 Deploy on cloud platform (AWS, Heroku, etc.)
💡 Add multi-language review support.
🏷️ Include product feature extraction.
📊 Advanced model architecture (e.g. BERT, transformers)
🌎 Add support for other e-commerce platforms.

