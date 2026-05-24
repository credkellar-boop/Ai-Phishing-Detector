import streamlit as st
import joblib
import os

st.set_page_config(page_title="AI Phishing Detector", page_icon="🎣")

st.title("🎣 AI-Powered Phishing Detector")
st.markdown("Analyze email text or messages to detect potential phishing attempts.")

@st.cache_resource
def load_models():
    """Loads the pre-trained model and vectorizer."""
    if not os.path.exists('model/phishing_model.pkl') or not os.path.exists('model/vectorizer.pkl'):
        return None, None
    clf = joblib.load('model/phishing_model.pkl')
    vectorizer = joblib.load('model/vectorizer.pkl')
    return clf, vectorizer

clf, vectorizer = load_models()

if clf is None or vectorizer is None:
    st.warning("⚠️ Model files not found. Please run `python train_model.py` first to generate the models.")
else:
    user_input = st.text_area("Paste the email or message text here:", height=200)
    
    if st.button("Analyze Text"):
        if user_input.strip() == "":
            st.error("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                text_vectorized = vectorizer.transform([user_input])
                
                prediction = clf.predict(text_vectorized)[0]
                probability = clf.predict_proba(text_vectorized)[0]
                
                st.markdown("---")
                if prediction == 1:
                    st.error("🚨 **PHISHING DETECTED**")
                    st.write(f"Confidence: **{probability[1]*100:.2f}%**")
                    st.markdown("> **Warning:** This message contains patterns commonly found in phishing attempts. Do not click any links.")
                else:
                    st.success("✅ **LEGITIMATE**")
                    st.write(f"Confidence: **{probability[0]*100:.2f}%**")
                    st.markdown("> This message appears safe, but always exercise caution with unexpected links.")
