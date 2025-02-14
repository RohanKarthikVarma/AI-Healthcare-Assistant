import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datetime import datetime

# Download minimal NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load models
query_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
response_model_name = "microsoft/BioGPT"
response_tokenizer = AutoTokenizer.from_pretrained(response_model_name)
response_model = AutoModelForCausalLM.from_pretrained(response_model_name, torch_dtype=torch.float16)
response_pipeline = pipeline("text-generation", model=response_model, tokenizer=response_tokenizer)

# Enhanced medical response system with embeddings
medical_responses = {
    "symptom": "I recommend consulting a doctor about these symptoms. Would you like me to find nearby clinics?",
    "pain": "Persistent pain requires medical evaluation. Can you describe it (throbbing, sharp, constant)?",
    "fever": "If fever exceeds 38.5°C (101.3°F) for 24hrs, seek medical help. Are you experiencing chills?",
    "medication": "Never alter dosage without consulting your doctor. Need reminder settings?",
    "appointment": "I can help schedule telehealth consultations. Preferred day/time?",
    "emergency": "🚨 For emergencies, contact:\n- US: 911\n- EU: 112\n- India: 108\n- General: 112",
    "diet": "While I suggest balanced meals, consult a nutritionist for personalized plans. Interested?",
    "allergy": "Are you experiencing breathing difficulties? This could be anaphylaxis - seek immediate care.",
    "sore throat": "For a sore throat, try warm salt water gargles, honey with tea, and staying hydrated. If symptoms persist, see a doctor."
}

medical_keys = list(medical_responses.keys())
medical_embeddings = query_model.encode(medical_keys, convert_to_tensor=True)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'reminders' not in st.session_state:
    st.session_state.reminders = []
if 'appointment_bookings' not in st.session_state:
    st.session_state.appointment_bookings = []
if 'main_section' not in st.session_state:
    st.session_state.main_section = "chatbot"  # Default to chatbot view

def process_input(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return [word for word in word_tokenize(text) if word not in stopwords.words('english')]

def health_response(user_input):
    input_embedding = query_model.encode(user_input, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(input_embedding, medical_embeddings)[0]
    best_match_index = torch.argmax(similarities).item()
    
    if similarities[best_match_index] > 0.5:
        response = medical_responses[medical_keys[best_match_index]]
    else:
        response = response_pipeline(
            user_input,
            max_length=100,
            num_return_sequences=1,
            temperature=0.6,
            top_p=0.85,
            repetition_penalty=1.3
        )[0]['generated_text'].strip()
    
    st.session_state.chat_history.append(f"You: {user_input}\nAssistant: {response}")
    return response

def get_next_appointment():
    if st.session_state.appointment_bookings:
        sorted_appointments = sorted(st.session_state.appointment_bookings, key=lambda x: datetime.strptime(x.split(', ')[6] + ' ' + x.split(', ')[7], "%Y-%m-%d %H:%M:%S"))
        return sorted_appointments[0]
    return "No upcoming appointments"

def main():
    st.title("⚕️ Health Assistant")
    st.caption("For general health info only")
    
    st.sidebar.title("Menu")
    if st.sidebar.button("💬 Chatbot"):
        st.session_state.main_section = "chatbot"
    if st.sidebar.button("📅 Book an Appointment"):
        st.session_state.main_section = "appointment"
    if st.sidebar.button("🔔 Set a Reminder"):
        st.session_state.main_section = "reminder"
    
    st.sidebar.markdown("---")
    st.sidebar.title("Next Appointment")
    st.sidebar.text(get_next_appointment())
    
    st.sidebar.title("Reminders")
    for reminder in st.session_state.reminders:
        st.sidebar.text(reminder)
    
    st.sidebar.title("Chat History")
    for chat in st.session_state.chat_history:
        st.sidebar.text(chat)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("[GitHub]()")
    
    if st.session_state.main_section == "chatbot":
        query = st.text_input("Describe your health concern:")
        if st.button("Get Advice"):
            if query:
                with st.spinner("Analyzing..."):
                    reply = health_response(query)
                st.markdown(f"**Assistant:** {reply}")
                st.info("Always consult a healthcare professional for medical advice")
            else:
                st.warning("Please describe your concern")
    
    elif st.session_state.main_section == "appointment":
        st.subheader("Book an Appointment")
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        mobile = st.text_input("Mobile Number")
        city = st.text_input("City")
        reason = st.text_area("Reason for Appointment")
        date = st.date_input("Select Date")
        time = st.time_input("Select Time")
        if st.button("Confirm Appointment"):
            st.session_state.appointment_bookings.append(f"{first_name}, {last_name}, {age}, {gender}, {mobile}, {city}, {date}, {time}, {reason}")
            st.success("Appointment booked successfully!")
    
    elif st.session_state.main_section == "reminder":
        st.subheader("Set a Reminder")
        reminder_text = st.text_input("Reminder")
        reminder_time = st.time_input("Reminder Time")
        if st.button("Add Reminder"):
            st.session_state.reminders.append(f"🔔 {reminder_text} at {reminder_time}")
            st.success("Reminder added!")
    
if __name__ == "__main__":
    main()
