import streamlit as st
import pickle
import numpy as np

with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

st.title("🚢 Titanic Survival Prediction App")
st.markdown("Predict survival of a Titanic passenger based on inputs")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 80, 30)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8)
parch = st.number_input("Parents/Children Aboard", 0, 6)
fare = st.number_input("Fare Paid", 0.0, 600.0)
embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])
title = st.selectbox("Title", ['Mr', 'Miss', 'Mrs', 'Rare'])


sex_encoded = 1 if sex == "Male" else 0
embarked_map = {"C": 0, "Q": 1, "S": 2}
title_map = {"Rare": 0, "Miss": 1, "Mr": 2, "Mrs": 3}

input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare,
                        embarked_map[embarked], title_map[title]]])

if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    if prediction == 1:
        st.success(f"✅ The passenger **would have survived**! (Survival Probability: {probability:.2f})")
    else:
        st.error(f"❌ The passenger **would not have survived**. (Survival Probability: {probability:.2f})")
