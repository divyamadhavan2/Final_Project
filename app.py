import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Hardcoded credentials
USERNAME = "admin"
PASSWORD = "password"

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to set a solid background color
def set_bg_color(color):
    page_bg_color = f'''
        <style>
        .stApp {{
            background-color: {color};
        }}
        .stApp .css-1d391kg, .stApp .css-1v0mbdj, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
            color: #333333;  /* Darker text color for better contrast */
            font-weight: bold; /* Make all text bold */
        }}
        </style>
        '''
    st.markdown(page_bg_color, unsafe_allow_html=True)

# Set the background to a light gray color
set_bg_color("#B3D7C3")  # Light gray background

# Simple authentication function
def check_password():
    """Returns `True` if the user has correct password."""
    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #99a3a4; /* Change this to your desired color */
        }
    </style>
    """, unsafe_allow_html=True)

    st.write("Please login with the Credentials and run the app for the student prediction app")
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state["authenticated"] = True
            st.sidebar.success("Login successful")
            return True
        else:
            st.sidebar.error("Invalid username or password")
            st.session_state["authenticated"] = False
            return False
    return False

def logout():
    st.session_state["authenticated"] = False
    st.experimental_rerun()

# Function to preprocess input data
def preprocess_input(input_data):
    # Categorical mappings
    yes_no_mapping = {'No': 0, 'Yes': 1}
    gender_mapping = {'Male': 0, 'Female': 1}
    ethnicity_mapping = {'Caucasian': 0, 'African American': 1, 'Asian': 2, 'Other': 3}
    parental_support_mapping = {'None': 0, 'Low': 1, 'Moderate': 2, 'High': 3, 'Very High': 4}
    parental_education_mapping = {'None': 0, 'High School': 1, 'Some College': 2, "Bachelor's": 3, 'Higher': 4}

    # Apply mappings
    try:
        input_data['ParentalEducation'] = parental_education_mapping.get(input_data.get('ParentalEducation', ''), 0)
        input_data['Tutoring'] = yes_no_mapping.get(input_data.get('Tutoring', ''), 0)
        input_data['ParentalSupport'] = parental_support_mapping.get(input_data.get('ParentalSupport', ''), 0)
        input_data['Gender'] = gender_mapping.get(input_data.get('Gender', ''), 0)
        input_data['Ethnicity'] = ethnicity_mapping.get(input_data.get('Ethnicity', ''), 0)
        input_data['Extracurricular'] = yes_no_mapping.get(input_data.get('Extracurricular', ''), 0)
        input_data['Sports'] = yes_no_mapping.get(input_data.get('Sports', ''), 0)
        input_data['Music'] = yes_no_mapping.get(input_data.get('Music', ''), 0)
        input_data['Volunteering'] = yes_no_mapping.get(input_data.get('Volunteering', ''), 0)

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Combine categorical features into final dataframe
        final_df = input_df[['StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 
                             'Extracurricular', 'Music', 'ParentalEducation', 'Sports', 'GPA', 
                             'Gender', 'Ethnicity', 'Volunteering']]

        return final_df
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Main app function
def main_app():
    # Load trained Random Forest model
    try:
        model = joblib.load('finalized_model.pkl')  # path to model pkl file
        st.write("Model loaded successfully")

        # Define the target variable encoder
        label_encoder = LabelEncoder()
        grade_classes = ['Grade F', 'Grade D', 'Grade C', 'Grade B', 'Grade A']  # grades
        label_encoder.fit(grade_classes)

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    st.title("Student Grade Prediction App")
    st.write(
    """
    This app predicts student grades and provides insights to help educators and management understand and improve student performance.
    """
    )

    # User Inputs in the center
    st.subheader("User Input")
    input_data = {}

    with st.form(key='input_form'):
        col1, col2, col3 = st.columns(3)

        with col1:
            input_data['Age'] = st.number_input('Enter Age', min_value=15, max_value=19, value=16)
            input_data['Gender'] = st.selectbox('Select Gender', ['Male', 'Female'])
            input_data['Ethnicity'] = st.selectbox('Select Ethnicity', ['Caucasian', 'African American', 'Asian', 'Other'])
            input_data['StudyTimeWeekly'] = st.number_input('Enter Study Time Weekly (hours)', min_value=0.0, max_value=50.0, value=10.0)
            input_data['GPA'] = st.number_input('Enter GPA', min_value=0.0, max_value=4.0, value=1.0, step=0.1, format="%.2f")
        with col2:
            input_data['Absences'] = st.number_input('Enter Number of Absences', min_value=0, max_value=30, value=0)
            input_data['ParentalEducation'] = st.selectbox('Select Parental Education', ['None', 'High School', 'Some College', "Bachelor's", 'Higher'])
            input_data['Tutoring'] = st.selectbox('Select Tutoring', ['Yes', 'No'])
            input_data['ParentalSupport'] = st.selectbox('Select Parental Support', ['None', 'Low', 'Moderate', 'High', 'Very High'])

        with col3:
            input_data['Extracurricular'] = st.selectbox('Select Extracurricular', ['Yes', 'No'])
            input_data['Sports'] = st.selectbox('Select Sports', ['Yes', 'No'])
            input_data['Music'] = st.selectbox('Select Music', ['Yes', 'No'])
            input_data['Volunteering'] = st.selectbox('Select Volunteering', ['Yes', 'No'])

        submit_button = st.form_submit_button(label='Predict Grade')

    if submit_button:
        
        # Create a DataFrame with the specified column order
        all_inputs_df = pd.DataFrame([input_data])[[
            'Age', 'Gender', 'Ethnicity', 'StudyTimeWeekly', 'Absences', 
            'ParentalEducation', 'Tutoring', 'ParentalSupport', 
            'Extracurricular', 'Sports', 'Music', 'Volunteering', 'GPA'
        ]]
        # Format the GPA column to two decimal places
        all_inputs_df['GPA'] = all_inputs_df['GPA'].map("{:.2f}".format)

        # Display the full input data in a table format
        st.write("### Input Data")
        st.table(all_inputs_df)

        # Preprocess the data for prediction
        input_df = preprocess_input(input_data)

        # Make prediction
        try:
            individual_prediction = model.predict(input_df)

            # Convert the numeric prediction to the corresponding grade label using the label encoder
            decoded_prediction = label_encoder.inverse_transform([int(individual_prediction[0])])[0]

            st.write(f"### The predicted grade for the student is: {decoded_prediction}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

   # Option for bulk prediction
    st.subheader("Bulk Prediction")

    uploaded_file = st.file_uploader("Upload a CSV file for bulk prediction", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read the uploaded CSV file
            original_df = pd.read_csv(uploaded_file)  # Keep the original data intact

            # Convert the dataframe to a list of dictionaries for processing
            process_dict = original_df.to_dict('records')
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(50, text=progress_text)
            predictions = []

            # Process each row to make predictions
            for item in process_dict:
                process_df = preprocess_input(item)

                # Make predictions
                processed_df = model.predict(process_df)
                
                # Ensure that the prediction is a single value
                prediction = processed_df[0] if hasattr(processed_df, '__getitem__') else processed_df
                
                # Append the decoded prediction
                predictions.append(label_encoder.inverse_transform([int(prediction)])[0])

            my_bar.empty()

            # Add predictions to the original DataFrame
            original_df['Predicted_Grade'] = predictions

            # Display only the first 10 rows with predictions
            st.write("### First 10 Rows with Predictions")
            st.dataframe(original_df.head(10))

            # Provide a download button for the full predicted data
            st.write("### Download Full Predicted Data")
            csv = original_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name='predicted_student_grades.csv',
                mime='text/csv'
            )
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Authentication check
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if st.session_state["authenticated"]:
    main_app()  # Call the main app function if authenticated
else:
    if check_password():  # If not authenticated, run the password check
        main_app()
    else:
        st.stop()  # Stop execution if not authenticated
