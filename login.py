import streamlit as st
from huggingface_hub import notebook_login, HfFolder

# Function for the Login Page
def login():
    st.title("🔒 Login to Audio2Art")
    st.write("Click the button below to authenticate with Hugging Face.")
    
    # Hugging Face Authentication
    if st.button("Login with Hugging Face"):
        notebook_login()
        hf_token = HfFolder.get_token()  # Get the saved token
        
        # Store the token in Streamlit's session state
        if hf_token:
            st.session_state['hf_token'] = hf_token
            st.success("Login Successful!")
            st.experimental_rerun()
        else:
            st.error("Authentication failed. Please try again.")

# Function to Check Login Status
def is_logged_in():
    return 'hf_token' in st.session_state

# Function to Get Hugging Face Token
def get_token():
    return st.session_state['hf_token'] if 'hf_token' in st.session_state else None

# Function to Logout
def logout():
    st.session_state.pop('hf_token', None)
    st.success("Logged out successfully.")
    st.experimental_rerun()
