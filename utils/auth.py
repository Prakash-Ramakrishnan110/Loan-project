import streamlit as st
from supabase import create_client, Client

@st.cache_resource
def get_supabase_client() -> Client:
    """Initialize the Supabase client."""
    if "SUPABASE_URL" not in st.secrets or "SUPABASE_KEY" not in st.secrets:
        st.error("Supabase credentials are missing. Please add them to .streamlit/secrets.toml")
        st.stop()
        
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

def authenticate_user(email, password):
    """
    Authenticate user using Supabase Auth.
    """
    try:
        supabase = get_supabase_client()
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if response.user:
            return True, "Login successful"
        return False, "Invalid credentials."
    except Exception as e:
        error_msg = getattr(e, 'message', str(e))
        return False, f"Authentication error: {error_msg}"

def register_user(email, password):
    """
    Register user using Supabase Auth.
    """
    try:
        supabase = get_supabase_client()
        response = supabase.auth.sign_up({"email": email, "password": password})
        if response.user:
            return True, "Account created successfully!"
        return False, "Failed to create account."
    except Exception as e:
        error_msg = getattr(e, 'message', str(e))
        return False, f"Registration error: {error_msg}"
