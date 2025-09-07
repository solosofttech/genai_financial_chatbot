import os
import streamlit as st
from bl.chatdb import ChatDB
from bl.aboutme import AboutMe
from dotenv import load_dotenv
from bl.ingestion import Ingestion
from bl.ingestion import Ingestion
from bl.chatollama import ChatOllama
from streamlit_option_menu import option_menu

load_dotenv()

with st.sidebar:
    st.image("assets/logo_new.png", width=200)
    selected = option_menu(
        menu_title=None,
        options=["Check Finances", "Create Embeddings", "About Me"],
        icons=["house", "gear", "fire"],
        menu_icon="cast",
        default_index=0,
    )

# Display content based on the selected option
if selected == "Create Embeddings":
    Ingestion.ingestion()

elif selected == "Check Finances":
    ChatDB.chat_interface()

elif selected == "Chat Ollama (Chroma)":
    ChatOllama.chat_interface()

elif selected == "About Me":
    AboutMe.show_cv()