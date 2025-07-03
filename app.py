# app.py
import streamlit as st
from typing import List, Dict, Any, Optional
import asyncio
from back import (ChatConfig, ChatLogger, ChatMemory, QuestionGenerator, 
                  GeminiRAG, ProductDatabase, UserManager, UserInfo)
import os 

# UI Text in English
UI_TEXT = {
    "welcome_message": """
    Welcome to the Career Guidance Chatbot! 🎓

    Hi! We are here to help you navigate your career journey.  
    
    Ask any questions you have about internships, placements, skills development, career paths, or anything else related to your future career. 
    
    Choose a question below or ask your own.
    """,
    "input_placeholder": "Enter your question here...",
    "input_label": "Ask Anything:",
    "clear_chat": "Clear Chat",
    "sidebar_title": "Tell Us About Yourself for Personalized Guidance",
    "form_name": "Your Name",
    "form_college": "College Name",
    "form_degree": "Degree",
    "form_year": "Year of Study",
    "form_goals": "Your Career Goals",
    "form_internship": "Have you done an internship?",
    "form_placement": "Have you secured a placement?",
    "form_submit": "Get Personalized Advice",
    "form_success": "✅ Personalized advice activated!",
    "form_error": "❌ Error saving information. Please try again.",
    "form_required": "Please fill in all required fields.",
    "initial_questions": [
        "What are some good career options in my field?",
        "How do I get an internship in my desired field?",
        "What skills are most important for success in this career path?",
        "What are some tips for preparing for job interviews?"
    ]
}

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'initialized': False,
        'chat_memory': ChatMemory(),
        'messages': [],
        'message_counter': 0,
        'processed_questions': set(),
        'trigger_rerun': False,
        'user_info': None,
        'show_suggestions': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
      
# Configure the page
st.set_page_config(
    page_title="Career Guidance Chatbot",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.user-message {
    background-color: #000000;
    color: #2e9ff;
    padding: 15px;
    border-radius: 15px;
    margin: 10px 0;
}
.assistant-message {
    background-color: #000000;
    color: #2e9ff;
    padding: 15px;
    border-radius: 15px;
    margin: 10px 0;
}
.stButton > button {
    background-color: #212B2A;
    color: white;
    border: none;
    padding: 10px 15px;
    border-radius: 5px;
    transition: background-color 0.3s;
}
.stButton > button:hover {
    background-color: #3d85c6;
}
</style>
""", unsafe_allow_html=True)

def initialize_components():
    """Initialize and cache application components"""
    try:
        config = ChatConfig()
        logger = ChatLogger(config.log_file)
        question_gen = QuestionGenerator(config.gemini_api_key)
        rag = GeminiRAG(config.gemini_api_key)
        user_manager = UserManager(config.user_data_file)
        db = ProductDatabase(config) # Initialize the database here
        return config, logger, question_gen, rag, user_manager, db
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        raise e

def load_initial_database():
    """Load the default database"""
    if not st.session_state.initialized:
        try:
            config, logger, question_gen, rag, user_manager, db = initialize_components()
            
            st.session_state.db = db
            st.session_state.config = config
            st.session_state.logger = logger
            st.session_state.question_gen = question_gen
            st.session_state.rag = rag
            st.session_state.user_manager = user_manager
            st.session_state.initialized = True
            
        except Exception as e:
            st.error(f"Error loading initial database: {str(e)}")
            return None

async def process_question(question: str):
    """Process a question and update the chat state"""
    try:
        relevant_docs = st.session_state.db.search(question)
        context = st.session_state.rag.create_context(relevant_docs)
        answer = await st.session_state.rag.get_answer(
            question=question,
            context=context,
            user_info=st.session_state.user_info
        )
        
        follow_up_questions = await st.session_state.question_gen.generate_questions(
            question, 
            answer,
            st.session_state.user_info
        )
        
        st.session_state.chat_memory.add_interaction(question, answer)
        st.session_state.logger.log_interaction(
            question, 
            answer,
            st.session_state.user_info
        )
        
        st.session_state.message_counter += 1
        
        st.session_state.messages.append({
            "role": "user",
            "content": question,
            "message_id": st.session_state.message_counter
        })
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "questions": follow_up_questions,
            "message_id": st.session_state.message_counter
        })
        
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def render_user_form():
    """Render the user information form in the sidebar"""
    st.sidebar.title(UI_TEXT["sidebar_title"])
    
    with st.sidebar.form("user_info_form"):
        name = st.text_input(UI_TEXT["form_name"])
        college = st.text_input(UI_TEXT["form_college"])
        degree = st.text_input(UI_TEXT["form_degree"])
        year = st.number_input(UI_TEXT["form_year"], min_value=1, max_value=5, step=1)
        career_goals = st.text_area(UI_TEXT["form_goals"])
        has_internship = st.checkbox(UI_TEXT["form_internship"])
        has_placement = st.checkbox(UI_TEXT["form_placement"])
        
        submitted = st.form_submit_button(UI_TEXT["form_submit"])
        
        if submitted:
            if name and college and degree and year and career_goals:
                user_info = UserInfo(
                    name=name,
                    college=college,
                    degree=degree,
                    year=year,
                    career_goals=career_goals,
                    has_internship=has_internship,
                    has_placement=has_placement
                )
                
                if st.session_state.user_manager.save_user_info(user_info):
                    st.session_state.user_info = user_info
                    st.sidebar.success(UI_TEXT["form_success"])
                else:
                    st.sidebar.error(UI_TEXT["form_error"])
            else:
                st.sidebar.warning(UI_TEXT["form_required"])

def main():
    # Initialize session state
    init_session_state()
    
    # Load initial database and components if not already initialized
    if not st.session_state.initialized:
        load_initial_database()

    # Render user form in sidebar
    render_user_form()

    # Display title
    st.title("Career Guidance Chatbot")

    # Welcome message
    if not st.session_state.messages:
        st.markdown(UI_TEXT["welcome_message"])
        
        # Display initial questions as buttons
        cols = st.columns(2)
        for i, question in enumerate(UI_TEXT["initial_questions"]):
            if cols[i % 2].button(question, key=f"initial_{i}", use_container_width=True):
                asyncio.run(process_question(question))
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f'<div class="user-message">👤 {message["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="assistant-message">🎓 {message["content"]}</div>',
                unsafe_allow_html=True
            )
            
            if message.get("questions"):
                cols = st.columns(2)
                for i, question in enumerate(message["questions"]):
                    if cols[i % 2].button(
                        question,
                        key=f"followup_{message['message_id']}_{i}",
                        use_container_width=True
                    ):
                        asyncio.run(process_question(question))
    
    # Input area
    with st.container():
        # Create a form for input
        with st.form(key='input_form'):
            question = st.text_input(
                UI_TEXT["input_label"],
                key="user_input",
                placeholder=UI_TEXT["input_placeholder"]
            )
            submit = st.form_submit_button("Send")
        
        # Process input when submitted
        if submit and question:
            with st.spinner("🔄 Processing your message..."):
                asyncio.run(process_question(question))
                if 'processed_questions' not in st.session_state:
                    st.session_state.processed_questions = set()
                st.session_state.processed_questions.add(question)
                st.rerun()
        
        # Clear chat controls
        cols = st.columns([4, 1])
        if cols[1].button(UI_TEXT["clear_chat"], use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_memory.clear_history()
            st.session_state.message_counter = 0
            if 'processed_questions' in st.session_state:
                st.session_state.processed_questions = set()
            st.rerun()

if __name__ == "__main__":
    main()