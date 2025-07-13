import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import streamlit as st
import os
import shutil
from ingestion.loader import load_documents_to_chroma, save_uploaded_files
from ingestion.pipeline import run_pipeline
from Vectorstore.index import buildChromaDB, get_index_from_chroma
from chat.engine import create_chat_engine
from config.settings import add_api_key
from google.genai.errors import ServerError
import nest_asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

nest_asyncio.apply()

# ---------------------------- SETUP ----------------------------------

def setup_chat_engine(course_name, gemini_api_key):
    try:
        if gemini_api_key:
            add_api_key(gemini_api_key)

        index = get_index_from_chroma(course_name)
        chat_engine = create_chat_engine(index)

        st.session_state.chat_engine = chat_engine
        st.session_state.index_loaded = True
        st.success(f"✅ Loaded course: {course_name}")
    except Exception as e:
        st.session_state.index_loaded = False
        st.error(f"❌ Failed to load index: {e}")

def check_course_exists(course_name):
    """Check if course vector database exists"""
    try:
        get_index_from_chroma(course_name)
        return True
    except:
        return False

def build_course_index(course_name, uploaded_files, gemini_api_key):
    """Build vector index for a course from uploaded files"""
    try:
        if gemini_api_key:
            add_api_key(gemini_api_key)
        
        # Create temporary directory for uploaded files
        temp_dir = f"data/{course_name}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded files
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Load documents and build index
        docs = load_documents_to_chroma()
        nodes = run_pipeline(docs)
        index = buildChromaDB(nodes, course_name)
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        
        return True
    except Exception as e:
        st.error(f"❌ Failed to build index: {e}")
        return False

# -------------------------- PAGE CONFIG ------------------------------

st.set_page_config(
    page_title="Modular RAG Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------- SESSION STATE INIT -----------------------

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None
if "index_loaded" not in st.session_state:
    st.session_state.index_loaded = False
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""
if "selected_course" not in st.session_state:
    st.session_state.selected_course = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "show_setup_mode" not in st.session_state:
    st.session_state.show_setup_mode = False
if "setup_course_name" not in st.session_state:
    st.session_state.setup_course_name = ""
if "custom_course_name" not in st.session_state:
    st.session_state.custom_course_name = ""
if "setup_password_verified" not in st.session_state:
    st.session_state.setup_password_verified = False
if "show_password_modal" not in st.session_state:
    st.session_state.show_password_modal = False

# --------------------------- HEADER ----------------------------------

st.title("🧠 Ragademic")
st.markdown("Select a course → Activate the knowledge engine → Let the dialogue begin.")

# -------------------------- SIDEBAR ----------------------------------

with st.sidebar:
    st.header("🔑 API Configuration")

    # Gemini API Key
    api_key = st.text_input(
        "Enter Gemini API Key:",
        type="password",
        value=st.session_state.gemini_api_key,
        help="Get your API key from: https://aistudio.google.com/app/apikey"
    )

    if api_key and api_key != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = api_key
        st.rerun()

    # Gate rest of app
    if not st.session_state.gemini_api_key:
        st.warning("🚫 Please enter your Gemini API key to begin.")
        st.stop()

    st.divider()
    st.header("📚 Course Management")

    # Get password from environment
    SETUP_PASSWORD = os.getenv("SETUP_PASSWORD", "admin123")

    # Password-protected setup mode
    if not st.session_state.setup_password_verified:
        if st.button("➕ Add New Course", key="add_course_btn"):
            st.session_state.show_password_modal = True
            st.rerun()
        
        # Password modal
        if st.session_state.show_password_modal:
            st.subheader("🔒 Enter Setup Password")
            password = st.text_input("Password:", type="password", key="password_input")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Verify", key="verify_password"):
                    if password == SETUP_PASSWORD:
                        st.session_state.setup_password_verified = True
                        st.session_state.show_password_modal = False
                        st.session_state.show_setup_mode = True
                        st.success("✅ Access granted!")
                        st.rerun()
                    else:
                        st.error("❌ Incorrect password")
            
            with col2:
                if st.button("❌ Cancel", key="cancel_password"):
                    st.session_state.show_password_modal = False
                    st.rerun()
    else:
        if st.button("➕ Add New Course", key="add_course_btn_verified"):
            st.session_state.show_setup_mode = True
            st.rerun()
        
        if st.button("🔒 Lock Setup", key="lock_setup"):
            st.session_state.setup_password_verified = False
            st.session_state.show_setup_mode = False
            st.rerun()

    # Existing courses
    existing_courses = ["Algorithms", "Computer-Networks", "Theory-of-Automata"]
    available_courses = [course for course in existing_courses if check_course_exists(course)]
    
    if available_courses:
        st.subheader("📖 Available Courses")
        
        course_options = ["Select course"] + available_courses
        
        # Ensure selected course is in course_options
        if st.session_state.selected_course not in course_options:
            st.session_state.selected_course = course_options[0]

        current_index = course_options.index(st.session_state.selected_course)
        selected_course = st.selectbox(
            "Select Course Collection:",
            course_options,
            index=current_index,
            key="course_selector"
        )

        # Handle course switch
        if selected_course != st.session_state.selected_course and selected_course != "Select course":
            # Save current chat to history
            if st.session_state.selected_course and st.session_state.messages:
                st.session_state.chat_history[st.session_state.selected_course] = st.session_state.messages

            # Restore previous messages or start fresh
            st.session_state.messages = st.session_state.chat_history.get(selected_course, [])
            st.session_state.selected_course = selected_course
            setup_chat_engine(selected_course, st.session_state.gemini_api_key)
            st.rerun()

        # Show previous course chats (if any)
        if st.session_state.chat_history:
            st.subheader("🧾 Previous Chats")
            for course, messages in st.session_state.chat_history.items():
                if course != st.session_state.selected_course and course in available_courses:
                    if st.button(f"💬 Continue: {course}", key=f"continue_{course}"):
                        st.session_state.chat_history[st.session_state.selected_course] = st.session_state.messages
                        st.session_state.messages = messages
                        st.session_state.selected_course = course
                        setup_chat_engine(course, st.session_state.gemini_api_key)
                        st.rerun()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        if st.session_state.selected_course:
            st.session_state.chat_history[st.session_state.selected_course] = []

# ------------------------ SETUP MODE ---------------------------------

if st.session_state.show_setup_mode and st.session_state.setup_password_verified:
    st.header("🛠️ Course Setup Mode")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📝 Course Information")
        
        # Course name input
        course_name_option = st.selectbox(
            "Select or create course:",
            ["Custom"] + ["Algorithms", "Computer-Networks", "Theory-of-Automata"],
            key="course_name_selector"
        )
        
        if course_name_option == "Custom":
            course_name = st.text_input(
                "Enter custom course name:",
                value=st.session_state.custom_course_name,
                key="custom_course_input"
            )
            if course_name:
                st.session_state.custom_course_name = course_name
        else:
            course_name = course_name_option
            st.session_state.custom_course_name = ""
        
        st.session_state.setup_course_name = course_name
        
        # File upload
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload course documents",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            help="Upload documents for this course"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files ready for processing")
            with st.expander("📋 Uploaded Files"):
                for file in uploaded_files:
                    st.write(f"• **{file.name}** ({file.size:,} bytes)")
        
        # Build button
        if st.button("🚀 Build Course Index", disabled=not (course_name and uploaded_files)):
            if course_name and uploaded_files:
                with st.spinner(f"Building index for {course_name}..."):
                    success = build_course_index(course_name, uploaded_files, st.session_state.gemini_api_key)
                    if success:
                        st.success(f"✅ Successfully created course: {course_name}")
                        st.info("You can now select this course from the dropdown above!")
                    else:
                        st.error("❌ Failed to create course index")
    
    with col2:
        st.subheader("🚪 Exit Setup")
        if st.button("❌ Close Setup Mode", key="close_setup"):
            st.session_state.show_setup_mode = False
            st.session_state.setup_course_name = ""
            st.session_state.custom_course_name = ""
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Note:** This setup mode is password protected.")

# ------------------------ CHAT INTERFACE -----------------------------

if not st.session_state.show_setup_mode:
    if st.session_state.index_loaded and st.session_state.selected_course != "Select course":
        st.header(f"💬 Chat with {st.session_state.selected_course}")
        
        for msg in st.session_state.messages:
            role = "🤖 Assistant: \n" if msg["role"] == "assistant" else "👤 You:\n"
            st.markdown(f"**{role}** {msg['content']}")

        if prompt := st.chat_input("Ask something from the selected course..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chat_engine.chat(prompt)
                    response_text = str(response.response) if hasattr(response, "response") else str(response)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.markdown(f"**🤖 Assistant:** {response_text}")
                except ServerError:
                    error_msg = "❌ Gemini LLM is overloaded (503). Please try again later."
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.error(error_msg)
                except Exception as e:
                    error_msg = f"❌ Unexpected error: {e}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.error(error_msg)
    else:
        st.info("⚠️ Select a course to initialize the chat engine, or add a new course using the setup mode.")

# ---------------------------- FOOTER ---------------------------------

st.markdown("---")
st.markdown("Made with ❤️ using LlamaIndex and ChromaDB 🔍🧠")