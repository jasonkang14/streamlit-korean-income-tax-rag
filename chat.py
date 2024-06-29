import streamlit as st
from dotenv import load_dotenv
from llm import get_ai_response

# Set the page configuration
st.set_page_config(page_title="소득세 챗봇", page_icon="🤖")

# Display the title and caption
st.title("🤖 소득세 챗봇")
st.caption("소득세에 관련된 모든것을 답해드립니다!")

def initialize_session_state():
    """Initialize session state variables."""
    if 'env_loaded' not in st.session_state:
        load_dotenv()
        st.session_state['env_loaded'] = True
    if 'message_list' not in st.session_state:
        st.session_state.message_list = []

def display_messages():
    """Display all previous messages."""
    for message in st.session_state.message_list:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def handle_user_input():
    """Handle new user input and generate AI response."""
    if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!"):
        # Display the user's message
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state.message_list.append({"role": "user", "content": user_question})

        # Generate and display AI response
        with st.spinner("답변을 생성하는 중입니다"):
            try:
                ai_response = get_ai_response(user_question)
                with st.chat_message("ai"):
                    st.write(ai_response)
                st.session_state.message_list.append({"role": "ai", "content": ai_response})
            except Exception as e:
                st.error(f"AI 응답 생성 중 오류가 발생했습니다: {e}")

# Initialize session state
initialize_session_state()

# Display all previous messages
display_messages()

# Handle new user input
handle_user_input()