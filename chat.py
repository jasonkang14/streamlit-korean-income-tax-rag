import streamlit as st
from dotenv import load_dotenv
from llm import get_ai_response

# Set the page configuration
st.set_page_config(page_title="소득세 챗봇", page_icon="🤖")

# Display the title and caption
st.title("🤖 소득세 챗봇")
st.caption("소득세에 관련된 모든것을 답해드립니다!")

# Load environment variables
if 'env_loaded' not in st.session_state:
    load_dotenv()
    st.session_state['env_loaded'] = True

# Initialize the message list in session state if it doesn't exist
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

# Display all previous messages
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle new user input
if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!"):
    # Display the user's message
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    # Generate and display AI response
    with st.spinner("답변을 생성하는 중입니다"):
        ai_response = get_ai_response(user_question)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({"role": "ai", "content": ai_message})