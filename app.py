import streamlit as st
from file_processing import process_file
from chatbot_service import ChatbotService

# Initialize Streamlit app title and other configurations
st.title("Medical Private GPT")

# Initialize chatbot service
chatbot_service = ChatbotService()
def main():
# Streamlit UI
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])

    if uploaded_file:
        try:
            content = process_file(uploaded_file)
            doc_id = uploaded_file.name
            chatbot_service.add_document(doc_id, content)
            st.session_state['last_doc_id'] = doc_id
            st.success(f"Document '{doc_id}' uploaded and processed successfully")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    st.header("Chat with the Question Answering Bot")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    if 'last_doc_id' in st.session_state:
        doc_id = st.session_state['last_doc_id']

        with st.form(key='chat_form', clear_on_submit=True):
            question = st.text_input("You:")
            submit_button = st.form_submit_button("Ask")

            if submit_button and question:
                if doc_id:
                    answer = chatbot_service.answer_question(question, doc_id)
                    st.session_state['chat_history'].append({"You": question, "Bot": answer})
                else:
                    st.error("Please upload a document first")

        for exchange in st.session_state['chat_history']:
            st.write("You:", exchange["You"])
            st.write("Bot:", exchange["Bot"])
    else:
        st.write("Please upload a document to start the chat.")

if __name__ == "__main__":
    main()
