import streamlit as st
import tempfile
import os
from app.graph.workflow import app_graph
from app.retrieval.ingestion import process_document
from app.retrieval.retriever import FinSightRetriever

st.set_page_config(page_title="FinSight", layout="wide")

st.title("FIN SIGHT")
st.subheader("AI-powered financial research")

# Initialize retriever
@st.cache_resource
def get_retriever():
    return FinSightRetriever()

retriever = get_retriever()

with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader("Upload 10-K/10-Q PDFs", type="pdf", accept_multiple_files=True)
    company_name = st.text_input("Company Name")
    doc_year = st.number_input("Year", min_value=1990, max_value=2030, value=2023)
    
    if st.button("Process Documents"):
        if uploaded_files and company_name:
            with st.spinner("Processing..."):
                for uploaded_file in uploaded_files:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    try:
                        docs = process_document(tmp_path, company_name, "Annual Report", doc_year)
                        retriever.add_documents(docs)
                        st.success(f"Processed {uploaded_file.name}")
                    finally:
                        os.unlink(tmp_path)
        else:
            st.warning("Please upload files and provide a company name.")
            
    st.markdown("---")
    st.markdown("### Example Questions")
    st.markdown("- Compare revenue growth between these companies.")
    st.markdown("- What are the biggest risks management identifies?")
    st.markdown("- How has operating profitability evolved?")
    st.markdown("- Compare both companies' AI strategy.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "reasoning" in msg:
            with st.expander("Agent reasoning flow"):
                st.markdown(msg["reasoning"])

if question := st.chat_input("Ask a financial research question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
        
    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            initial_state = {
                "question": question,
                "chat_history": st.session_state.messages[:-1]
            }
            
            final_state = app_graph.invoke(initial_state)
            
            answer = final_state.get("final_answer", "Sorry, I could not generate an answer.")
            
            # Construct reasoning trace
            reasoning = ""
            if final_state.get("plan"):
                reasoning += f"✓ Query decomposed into {len(final_state['plan'].tasks)} research tasks\n"
            if final_state.get("retrieved_chunks"):
                reasoning += f"✓ {len(final_state['retrieved_chunks'])} document passages retrieved\n"
            if final_state.get("verification_result"):
                vr = final_state["verification_result"]
                supported_count = sum(1 for v in vr.verifications if v.status == "SUPPORTED")
                reasoning += f"✓ {len(vr.verifications)} claims checked\n"
                reasoning += f"✓ {supported_count}/{len(vr.verifications)} claims supported\n"
            
            st.markdown(answer)
            with st.expander("Agent reasoning flow"):
                st.markdown(reasoning)
                
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "reasoning": reasoning
            })
