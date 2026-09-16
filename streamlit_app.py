import streamlit as st
import tempfile
import os
from app.graph.workflow import app_graph
from app.retrieval.ingestion import process_document
from app.retrieval.retriever import global_retriever as retriever

st.set_page_config(page_title="FinSight", layout="wide")

st.title("FIN SIGHT")
st.subheader("AI-powered financial research")

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
                        # Pass original filename instead of tmp file for metadata
                        docs = process_document(tmp_path, company_name, "Annual Report", doc_year, original_filename=uploaded_file.name)
                        retriever.add_documents(docs)
                        st.success(f"Processed {uploaded_file.name}")
                    finally:
                        os.unlink(tmp_path)
        else:
            st.warning("Please upload files and provide a company name.")
            
    if st.button("🗑️ Clear Indexed Documents"):
        try:
            retriever.vectorstore.delete_collection()
            retriever._init_db()  # Re-initialize
            st.success("Database cleared!")
            st.experimental_rerun()
        except Exception as e:
            st.error("Error clearing database.")
            
    st.markdown("---")
    st.markdown("### Database Stats")
    
    # Query Chroma for unique companies and chunk counts
    try:
        all_docs = retriever.vectorstore.get(include=["metadatas"])
        if all_docs and all_docs.get("metadatas"):
            stats = {}
            for meta in all_docs["metadatas"]:
                if not meta: continue
                company = meta.get("company", "Unknown")
                year = meta.get("year", "Unknown")
                filename = meta.get("source_filename", "Unknown")
                key = f"{company} {year} ({filename})"
                stats[key] = stats.get(key, 0) + 1
            
            if stats:
                for key, count in stats.items():
                    st.markdown(f"✅ **{key}** — {count} chunks")
            else:
                st.markdown("*Database is empty.*")
        else:
            st.markdown("*Database is empty.*")
    except Exception as e:
        st.markdown("*Could not load stats.*")
            
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
            
            # Fix if the answer is still a stringified list of dicts from Gemini
            import re
            if isinstance(answer, str) and answer.strip().startswith("[{") and "'type': 'text'" in answer:
                # Use regex to extract the text content robustly
                match = re.search(r"'text':\s*['\"](.*?)['\"],\s*'extras'", answer, re.DOTALL)
                if match:
                    answer = match.group(1)
                    # Unescape newlines
                    answer = answer.replace('\\n', '\n')
            
            # Construct richer reasoning trace
            reasoning = "### Agentic Workflow Trace\n\n"
            iters = final_state.get("iteration_count", 0)
            if iters > 0:
                reasoning += f"🔄 **Self-Correction Loops Executed: {iters}** (Planner adjusted keywords based on missing evidence)\n\n"
                
            if final_state.get("plan"):
                reasoning += f"**1. Planner (Task Decomposition)**\n"
                for i, t in enumerate(final_state['plan'].tasks):
                    reasoning += f"- Task {i+1}: {t.description} *(Keywords: {', '.join(t.keywords)})*\n"
                reasoning += "\n"
                
            if final_state.get("retrieval_traces"):
                reasoning += f"**2. RAG Retrieval Diagnostics**\n"
                for i, trace in enumerate(final_state['retrieval_traces']):
                    reasoning += f"**Attempt {trace.get('attempt', 1)} | Task: {trace.get('company', 'Any')} ({trace.get('year', 'Any')})**\n"
                    reasoning += f"- **Query**: `{trace.get('query', '')}`\n"
                    reasoning += f"- **Method**: `{trace.get('method', '')}`\n"
                    
                    if not trace.get('matched_files'):
                        reasoning += f"- ❌ **Error**: No indexed documents matched the metadata filters.\n\n"
                    else:
                        reasoning += f"- **Target Files**: {', '.join(trace.get('matched_files', []))}\n"
                        reasoning += f"- **Candidates Retrieved**: {trace.get('candidates_retrieved', 0)}\n\n"

            if final_state.get("retrieved_chunks"):
                chunks = final_state['retrieved_chunks']
                reasoning += f"**3. Context Selection**\n"
                reasoning += f"- Sent top {len(chunks)} best reranked pages to the Analyst.\n"
                sources = list(set([f"{c.metadata.source_filename} (p.{c.metadata.page})" for c in chunks]))[:5]
                reasoning += f"- Primary sources used: {', '.join(sources)}\n\n"
                
            if final_state.get("verification_result"):
                vr = final_state["verification_result"]
                supported_count = sum(1 for v in vr.verifications if v.status == "SUPPORTED")
                reasoning += f"**3. Verifier (Anti-Hallucination Loop)**\n"
                reasoning += f"- Checked {len(vr.verifications)} numerical/factual claims against raw documents.\n"
                for v in vr.verifications:
                    if v.status == "SUPPORTED":
                        icon = "✅"
                        reasoning += f"  - {icon} Claim: *{v.claim}*\n"
                    elif v.status == "INSUFFICIENT_EVIDENCE":
                        icon = "⚠️"
                        reasoning += f"  - {icon} Missing Evidence: *{v.claim}* — additional retrieval required\n"
                    else:
                        icon = "❌"
                        reasoning += f"  - {icon} Unsupported Claim: *{v.claim}*\n"
                        
                if not vr.all_supported:
                    if any(v.status == "INSUFFICIENT_EVIDENCE" for v in vr.verifications):
                        reasoning += f"\n*Self-Correction Triggered:* Sent back to Planner to fetch missing documents.\n"
                    else:
                        reasoning += f"\n*Self-Correction Triggered:* Sent back to Analyst to fix unsupported claims.\n"
            
            st.markdown(answer)
            with st.expander("Agent reasoning flow"):
                st.markdown(reasoning)
                
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "reasoning": reasoning
            })
