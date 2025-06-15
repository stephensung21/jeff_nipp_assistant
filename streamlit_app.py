import streamlit as st
from app import create_fitness_graph

st.set_page_config(page_title="Jeff Nippard Fitness Assistant", page_icon="💪")
st.title("💪 Fitness Coach Assistant")
st.write("Ask about workouts, diet, or specific YouTube videos.")

query = st.text_area("Enter your question:")

if st.button("Get Answer"):

    if query.strip():
        agent = create_fitness_graph()
        with st.spinner("Thinking..."):
            response = agent.invoke({"question": query})
        st.success("Done!")

        # Final answer
        if "final_answer" in response:
            st.markdown("### 🧠 Answer:")
            st.write(response["final_answer"])

        # Retrieved documents (only if present)
        documents = response.get("documents", [])
        if documents:
            st.markdown("### 📄 Retrieved Documents:")
            for i, doc in enumerate(documents, 1):
                st.markdown(f"**{i}.** {doc}")
        else:
            st.markdown("### No Docs:")

        if "final_answer" not in response and not documents:
            st.info("No answer or documents were returned.")
    else:
        st.warning("Please type a question.")
