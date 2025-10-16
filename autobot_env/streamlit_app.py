import streamlit as st
from main import compiled_graph  # Assicurati che main.py esporti compiled_graph

st.set_page_config(page_title="Autobot - Chinese Cars in Italy", page_icon="🚗")

st.title("Autobot")
st.write("Ask anything about Chinese cars available in Italy!")

user_input = st.text_input("Your question:", "")

if st.button("Ask Autobot") and user_input.strip():
    with st.spinner("Autobot is thinking..."):
        result = compiled_graph.invoke({"input": user_input})
        st.markdown(result["output"])