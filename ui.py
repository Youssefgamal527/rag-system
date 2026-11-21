
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("📚 RAG Assistant")
st.caption("اسألي عن المعلومات الموجودة في ملفاتك")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            st.markdown("**المصادر:**")
            for source in message["sources"]:
                st.write(
                    f"- {source['source']} - صفحة {source['page']}"
                )

question = st.chat_input("اكتبي سؤالك هنا...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث وتجهيز الإجابة..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question},
                    timeout=180
                )

                response.raise_for_status()
                result = response.json()

                answer = result["answer"]
                sources = result.get("sources", [])

                st.markdown(answer)

                if sources:
                    st.markdown("**المصادر:**")
                    for source in sources:
                        st.write(
                            f"- {source['source']} - صفحة {source['page']}"
                        )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except requests.RequestException as error:
                st.error(f"حدث خطأ أثناء الاتصال بالـ API: {error}")
