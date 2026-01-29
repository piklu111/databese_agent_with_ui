import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from urllib.parse import quote_plus
from dotenv import load_dotenv
import os
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import tempfile
import shutil

load_dotenv()

st.set_page_config(page_title="Text to SQL Agent", layout="wide")

# ---------------- LLM LOADING ---------------- #

@st.cache_resource
def load_llm():
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        task="text-generation",
        max_new_tokens=500,
        #do_sample=False,
        # repetition_penalty=1.03,
        provider="auto",  # let Hugging Face choose the best provider for you
        temperature=0
    )
    return ChatHuggingFace(llm=llm)

model = load_llm()

# ---------------- SQL AGENT ---------------- #

def get_sql_agent(db_uri):
    db = SQLDatabase.from_uri(db_uri)
    agent = create_sql_agent(
        llm=model,
        db=db,
        verbose=True,
        agent_type="openai-tools"
    )
    return agent

def create_sqlite_agent_from_file(uploaded_file):

    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "data.db")

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    engine = create_engine(f"sqlite:///{db_path}")
    df.to_sql("data", engine, index=False, if_exists="replace")

    agent = get_sql_agent(f"sqlite:///{db_path}")
    return agent, df, temp_dir
# ---------------- UI ---------------- #

# st.title("🤖 Text to SQL Agent using HuggingFace + LangChain")
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 35px 40px;
        border-radius: 22px;
        box-shadow: 0 0 25px rgba(98,0,234,0.4);
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }

    .hero:before {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at top right, rgba(0,255,255,0.25), transparent 40%),
                    radial-gradient(circle at bottom left, rgba(255,0,255,0.25), transparent 40%);
        z-index: 0;
    }

    .hero h1 {
        font-size: 44px;
        font-weight: 900;
        background: linear-gradient(90deg, #00f5ff, #ff00f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        position: relative;
        z-index: 1;
    }

    .hero p {
        font-size: 18px;
        color: #cbd5f5;
        margin-top: 8px;
        position: relative;
        z-index: 1;
    }

    .badges {
        margin-top: 12px;
        position: relative;
        z-index: 1;
    }

    .badge {
        display: inline-block;
        background: rgba(255,255,255,0.08);
        padding: 6px 14px;
        border-radius: 12px;
        margin-right: 8px;
        font-size: 13px;
        color: #fff;
        border: 1px solid rgba(255,255,255,0.15);
        backdrop-filter: blur(8px);
    }
    </style>

    <div class="hero">
        <h1>⚡ Data Pilot</h1>
        <p>Talk to your Databases & Files using Natural Language</p>
    </div>
    """,
    unsafe_allow_html=True
)



tabs = st.tabs(["PostgreSQL", "MySQL", "SQLite", "CSV / Excel"])

# ============ PostgreSQL TAB ============ #
with tabs[0]:
    st.subheader("PostgreSQL Connection")

    user = st.text_input("User", "postgres")
    password = st.text_input("Password", type="password")
    host = st.text_input("Host", "localhost")
    port = st.text_input("Port", "5432")
    database = st.text_input("Database")

    if "pg_connected" not in st.session_state:
        st.session_state["pg_connected"] = False

    if st.button("Test PostgreSQL Connection"):
        if not all([user, password, host, port, database]):
            st.warning("⚠️ Please fill all DB fields")
        else:
            try:
                pwd = quote_plus(password)
                pg_uri = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{database}"

                db = SQLDatabase.from_uri(pg_uri)
                db.run("SELECT 1")

                st.session_state["pg_connected"] = True
                st.session_state["pg_uri"] = pg_uri

                st.success("✅ PostgreSQL connection successful!")

            except Exception as e:
                st.session_state["pg_connected"] = False
                st.error(f"❌ Connection failed: {str(e)}")

    # Show query box only if connection successful
    if st.session_state["pg_connected"]:

        st.markdown("### Ask Your Question")

        query = st.text_input(
            "Enter your question",
            placeholder="Who spent the highest amount on movie rent?",
            key="pg_query"
        )

        if st.button("Run Query on PostgreSQL"):
            if not query:
                st.warning("⚠️ Please enter a question")
            else:
                with st.spinner("Executing query..."):
                    try:
                        agent = get_sql_agent(st.session_state["pg_uri"])
                        res = agent.invoke({"input": query})

                        st.success("Query executed successfully!")
                        st.markdown("### Answer")
                        st.write(res["output"])

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")


# ============ MySQL TAB ============ #
with tabs[1]:
    st.subheader("MySQL Connection")

    user = st.text_input("User", key="mysql_user")
    password = st.text_input("Password", type="password", key="mysql_pass")
    host = st.text_input("Host", "localhost", key="mysql_host")
    port = st.text_input("Port", "3306", key="mysql_port")
    database = st.text_input("Database", key="mysql_db")

    if st.button("Run on MySQL"):
        if not all([user, password, host, port, database]):
            st.warning("⚠️ Please fill all fields + question")
        else:
            query = st.text_input("Enter your question:", placeholder="Who spent the highest amount on movie rent?")
            with st.spinner("Executing query..."):
                try:
                    pwd = quote_plus(password)
                    mysql_uri = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{database}"

                    agent = get_sql_agent(mysql_uri)
                    res = agent.invoke({"input": query})

                    st.success("Query executed successfully!")
                    st.markdown("### Answer")
                    st.write(res["output"])

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# ============ SQLite TAB ============ #
with tabs[2]:
    st.subheader("SQLite Connection")

    sqlite_path = st.text_input("Database File Path", "example.db")

    if st.button("Run on SQLite"):
        if not all([sqlite_path]):
            st.warning("⚠️ Please fill database path + question")
        else:
            query = st.text_input("Enter your question:", placeholder="Who spent the highest amount on movie rent?")
            with st.spinner("Executing query..."):
                try:
                    sqlite_uri = f"sqlite:///{sqlite_path}"

                    agent = get_sql_agent(sqlite_uri)
                    res = agent.invoke({"input": query})

                    st.success("Query executed successfully!")
                    st.markdown("### Answer")
                    st.write(res["output"])

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
# ============ CSV / EXCEL TAB ============ #
with tabs[3]:
    st.subheader("Upload CSV or Excel & Query")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    if uploaded_file:
        with st.spinner("Processing file..."):
            if "agent" not in st.session_state:
                agent, df, temp_dir = create_sqlite_agent_from_file(uploaded_file)

                st.session_state["agent"] = agent
                st.session_state["df"] = df
                st.session_state["temp_dir"] = temp_dir

            st.success("File uploaded successfully!")

            st.markdown("### Data Preview")
            st.dataframe(st.session_state["df"].head(20))

        question = st.text_input(
            "Ask question about this dataset",
            placeholder="Which product has highest sales?"
        )

        if st.button("Run Query"):
            if not question:
                st.warning("⚠️ Please enter a question")
            else:
                with st.spinner("Running query..."):
                    try:
                        res = st.session_state["agent"].invoke({"input": question})
                        st.success("Query executed successfully!")
                        st.markdown("### Answer")
                        st.write(res["output"])

                        st.session_state["show_popup"] = True

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
# ========= POPUP SECTION (ADD HERE AT BOTTOM) ========= #

if st.session_state.get("show_popup", False):

    st.warning("Do you want to continue querying this uploaded file?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Yes", key="continue_yes"):
            st.session_state["show_popup"] = False
            st.success("You can continue querying.")

    with col2:
        if st.button("No", key="continue_no"):
            temp_dir = st.session_state.get("temp_dir")

            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

            st.session_state.clear()
            st.success("Temporary files deleted successfully.")
