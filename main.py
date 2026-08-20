
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import yfinance as yf
import chromadb
from chunc import chunk_text
from embed import create_embedding
import os


load_dotenv()

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background-color: #0F172A;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1E293B;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #F8FAFC;
}

/* Main title */
h1 {
    color: #F8FAFC;
    font-weight: 700;
}

/* Headings */
h2, h3 {
    color: #F8FAFC;
    font-weight: 600;
}

/* Normal text */
p, label {
    color: #CBD5E1;
}

/* Buttons */
.stButton > button {
    background-color: #10B981;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: 0.2s;
}

/* Button hover */
.stButton > button:hover {
    background-color: #059669;
    color: white;
    border: none;
}

/* Text input */
.stTextInput input {
    background-color: #1E293B;
    color: #F8FAFC;
    border: 1px solid #334155;
    border-radius: 8px;
}

/* Selectbox */
.stSelectbox > div > div {
    background-color: #1E293B;
    color: #F8FAFC;
    border-radius: 8px;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #1E293B;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #334155;
}

/* Metric value */
[data-testid="stMetricValue"] {
    color: #10B981;
    font-weight: 700;
}

/* Dividers */
hr {
    border-color: #334155;
}

</style>
""", unsafe_allow_html=True)


chroma_client = chromadb.PersistentClient(
    path="./stock_database"
)

collection = chroma_client.get_or_create_collection(
    name="stock_data"
)

def load_text():
    with open(
        "stock_market_overview.txt",
        "r",
        encoding="utf-8"
    ) as file:
        return file.read()

txt_file = load_text()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("📈 AI Stock Analyzer")

st.divider()

st.sidebar.title("Stock Settings")

st.sidebar.write("Enter a stock's ticker to find it's price history.")

ticker = st.sidebar.text_input("Stock Ticker:","AAPL").upper()

period = st.sidebar.selectbox(
    "Select how much data do you want?(from how long ago): ",
    [
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "2y"
    ]
)

save_settings = st.sidebar.button("Select Stock")

if "stock_data" not in st.session_state:
    st.session_state.stock_data = None

if "ticker" not in st.session_state:
    st.session_state.ticker = None

if "period" not in st.session_state:
    st.session_state.period = None

if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

if "analysis" not in st.session_state:
    st.session_state.analysis = ""

if save_settings:

    try:
        stock = yf.Ticker(
            ticker
        )


        data = stock.history(
            period=period,
        )
    except Exception as e:
        st.error(f"Could not get stock info: {e}.")
        st.stop()

    if data.empty:
        st.error(
            "Stock not found, please check your ticker."
        )

    else:

        st.session_state.stock_data = data
        st.session_state.ticker = ticker
        st.session_state.period = period

        # Reset old analysis
        st.session_state.analyzed = False
        st.session_state.analysis = ""

        st.sidebar.success("Stock Chosen, Stock Info Found!")

        #Turn info into strings so I can save it as chunks in a database
        stock_text = ""

        for date, row in data.iterrows():
            stock_text += f"""
        Ticker: {ticker}
        Date: {date}
        Open: ${row["Open"]:.2f}
        High: ${row["High"]:.2f}
        Low: ${row["Low"]:.2f}
        Close: ${row["Close"]:.2f}
        Volume: {row["Volume"]}
        """

        chunks = chunk_text(stock_text)

        embeddings = []

        for chunk in chunks:
            embedding = create_embedding(chunk)
            embeddings.append(embedding)

        ids = []

        for i in range(len(chunks)):
            ids.append(f"{ticker}_{i}")

        # Store in ChromaDB
        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings
        )

if st.session_state.stock_data is not None:

    data = st.session_state.stock_data
    ticker = st.session_state.ticker
    period = st.session_state.period

    current_price = data["Close"].iloc[-1]

    st.metric(
        "Current Price",
        f"${current_price:.2f} USD"
    )

    st.subheader(
        f"{ticker} Stock Price:"
    )

    st.line_chart(
        data["Close"]
    )

    first_price = data["Close"].iloc[0]

    change = (
        (current_price - first_price)
        / first_price
        * 100
    )

    st.subheader(
        f"Change within selected time range: {change:.2f}%"
    )

    #Ai Help

    st.subheader("AI Recommendations:")
    st.write("This system is an AI and AI can make mistakes")

    if st.button("Analyze Stock"):
        prompt = f"""
        Analyze this stock and give recommendations and suggestions to the user:

        {ticker}

        STARTING_PRICE:
        {first_price:.2f}

        CURRENT_PRICE:
        {current_price:.2f}

        CHANGE:
        {change:.2f}%

        SELECTED TIME RANGE:
        {period}

        EXTRA INFO:
        {txt_file}

        DO NOT tell the user to buy or sell any stock.
        DO NOT tell the user what to do.
        DO NOT use information from the web.
        DO NOT answer unrelated questions.
        """

        response = client.responses.create(
            model="gpt-4.1-nano",
            input=prompt
        )

        st.session_state.analysis = response.output_text
        st.session_state.analyzed = True

    # Display analysis
    if st.session_state.analyzed:
        st.divider()

        st.subheader("AI Analysis")

        st.write(
            st.session_state.analysis
        )

    # Chat is OUTSIDE the button
    st.divider()

    st.subheader("AI Recommendations and Help:")

    st.write(
        "This system is an AI and AI can make mistakes"
    )

    chat = st.text_input(
        "Ask a question about this stock:"
    )

    if chat:
        question_embedding = create_embedding(chat)

        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=3
        )

        documents = results["documents"][0]

        context = "\n\n".join(documents)

        prompt = f"""
        Answer the user's question using the stock information below.

        STOCK INFORMATION:
        {context}

        USER QUESTION:
        {chat}

        EXTRA INFO:
        {txt_file}

        Do not tell the user to buy or sell stocks.
        Do not give financial advice.
        Do not use information from the web.

        If the information isn't available, say you don't have
        enough information to answer.

        DO NOT EVER answer question that AREN'T related to stocks
        """

        response1 = client.responses.create(
            model="gpt-4.1-nano",
            input=prompt
        )

        st.write(response1.output_text)