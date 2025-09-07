from bl import *
import streamlit as st
from PIL import Image
from langchain_groq import ChatGroq
from bl.chatollama import ChatOllama
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from sqlalchemy import create_engine,inspect, text
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy.orm import sessionmaker, declarative_base
from langchain_core.runnables.config import RunnableConfig
from langchain.chains.question_answering import load_qa_chain

class AgentState(TypedDict):
    question: str
    sql_query: str
    query_result: str
    query_rows: list
    attempts: int
    relevance: str
    sql_error: bool

class CheckRelevance(BaseModel):
    relevance: str = Field(
        description="Indicates whether the question is related to the database schema. 'relevant' or 'not_relevant'."
    )

class ConvertToSQL(BaseModel):
    sql_query: str = Field(
        description="The SQL query corresponding to the user's natural language question."
    )

class RewrittenQuestion(BaseModel):
    question: str = Field(description="The rewritten question.")

class ChatDB:

    # Initializing LLM            
    llm = ChatGroq(
    temperature=0,
    model_name=G_GROQ_LLM_MODEL,
        )    
    DATABASE_URL = f"mysql+pymysql://{G_DBUSERNAME}:{G_PASSWORD}@{G_SERVER}/{G_DATABASE}"
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    description = """
    "Year": "The fiscal year for which the financial data is reported.",
    "Company": "The name of the company like [AAPL,MSFT,GOOG,PYPL,AIG,PCG,SHLDQ,,MCD,BCS,NVDA,INTC,AMZN].",
    "Category": "The industry or sector the company operates in.",
    "Market Cap(in B USD)": "Market capitalization in billions of US dollars.",
    "Revenue": "Total revenue generated in the fiscal year (in USD).",
    "GrossProfit": "Gross profit earned in the fiscal year (in USD).",
    "NetIncome": "Net income after all operating expenses, taxes, and interest (in USD).",
    "Earning Per Share": "Earnings per share (EPS), indicating profitability per share.",
    "EBITDA": "Earnings before interest, taxes, depreciation, and amortization (in USD).",
    "Share Holder Equity": "Total equity held by shareholders at the end of the fiscal year (in USD).",
    "Cash Flow from Operating": "Net cash generated from core operating activities (in USD).",
    "Cash Flow from Investing": "Net cash used or generated through investing activities (in USD).",
    "Cash Flow from Financial Activities": "Net cash inflow or outflow from financing activities (in USD).",
    "Current Ratio": "Liquidity ratio calculated as current assets divided by current liabilities.",
    "Debt/Equity Ratio": "Financial leverage ratio measuring the proportion of debt to equity.",
    "ROE": "Return on Equity – percentage return on shareholders’ equity.",
    "ROA": "Return on Assets – percentage return relative to total assets.",
    "ROI": "Return on Investment – a measure of profitability relative to invested capital.",
    "Net Profit Margin": "Percentage of revenue that remains as profit after all expenses.",
    "Free Cash Flow per Share": "Cash available to equity holders on a per-share basis (in USD).",
    "Return on Tangible Equity": "Return based on equity excluding intangible assets.",
    "Number of Employees": "Total number of full-time employees in the company during the fiscal year.",
    "Inflation Rate(in US)": "Annual inflation rate in the United States for the corresponding year."
    """    
    def get_database_schema():
        inspector = inspect(ChatDB.engine)
        schema = ""
        for table_name in inspector.get_table_names():
            schema += f"Table: {table_name}\n"
            for column in inspector.get_columns(table_name):
                col_name = column["name"]
                col_type = str(column["type"])
                if column.get("primary_key"):
                    col_type += ", Primary Key"
                if column.get("foreign_keys"):
                    fk = list(column["foreign_keys"])[0]
                    col_type += f", Foreign Key to {fk.column.table.name}.{fk.column.name}"
                schema += f"- {col_name}: {col_type}\n"
            schema += "\n"
            
        #st.write("Retrieved database schema.")
        return schema + ChatDB.description
    
    def check_relevance(state: AgentState, config: RunnableConfig):
        question = state["question"]
        schema = ChatDB.get_database_schema()
        #st.write(f"Checking relevance of the question: {question}")
        system = """You are an assistant that determines whether a given question is related to the following database schema.

        Schema:
        {schema}

        Respond with only "relevant" or "not_relevant".
        """.format(schema=schema)
        human = f"Question: {question}"
        check_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    ("human", human),
                ]
            )
        structured_llm = ChatDB.llm.with_structured_output(CheckRelevance)
        relevance_checker = check_prompt | structured_llm
        relevance = relevance_checker.invoke({})
        state["relevance"] = relevance.relevance
        #st.write(f"Relevance determined: {state['relevance']}")
        return state
    
    def convert_nl_to_sql(state: AgentState, config: RunnableConfig):
        question = state["question"]
        # schema = get_database_schema(engine)
        #st.write(f"Converting question to SQL: {question}")
        system = """You are an assistant that converts natural language questions into SQL queries based.Use the exact DataBase Name,Table and Columns Name:
                DataBase Name ="finance"
            Table Name = ['finance_report']
            Column Name and Data Type:
                - 'Year' bigint
                - 'Company' text
                - Category text
                - 'Market Cap(in B USD)' double
                - 'Revenue' double
                - 'GrossProfit' double
                - 'NetIncome' double
                - 'Earning Per Share' double
                - 'EBITDA' double
                - 'Share Holder Equity' double
                - 'Cash Flow from Operating' double
                - 'Cash Flow from Investing' double
                - 'Cash Flow from Financial Activities' double
                - 'Current Ratio' double
                - 'DebtEquityRatio' double
                - 'ROE' double
                - 'ROA' double
                - 'ROI' double
                - 'Net Profit Margin' double
                - 'Free Cash Flow per Share' double
                - 'Return on Tangible Equity' double
                - 'Number of Employees' bigint
                - 'Inflation Rate(in US)' double
            Columns Description:
                "Year": "The fiscal year for which the financial data is reported.",
                "Company": "The name of the company like [AAPL,MSFT,GOOG,PYPL,AIG,PCG,SHLDQ,,MCD,BCS,NVDA,INTC,AMZN].",
                "Category": "The industry or sector the company operates in.",
                "Market Cap(in B USD)": "Market capitalization in billions of US dollars.",
                "Revenue": "Total revenue generated in the fiscal year (in USD).",
                "Gross Profit": "Gross profit earned in the fiscal year (in USD).",
                "NetIncome": "Net income after all operating expenses, taxes, and interest (in USD).",
                "Earning Per Share": "Earnings per share (EPS), indicating profitability per share.",
                "EBITDA": "Earnings before interest, taxes, depreciation, and amortization (in USD).",
                "Share Holder Equity": "Total equity held by shareholders at the end of the fiscal year (in USD).",
                "Cash Flow from Operating": "Net cash generated from core operating activities (in USD).",
                "Cash Flow from Investing": "Net cash used or generated through investing activities (in USD).",
                "Cash Flow from Financial Activities": "Net cash inflow or outflow from financing activities (in USD).",
                "Current Ratio": "Liquidity ratio calculated as current assets divided by current liabilities.",
                "DebtEquityRatio": "Financial leverage ratio measuring the proportion of debt to equity.",
                "ROE": "Return on Equity – percentage return on shareholders’ equity.",
                "ROA": "Return on Assets – percentage return relative to total assets.",
                "ROI": "Return on Investment – a measure of profitability relative to invested capital.",
                "Net Profit Margin": "Percentage of revenue that remains as profit after all expenses.",
                "Free Cash Flow per Share": "Cash available to equity holders on a per-share basis (in USD).",
                "Return on Tangible Equity": "Return based on equity excluding intangible assets.",
                "Number of Employees": "Total number of full-time employees in the company during the fiscal year.",
                "Inflation Rate(in US)": "Annual inflation rate in the United States for the corresponding year."
                
                Ensure that all query-related data is scoped. 
                Provide only the SQL query without any explanations. Alias columns appropriately to match the expected keys in the result.
                Note Do not use '_' in columns. Use the same column name that are mentioned above
            """
        convert_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Question: {question}"),
            ]
        )
        structured_llm = ChatDB.llm.with_structured_output(ConvertToSQL)
        sql_generator = convert_prompt | structured_llm
        result = sql_generator.invoke({"question": question})
        state["sql_query"] = result.sql_query
        #print(f"Generated SQL query: {state['sql_query']}")
        return state

    def execute_sql(state: AgentState):
        sql_query = state["sql_query"].strip()
        session = ChatDB.SessionLocal()
        #st.write(f"Executing SQL query: {sql_query}")
        try:
            result = session.execute(text(sql_query))
            if sql_query.lower().startswith("select"):
                rows = result.fetchall()
                columns = result.keys()
                if rows:
                    header = ", ".join(columns)
                    state["query_rows"] = [dict(zip(columns, row)) for row in rows]
                    #print(f"Raw SQL Query Result: {state['query_rows']}")
                    # Format the result for readability
                    data = "; ".join([
                        ", ".join([f"{key}: {value}" for key, value in row.items()])
                        for row in state["query_rows"]
                    ])
                    formatted_result = f"{header}\n{data}"
                else:
                    state["query_rows"] = []
                    formatted_result = "No results found."
                state["query_result"] = formatted_result
                state["sql_error"] = False
                #print("SQL SELECT query executed successfully.")
            else:
                session.commit()
                state["query_result"] = "The action has been successfully completed."
                state["sql_error"] = False
                #print("SQL command executed successfully.")
        except Exception as e:
            state["query_result"] = f"Error executing SQL query: {str(e)}"
            state["sql_error"] = True
            #st.write(f"Error executing SQL query: {str(e)}")
        finally:
            session.close()
        return state
    
    def generate_human_readable_answer(state: AgentState):
        sql = state["sql_query"]
        result = state["query_result"]
        #st.write(result)
        query_rows = state.get("query_rows", [])
        sql_error = state.get("sql_error", False)
        #st.write("Generating a human-readable answer.")
        system = """
        You are a professional  financial data analyst assistant. Your role is to interpret SQL query results and convert them into clear, executive-level summaries that are ready for dashboards, reports, or stakeholder communication.
        Column Definitions:
            "Year": "The fiscal year for which the financial data is reported.",
                "Company": "The name of the company like [AAPL,MSFT,GOOG,PYPL,AIG,PCG,SHLDQ,,MCD,BCS,NVDA,INTC,AMZN].",
                "Category": "The industry or sector the company operates in.",
                "Market Cap(in B USD)": "Market capitalization in billions of US dollars.",
                "Revenue": "Total revenue generated in the fiscal year (in USD).",
                "Gross Profit": "Gross profit earned in the fiscal year (in USD).",
                "NetIncome": "Net income after all operating expenses, taxes, and interest (in USD).",
                "Earning Per Share": "Earnings per share (EPS), indicating profitability per share.",
                "EBITDA": "Earnings before interest, taxes, depreciation, and amortization (in USD).",
                "Share Holder Equity": "Total equity held by shareholders at the end of the fiscal year (in USD).",
                "Cash Flow from Operating": "Net cash generated from core operating activities (in USD).",
                "Cash Flow from Investing": "Net cash used or generated through investing activities (in USD).",
                "Cash Flow from Financial Activities": "Net cash inflow or outflow from financing activities (in USD).",
                "Current Ratio": "Liquidity ratio calculated as current assets divided by current liabilities.",
                "DebtEquityRatio": "Financial leverage ratio measuring the proportion of debt to equity.",
                "ROE": "Return on Equity – percentage return on shareholders’ equity.",
                "ROA": "Return on Assets – percentage return relative to total assets.",
                "ROI": "Return on Investment – a measure of profitability relative to invested capital.",
                "Net Profit Margin": "Percentage of revenue that remains as profit after all expenses.",
                "Free Cash Flow per Share": "Cash available to equity holders on a per-share basis (in USD).",
                "Return on Tangible Equity": "Return based on equity excluding intangible assets.",
                "Number of Employees": "Total number of full-time employees in the company during the fiscal year.",
                "Inflation Rate(in US)": "Annual inflation rate in the United States for the corresponding year."
        Instructions for the LLM:
            Analyze dynamically: Use only the columns provided in the result set. Do not refer to missing data.
            Avoid filler language: Never use phrases like "more data is needed", "limited data", or "could not determine". Always write with confidence.
            Always generate a clean, executive-friendly summary, as if it were written by a senior analyst for business stakeholders.
            Highlight key financial metrics, year-over-year trends, and noteworthy performance changes.
            Focus on clarity, business value, and actionable insight — not technical jargon.  
         Output Style Examples:
        
        DO:
        - “In FY2022, Apple reported $394B in revenue and $99.8B in net income, delivering an EPS of $6.11. Despite a lower market cap compared to 2021, Apple maintained a strong ROE of 197% and added 10K new employees.”
        - “Over a 5-year period, Apple’s net profit margin remained above 20%, with the highest EBITDA recorded in 2022 at $130.5B, signaling efficient operations despite inflation pressures.”
        
        DO NOT:
        - “Based on the limited data available…”
        - “It would be helpful to have more information…”
        - “The company seems profitable but cannot confirm without more metrics…”
        """
        if sql_error:
        # Directly relay the error message
            generate_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    f"""SQL Query:
            {sql}

        Result:
        {result}
        Formulate a clear and understandable error message in a single sentence,informing them about the issue."""
                        ),
                    ]
                )
        elif sql.lower().startswith("select"):
            if not query_rows:
                # Handle cases with no orders
                generate_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system),
                        (
                            "human",
                            f"""SQL Query:
                            {sql}
                            Result:
                            {result}
                            """
                        ),
                    ]
                )
            else:
                
                generate_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system),
                        (
                            "human",
                            f"""SQL Query:
                                {sql}                            
                                Result:
                                {result}
                                '"""
                        ),
                    ]
                )
        else:
            # Handle non-select queries
            generate_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    (
                        "human",
                        f"""SQL Query:
                        {sql}
                        Result:
                        {result}
                        """
                    ),
                ]
            )

        human_response = generate_prompt | ChatDB.llm | StrOutputParser()
        answer = human_response.invoke({})
        state["query_result"] = answer
        #st.write("Generated human-readable answer.")
        return state
    
    def regenerate_query(state: AgentState):
        question = state["question"]
        #st.write("Regenerating the SQL query by rewriting the question.")
        system = """You are an assistant that reformulates an original question to enable more precise SQL queries. Ensure that all necessary details, such as table joins, are preserved to retrieve complete and accurate data.
        """
        rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    f"Original Question: {question}\nReformulate the question to enable more precise SQL queries, ensuring all necessary details are preserved.",
                ),
            ]
        )
    
        structured_llm = ChatDB.llm.with_structured_output(RewrittenQuestion)
        rewriter = rewrite_prompt | structured_llm
        rewritten = rewriter.invoke({})
        state["question"] = rewritten.question
        state["attempts"] += 1
        #st.write(f"Rewritten question: {state['question']}")
        return state
    
    def generate_knowledgebase_response(state: AgentState):
        question = state["question"]

        # step: initialize embeddings
        embeddings = ChatOllama.initialize_embedding_model()

        # step7: initialize qa chain
        qa = load_qa_chain(llm=ChatDB.llm, chain_type="stuff")    
        
        if embeddings is None:
            state["query_result"] = "Ollama embeddings failed to initialize "
            return state

        # step2: Load vector store
        vectorstore = ChatOllama.load_vectorstore(embeddings)

        if vectorstore is None:
            state["query_result"] = "Failed to load vector store"
            return state

        relevant_docs = ChatOllama.get_relevant_documents(vectorstore, question)

        if relevant_docs is None:
            state["query_result"] = "Failed to load relevant docs"
            return state
        
        if relevant_docs:
            llm_response = qa.invoke( {"input_documents": relevant_docs, "question": question})
            # Check if the answer is found in the local context
            response = llm_response["output_text"]

            # Check if response indicates no relevant info found
            if "sorry, answer not found in knowledge base" in response.lower():
                response = "Sorry, answer not found in knowledge base."
        else:
            response = "Sorry, answer not found in knowledge base."

        state["query_result"] = response

        return state
    
    def generate_fallback_response(state: AgentState):
        st.write("LLM could not find an answer in DB. Query redirecting toward Knowledgebase")
        state["query_result"] = "Sorry, Unable to find answer in Database."
        return state
    
    def end_max_iterations(state: AgentState):
        state["query_result"] = "Please try again."
        st.write("Maximum attempts reached. Ending the workflow.")
        return state

    def relevance_router(state: AgentState):
        if state["relevance"].lower() == "relevant":
            return "convert_to_sql"
        else:
            return "generate_fallback_response"
        
    def check_attempts_router(state: AgentState):
        if state["attempts"] < 3:
            return "convert_to_sql"
        else:
            return "end_max_iterations"
    
    def execute_sql_router(state: AgentState):
        if not state.get("sql_error", False):
            return "generate_human_readable_answer"
        else:
            return "regenerate_query"
    
    def initialize_agent():
        try:

            workflow = StateGraph(AgentState)
            workflow.add_node("check_relevance", ChatDB.check_relevance)
            workflow.add_node("convert_to_sql", ChatDB.convert_nl_to_sql)
            workflow.add_node("execute_sql", ChatDB.execute_sql)
            workflow.add_node("generate_human_readable_answer", ChatDB.generate_human_readable_answer)
            workflow.add_node("regenerate_query", ChatDB.regenerate_query)
            workflow.add_node("generate_fallback_response", ChatDB.generate_fallback_response)
            workflow.add_node("end_max_iterations", ChatDB.end_max_iterations)
            workflow.add_node("generate_knowledgebase_response", ChatDB.generate_knowledgebase_response)

            #workflow.add_edge("check_relevance","convert_to_sql")

            workflow.add_conditional_edges("check_relevance",
                                ChatDB.relevance_router,{
                                    "convert_to_sql":"convert_to_sql",
                                    "generate_fallback_response":"generate_fallback_response"
                                        })
            
            workflow.add_edge("convert_to_sql", "execute_sql")
            workflow.add_edge("generate_fallback_response", "generate_knowledgebase_response")

            workflow.add_conditional_edges(
                "execute_sql",
                ChatDB.execute_sql_router,
                {
                    "generate_human_readable_answer": "generate_human_readable_answer",
                    "regenerate_query": "regenerate_query",
                },
            )

            workflow.add_conditional_edges(
                "regenerate_query",
                ChatDB.check_attempts_router,
                {
                    "convert_to_sql": "convert_to_sql",
                    "max_iterations": "end_max_iterations",
                },
            )

            workflow.add_edge("generate_human_readable_answer", END)
            workflow.add_edge("generate_knowledgebase_response", END)
            workflow.add_edge("end_max_iterations", END)

            workflow.set_entry_point("check_relevance")
            app = workflow.compile()

            
            try:
                #st.image(Image(app.get_graph(xray=True).draw_mermaid_png()))
                app.get_graph(xray=True).print_ascii()
            except Exception as e:
                #st.error(f"Error Creating Image:{e}")
                pass

            return app
        except Exception as ex:
            st.write(f"Initialize Agent: {ex}")
            return None
    

    def chat_interface():

        """Main chat interface"""
        st.title("🤖 Financial AI Chatbot ")
        st.write("Ask questions about your companies finances using Groq/MySQL")

        #st.write(f"Database URL Information:{ChatDB.DATABASE_URL}")
        #st.write(f"Schema Information:{ChatDB.get_database_schema()}")

        # step 1: Initialize embeddings and llama models
        with st.spinner("Initializing Agent..."):
            app = ChatDB.initialize_agent()
      
        if app is None:
            st.error("Error Initializing Agent")
            st.stop()

        # step3: setup chat history    
        # Initialize session state for chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

         # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get relevant documents
            with st.spinner("Generating response..."):                
                result = app.invoke({"question": prompt, "attempts": 0})                

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": result['query_result']})
            with st.chat_message("assistant"):
                st.markdown(result['query_result'])


        







