

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader ,PyPDFDirectoryLoader # load document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

#data in st session -> no to loose previous conversation during refresh


if "document_uploaded" not in st.session_state: 
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent =None

if "vector_store" not in st.session_state:
    st.session_state.vector_store= None

if "messages" not in st.session_state:
    st.session_state.messages=[]

def process_document(path):
    
    ##load documents

    #loader=PyPDFLoader("../data/medical_report.pdf") # for single file
    loader=PyPDFDirectoryLoader(path) # for multiple files
    docs=loader.load()

    #splitting into multiple chunks
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=200)
    docs=splitter.split_documents(documents=docs)

    #emeddings and Vector DB

    embeddings=OpenAIEmbeddings(model="text-embedding-3-large")
    vector_db=InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )

    #create agent -> tool , llm , prompt

    #llm
    llm=ChatGroq(model="openai/gpt-oss-20b")

    #tool
    @tool
    def retreive_context(query:str):
        """
            Retreive documents relevant to a query from the knowlwdge base.
        """
        context=""

        docs=vector_db.similarity_search(query=query,k=3) # 4 by default , can customize
        for doc in docs:
            context+= doc.page_content + "\n\n"

        return context
        #return "\n\n".join(doc.page_content for doc in docs)

    #prompt
    system_prompt="""
        You are a helpful assistant that answers questions using retreived context.
        My kowledge base consists of the details from the uploaded document.
        Alwyas use the 'retreive_context' tool for questions requiring external knowledge .
    """
    #memory by langraph
    memory= InMemorySaver()

    #agent
    agent=create_agent(
        model=llm,
        tools=[retreive_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    st.session_state.agent=agent
    st.session_state.document_uploaded=True


    #->>>>>>>>>>>>without UI interaction
    # while True:
    #     query =input("User: ")
    #     if query.lower()=="quit":
    #         break

    #     res=agent.invoke(
    #         {"messages":[{"role":"user","content":query}]},
    #         {"configurable":{"thread_id":1}}
    #         )
    #     result=res["messages"][-1].content

    #     print("AI : ",result)



#upload UI->till pdf not uploaded

if not st.session_state.document_uploaded:
    uploaded=st.file_uploader(label="Select PDF Files",type=["pdf"],accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing ...."):
            # uploaded pdf to local folder doc_files
            path="./doc_files/" 
            for file in uploaded:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())

            process_document(path)
            st.rerun()


    #chat UI->whrn pdf uploaded and vector DB stored

    #when doc is uploaded and agent value is true the only
if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role= message.get("role")
        content=message.get("content")
        st.chat_message(role).markdown(content)


    query= st.chat_input("Ask Anything related to uploaded documents....")
    #now acc. to query
    if query:
        st.session_state.messages.append({"role":"user","content":query})
        st.chat_message("user").markdown(query)
        res=st.session_state.agent.invoke(
            {"messages":[{"role":"user","content":query}]},
            {"configurable":{"thread_id":1}}
        )

        answer=res["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role":"ai","content":answer})

