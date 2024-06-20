import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    print("Retrieving...")
    embedding = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4")
    query = "What is Pinecone in machine learning"

    vectorstore = PineconeVectorStore(
        index_name=os.environ['INDEX_NAME'], embedding=embedding
    )

    template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        Always say 'Thanks for asking!' at the end of the answer.
        
        {context}
        
        Question: {question}
        
        Helpful Answer: 
    """

    custom_rag_prompt = PromptTemplate.from_template(template=template)

    rag_chain = (
            {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
    )

    result = rag_chain.invoke(query)
    print(result)

    # vectorstore = PineconeVectorStore(
    #     index_name=os.environ['INDEX_NAME'], embedding=embedding
    # )
    #
    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    # retrieval_chain = create_retrieval_chain(
    #     retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    # )
    #
    # result = retrieval_chain.invoke(input={"input": query})
    # print(result)