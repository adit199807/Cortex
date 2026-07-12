from typing import List
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

class  VectorStore:
    """
    
    """
    def __init__(self) -> None:
        self.embedding = OpenAIEmbeddings()
        self.vectorstore = None
        self.retriever = None


    def create_vectorstore(self, documents:List[Document]):

        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorstore.as_retriever()

    def get_retriever(self):

        if not self.retriever:
            raise ValueError("Vectore store is not initialized, please initialize it by calling vedcote store")

        return self.retriever

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever.invoke(query)


