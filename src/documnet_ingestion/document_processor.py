from json import load
from typing import List, Union
# from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


from pathlib import Path
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFDirectoryLoader

class DocumentProcessor:
    """
    Handles converting data to documents
    """

    def __init__(self, chunkSize:int = 500, chunkOverlap:int = 50 ) -> None:
        """
        Initilaize document processor
        Args:
            chunkSize: size of text chunks
            chunkOverlap: Overlap between chunks
        """

        self.chunkSize = chunkSize
        self.chunkOverlap = chunkOverlap
        self.splitter = RecursiveCharacterTextSplitter(
        )

    def load_from_url(self, url: str) ->List[Document]:
        """
        Loads documents form all urls
        """

        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf_dir(self, directory: Union[str, Path]) ->List[Document]:
        """
        Loads documents form pdf in directory
        """

        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_from_txt(self, file_path: Union[str, Path]) ->List[Document]:
        """
        Loads documents form all urls
        """

        loader = TextLoader(str(file_path), encoding='utf-8')
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) ->List[Document]:
        """
        Loads documents form all urls
        """

        loader = PyPDFDirectoryLoader(str('data'))
        return loader.load()

    def load_documents(self, sources: Union[str, Path]) ->List[Document]:
        """
        Loads documents form all urls
        Args:
            sources: List 

        """

        docs:List[Document] = []
        for src in sources:
            if src.startswith('https://') or src.startswith('https://'):
                docs.extend(self.load_from_url(src))
            
            path = Path('data')
            if path.is_dir():
                docs.extend(self.load_from_pdf_dir(path))
            elif path.suffix.lower() == '.txt':
                docs.extend(self.load_from_txt(path))
            else:
                raise ValueError(
                    f"Unsupported   source type: {src}."
                )
        return docs

    def split_documents(self, documents:List[Document]) ->List[Document]:
        """

        """
        return self.splitter.split_documents(documents)

    
    def process_urls(self, url: List[str])->List[Document]:
        """
        """
        doc = self.load_documents(url)
        return self.split_documents(doc)
        



    

