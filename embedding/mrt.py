from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
import os

# load env variables
load_dotenv()
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = os.getenv("QDRANT_PORT", "6333")
openai_api_key = os.getenv("OPENAI_API_KEY")
embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# load the document
loader = PyPDFLoader("./MRT.pdf")
docs = loader.load()
# print(docs)

# set up splitter and embeddings model
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)
embeddings_model = OpenAIEmbeddings(api_key=openai_api_key, model=embedding_model)

# add metadata to the chunks
metadatas = []
for text in chunks:
    metadatas.append({
        "department": "MRT",
        "account_manager": "David",
        "source": text.metadata["source"],
        "page": text.metadata["page"]
    })

# create the Qdrant vectorstore
qdrant = Qdrant.from_texts(
    [t.page_content for t in chunks],
    embeddings_model,
    url=qdrant_host, 
    collection_name="mrt",
    force_recreate=True,
    metadatas = metadatas
)
