from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
import os

# load env variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# load the document
loader = PyPDFLoader("./MRT.pdf")
docs = loader.load()
# print(docs)

# set up splitter and embeddings model
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)
embeddings_model = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-large")

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
    url="localhost", 
    collection_name="mrt",
    force_recreate=True,
    metadatas = metadatas
)
