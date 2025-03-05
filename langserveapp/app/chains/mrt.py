from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.vectorstores import Qdrant
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# 載入 env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = os.getenv("QDRANT_PORT", "6333")

# 使用 OpenAIEmbeddings
embeddings_model = OpenAIEmbeddings(api_key=openai_api_key,model=embedding_model)

# 連線 Qdrant
client = QdrantClient(host=qdrant_host, port=int(qdrant_port))

# 確保 collection 存在避免報錯，又 text-embedding-3-large 模型的向量大小為 3072 維度
collection_name = "mrt"
try:
    client.get_collection(collection_name=collection_name)
except Exception:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=3072, distance=models.Distance.COSINE)
    )
qdrant = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings_model
)

# 設定 qdrant retriever，只有當 metadata.department 為 MRT 的資料才會被取得
retriever = qdrant.as_retriever(
    search_kwargs=dict(
    filter=models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.department",
                match=models.MatchValue(value="MRT")
            )
        ]
    )
))

# 設定 openai model
model = ChatOpenAI(api_key=openai_api_key, model=chat_model)

# 設定 prompt
prompt = ChatPromptTemplate.from_template("""請回答依照 context 裡的資訊來回答問題:
<context>
{context}
</context>
Question: {input}""")
# print(prompt)
# input_variables=['context', 'input'] input_types={} partial_variables={} messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'input'], input_types={}, partial_variables={}, template='請回答依照 context 裡的資訊來回答問題:\n<context>\n{context}\n</context>\nQuestion: {input}'), additional_kwargs={})]

# 依照傳入的 model 和 prompt 建立 LCEL 鏈，結構如下
document_chain = create_stuff_documents_chain(model, prompt)
# print(document_chain)
# bound = RunnableBinding(
#     bound=RunnableAssign(
#         mapper={
#             'context': RunnableLambda(format_docs)
#         }
#     ),
#     kwargs={},
#     config={'run_name': 'format_inputs'},
#     config_factories=[]
# ) | ChatPromptTemplate(
#     input_variables=['context', 'input'],
#     input_types={},
#     partial_variables={},
#     messages=[
#         HumanMessagePromptTemplate(
#             prompt=PromptTemplate(
#                 input_variables=['context', 'input'],
#                 input_types={},
#                 partial_variables={},
#                 template='請回答依照 context 裡的資訊來回答問題:\n<context>\n{context}\n</context>\nQuestion: {input}'
#             )
#         )
#     ]
# ) | ChatOpenAI(
#     client=<openai.resources.chat.completions.completions.Completions object>,
#     async_client=<openai.resources.chat.completions.completions.AsyncCompletions object>,
#     root_client=<openai.OpenAI object>,
#     root_async_client=<openai.AsyncOpenAI object>,
#     model_name='o3-mini',
#     model_kwargs={},
#     openai_api_key=SecretStr('**********')
# ) | StrOutputParser(
#     kwargs={},
#     config={'run_name': 'stuff_documents_chain'},
#     config_factories=[]
# )

# 依照傳入的 qdrant retriever 和剛才建立的 document chain 建立 retrieval chain，結構如下
retrieval_chain = create_retrieval_chain(retriever, document_chain)
# bound = RunnableAssign(
#     mapper={
#         'context': RunnableBinding(
#             bound=RunnableLambda(lambda x: x['input']) | VectorStoreRetriever(
#                 tags=['QdrantVectorStore', 'OpenAIEmbeddings'],
#                 vectorstore=<langchain_qdrant.qdrant.QdrantVectorStore object>,
#                 search_kwargs={
#                     'filter': Filter(
#                         should=None, 
#                         min_should=None, 
#                         must=[
#                             FieldCondition(
#                                 key='metadata.department', 
#                                 match=MatchValue(value='MRT'), 
#                                 range=None, 
#                                 geo_bounding_box=None, 
#                                 geo_radius=None, 
#                                 geo_polygon=None, 
#                                 values_count=None
#                             )
#                         ], 
#                         must_not=None
#                     )
#                 }
#             ),
#             kwargs={},
#             config={'run_name': 'retrieve_documents'},
#             config_factories=[]
#         )
#     }
# ) | RunnableAssign(
#     mapper={
#         'answer': RunnableBinding(
#             bound=RunnableBinding(
#                 bound=RunnableAssign(
#                     mapper={
#                         'context': RunnableLambda(format_docs)
#                     }
#                 ),
#                 kwargs={},
#                 config={'run_name': 'format_inputs'},
#                 config_factories=[]
#             ) | ChatPromptTemplate(
#                 input_variables=['context', 'input'],
#                 input_types={},
#                 partial_variables={},
#                 messages=[
#                     HumanMessagePromptTemplate(
#                         prompt=PromptTemplate(
#                             input_variables=['context', 'input'],
#                             input_types={},
#                             partial_variables={},
#                             template='請回答依照 context 裡的資訊來回答問題:\n<context>\n{context}\n</context>\nQuestion: {input}'
#                         )
#                     )
#                 ]
#             ) | ChatOpenAI(
#                 client=<openai.resources.chat.completions.completions.Completions object>,
#                 async_client=<openai.resources.chat.completions.completions.AsyncCompletions object>,
#                 root_client=<openai.OpenAI object>,
#                 root_async_client=<openai.AsyncOpenAI object>,
#                 model_name='o3-mini',
#                 model_kwargs={},
#                 openai_api_key=SecretStr('**********')
#             ) | StrOutputParser(),
#             kwargs={},
#             config={'run_name': 'stuff_documents_chain'},
#             config_factories=[]
#         )
#     },
#     kwargs={},
#     config={'run_name': 'retrieval_chain'},
#     config_factories=[]
# )

# 建立 chain 並指定問題的型別為字串
class Question(BaseModel): 
    input: str
chain = retrieval_chain.with_types(input_type=Question)