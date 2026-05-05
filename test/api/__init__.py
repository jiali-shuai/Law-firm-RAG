from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI

from config import setup_env, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from database.milvus import get_client

setup_env()

_upload_chat_llm = None
_upload_reasoner_llm = None


def _get_upload_llms():
    global _upload_chat_llm, _upload_reasoner_llm
    if _upload_chat_llm is None:
        _upload_chat_llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.3,
            openai_api_key=DEEPSEEK_API_KEY,
            openai_api_base=DEEPSEEK_BASE_URL,
        )
        _upload_reasoner_llm = ChatOpenAI(
            model="deepseek-reasoner",
            temperature=0.7,
            openai_api_key=DEEPSEEK_API_KEY,
            openai_api_base=DEEPSEEK_BASE_URL,
        )
    return _upload_chat_llm, _upload_reasoner_llm


app = FastAPI(title="RAG向量库管理系统", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    try:
        get_client()
        print("✅ Milvus 连接成功！")
    except Exception as e:
        print(f"⚠️ Milvus 连接失败: {e}")


from api.collections import router as collections_router
from api.documents import router as documents_router
from api.chat import router as chat_router

app.include_router(collections_router)
app.include_router(documents_router)
app.include_router(chat_router)
