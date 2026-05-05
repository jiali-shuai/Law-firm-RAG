from langchain_openai import ChatOpenAI

from agent.legal_graph import LegalGraph
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, setup_env


setup_env()

CHAT_LLM = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.3,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL
)

REASONER_LLM = ChatOpenAI(
    model="deepseek-reasoner",
    temperature=0.7,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL
)


def main():
    question = input("请输入您的法律问题：")
    graph = LegalGraph(CHAT_LLM, REASONER_LLM)
    result = graph.invoke(question)
    print(f"\n案件类型：{result['case_type']}")
    print(f"\n法律意见：\n{result['answer']}")


if __name__ == '__main__':
    main()
