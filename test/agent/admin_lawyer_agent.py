import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

from agent.prompt_loader import load_prompt
from chongpai.reranker import bge_rerank
from jiansuo.qurey import query

CASE_TYPE = "行政诉讼案件"
COLLECTION_NAME = "xingzheng"
PROMPT_FILE = "legal_admin_lawyer.txt"
PROMPT_JIANSUO = "legal_search_strategy.txt"
LABEL = "行政诉讼律师"
CASE_PARAMS = {"dense_top_k": 18, "sparse_top_k": 18, "alpha": 0.40}

"""兜底json解析"""
def _parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    if not text:
        raise ValueError("LLM 返回空内容")
    return json.loads(text)


"""行政诉讼律师智能体"""
class AdminLawyerAgent:

    def __init__(self, chat_llm: ChatOpenAI, reasoner_llm: ChatOpenAI):
        self.chat_llm = chat_llm
        self.reasoner_llm = reasoner_llm
        self.strategy_prompt = load_prompt(PROMPT_JIANSUO)
        self.lawyer_prompt = load_prompt(PROMPT_FILE)

    def run(self, state: dict) -> dict:
        user_demands = state["user_demands"]
        basic_info = state["basic_info"]
        
        """根据param_overrides调整参数"""
        params = dict(CASE_PARAMS)
        overrides = state.get("_param_overrides", {})
        if CASE_TYPE in overrides:
            p = overrides[CASE_TYPE]
            if p.get("dense_top_k") is not None:
                params["dense_top_k"] = p["dense_top_k"]
            if p.get("sparse_top_k") is not None:
                params["sparse_top_k"] = p["sparse_top_k"]
            if p.get("alpha") is not None:
                params["alpha"] = p["alpha"]

        """根据用户诉求和案件基本信息生成搜索策略"""
        strategy_input = f"案件类型：{CASE_TYPE}\n用户诉求：{user_demands}\n案件基本信息：{basic_info}"
        strategy_response = self.chat_llm.invoke([
            ("system", self.strategy_prompt),
            ("user", strategy_input)
        ])
        strategy = _parse_json(strategy_response.content)
        
        """提取三要素"""
        main_query = strategy["main_query"]
        element_queries = strategy["element_queries"]
        final_top_k = strategy["final_top_k"]
        
        """根据搜索策略进行查询"""
        all_queries = [main_query] + element_queries
        all_docs = []
        seen_ids = set() #去重
        """多query查询"""
        for q in all_queries:
            results = query(q, params["dense_top_k"], params["sparse_top_k"], params["alpha"], collection_name=COLLECTION_NAME)
            for item in results[0]:
                doc_id = item["id"]
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append({"id": doc_id, "text": item["text"]})

        reranked = bge_rerank(main_query, all_docs, final_top_k)

        context = "\n\n".join([
            f"【参考案例{i+1}】\n{doc.get('text', '')}"
            for i, doc in enumerate(reranked)
        ])

        user_message = f"参考案例：\n{context}\n\n用户诉求：{user_demands}\n\n案件基本信息：{basic_info}\n\n请给出专业法律分析意见："

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.lawyer_prompt),
            ("user", user_message)
        ])
        chain = prompt | self.reasoner_llm
        answer = chain.invoke({}).content

        return {
            "final_answer": answer,
            "messages": [AIMessage(content=f"[{LABEL}]：\n{answer}")]
        }
