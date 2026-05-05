from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ParamOverride(BaseModel):
    dense_top_k: int | None = None
    sparse_top_k: int | None = None
    alpha: float | None = None


class LegalChatRequest(BaseModel):
    question: str
    session_id: str
    param_overrides: dict[str, ParamOverride] = {}


class LegalChatResponse(BaseModel):
    answer: str
    case_type: str
    session_id: str
    needs_followup: bool = False


_legal_sessions = {}


@router.post("/api/legal/chat", response_model=LegalChatResponse)
def legal_chat(req: LegalChatRequest):
    from agent.legal_graph import LegalGraph
    from api import _get_upload_llms

    session = _legal_sessions.get(req.session_id)

    if session and session.get("completed"):
        return LegalChatResponse(
            answer="本次咨询已结束，请刷新页面开始新的会话。",
            case_type=session.get("case_type", ""),
            session_id=req.session_id,
            needs_followup=False,
        )

    chat_llm, reasoner_llm = _get_upload_llms()
    legal_graph = LegalGraph(chat_llm, reasoner_llm)

    history_parts = []
    if session:
        for turn in session.get("history", []):
            history_parts.append(f"用户：{turn['user']}")
            history_parts.append(f"接待员：{turn['assistant']}")
    conversation_context = "\n".join(history_parts)

    overrides = {}
    for case_type, p in req.param_overrides.items():
        overrides[case_type] = {
            "dense_top_k": p.dense_top_k,
            "sparse_top_k": p.sparse_top_k,
            "alpha": p.alpha,
        }

    result = legal_graph.invoke(
        req.question,
        conversation_context=conversation_context,
        param_overrides=overrides,
    )

    if result["needs_followup"]:
        if not session:
            session = {"history": [], "completed": False, "case_type": ""}
        session["history"].append({
            "user": req.question,
            "assistant": result["followup_question"],
        })
        _legal_sessions[req.session_id] = session
        return LegalChatResponse(
            answer=result["followup_question"],
            case_type="",
            session_id=req.session_id,
            needs_followup=True,
        )

    if not session:
        session = {"history": [], "completed": False, "case_type": ""}
    session["completed"] = True
    session["case_type"] = result["case_type"]
    _legal_sessions[req.session_id] = session
    return LegalChatResponse(
        answer=result["answer"],
        case_type=result["case_type"],
        session_id=req.session_id,
        needs_followup=False,
    )
