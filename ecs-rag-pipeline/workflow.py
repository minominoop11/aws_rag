import operator
import chat
import json
import traceback
import logging

from pathlib import Path
from typing import Any, Dict, List, Tuple
from typing_extensions import Annotated, TypedDict
from typing import List, Tuple 
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_NAME = "claude_3_5_sonnet"
RECURSION_LIMIT = 50
TOP_K = 4           

class State(TypedDict, total=False):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    info: Annotated[List[Tuple], operator.add]
    reference_docs: List[Tuple[Any, float]] 
    answer: str

def _build_prompt(system: str, human: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", system), ("human", human)])

def query_planner(state: State) -> Dict[str, Any]:
    logging.info("###### query plan ######\ninput: %s", state["input"])

    system_msg = (
        "You are a retrieval query planner for an industrial safety RAG system "
        "backed by a vector store. "
        "Given an intrusion/safety event JSON, generate *three* high-recall "
        "natural-language questions that retrieve the most relevant "
        "work-instruction documents."
    )
    human_msg = "Event JSON:\n{event_json}"

    planner_prompt = _build_prompt(system_msg, human_msg)
    llm = chat.get_chat(model=MODEL_NAME)
    response = (planner_prompt | llm).invoke({"event_json": state["input"]})
    raw_text: str = response.content
    logging.info("LLM raw response: %s", raw_text)

    queries = [
        line.strip()
        for line in raw_text.splitlines()
        if line.strip()
        and not line.lower().startswith(("event json", "natural-language questions"))
    ]
    logging.info("parsed queries: %s", queries)

    return {"input": state["input"], "plan": queries}

def retriever(state: State) -> Dict[str, Any]:
    logging.info("###### retriever ######\nplan: %s", state["plan"])

    vectorstore = chat.build_or_load_chroma()
    retrieved: List[Tuple[Any, float]] = sum(
        (vectorstore.similarity_search_with_score(q, k=TOP_K) for q in state["plan"]),
        start=[],
    )

    logging.info("raw docs: %d", len(retrieved))
    filtered = chat.check_duplication(retrieved)
    logging.info("dedup docs: %d", len(filtered))

    return {
        "input": state["input"],
        "plan": state["plan"],
        "past_steps": [state["plan"]],
        "reference_docs": filtered,
    }

def generate_answer(state: State) -> Dict[str, Any]:
    logging.info("#### generating answer ####")

    context = "\n\n".join(doc.page_content for doc, _ in state.get("reference_docs", []))
    query = state["input"]

    system_msg = (
        "Here is pieces of context, contained in <context> tags. "
        "Provide a concise answer to the question at the end. "
        "Explain clearly the reasoning. "
        "If you don't know the answer, just say so."
    )
    human_msg = "Reference texts:\n{context}\n\nQuestion: {input}"
    prompt = _build_prompt(system_msg, human_msg)

    llm = chat.get_chat(model=MODEL_NAME)
    try:
        response = (prompt | llm).invoke({"context": context, "input": query})
        answer = response.content
        logging.info("LLM answer: %s", answer)
    except Exception:  # 세부 Exception 타입이 있다면 교체
        logging.error("LLM invocation failed:\n%s", traceback.format_exc())
        answer = "Sorry, an internal error occurred while generating the answer."

    return {"answer": answer}

def run_workflow(query: Dict[str, Any]) -> str:
    wf = StateGraph(State)
    wf.add_node("planner", query_planner)
    wf.add_node("retriever", retriever)
    wf.add_node("generate", generate_answer)

    wf.set_entry_point("planner")
    wf.add_edge("planner", "retriever")
    wf.add_edge("retriever", "generate")
    wf.add_edge("generate", END)
    app = wf.compile()

    inputs = {"input": query}
    config = {"recursion_limit": RECURSION_LIMIT}

    last_value: Dict[str, Any] = {}
    for output in app.stream(inputs, config):
        for key, value in output.items():
            logging.debug("Finished: %s", key)
            last_value = value  # END 단계에서 answer 포함

    return last_value.get("answer", "No answer produced.")

def run_rag_pipeline(event: Dict[str, Any]) -> None:
    rag_result = run_workflow(event)
    logging.info("final:", rag_result)

    anomaly_file = Path("data_source/dynamodb_anomaly_data/dummy_safety_events_2025.json")
    if not anomaly_file.exists():
        raise FileNotFoundError(f"{anomaly_file} not found.")

    with anomaly_file.open(encoding="utf-8") as fp:
        records = json.load(fp)

    for record in records:
        if record.get("eventId") == event["eventId"]:
            record["ragAdvisor"] = rag_result
            break
    else:  # runs only if the for‑loop did NOT break
        raise KeyError(f"eventId {event['eventId']} not found. No changes made.")

    with anomaly_file.open("w", encoding="utf-8") as fp:
        json.dump(records, fp, indent=2, ensure_ascii=False)

    logging.info(f"ragAdvisor added to eventId {event['eventId']} and saved to '{anomaly_file}'")
    