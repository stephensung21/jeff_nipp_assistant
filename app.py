from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from typing import TypedDict, Annotated, List, Union

# Embeddings and LLM
persist_directory = "chroma_fitness_db"
embedding_model = OllamaEmbeddings(model="mxbai-embed-large:latest")
llm = OllamaLLM(model="llama3.2")

# Vector DB
vectordb = Chroma(
    collection_name="fitness_docs",
    embedding_function=embedding_model,
    persist_directory=persist_directory
)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# State type for LangGraph
class AgentState(TypedDict):
    question: str
    final_answer: str
    documents: Annotated[List[str], "docs"]

# Define tool functions
def diet_tool(q):
    return llm.invoke(f"You are a nutrition coach. Answer this clearly:\n{q}")

def exercise_tool(q):
    return llm.invoke(f"""You are a fitness and exercise coach. You understand and know different exercises, exercise types, fitness splits and routines, machines, and the optimal way to work out.
                       Help the user with:\n{q}""")

def retrieve_tool(q):
    docs = retriever.get_relevant_documents(q)
    print(f"Retrieved {len(docs)} docs for: {q}")
    return [doc.page_content for doc in docs]

# Router logic (can be replaced with an LLM-based one)
def route_tool(state: AgentState) -> dict:
    q = state["question"]
    decision = llm.invoke(
        f"""You are a router for a fitness assistant. Given the user's question below, choose the best expert to answer it.

User question: "{q}"

Options:
- DietExpert: for nutrition, food, diet, supplements
- ExerciseExpert: for workouts, training, physical routines
- TranscriptRetriever: for general fitness info, YouTube transcripts, or anything else

Respond with ONLY one of the following EXACTLY: DietExpert, ExerciseExpert, or TranscriptRetriever.
Do not include any explanation or punctuation.
""").strip()

    # Clean and validate response
    decision = decision.split()[0].strip().replace('"', '')
    valid_tools = {"DietExpert", "ExerciseExpert", "TranscriptRetriever"}
    if decision not in valid_tools:
        decision = "TranscriptRetriever"  # fallback

    return {"next": decision}


# Create the graph
def create_fitness_graph():
    graph = StateGraph(AgentState)

    tools = {
        "TranscriptRetriever": RunnableLambda(lambda s: {
            "final_answer": "See retrieved documents.",
            "documents": s["documents"]
        }),
        "DietExpert": RunnableLambda(lambda s: {
            "final_answer": diet_tool(s["question"]),
            "documents": s["documents"]
        }),
        "ExerciseExpert": RunnableLambda(lambda s: {
            "final_answer": exercise_tool(s["question"]),
            "documents": s["documents"]
        }),
    }

    graph.add_node("retriever", RunnableLambda(lambda s: {
        "question": s["question"],
        "documents": retrieve_tool(s["question"])
        }))
    graph.add_edge("retriever", "router")
    graph.set_entry_point("retriever")

    graph.add_node("router", RunnableLambda(route_tool))
    for name, tool_node in tools.items():
        graph.add_node(name, tool_node)
        graph.add_edge(name, END)  # Still connect all tools to END

    # 🟡 Use conditional edges based on what the router returns
    graph.add_conditional_edges(
        "router",
        lambda state: state["next"],
        {
            "DietExpert": "DietExpert",
            "ExerciseExpert": "ExerciseExpert",
            "TranscriptRetriever": "TranscriptRetriever"
        }
    )

    graph.set_entry_point("router")

    return graph.compile()