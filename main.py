from vector import ingest_video_to_chroma
from langchain_ollama.llms import OllamaLLM
from app import create_fitness_graph

__import__('pysqlite3')
import sys,os
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

llm = OllamaLLM(model="llama3.2")
video_ids = ["jLvqKgW-_G8"
             , "fGm-ef-4PVk"
             , "hRZ5MM6gmlE"
             , "bvwg4D9UWGI"
             , "3ryh7PNhz3E"
             , "3Y2vxKsaK9A"
             , "928aRhhPP8I"
             , "DXL18E7QRbk"
             , "jf9PBwwNAMs"
             , "OpRMRhr0Ycc"
             , "SgyUoY0IZ7A"
             , "KV4D8MQrdhw"
             , "b6ouj88iBZs"
             , "GNO4OtYoCYk"
             , "kIXcoivzGf8"
             , "-A25orqRydQ"
             , "fGm-ef-4PVk"
             , "jf9PBwwNAMs"
             , "OpRMRhr0Ycc"
             , "SgyUoY0IZ7A"
             , "b6ouj88iBZs"
             , "GNO4OtYoCYk"
             , "kIXcoivzGf8"
             , "-A25orqRydQ"
             , "fGm-ef-4PVk"]
             

for vid in video_ids:
    ingest_video_to_chroma(vid, llm)


# Create the graph
agent = create_fitness_graph()

question = "What is the best Push Pull Leg exercises?"
response = agent.invoke({"question": question})

# Check the keys in the response and print accordingly
if "final_answer" in response:
    print("\n✅ FINAL ANSWER:")
    # print(response["final_answer"])
elif "documents" in response:
    print("\n📄 RETRIEVED DOCUMENTS:")
    for i, doc in enumerate(response["documents"], 1):
        print(f"{i}. {doc}")
else:
    print("⚠️ No answer or documents found.")

# agent = create_main_agent()
# print(agent.run("What’s a good workout plan for beginners?"))
