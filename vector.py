import os
import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

__import__('pysqlite3')
import sys,os
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# === CONFIG ===
embedding_model = OllamaEmbeddings(model="mxbai-embed-large:latest")
persist_directory = "chroma_fitness_db"
collection_name = "fitness_docs"
# TRANSCRIPT_API_KEY = os.getenv("TRANSCRIPT_API_KEY")  # your Basic token
TRANSCRIPT_API_KEY = '684d3ab82514cbe8838df997'

# api_token = '684d3ab82514cbe8838df997'


# === HELPERS ===
def get_video_metadata(data):

    video_data = data[0]

    # Extract title
    title = video_data.get("title", "No Title Found")
    desc = video_data.get("microformat", [])

    for i in desc['playerMicroformatRenderer']:
        if i == 'description':
            description = desc['playerMicroformatRenderer'][i]
        
        if i == 'ownerChannelName':
            channel = desc['playerMicroformatRenderer'][i]

    return {
        "title": title,
        "channel": channel,
        "description": description
    }


def get_transcript_from_api(video_id):
    url = "https://www.youtube-transcript.io/api/transcripts"
    headers = {
        "Authorization": f"Basic {TRANSCRIPT_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json={"ids": [video_id]})
    response.raise_for_status()
    data = response.json()

    if not data:
        print(f"No data found for video ID: {video_id}")
        return None, None
    
    metadata = get_video_metadata(data)

    tracks = data[0].get('tracks', [])
    english_chunks = tracks[0]['transcript']
    english_text = " ".join(chunk['text'].replace('\n', ' ') for chunk in english_chunks)

    return english_text, metadata

    # return None, None


def video_already_ingested(video_id):
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory
    )
    existing_docs = vectordb.get(include=["metadatas"])
    for meta in existing_docs["metadatas"]:
        if meta.get("source") == video_id:
            return True
    return False


def ingest_video_to_chroma(video_id, llm):
    if video_already_ingested(video_id):
        print(f"⚠️  Video {video_id} already ingested. Skipping.")
        return

    transcript, metadata = get_transcript_from_api(video_id)
    if not transcript:
        print(f"❌ No transcript available for video {video_id}")
        return
    
    # Get summary
    prompt = f"""
You are a metadata assistant. Summarize the main topic of this fitness video titled '{metadata['title']}' from '{metadata['channel']}'. 
Transcript:\n{transcript[:2000]}
"""
    summary = llm.invoke(prompt)

    # Chunk and embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.create_documents(
        [transcript],
        metadatas=[{
            "source": video_id,
            "title": metadata["title"],
            "channel": metadata["channel"],
        }]
    )

    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        # persist_directory=persist_directory # comment out to not use sqlite3?
    )
    vectordb.add_documents(chunks)
    print(f"✅ Ingested: {metadata['title']}")





