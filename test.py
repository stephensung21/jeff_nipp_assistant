import requests
import pandas as pd

api_token = '684d3ab82514cbe8838df997'

response = requests.post(
    "https://www.youtube-transcript.io/api/transcripts",
    headers={
        "Authorization": f"Basic {api_token}",
        "Content-Type": "application/json"
    },
    json={"ids": ["jLvqKgW-_G8"]}  # Example YouTube video ID
)

# # Output response
# print("Status Code:", response.status_code)
# try:
#     print("JSON Response:", response.json())
# except Exception:
#     print("Raw Response:", response.text)


data = response.json()

video_data = data[0]

# Extract title
title = video_data.get("title", "No Title Found")

# Extract Script
tracks = video_data.get('tracks', [])

# Extract English transcript text chunks and join them
# english_text = ""
# for track in tracks:
#     if track.get('language') == 'English':
#         english_chunks = track.get('transcript', [])
#         english_text = " ".join(chunk['text'].replace('\n', ' ') for chunk in english_chunks)
#         # print(english_chunks)


# track = tracks[0]['transcript']
english_chunks = tracks[0]['transcript']
english_text = " ".join(chunk['text'].replace('\n', ' ') for chunk in english_chunks)

print(english_text)



# print(tracks[0]['transcript'])



# desc = video_data.get("microformat", [])

# for i in desc['playerMicroformatRenderer']:

#     if i == 'ownerChannelName':
#         print(desc['playerMicroformatRenderer'][i])

# print(desc['playerMicroformatRenderer'])

#keywords

# playerMicroformatRenderer