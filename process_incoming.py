import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json = {
        "model": "bge-m3",
        "input": text_list
    })
    
    embedding = r.json()['embeddings']
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json = {
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })
    response = r.json()
    # print(response)
    return response

df = joblib.load('embeddings.joblib')

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]
# print(question_embedding)

#Find similarities of question_embedding with other embeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding'].shape))
# cosine similarity => 1 - exactly similar, 0 - not similar at all, 
# argsort gives the indices of the most similar vector in ascending order (most similar index in the end of the list)
# [::-1] reverses the array
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
top_results = 5
max_indx = similarities.argsort()[::-1][0: top_results]
# print(max_indx)
new_df = df.loc[max_indx]
# print(new_df[["title", "number", "text"]])
#  ''' in python this is used as multiline string
#       like this '''
# orient=records gives list of dictionaries
prompt = f'''I am teaching microprocessor 8086 in my course. Here are video 
subtitle chunks containing video title, video number, start time and end time in seconds,
, the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
------------------------------------------------
{incoming_query}
User asked this question related to the video chunks, you have to answer in a human way(dont mention the above format, its just for you) where and how much
content is taught in which video (in which video and at what timestamp) and and guide the user to go 
to that particular video. If user asked unrelated question, tell him 
that you can only answer questions related to the course
give the exact timestamp of the instance in min:sec format
'''

with open("prompt.txt", "w") as f:
    f.write(prompt)
    
response = inference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)
# for index, item in new_df.iterrows():
    # print(index, item["title"], item["number"], item["text"], item["start"], item["end"])