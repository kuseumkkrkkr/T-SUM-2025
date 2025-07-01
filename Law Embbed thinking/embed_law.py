import openai
import numpy as np
import faiss
import pickle
import tkinter as tk
from tkinter import filedialog
import win32com.client

openai.api_key = ""
# 파일 다이얼로그로 doc 파일 선택
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="법률 데이터(doc) 파일을 선택하세요",
    filetypes=[("Word Documents", "*.doc")]
)
if not file_path:
    print("파일을 선택하지 않았습니다.")
    exit()

# .doc 파일에서 텍스트 추출
def extract_law_texts_from_doc(path):
    word = win32com.client.Dispatch("Word.Application")
    doc = word.Documents.Open(path)
    text = doc.Content.Text
    doc.Close()
    word.Quit()
    paragraphs = [p.strip() for p in text.split('\r') if p.strip()]
    return paragraphs

law_texts = extract_law_texts_from_doc(file_path)
print(f"{len(law_texts)}개의 조문/문단을 추출했습니다.")

def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-3-small"
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)

embeddings = np.array([get_embedding(law) for law in law_texts], dtype=np.float32)
faiss.normalize_L2(embeddings)
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings)

with open("law_embeddings.npy", "wb") as f:
    np.save(f, embeddings)
with open("law_texts.pkl", "wb") as f:
    pickle.dump(law_texts, f)
faiss.write_index(index, "law_faiss.index")
print("Saved") 