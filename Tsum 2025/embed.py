# requirements:
#   pip install openai faiss-cpu numpy matplotlib scikit-learn

import openai
import numpy as np
import faiss
import pickle

openai.api_key = "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"  # 여기에 본인의 OpenAI API 키를 입력하세요

documents = [
    "이재명 더불어민주당 대표",
    "윤석열 대통령 국민의힘",
    "한동훈 전 법무부 장관 국민의힘",
    "김기현 국민의힘 대표",
    "이준석 전 국민의힘 대표",
    "홍준표 대구시장 국민의힘",
    "조국 전 법무부 장관 더불어민주당",
    "박용진 더불어민주당 의원",
    "추미애 전 법무부 장관 더불어민주당",
    "심상정 정의당 의원",
    "안철수 국민의힘 의원 (전 국민의당 대표)",
    "유승민 전 국민의힘 의원",
    "오세훈 서울시장 국민의힘",
    "김동연 경기도지사 더불어민주당",
    "박영선 전 중소벤처기업부 장관 더불어민주당"
]

def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)

embeddings = np.array([get_embedding(doc) for doc in documents], dtype=np.float32)

faiss.normalize_L2(embeddings)
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings)

with open("embeddings.npy", "wb") as f:
    np.save(f, embeddings)
with open("documents.pkl", "wb") as f:
    pickle.dump(documents, f)
faiss.write_index(index, "faiss.index")
print("Saved")
