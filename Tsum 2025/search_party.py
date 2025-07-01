import openai
import numpy as np
import faiss
import pickle

openai.api_key = 

# 임베딩, 문서, 인덱스 로드
documents = None
with open("documents.pkl", "rb") as f:
    documents = pickle.load(f)
embeddings = np.load("embeddings.npy")
index = faiss.read_index("faiss.index")

# 쿼리 임베딩 생성 함수
def get_query_embedding(query):
    response = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)

def search_party(party_name, threshold=0.85):
    query_embedding = get_query_embedding(party_name)
    query_embedding = query_embedding.reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    D, I = index.search(query_embedding, len(documents))
    results = [(documents[i], float(score)) for i, score in zip(I[0], D[0]) if score >= threshold]
    return results

if __name__ == "__main__":
    party = input("정당명을 입력하세요: ")
    results = search_party(party, threshold=0.85)
    if results:
        print(f"유사도 0.85 이상 인원:")
        for name, score in results:
            print(f"{name} (유사도: {score:.2f})")
        # 가장 유사도가 높은 인물에 대해 GPT에게 설명 요청
        top_name, top_score = results[0]
        prompt = f"'{top_name}'에 대해 간단히 설명해줘. 이 인물과 '{party}'의 관계도 알려줘."
        chat_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 한국 정치 전문가입니다."},
                {"role": "user", "content": prompt}
            ]
        )
        print("\nGPT 설명:")
        print(chat_response["choices"][0]["message"]["content"].strip())
    else:
        print("해당 정당에 대해 유사도 0.85 이상인 인원이 없습니다.") 
