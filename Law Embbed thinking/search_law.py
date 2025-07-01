import openai
import numpy as np
import faiss
import pickle

openai.api_key = ""

with open("law_texts.pkl", "rb") as f:
    law_texts = pickle.load(f)
embeddings = np.load("law_embeddings.npy")
index = faiss.read_index("law_faiss.index")

def get_query_embedding(query):
    response = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)

def search_law_for_keywords(keywords, threshold=0.7, topk=2):
    results_per_keyword = []
    for kw in keywords:
        query_embedding = get_query_embedding(kw).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        D, I = index.search(query_embedding, len(law_texts))
        # threshold 이상만, 없으면 topk개
        filtered = [(law_texts[i], float(score)) for i, score in zip(I[0], D[0]) if score >= threshold]
        if not filtered:
            D, I = index.search(query_embedding, topk)
            filtered = [(law_texts[i], float(score)) for i, score in zip(I[0], D[0])]
        results_per_keyword.append((kw, filtered[:topk]))
    return results_per_keyword

def is_query_complete(context):
    prompt = (
        "아래 사건 설명이 법률적 판단을 내리기에 충분히 구체적인가? "
        "부족하다면 어떤 추가 정보가 필요한지 한 문장으로 질문을 생성해줘. "
        "충분하다면 '충분'이라고만 답해."
        f"\n\n사건 설명: {context}"
    )
    chat_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 법률 상담 전문가입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    return chat_response["choices"][0]["message"]["content"].strip()

def extract_keywords(context):
    prompt = (
        "아래 사건 설명과 답변들을 바탕으로, 법률 검색에 적합한 핵심 키워드 또는 한두 문장(최대 5개)을 줄바꿈으로 구분하여 출력해줘."
        f"\n\n사건 및 답변: {context}"
    )
    chat_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 법률 검색 전문가입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    # 줄바꿈 기준 분리
    return [kw.strip() for kw in chat_response["choices"][0]["message"]["content"].strip().split('\n') if kw.strip()]

def make_judgement(context, law_results):
    # law_results: [(키워드, [(법률, 유사도), ...]), ...]
    law_summary = ""
    for kw, laws in law_results:
        law_summary += f"\n키워드: {kw}\n"
        for law, score in laws:
            law_summary += f"- {law} (유사도: {score:.2f})\n"
    prompt = (
        f"사건 설명: {context}\n"
        f"키워드별 관련 법률:\n{law_summary}\n"
        "위 정보를 바탕으로 법률적 판단(또는 설명)을 내려줘."
    )
    chat_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 법률 판결 전문가입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    return chat_response["choices"][0]["message"]["content"].strip()

if __name__ == "__main__":
    # 1. 사건 입력
    context = input("사건을 입력하세요: ")
    answers = []
    # 2~3. 정보 보충 루프
    while True:
        followup = is_query_complete(context + '\n' + '\n'.join(answers))
        if followup.strip() == "충분":
            break
        print("추가 질문:", followup)
        answer = input("답변: ")
        answers.append(f"Q: {followup}\nA: {answer}")
    # 4. 키워드/핵심문장 추출
    full_context = context + '\n' + '\n'.join(answers)
    keywords = extract_keywords(full_context)
    print(f"\n[추출된 키워드/문장]")
    for kw in keywords:
        print("-", kw)
    # 5. 각 키워드별 법률 검색
    law_results = search_law_for_keywords(keywords, threshold=0.7, topk=2)
    print("\n[키워드별 검색된 법률]")
    for kw, laws in law_results:
        print(f"키워드: {kw}")
        for law, score in laws:
            print(f"  - {law} (유사도: {score:.2f})")
    # 6. 판결/설명
    print("\n[최종 판결/설명]")
    print(make_judgement(full_context, law_results)) 