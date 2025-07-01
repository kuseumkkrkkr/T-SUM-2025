# requirements:
#   pip install numpy faiss-cpu matplotlib scikit-learn plotly dash pickle

import numpy as np
import faiss
import pickle
from sklearn.decomposition import PCA
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import os
import openai


embeddings = np.load("embeddings.npy")
with open("documents.pkl", "rb") as f:
    documents = pickle.load(f)

index = faiss.read_index("faiss.index")

# 3D 투영
pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings)

# OpenAI API 키 설정
openai.api_key = "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"  # 여기에 본인의 OpenAI API 키를 입력하세요

def get_query_embedding(query):
    try:
        response = openai.Embedding.create(
            input=query,
            model="text-embedding-ada-002"
        )
        return np.array(response['data'][0]['embedding'], dtype=np.float32)
    except Exception as e:
        print(f"임베딩 생성 오류: {e}")
        return None

# Dash 앱 초기화
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Embeddings 3D Visualization"),
    dcc.Input(id="query", type="text", placeholder="검색어 입력"),
    dcc.Input(id="threshold", type="number", value=0.7, min=0.0, max=1.0, step=0.05, placeholder="유사도 임계값"),
    html.Button("검색", id="search-btn"),
    dcc.Graph(id="scatter-plot"),
    html.Div(id="doc-info")
])

@app.callback(
    Output("scatter-plot", "figure"),
    Output("doc-info", "children"),
    Input("search-btn", "n_clicks"),
    State("query", "value"),
    State("threshold", "value")
)
def update_plot(n_clicks, query, threshold):
    if not query:
        # 기본 플롯
        fig = px.scatter_3d(
            x=embeddings_3d[:, 0],
            y=embeddings_3d[:, 1],
            z=embeddings_3d[:, 2],
            hover_name=documents
        )
        return fig, "단어를 클릭하면 정보가 표시됩니다."

    # 쿼리 임베딩 생성 (실제 API 사용)
    query_embedding = get_query_embedding(query)
    if query_embedding is None:
        fig = px.scatter_3d(
            x=embeddings_3d[:, 0],
            y=embeddings_3d[:, 1],
            z=embeddings_3d[:, 2],
            hover_name=documents
        )
        return fig, "임베딩 생성 오류: 쿼리 임베딩을 생성할 수 없습니다."
    # 1차원 → 2차원으로 변환
    query_embedding = query_embedding.reshape(1, -1)
    faiss.normalize_L2(query_embedding)

    # 검색
    k = len(documents)
    D, I = index.search(query_embedding, k)

    # 유사도 필터링
    filtered_indices = [i for i, score in zip(I[0], D[0]) if score >= threshold]
    filtered_docs = [documents[i] for i in filtered_indices]

    # 필터링된 점만 그리기
    fig = px.scatter_3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        hover_name=documents,
        opacity=0.2
    )

    if filtered_indices:
        fig.add_scatter3d(
            x=embeddings_3d[filtered_indices, 0],
            y=embeddings_3d[filtered_indices, 1],
            z=embeddings_3d[filtered_indices, 2],
            mode="markers",
            marker=dict(size=6, color="red"),
            name="검색 결과"
        )
        info = f"검색어: '{query}' → {len(filtered_indices)}개 문서가 유사도 {threshold} 이상입니다."
    else:
        info = f"검색어: '{query}'에 대해 유사도 {threshold} 이상인 문서가 없습니다."
    return fig, info

if __name__ == "__main__":
    app.run(debug=True)
