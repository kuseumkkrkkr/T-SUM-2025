import openai
import numpy as np
import faiss
import pickle
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import dash
from dash import dcc, html, Input, Output, State

openai.api_key = ""

with open("law_texts.pkl", "rb") as f:
    law_texts = pickle.load(f)
embeddings = np.load("law_embeddings.npy")
index = faiss.read_index("law_faiss.index")

# 3D 투영
pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings)

def get_query_embedding(query):
    response = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("법률 키워드 3D 임베딩 시각화 및 유사도 비교"),
    dcc.Input(id="keyword", type="text", placeholder="법률 키워드 또는 문장 입력", style={"width": "60%"}),
    html.Button("검색", id="search-btn"),
    dcc.Graph(id="scatter-plot"),
    html.Div(id="top-laws")
])

@app.callback(
    Output("scatter-plot", "figure"),
    Output("top-laws", "children"),
    Input("search-btn", "n_clicks"),
    State("keyword", "value")
)
def update_plot(n_clicks, keyword):
    if not keyword:
        fig = go.Figure(data=[
            go.Scatter3d(
                x=embeddings_3d[:, 0], y=embeddings_3d[:, 1], z=embeddings_3d[:, 2],
                mode="markers",
                marker=dict(size=4, color="blue"),
                text=law_texts,
                name="법률 조문"
            )
        ])
        fig.update_layout(title="법률 조문 3D 임베딩")
        return fig, "키워드를 입력하세요."

    # 쿼리 임베딩 및 3D 투영
    query_emb = get_query_embedding(keyword)
    query_emb_3d = pca.transform(query_emb.reshape(1, -1))

    # 유사도 계산
    query_emb_norm = query_emb.reshape(1, -1)
    faiss.normalize_L2(query_emb_norm)
    D, I = index.search(query_emb_norm, 5)
    top_indices = I[0]
    top_scores = D[0]
    top_texts = [law_texts[i] for i in top_indices]

    # 전체 + 쿼리 + Top N 표시
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=embeddings_3d[:, 0], y=embeddings_3d[:, 1], z=embeddings_3d[:, 2],
        mode="markers",
        marker=dict(size=4, color="blue"),
        text=law_texts,
        name="법률 조문"
    ))
    fig.add_trace(go.Scatter3d(
        x=query_emb_3d[:, 0], y=query_emb_3d[:, 1], z=query_emb_3d[:, 2],
        mode="markers+text",
        marker=dict(size=8, color="red"),
        text=["입력 키워드"],
        name="입력 키워드"
    ))
    fig.add_trace(go.Scatter3d(
        x=embeddings_3d[top_indices, 0], y=embeddings_3d[top_indices, 1], z=embeddings_3d[top_indices, 2],
        mode="markers+text",
        marker=dict(size=7, color="orange"),
        text=[f"Top{idx+1}" for idx in range(len(top_indices))],
        name="유사도 Top5"
    ))
    fig.update_layout(title="법률 조문 3D 임베딩 + 입력 키워드", legend=dict(x=0, y=1))

    # Top N 결과 텍스트
    top_law_info = "<br>".join([f"{idx+1}. {law} (유사도: {score:.2f})" for idx, (law, score) in enumerate(zip(top_texts, top_scores))])
    return fig, f"<b>유사한 법률 조문 Top 5:</b><br>{top_law_info}"

if __name__ == "__main__":
    app.run(debug=True) 