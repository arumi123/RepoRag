import os
import ast
from sentence_transformers import SentenceTransformer
import faiss
import openai
import numpy as np
import json

# 1. ソースコード解析（Python専用）
def parse_python_code(repo_path):
    """
    指定されたリポジトリパスからPythonコードを解析し、
    クラス、関数名、ドキュメント文字列、ソースコードスニペットを抽出します。
    """
    code_data = []  # 解析したデータを格納するリスト
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):  # Pythonファイルを対象にする
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    source_code = f.read()
                try:
                    # 抽象構文木（AST）で解析
                    tree = ast.parse(source_code)
                    for node in ast.walk(tree):  # AST内を探索
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):  # 関数またはクラスを検出
                            code_data.append({
                                "type": "class" if isinstance(node, ast.ClassDef) else "function",  # 種類
                                "name": node.name,  # 名前
                                "docstring": ast.get_docstring(node),  # ドキュメント文字列
                                "code": ast.get_source_segment(source_code, node)  # ソースコードスニペット
                            })
                except SyntaxError:
                    print(f"Skipping {file_path} due to parsing error.")  # パースエラーが発生した場合はスキップ
    return code_data

# 2. ベクトル化とデータベース格納
def create_vector_database(repo_data):
    """
    ソースコードのデータを埋め込みベクトルに変換し、
    FAISSベクトルデータベースに保存します。
    """
    # 埋め込みモデルの初期化
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 各コードスニペットをベクトル化
    print("Embedding data...")
    vectors = []
    metadata = []
    for item in repo_data:
        text = item["docstring"] or item["code"]  # ドキュメント文字列があれば優先
        vector = model.encode(text)  # ベクトル化
        vectors.append(vector)
        metadata.append(item)  # メタデータを保存

    # FAISSインデックスの初期化
    print("Creating FAISS index...")
    dimension = len(vectors[0])  # ベクトルの次元数
    index = faiss.IndexFlatL2(dimension)  # L2ノルムを使用したインデックス

    # ベクトルをFAISSインデックスに追加
    index.add(np.array(vectors))

    # FAISSインデックスとメタデータを保存
    faiss.write_index(index, "faiss_index.bin")  # インデックスの保存
    with open("metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)  # メタデータの保存

    print("FAISS database created and saved locally.")

# 3. クエリ検索
def query_code_database(query):
    """
    ユーザーのクエリをベクトルに変換し、
    FAISSで関連コードスニペットを検索します。
    """
    # 埋め込みモデルのロード
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = model.encode(query).reshape(1, -1)  # クエリをベクトル化

    # FAISSインデックスとメタデータのロード
    index = faiss.read_index("faiss_index.bin")
    with open("metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # FAISS検索
    print("Searching FAISS database...")
    distances, indices = index.search(query_vector, k=5)  # 上位5件を取得

    # 検索結果を整形して返す
    results = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx < len(metadata):  # インデックスが範囲内か確認
            result = metadata[idx]
            result["distance"] = distance
            results.append(result)

    return results

# 4. 自然言語生成
def generate_response_from_matches(query, matches):
    """
    検索結果を基に、GPTを使って自然言語による回答を生成します。
    """
    if not matches:
        return "関連するコードは見つかりませんでした。"

    # 検索結果を文脈として整形
    context = "\n\n".join([f"Name: {res['name']}\nCode:\n{res['code']}" for res in matches])

    # OpenAI APIを使用して応答を生成
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # または "gpt-4"
        messages=[
            {"role": "system", "content": "あなたはコード解析の専門家です。"},
            {"role": "user", "content": f"以下のコードスニペットに基づいて、質問に回答してください。\n\n{context}\n\n質問: {query}"}
        ],
        max_tokens=600
    )
    return response.choices[0].message.content


# メイン処理
if __name__ == "__main__":
    # ステップ 1: ソースコードを解析
    repo_path = "/home/user/homehub"  # 対象リポジトリのパス
    print("Parsing repository...")
    repo_data = parse_python_code(repo_path)

    # ステップ 2: データベースに格納
    print("Creating vector database...")
    create_vector_database(repo_data)

    # ステップ 3: ユーザークエリを受け取って検索
    user_query = input("質問を入力してください: ")
    print("Searching for related code...")
    search_results = query_code_database(user_query)

    # ステップ 4: 結果を元に自然言語応答を生成
    print("Generating response...")
    answer = generate_response_from_matches(user_query, search_results)
    print("\n--- AIの回答 ---")
    print(answer)
    print("\n--- AIの回答 ---")