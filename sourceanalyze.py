import os
import ast
from sentence_transformers import SentenceTransformer
import pinecone
import openai

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
    Pineconeベクトルデータベースに保存します。
    """
    # 埋め込みモデルの初期化
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Pineconeの初期化
    pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")
    index = pinecone.Index("code-search")  # データベース名（事前に作成する必要があります）

    # 各コードスニペットをベクトル化して保存
    for item in repo_data:
        vector = model.encode(item["docstring"] or item["code"])  # ドキュメント文字列があれば優先
        index.upsert([(item["name"], vector, {"type": item["type"], "code": item["code"]})])

# 3. クエリ検索
def query_code_database(query):
    """
    ユーザーのクエリをベクトルに変換し、
    Pineconeで関連コードスニペットを検索します。
    """
    # モデルのロード
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = model.encode(query)  # クエリをベクトル化

    # Pineconeで検索
    index = pinecone.Index("code-search")
    results = index.query(query_vector, top_k=5, include_metadata=True)  # 上位5件を取得

    # 検索結果を返す
    return results["matches"]

# 4. 自然言語生成
def generate_response_from_matches(query, matches):
    """
    検索結果を基に、GPTを使って自然言語による回答を生成します。
    """
    # Pineconeから取得した結果を文脈として整形
    context = "\n\n".join([f"Name: {res['id']}\nCode: {res['metadata']['code']}" for res in matches])
    
    # OpenAI APIを使用して応答を生成
    openai.api_key = "YOUR_API_KEY"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"以下のコードスニペットに基づいて、質問に回答してください。\n\n{context}\n\n質問: {query}",
        max_tokens=300
    )
    return response["choices"][0]["text"].strip()

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
    user_query = input("質問を入力してください: ")  # 例: "このクラスの役割は？"
    print("Searching for related code...")
    search_results = query_code_database(user_query)

    # ステップ 4: 結果を元に自然言語応答を生成
    print("Generating response...")
    answer = generate_response_from_matches(user_query, search_results)
    print("\n--- AIの回答 ---")
    print(answer)
