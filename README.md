# RepoRag
## 機能
- 特定のリポジトリのソースコードの情報をチャット形式で得られる。

## 動作環境(検証済み環境)
- WSL上(winodws11)

## 環境構築手順
1. このリポジトリをclone
2. 仮想環境を構築
~~~bash
python3 -m venv venv
~~~
3. ライブラリのインストール
~~~bash
pip install -r requirements.txt
~~~
4. OpenAIのAPIキーを環境変数にエクスポート
~~~bash
export OPENAI_API_KEY=〇〇〇
~~~
5. 実行
~~~bash
python3 sourceanalyze.py
~~~
