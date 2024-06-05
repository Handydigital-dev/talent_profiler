import streamlit as st
import requests
import time
from fake_useragent import UserAgent
import logging
from bs4 import BeautifulSoup
import json
import os
from dotenv import load_dotenv
import pyperclip
import uuid
import pandas as pd
st.set_page_config(page_title="AICS提案文章作成")

# 環境変数を読み込む
load_dotenv()

# 定数の定義
API_ENDPOINTS = {
    "CLAUDE": "https://api.anthropic.com/v1/messages",
}
RETRY_ATTEMPTS = 5

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")

def get_webpage_content(url):
    """
    指定されたURLのWebページのコンテンツを取得し、HTMLタグを除去して返します。
    """
    try:
        ua = UserAgent()
        headers = {'User-Agent': ua.random}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return strip_html_tags(response.text)
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            st.warning(f"404 Not Found for url: {url}")
        elif e.response.status_code == 403:
            st.warning(f"403 Forbidden for url: {url}")
        else:
            st.error(f"Error fetching webpage: {e}")
        return ""
    except requests.RequestException as e:
        st.error(f"Error fetching webpage: {e}")
        return ""

def strip_html_tags(html):
    """
    HTMLコンテンツからHTMLタグを除去します。
    """
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

def summarize_content(content, max_length, model, api_key):
    """
    指定された文章を要約します。
    """
    url = API_ENDPOINTS["CLAUDE"]
    payload = {
        "model": model,
        "max_tokens": min(max_length, 4096),
        "temperature": 0.7,
        "messages": [
            {"role": "user", "content": f"以下の文章を{max_length}文字程度に要約してください。\n\n{content}"}
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
        "anthropic-version": "2023-06-01"
    }
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            if 'content' in response_data:
                return response_data['content'][0]['text']
            else:
                st.error(f"Unexpected response format: {response_data}")
                return ""
        except requests.HTTPError as e:
            if e.response.status_code == 429:
                retry_wait_time = 5 * (attempt + 1)  # リトライ回数に応じて待機時間を増やす
                st.warning(f"Rate limit exceeded. Retrying in {retry_wait_time} seconds... (Attempt {attempt + 1})")
                time.sleep(retry_wait_time)
            else:
                handle_api_error(e, response)
                return ""
        except requests.RequestException as e:
            st.error(f"Error with Claude API: {e}")
            return ""
    st.error("Max retry attempts reached. Please try again later.")
    return ""


def improve_introduction(introduction, talent_name, prompt, model, api_key):
    """
    指定された導入文を改善します。
    """
    prompt_text = f"{prompt.replace('{talentName}', talent_name)}\n\n{introduction}"
    url = API_ENDPOINTS["CLAUDE"]
    payload = {
        "model": model,
        "max_tokens": 4096,
        "temperature": 0.7,
        "messages": [
            {"role": "user", "content": prompt_text}
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
        "anthropic-version": "2023-06-01"
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        if 'content' in response_data:
            return response_data['content'][0]['text']
        else:
            st.error(f"Unexpected response format: {response_data}")
            return ""
    except requests.HTTPError as e:
        handle_api_error(e, response)
        return ""
    except requests.RequestException as e:
        st.error(f"Error with Claude API: {e}")
        return ""

def handle_api_error(e, response):
    """
    APIエラーをハンドリングします。
    """
    if e.response.status_code == 401:
        st.error("Unauthorized: Please check your API key.")
    elif e.response.status_code == 400:
        st.error("Bad Request: Please check your input data and try again.")
        st.error(f"Response content: {response.text}")
    else:
        st.error(f"Error with API: {e}")

def collect_talent_info(talent_name, prompt, max_results, api_key):
    """
    指定されたタレントの情報を収集します。
    """
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": prompt.replace("{talentName}", talent_name),
        "search_depth": "advanced",
        "include_answer": True,
        "include_images": False,
        "include_raw_content": False,
        "max_results": max_results,
        "include_domains": [],
        "exclude_domains": []
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        handle_api_error(e, response)
        return None
    except requests.RequestException as e:
        st.error(f"Error with Tavily API: {e}")
        return None

def process_talent(TAVILY_API_KEY, CLAUDE_API_KEY, talent_name, tavily_prompt, max_results, summarization_length, introduction_prompt, claude_model_summarization, claude_model_introduction):
    """
    指定されたタレントの情報を処理します。
    """
    summarized_articles = []
    st.write(f"### Processing {talent_name}...")
    talent_info = collect_talent_info(talent_name, tavily_prompt, max_results, TAVILY_API_KEY)
    if talent_info:
        answer = talent_info.get("answer", "")
        st.write(f"**simple answer:** {answer}")
        references = [{"title": result["title"], "url": result["url"], "summary": None, "status": "pending"} for result in talent_info["results"][:max_results]]
        
        # 各参照に番号を割り当てる
        for i, ref in enumerate(references):
            ref["index"] = i + 1
        
        num_articles = len(references)
        st.write(f"Summarizing {num_articles} articles...")

        for reference in references:
            with st.spinner(f'Summarizing: [{reference["index"]}] {reference["title"]}'):
                webpage_content = get_webpage_content(reference["url"])
                if webpage_content:
                    summarized_content = summarize_content(webpage_content, summarization_length, claude_model_summarization, CLAUDE_API_KEY)
                    if summarized_content:
                        summarized_articles.append(summarized_content)
                        reference["summary"] = summarized_content
                        reference["status"] = "completed"
                    else:
                        st.warning(f'Summary failed for reference: [{reference["index"]}] {reference["title"]}. Waiting for 10 seconds...')
                        reference["status"] = "failed"
                        time.sleep(10)  # 要約に失敗した場合、5秒間待機する

            st.write(f'{"✅" if reference["status"] == "completed" else "❌"} [{reference["index"]}] [{reference["title"]}]({reference["url"]})')

            with st.spinner("Processing... summary"):
                time.sleep(13)  # 13秒間処理を停止し、スピナーを表示

    if summarized_articles:
        with st.spinner("Generating Improved Introduction..."):
            overall_summary = summarize_content("\n\n".join(summarized_articles), 4096, claude_model_introduction, CLAUDE_API_KEY)
            improved_introduction = improve_introduction(overall_summary, talent_name, introduction_prompt, claude_model_introduction, CLAUDE_API_KEY)
        result = {
            "talent_name": talent_name,
            "answer": answer,
            "references": references,
            "improved_introduction": improved_introduction
        }
        display_result(result)
        
        return result  # resultを返すように修正
        
    return None

def display_result(result):
    """
    結果を表示します。
    """
    st.write(f"## {result['talent_name']}")
    st.write(f"**Improved Introduction:** {result['improved_introduction']}")
    st.write("**References:**")
    for ref in result['references']:
        if ref["summary"]:
            st.write(f"[{ref['index']}] [{ref['title']}]({ref['url']})")


# Main App
def main():
    session_id = str(uuid.uuid4())  # セッションIDを生成
    session_key = f"processing_{session_id}"

    if session_key not in st.session_state:
        st.session_state[session_key] = False

    if not st.session_state[session_key]:
        st.session_state[session_key] = True
        try:
            start_time = time.time()  # 処理開始時間を記録
            end_time = None  # end_time変数を初期化

            st.title("AICSタレント提案文作成")
            st.text("Settingsのタレント名に名前を入力して「簡易提案文作成」「提案文作成」ボタンを押す")
            st.text("「簡易提案文作成」： WEB上から記事を取得(20記事固定)→概要を作成→概要をCSVでダウンロード")
            st.markdown("<small>↑ 精度よりもとにかくざっとでも提案文が必要のとき、精度のバラツキが大きい</small>", unsafe_allow_html=True)
            st.text("「提案文作成」： WEB上から記事を取得→記事を要約→要約した内容で提案文を作成→提案文をCSVでダウンロード")
            st.text("Settingsの項目目安 5記事300文字要約くらいで一定の精度が出ると思いますが、\n微妙だなとなったら10記事400文字とかにしてみてもいいかもです。")
            st.text("5記事300文字要約の場合 処理時間:3~4分/人 費用:2~3円")
            st.markdown("<small>AIなので正しい情報とは限らないのと回答はバラツキます。あくまで取っ掛かりやヒントとして使ってください。</small>", unsafe_allow_html=True)
            st.sidebar.title("Settings")
            # セッションステートから設定項目の値を取得
            talent_names = st.session_state.get("talent_names", "")
            tavily_prompt = st.session_state.get("tavily_prompt", "{talentName}の出演に関する情報(2023年以降)")
            max_results = st.session_state.get("max_results", 5)
            claude_model_summarization = st.session_state.get("claude_model_summarization", "claude-3-haiku-20240307")
            summarization_length = st.session_state.get("summarization_length", 300)
            claude_model_introduction = st.session_state.get("claude_model_introduction", "claude-3-haiku-20240307")
            introduction_prompt = st.session_state.get("introduction_prompt", "以下の情報をもとに、{talentName}を広告起用する際の推しポイントを紹介する文章を作成してください。")

            talent_names = st.sidebar.text_area("タレント名\n[改行して複数人入力可]", value=talent_names, key="talent_names")
            tavily_prompt = st.sidebar.text_area("情報収集AIの指示", value=tavily_prompt, key="tavily_prompt")
            max_results = st.sidebar.selectbox("情報収集で取得する記事数", list(range(1, 21)), index=max_results-1, key="max_results")
            claude_model_summarization = st.sidebar.selectbox("記事要約のモデル(今はhaikuでよい)", ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"], index=["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"].index(claude_model_summarization), key="claude_model_summarization")
            summarization_length = st.sidebar.number_input("要約する際の文字数", min_value=100, max_value=1000, value=summarization_length, key="summarization_length")
            claude_model_introduction = st.sidebar.selectbox("提案文作成のモデル(今はhaikuでよい)", ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"], index=["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"].index(claude_model_introduction), key="claude_model_introduction")
            introduction_prompt = st.sidebar.text_area("提案文作成時のAIへの指示", value=introduction_prompt, key="introduction_prompt")

            if st.button("簡易提案文作成"):
                if not TAVILY_API_KEY:
                    st.error("Tavily API key is required.")
                elif not talent_names.strip():
                    st.error("Please enter at least one talent name.")
                elif not tavily_prompt.strip():
                    st.error("Tavily API prompt is required.")
                else:
                    talent_names_list = [name.strip() for name in talent_names.split('\n') if name.strip()]
                    total_talents = len(talent_names_list)
                    simple_results = []
                    for i, talent_name in enumerate(talent_names_list, start=1):
                        with st.spinner(f"Processing {talent_name} ({i}/{total_talents})..."):
                            simple_result = collect_talent_info(talent_name, tavily_prompt, 20, TAVILY_API_KEY)
                            if simple_result:
                                formatted_references = [f"[{j+1}] {result['title']}\n{result['url']}" for j, result in enumerate(simple_result['results'])]
                                simple_results.append([talent_name, simple_result.get("answer", ""), "\n".join(formatted_references)])

                    if simple_results:
                        simple_df = pd.DataFrame(simple_results, columns=['Talent Name', 'Simple Answer', 'References'])
                        st.write("## Simple Results")
                        st.dataframe(simple_df, width=800)

            if st.button("提案文作成"):
                if not TAVILY_API_KEY:
                    st.error("Tavily API key is required.")
                elif not talent_names.strip():
                    st.error("Please enter at least one talent name.")
                elif not tavily_prompt.strip():
                    st.error("Tavily API prompt is required.")
                elif not CLAUDE_API_KEY:
                    st.error("Claude API key is required.")
                else:
                    talent_names_list = [name.strip() for name in talent_names.split('\n') if name.strip()]
                    results = []  # resultsリストを再導入
                    for talent_name in talent_names_list:
                        result = process_talent(TAVILY_API_KEY, CLAUDE_API_KEY, talent_name, tavily_prompt, max_results, summarization_length, introduction_prompt, claude_model_summarization, claude_model_introduction)
                        if result:
                            results.append(result)

                    if results:
                        table_data = []
                        for result in results:
                            # referenceの形式を変更
                            formatted_references = [f"[{ref['index']}]{ref['title']}\n{ref['url']}" for ref in result['references']]
                            table_data.append([result['talent_name'], result['improved_introduction'], "\n".join(formatted_references), result['answer']])

                        df = pd.DataFrame(table_data, columns=['Talent Name', 'Improved Introduction','reference','simple answer'])

                        st.write("## Results")
                        st.dataframe(df, width=800)
            end_time = time.time()  # 処理終了時間を記録
            execution_time = end_time - start_time  # 処理時間を計算
            st.write(f"Execution time: {execution_time:.2f} seconds")  # 処理時間を表示
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            if end_time is None:
                end_time = time.time()  # エラーが発生した場合でもend_timeを設定
            st.session_state[session_key] = False
    else:
        st.warning("Another user is currently using the application. Please wait and try again later.")

if __name__ == "__main__":
    main()