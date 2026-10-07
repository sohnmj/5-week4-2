"""
RAG 과제: 웹 크롤링 -> Chroma 벡터DB -> RAG 기반 FAQ 생성 & 챗봇

rag/build_vector_db.ipynb + rag/rag_chatbot.ipynb 를 하나로 합치고,
과제 필수 요소인 "웹 크롤링"과 "FAQ 생성" 단계를 추가한 파일.

실행 방법 (sohnmj 폴더에서):
    pip install openai python-dotenv chromadb requests beautifulsoup4
    python rag_faq.py build      # 1) 크롤링 + 2) 청킹/임베딩/DB 저장
    python rag_faq.py faq        # 3) FAQ 자동 생성 -> faq.md
    python rag_faq.py chat       # 4) 터미널 챗봇
    python rag_faq.py gui        # 4) tkinter 챗봇
    python rag_faq.py all        # build + faq 한 번에
"""

import os  # 파일 시스템 접근, 환경 변수 읽기
import sys
import re

import requests  # 웹 페이지 요청
from bs4 import BeautifulSoup, NavigableString  # HTML 파싱
import chromadb  # 벡터 DB
from openai import OpenAI  # OpenAI API
from dotenv import load_dotenv  # .env 에서 API 키 로드

# ==========================================================
# 0. 설정
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 이 파일이 있는 폴더 기준으로 경로 설정
CRAWL_DIR = os.path.join(BASE_DIR, "crawled_data")  # 크롤링한 원천 데이터 저장 폴더
DB_PATH = os.path.join(BASE_DIR, "chroma_db")  # 벡터 DB 폴더 (rag/chroma_db 와 분리)
FAQ_PATH = os.path.join(BASE_DIR, "faq.md")  # 생성된 FAQ 저장 파일
COLLECTION_NAME = "rag_collection"

# 크롤링할 페이지 목록 (주제에 맞게 자유롭게 바꾸기)
URLS = [
    "https://pubg.com/ko/news/11057?category=patch_notes",  # PUBG 업데이트 43.1 패치노트
]

# 사이트별 본문 영역 CSS 선택자 (앞에서부터 찾아서 처음 걸리는 것 사용, 없으면 <body>)
CONTENT_SELECTORS = [
    "article.news-detail__content-template",  # pubg.com 뉴스/패치노트
    "#mw-content-text",  # 위키백과
]

EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o-mini"

# 환경 변수 Load 해서 api_key 가져오고 OpenAI 클라이언트 초기화
load_dotenv()  # 현재 폴더부터 상위로 .env 탐색
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ==========================================================
# 1. 웹 크롤링 (원천 데이터 확보)
# ==========================================================
BLOCK_TAGS = ["h1", "h2", "h3", "h4", "p", "li", "tr"]  # 한 줄로 뽑아낼 블록 태그
NESTED_TAGS = BLOCK_TAGS + ["ul", "ol", "table", "tbody"]  # 따로 처리되는 하위 블록


# 태그 자신의 텍스트만 추출 (하위 목록/문단은 따로 한 줄씩 뽑히므로 제외 -> 중복 방지)
def own_text(tag):
    parts = []
    for child in tag.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif child.name not in NESTED_TAGS:
            parts.append(child.get_text(" "))  # <strong>, <span>, <td> 같은 인라인 요소
    return " ".join(" ".join(parts).split())  # 공백/줄바꿈 정리


def crawl_url(url):
    # 일부 사이트는 User-Agent 없으면 차단하므로 브라우저처럼 헤더 지정
    headers = {"User-Agent": "Mozilla/5.0 (RAG-homework crawler)"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()  # 200이 아니면 예외 발생

    soup = BeautifulSoup(response.text, "html.parser")
    # 본문과 상관없는 태그 제거
    for tag in soup(["script", "style", "nav", "footer", "header", "sup"]):
        tag.decompose()

    # 사이트별 본문 영역 찾기
    main = None
    for selector in CONTENT_SELECTORS:
        main = soup.select_one(selector)
        if main:
            break

    lines = []
    for block in (main or soup.body).find_all(BLOCK_TAGS):
        line = own_text(block)
        if not line:
            continue
        if block.name.startswith("h"):
            line = f"## {line}"  # 소제목 표시 (청크에 섹션 맥락이 남도록)
        elif block.name == "li":
            line = f"- {line}"
        lines.append(line)

    if main is None:  # 본문 영역을 못 찾았으면 짧은 줄(메뉴 등) 제거
        lines = [l for l in lines if len(l) > 20]

    text = "\n".join(lines)
    text = re.sub(r"\[\d+\]", "", text)  # 각주 번호 [1] 제거
    return text


def crawl_and_save(urls, save_dir=CRAWL_DIR):
    os.makedirs(save_dir, exist_ok=True)
    for i, url in enumerate(urls, start=1):
        try:
            text = crawl_url(url)
        except Exception as e:
            print(f"[크롤링 실패] {url} -> {e}")
            continue
        file_path = os.path.join(save_dir, f"web_{i}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"출처: {url}\n\n{text}")  # 첫 줄에 출처 기록
        print(f"[크롤링 완료] {url} -> {file_path} ({len(text)}자)")


# ==========================================================
# 2. 벡터 DB 구축 (build_vector_db.ipynb)
# ==========================================================
# DB 초기화 함수 (매번 컬렉션을 지우고 새로 생성 -> id 중복 방지)
def init_db(db_path=DB_PATH):
    dbclient = chromadb.PersistentClient(path=db_path)
    try:
        dbclient.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = dbclient.create_collection(name=COLLECTION_NAME)
    return dbclient, collection


# 기존 DB 불러오기 (챗봇/FAQ 단계에서 사용)
def load_db(db_path=DB_PATH):
    dbclient = chromadb.PersistentClient(path=db_path)
    return dbclient.get_or_create_collection(COLLECTION_NAME)


# 텍스트 로딩 함수: 폴더 안의 .txt 파일들을 (파일명, 내용) 튜플 리스트로 반환
def load_text_files(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                docs.append((filename, f.read()))
    return docs


# 주어진 text를 임베딩 벡터로 변환하는 함수
def get_embedding(text, model=EMBED_MODEL):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


# 원천 데이터를 청크 단위로 나누고 overlap 적용하는 함수
def chunk_text(text, chunk_size=400, chunk_overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap  # overlap 적용
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return chunks


# 문서 로드 -> 청크 나누기 -> 임베딩 생성 후 DB 삽입
def build_vector_db(folder_path=CRAWL_DIR):
    _, collection = init_db()
    docs = load_text_files(folder_path)

    doc_id = 0
    for filename, text in docs:
        chunks = chunk_text(text, chunk_size=400, chunk_overlap=50)
        for idx, chunk in enumerate(chunks):
            doc_id += 1
            collection.add(
                documents=[chunk],  # 실제 청크 text
                embeddings=[get_embedding(chunk)],  # 임베딩 벡터
                metadatas=[{"filename": filename, "chunk_index": idx}],  # 출처 정보
                ids=[str(doc_id)],  # 청크마다 고유 id
            )
        print(f"[DB 저장] {filename}: {len(chunks)}개 청크")
    print(f"모든 문서 벡터DB에 저장 완료 (총 {doc_id}개 청크)")


# ==========================================================
# 3. 검색 + 답변 생성 (rag_chatbot.ipynb)
# ==========================================================
SYSTEM_PROMPT = """
당신은 주어진 문서 정보를 바탕으로 사용자 질문에 답변하는
지능형 어시스턴트입니다. 다음 원칙을 엄격히 지키세요:

1. 반드시 제공된 문서 내용에 근거해서만 답변을 작성하세요.
2. 문서에 언급되지 않은 내용이라면, 함부로 추측하거나 만들어내지 마세요.
   - 관련 내용이 없다면 "관련 문서를 찾지 못했습니다"라고 답변하세요.
3. 사실 관계를 명확히 기술하고, 불확실한 부분은 "정확한 정보를 찾지 못했습니다"라고 말하세요.
4. 지나치게 장황하지 않게, 간결하고 알기 쉽게 설명하세요.
5. 사용자가 한국어로 질문하면 한국어로, 다른 언어로 질문하면 해당 언어로 답변하세요.
6. 문서 출처나 연도가 중요하다면, 가능한 정확하게 전달하세요.
"""


# query를 임베딩해 chroma에서 가장 유사도가 높은 top-k개의 문서를 가져오는 함수
def retrieve(collection, query, top_k=3):
    query_embedding = get_embedding(query)
    return collection.query(query_embeddings=[query_embedding], n_results=top_k)


# 1) top_k개 문서 검색 2) context로 묶어 GPT에 전달 3) 최종 답변 반환
def generate_answer_with_context(collection, query, top_k=3):
    results = retrieve(collection, query, top_k)
    found_docs = results["documents"][0]
    found_metadatas = results["metadatas"][0]

    # 검색된 문서들을 하나의 문맥으로 결합
    context_texts = []
    for doc_text, meta in zip(found_docs, found_metadatas):
        context_texts.append(f"<<filename: {meta['filename']}>>\n{doc_text}")
    context_str = "\n\n".join(context_texts)

    user_prompt = f"아래는 검색된 문서들의 내용입니다:\n{context_str}\n\n질문: {query}"

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


# ==========================================================
# 4. FAQ 생성 (과제 핵심)
# ==========================================================
# 크롤링 문서를 GPT에게 보여주고 "자주 물어볼 만한 질문" 목록을 뽑는 함수
def generate_questions(folder_path=CRAWL_DIR, n_questions=5):
    docs = load_text_files(folder_path)
    # 토큰 절약을 위해 각 문서 앞부분만 사용
    sample = "\n\n".join(f"[{name}]\n{text[:3000]}" for name, text in docs)

    prompt = f"""다음 문서를 읽고, 이 주제에 대해 사람들이 자주 물어볼 만한 질문 {n_questions}개를 만들어주세요.
- 반드시 문서 안에서 답을 찾을 수 있는 질문만 만드세요.
- 한 줄에 질문 하나씩, 번호나 기호 없이 질문 문장만 출력하세요.

{sample}"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    lines = response.choices[0].message.content.strip().split("\n")
    # 혹시 붙은 번호/기호("1.", "-") 제거
    questions = [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", q).strip() for q in lines]
    return [q for q in questions if q]


# 질문마다 RAG로 답변을 생성해 FAQ 마크다운 파일로 저장
def generate_faq(n_questions=5):
    collection = load_db()
    questions = generate_questions(n_questions=n_questions)

    lines = ["# FAQ (웹 크롤링 데이터 기반 RAG 자동 생성)\n"]
    for i, q in enumerate(questions, start=1):
        answer = generate_answer_with_context(collection, q, top_k=3)
        print(f"Q{i}. {q}\nA. {answer}\n")
        lines.append(f"### Q{i}. {q}\n\n{answer}\n")

    with open(FAQ_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"FAQ 저장 완료 -> {FAQ_PATH}")


# ==========================================================
# 5. 챗봇 (터미널 / tkinter GUI)
# ==========================================================
def chat_cli():
    collection = load_db()
    while True:
        user_query = input("질문을 입력하세요(종료: quit): ")
        if user_query.lower() == "quit":
            break
        answer = generate_answer_with_context(collection, user_query, top_k=3)
        print("===답변===")
        print(answer)
        print("==========\n")


def chat_gui():
    from tkinter import Tk, Frame, Text, END
    import tkinter.ttk as ttk

    collection = load_db()

    def reset_status():
        label_status.config(text="", foreground="black")

    def process_query():
        query = text_input.get("1.0", END).strip()
        if not query:
            label_status.config(text="질문을 입력해주세요.", foreground="red")
            return
        label_status.config(text="질문 처리중...", foreground="blue")
        root.update()  # 상태 문구가 바로 보이도록 화면 갱신
        answer = generate_answer_with_context(collection, query)
        label_status.config(text="처리 완료", foreground="green")
        root.after(2000, reset_status)
        text_input.delete("1.0", "end-1c")  # 입력창 비우기
        text_output.config(state="normal")
        text_output.delete("1.0", END)
        text_output.insert(END, answer)
        text_output.config(state="disabled")  # 출력창 편집 불가

    root = Tk()
    root.title("RAG FAQ 챗봇")
    root.geometry("500x700")
    root.resizable(False, False)
    root.configure(bg="lavender")

    # ====== 입력 영역 ======
    frame_input = Frame(root, padx=10, pady=10)
    frame_input.pack(fill="x")
    ttk.Label(frame_input, text="질문 입력", font=("맑은 고딕", 12, "bold")).pack(anchor="w")
    text_input = Text(frame_input, height=6, font=("맑은 고딕", 11))
    text_input.pack(pady=5)
    ttk.Button(frame_input, text="전송", command=process_query).pack(pady=5)
    label_status = ttk.Label(frame_input, text="", font=("맑은 고딕", 10))
    label_status.pack(anchor="w", pady=5)

    ttk.Separator(root, orient="horizontal").pack(fill="x", padx=10, pady=10)

    # ====== 출력 영역 ======
    frame_output = Frame(root, padx=10, pady=10)
    frame_output.pack(fill="both", expand=True)
    ttk.Label(frame_output, text="답변", font=("맑은 고딕", 12, "bold")).pack(anchor="w")
    text_output = Text(frame_output, wrap="word", font=("맑은 고딕", 11),
                       state="disabled", height=20, bg="#f9f9f9")
    text_output.pack(side="left", fill="both", expand=True)
    scrollbar = ttk.Scrollbar(frame_output, command=text_output.yview)
    scrollbar.pack(side="right", fill="y")
    text_output.config(yscrollcommand=scrollbar.set)

    root.mainloop()


# ==========================================================
# 실행
# ==========================================================
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("build", "all"):
        crawl_and_save(URLS)  # 1) 웹 크롤링
        build_vector_db()  # 2) 청킹 + 임베딩 + Chroma 저장
    if mode in ("faq", "all"):
        generate_faq(n_questions=10)  # 3) FAQ 생성
    if mode == "chat":
        chat_cli()
    if mode == "gui":
        chat_gui()
