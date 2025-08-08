import os
import re
# import pinecone ##--For the production environment--##
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import time
import requests
import threading
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
# from langchain.vectorstores import Chroma, Pinecone as PineconeStore ##--Production environment--##

# Config
# RUN_ENV = os.getenv("RUN_ENV", "testing")
RUN_ENV = "testing"
embedding = OpenAIEmbeddings()
chunk_size = 500
chunk_overlap = 100
max_chars = 100_000
max_depth = 6
rate_limit = 1  # seconds between requests
seen_paginated_ids = set()

visited = set()
documents = []

# Set USER_AGENT
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LangChainCrawler/1.0"

def fetch_html(url):
    try:
        time.sleep(rate_limit)
        headers = {"User-Agent": os.environ["USER_AGENT"]}
        # headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        return BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print(f" Failed to fetch {url}: {e}")
        return None

def extract_internal_links(soup, base_url, domain):
    links = set()
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        full_url = urljoin(base_url, href)
        if domain in urlparse(full_url).netloc:
            links.add(full_url)
    return links

##--This can be used to normalize the urls to make encourage uniqueness in the queue
# def normalize_url(url):
#     parsed = urlparse(url)
#     return parsed._replace(query="", fragment="").geturl().rstrip("/")

##--This and the one above are helper functions
def is_html_page(url):
    return not re.search(r"\.(jpg|jpeg|png|gif|tif|bmp|zip|docx?)$", url, re.IGNORECASE)

##--This deals with pdf parsing
def extract_text_from_pdf(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open("temp.pdf", "wb") as f:
            f.write(response.content)

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        return "\n".join(doc.page_content for doc in documents)

    except Exception as e:
        print(f"⚠️ Failed to extract PDF: {url} — {e}")
        return ""

##--This to find the url that is paginated--##
def is_paginated_url(url):
    return re.search(r"[&?]Page=\d+", url, re.IGNORECASE)

##--This handles the pagination--##
def handle_pagination(base_url, max_page=5):
    links = []
    for page in range(1, max_page + 1):
        url = base_url if page == 1 else f"{base_url}&page={page}"
        soup = fetch_html(url)
        if not soup:
            continue
        page_links = extract_internal_links(soup, url, urlparse(base_url).netloc)
        links.extend(page_links)

        text = soup.get_text(separator=" ", strip=True)
        if len(text.strip()) > 100:
            doc = Document(page_content=text, metadata={"source": url, "source_type": "paginated"})
            documents.append(doc)

    return links

##--Extract the IDs to check if it has already been paginated--##
def extract_si_id(url):
    query = urlparse(url).query
    params = parse_qs(query)
    return params.get("SI_Id", [None])[0]


def crawl_recursive(url, domain, depth):
    if url in visited or depth > max_depth:
        return
    visited.add(url)

    soup = fetch_html(url)
    if not soup:
        return

    print(f"[{depth}] Crawled: {url}")
    text = soup.get_text(separator=" ", strip=True)
    if len(text) < max_chars:
        doc = Document(page_content=text, metadata={"source": url})
        documents.append(doc)

    sub_links = extract_internal_links(soup, url, domain)
    for link in sub_links:
        crawl_recursive(link, domain, depth + 1)

def init_vectorstore(chunks):
    if RUN_ENV == "testing":
        persist_dir = "chroma_test_db2"
        if os.path.exists(persist_dir):
            print(" Loading Chroma from disk...")
            return Chroma(persist_directory=persist_dir, embedding_function=embedding)
        else:
            print(" Creating new Chroma index...")
            return Chroma.from_documents(chunks, embedding, persist_directory=persist_dir)
    # else: ##--For Production Environment--##
    #     pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
    #     index_name = "your-production-index"
    #     if index_name not in pinecone.list_indexes():
    #         pinecone.create_index(index_name, dimension=1536)
    #     print(" Sending to Pinecone...")
    #     return PineconeStore.from_documents(chunks, embedding, index_name=index_name)


# def crawl_site(start_url, force_refresh=False):
#     persist_dir = "chroma_test_db2"
#     domain = urlparse(start_url).netloc

#     if os.path.exists(persist_dir) and not force_refresh:
#         print(" Loading existing Chroma vector store")
#         return Chroma(persist_directory=persist_dir, embedding_function=embedding)
    
#     else:
#         print(" Crawling site and rebuilding vector store")
#         crawl_recursive(start_url, domain, depth=0)

#         print(f"\n Successfully crawled {len(visited)} unique pages")

#         splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#         chunks = splitter.split_documents(documents)
#         return init_vectorstore(chunks)


def crawl_site(start_url, force_refresh=False, max_depth=10, max_workers=8):
    persist_dir = "chroma_test_db2"
    domain = urlparse(start_url).netloc

    if os.path.exists(persist_dir) and not force_refresh:
        print(" Loading existing Chroma vector store")
        return Chroma(persist_directory=persist_dir, embedding_function=embedding)
    elif force_refresh:
        print(" Force refresh enabled — rebuilding vector store")
    else:
        print(" No existing vector store — starting fresh crawl")

    print(" Crawling site and rebuilding vector store")
    queue = deque([(start_url, 0)])
    visited.clear()
    documents.clear()

    def process_url(url, depth):
        if url in visited or depth > max_depth:
            return []

        visited.add(url)
        print(f"🔍 [{depth}] Crawling: {url}")

        if url.endswith(".pdf"):
            text = extract_text_from_pdf(url)
            if text:
                doc = Document(page_content=text, metadata={"source": url, "source_type": "pdf"})
                documents.append(doc)
            return []

        if is_paginated_url(url):
            si_id = extract_si_id(url)
            if si_id in seen_paginated_ids:
                print(f"⏭️ Skipping duplicate pagination for SI_Id={si_id}")
                return []
            seen_paginated_ids.add(si_id)

            print(f"📄 Detected paginated URL: {url}")
            return handle_pagination(url)

        soup = fetch_html(url)
        if not soup:
            return []

        text = soup.get_text(separator=" ", strip=True)
        if len(text.strip()) > 100:
            doc = Document(page_content=text, metadata={"source": url, "source_type": "recursive"})
            documents.append(doc)

        return extract_internal_links(soup, url, domain)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while queue:
            batch = []
            while queue and len(batch) < max_workers:
                batch.append(queue.popleft())

            futures = {executor.submit(process_url, url, depth): (url, depth) for url, depth in batch}

            for future in as_completed(futures):
                url, depth = futures[future]
                try:
                    sub_links = future.result()
                    for link in sub_links:
                        # normalized = normalize_url(link)

                        # ✅ Filter: Only allow .html and .pdf
                        if (
                            link not in visited
                            and (link.endswith(".pdf") or is_html_page(link))
                        ):
                            queue.append((link, depth + 1))

                        # if link not in visited:
                        #     queue.append((link, depth + 1))
                except Exception as e:
                    print(f" Error processing {url}: {e}")

            print(f" Queue: {len(queue)} | Visited: {len(visited)} | Docs: {len(documents)}")

    print(f"\n Successfully crawled {len(visited)} unique pages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    return init_vectorstore(chunks)



start_url = "https://www.eib.org.tr/"
vectorstore = crawl_site(start_url)


def search_webpage_final(query):
    docs = vectorstore.similarity_search(query, k=4)  
    return docs[0].page_content