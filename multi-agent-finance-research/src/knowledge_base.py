# =============================================================================
# KNOWLEDGE BASE — builds the Chroma vector store for the Document-Q&A agent
# =============================================================================
# This module is adapted from the course reference (module5/01-multi-agent_system
# /agent.py). We keep its async aiohttp + BeautifulSoup scraper (a modern
# replacement for the deprecated AsyncHtmlLoader) but:
#   * point it at English Wikipedia company / finance pages,
#   * add a citation-friendly "title" to each chunk's metadata, and
#   * persist the Chroma store to disk so we embed the corpus only once.
# =============================================================================

import asyncio
import os

import aiohttp
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


# -----------------------------------------------------------------------------
# Async web scraper (kept close to the course reference for clear lineage)
# -----------------------------------------------------------------------------
async def _fetch_and_parse(session: aiohttp.ClientSession, url: str) -> str:
    """Fetch one Wikipedia page and return cleaned plain text."""
    # Be polite to Wikipedia's servers.
    await asyncio.sleep(0.5)
    async with session.get(url) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} for {url}")
        html = await response.text()

    soup = BeautifulSoup(html, "html.parser")
    # Strip non-content elements that would only add noise to the embeddings.
    for tag in soup(["script", "style", "nav", "footer", "header", "table"]):
        tag.extract()
    # Wikipedia wraps the article body in <div id="mw-content-text">; fall back
    # to the whole document if the structure ever changes.
    body = soup.find(id="mw-content-text") or soup
    text = " ".join(body.get_text().split())
    return text


async def _load_documents(slugs: list[str]) -> list[Document]:
    """Download all corpus pages concurrently and wrap them as Documents."""
    headers = {
        "User-Agent": os.environ.get(
            "USER_AGENT", "BBS-AIIM-Finance-Assistant/1.0 (Educational Project)"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    urls = [config.WIKIPEDIA_BASE_URL + slug for slug in slugs]

    documents: list[Document] = []
    async with aiohttp.ClientSession(headers=headers) as session:
        results = await asyncio.gather(
            *(_fetch_and_parse(session, url) for url in urls),
            return_exceptions=True,
        )

    for slug, url, result in zip(slugs, urls, results):
        if isinstance(result, Exception):
            print(f"  [warn] skipping {url}: {result}")
            continue
        # A readable title for citations, e.g. "Apple_Inc." -> "Apple Inc."
        title = slug.replace("_", " ")
        documents.append(
            Document(page_content=result, metadata={"source": url, "title": title})
        )
    return documents


# -----------------------------------------------------------------------------
# Vector store construction + persistence
# -----------------------------------------------------------------------------
def _embeddings():
    """Return the embedding model for the configured backend.

    Defaults to a local HuggingFace model so the corpus can be embedded without
    consuming the Gemini free-tier embedding quota (see config.EMBEDDINGS_BACKEND).
    """
    if config.EMBEDDINGS_BACKEND == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=config.GOOGLE_EMBEDDING_MODEL)

    # Default: local sentence-transformers (free, offline, unlimited).
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=config.HF_EMBEDDING_MODEL)


def _persist_dir() -> str:
    return str(config.PROJECT_ROOT / ".chroma")


def _build_from_web() -> Chroma:
    """Scrape the corpus, chunk it, embed it, and persist a Chroma store."""
    print(f"[kb] downloading {len(config.WIKI_PAGES)} Wikipedia pages ...")
    docs = asyncio.run(_load_documents(config.WIKI_PAGES))
    print(f"[kb] loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"[kb] embedding {len(chunks)} chunks (one-time cost) ...")

    store = Chroma.from_documents(
        documents=chunks,
        embedding=_embeddings(),
        persist_directory=_persist_dir(),
        collection_name="finance_kb",
    )
    print("[kb] vector store ready and persisted to disk")
    return store


# Singleton so the (expensive) store is built at most once per process.
_store: Chroma | None = None


def get_vectorstore(force_rebuild: bool = False) -> Chroma:
    """Return the corpus vector store, building or loading from disk as needed.

    On first ever run this scrapes Wikipedia and embeds the corpus (slow). On
    every later run it loads the persisted store from `.chroma/` (fast).
    """
    global _store
    if _store is not None and not force_rebuild:
        return _store

    config.require_api_key()
    persist_dir = _persist_dir()

    if not force_rebuild and os.path.isdir(persist_dir) and os.listdir(persist_dir):
        print("[kb] loading persisted vector store from disk")
        _store = Chroma(
            persist_directory=persist_dir,
            embedding_function=_embeddings(),
            collection_name="finance_kb",
        )
    else:
        _store = _build_from_web()
    return _store


def get_retriever():
    """Convenience: a retriever returning the top-k most relevant chunks."""
    return get_vectorstore().as_retriever(search_kwargs={"k": config.RETRIEVER_K})
