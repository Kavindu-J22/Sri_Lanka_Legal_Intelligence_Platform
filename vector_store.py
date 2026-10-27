"""
Sri Lanka Legal Intelligence Platform (SLLIP V3)
Managed Vector Database Storage Abstraction
===========================================
Decouples application code from local PersistentClient. Supports:
1. Local ChromaDB (default for offline/local development)
2. Chroma Cloud (managed HTTP client)
3. Qdrant Cloud (managed cloud vector store)

Credentials and endpoints are loaded securely via environment variables (.env)
or Streamlit secrets.
"""

import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional

WORKSPACE_DIR = Path(__file__).parent.resolve()
DEFAULT_LOCAL_PATH = WORKSPACE_DIR / "vector_db"

# Ensure .env is loaded
try:
    from dotenv import load_dotenv
    load_dotenv(WORKSPACE_DIR / ".env")
except Exception:
    pass


def get_secret(key: str, default: str = "") -> str:
    """Retrieve secret from environment variable or streamlit secrets if available."""
    val = os.getenv(key)
    if val:
        return val.strip()
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return default


def string_to_uuid(id_str: str) -> str:
    """Deterministically convert an arbitrary string ID into a valid UUID string (RFC 4122)."""
    try:
        uuid.UUID(str(id_str))
        return str(id_str)
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(id_str)))


class BaseVectorCollection(ABC):
    """Abstract interface for interacting with a vector database collection."""

    @abstractmethod
    def add(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        pass

    @abstractmethod
    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    @abstractmethod
    def get(self, ids: Optional[List[str]] = None, include: Optional[List[str]] = None) -> Dict[str, Any]:
        pass


class LocalChromaCollectionWrapper(BaseVectorCollection):
    """Wraps native ChromaDB collection with resilient batching."""

    def __init__(self, collection):
        self._col = collection

    def add(self, ids, documents, embeddings, metadatas):
        self._col.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def query(self, query_embeddings, n_results=10, where=None, include=None):
        kwargs = {
            "query_embeddings": query_embeddings,
            "n_results": n_results
        }
        if where:
            kwargs["where"] = where
        if include:
            kwargs["include"] = include
        return self._col.query(**kwargs)

    def count(self) -> int:
        return self._col.count()

    def get(self, ids=None, include=None) -> Dict[str, Any]:
        kwargs = {}
        if ids:
            kwargs["ids"] = ids
        if include:
            kwargs["include"] = include
        return self._col.get(**kwargs)


class QdrantCollectionWrapper(BaseVectorCollection):
    """Wraps Qdrant cloud client with drop-in ChromaDB API compatibility."""

    def __init__(self, client: Any, collection_name: str, vector_size: int = 384):
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self):
        from qdrant_client.models import Distance, VectorParams
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )

    def add(self, ids: List[str], documents: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        from qdrant_client.models import PointStruct
        points = []
        for i, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
            pt_id = string_to_uuid(i)
            payload = dict(meta) if meta else {}
            payload["document"] = str(doc)
            payload["_original_id"] = str(i)
            emb_list = [float(x) for x in emb]
            points.append(PointStruct(id=pt_id, vector=emb_list, payload=payload))

        if points:
            for attempt in range(4):
                try:
                    self.client.upsert(collection_name=self.collection_name, points=points)
                    break
                except Exception as e:
                    if attempt == 3:
                        raise e
                    import time
                    time.sleep(2 * (attempt + 1))

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        all_ids = []
        all_docs = []
        all_metas = []
        all_distances = []

        q_filter = None
        if where:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            conditions = []
            for k, v in where.items():
                conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
            if conditions:
                q_filter = Filter(must=conditions)

        for q_emb in query_embeddings:
            q_emb_list = [float(x) for x in q_emb]
            res = self.client.query_points(
                collection_name=self.collection_name,
                query=q_emb_list,
                query_filter=q_filter,
                limit=n_results,
                with_payload=True
            )

            cur_ids = []
            cur_docs = []
            cur_metas = []
            cur_dists = []

            for pt in res.points:
                orig_id = pt.payload.get("_original_id", str(pt.id)) if pt.payload else str(pt.id)
                doc = pt.payload.get("document", "") if pt.payload else ""
                meta = {k: v for k, v in pt.payload.items() if k not in ("document", "_original_id")} if pt.payload else {}
                dist = max(0.0, 1.0 - float(pt.score)) if pt.score is not None else 0.0

                cur_ids.append(orig_id)
                cur_docs.append(doc)
                cur_metas.append(meta)
                cur_dists.append(dist)

            all_ids.append(cur_ids)
            all_docs.append(cur_docs)
            all_metas.append(cur_metas)
            all_distances.append(cur_dists)

        return {
            "ids": all_ids,
            "documents": all_docs,
            "metadatas": all_metas,
            "distances": all_distances
        }

    def count(self) -> int:
        try:
            if not self.client.collection_exists(self.collection_name):
                return 0
            return self.client.count(collection_name=self.collection_name, exact=True).count
        except Exception:
            return 0

    def get(self, ids: Optional[List[str]] = None, include: Optional[List[str]] = None) -> Dict[str, Any]:
        with_vectors = "embeddings" in (include or [])
        if ids:
            mapped = [string_to_uuid(i) for i in ids]
            pts = self.client.retrieve(
                collection_name=self.collection_name,
                ids=mapped,
                with_payload=True,
                with_vectors=with_vectors
            )
        else:
            pts, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=with_vectors
            )

        ret_ids = []
        ret_docs = []
        ret_metas = []
        ret_embs = []

        for pt in pts:
            orig_id = pt.payload.get("_original_id", str(pt.id)) if pt.payload else str(pt.id)
            doc = pt.payload.get("document", "") if pt.payload else ""
            meta = {k: v for k, v in pt.payload.items() if k not in ("document", "_original_id")} if pt.payload else {}
            ret_ids.append(orig_id)
            ret_docs.append(doc)
            ret_metas.append(meta)
            if with_vectors:
                ret_embs.append(pt.vector)

        res = {
            "ids": ret_ids,
            "documents": ret_docs,
            "metadatas": ret_metas
        }
        if with_vectors:
            res["embeddings"] = ret_embs
        return res


class VectorStoreFactory:
    """Factory to instantiate and manage configured vector store clients."""

    @staticmethod
    def get_collection(
        collection_name: str,
        backend: Optional[str] = None,
        local_path: Optional[Path] = None
    ) -> BaseVectorCollection:
        """
        Return a wrapped collection instance based on configured backend:
        'local', 'chroma_cloud', or 'qdrant'.
        If backend is None, defaults to VECTOR_DB_BACKEND environment variable or 'local'.
        """
        backend = (backend or get_secret("VECTOR_DB_BACKEND", "local")).lower()

        if backend == "local":
            import chromadb
            db_path = local_path or DEFAULT_LOCAL_PATH
            db_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(db_path))
            native_col = client.get_or_create_collection(name=collection_name)
            return LocalChromaCollectionWrapper(native_col)

        elif backend == "chroma_cloud":
            import chromadb
            api_key = get_secret("CHROMA_CLOUD_API_KEY")
            tenant = get_secret("CHROMA_CLOUD_TENANT", "default_tenant")
            database = get_secret("CHROMA_CLOUD_DATABASE", "default_database")
            host = get_secret("CHROMA_CLOUD_HOST", "api.trychroma.com")
            port = int(get_secret("CHROMA_CLOUD_PORT", "443"))
            ssl = get_secret("CHROMA_CLOUD_SSL", "true").lower() == "true"

            client = chromadb.HttpClient(
                host=host,
                port=port,
                ssl=ssl,
                headers={"X-Chroma-Token": api_key} if api_key else {},
                tenant=tenant,
                database=database
            )
            native_col = client.get_or_create_collection(name=collection_name)
            return LocalChromaCollectionWrapper(native_col)

        elif backend == "qdrant":
            try:
                from qdrant_client import QdrantClient
                url = get_secret("QDRANT_URL")
                api_key = get_secret("QDRANT_API_KEY")
                if not url or not api_key:
                    raise ValueError("QDRANT_URL and QDRANT_API_KEY must be configured.")
                client = QdrantClient(url=url, api_key=api_key, timeout=60.0)
                return QdrantCollectionWrapper(client=client, collection_name=collection_name)
            except ImportError:
                raise ImportError("qdrant-client not installed. Run 'pip install qdrant-client'")

        else:
            raise ValueError(f"Unknown VECTOR_DB_BACKEND: '{backend}'. Supported: 'local', 'chroma_cloud', 'qdrant'")


if __name__ == "__main__":
    print("=" * 70)
    print("SLLIP V3 - VECTOR STORE ABSTRACTION TEST")
    print("=" * 70)
    backend = get_secret("VECTOR_DB_BACKEND", "local")
    print(f"Configured backend: {backend}")
    store = VectorStoreFactory.get_collection("case_law_collection")
    print(f"Connected to collection: case_law_collection")
    print(f"Current Vector Count:   {store.count():,}")
