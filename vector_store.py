"""
Sri Lanka Legal Intelligence Platform (SLLIP V3)
Managed Vector Database Storage Abstraction
===========================================
Decouples application code from local PersistentClient. Supports:
1. Local ChromaDB (default for offline/local development)
2. Chroma Cloud (managed HTTP client)
3. Qdrant Cloud (managed cloud vector store)

Credentials and endpoints are loaded securely via environment variables (.env)
or Streamlit secrets (st.secrets).

Never silently returns 0 on exceptions; raises typed errors or reports structured
health status for the diagnostic panel.
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


class VectorStoreError(Exception):
    """Base exception for all vector store failures."""
    pass


class VectorStoreAuthError(VectorStoreError):
    """Authentication or authorization failure against the vector store."""
    pass


class VectorStoreConnectionError(VectorStoreError):
    """Network connection, endpoint unreachable, or timeout error."""
    pass


class CollectionNotFoundError(VectorStoreError):
    """The requested collection does not exist on the vector backend."""
    pass


def get_secret(key: str, default: str = "") -> str:
    """Retrieve secret from Streamlit secrets (st.secrets) or environment variables."""
    # First check streamlit secrets if running inside Streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            val = str(st.secrets[key]).strip()
            if val:
                return val
    except Exception:
        pass

    # Then check environment variables
    val = os.getenv(key)
    if val:
        return val.strip()

    return default


def string_to_uuid(id_str: str) -> str:
    """Deterministically convert an arbitrary string ID into a valid RFC 4122 UUID string."""
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

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return structured sanitized health status for diagnostic panels."""
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

    def get_status(self) -> Dict[str, Any]:
        col_name = getattr(self._col, "name", "local_chroma")
        try:
            cnt = self._col.count()
            return {
                "status": "healthy" if cnt > 0 else "empty_confirmed",
                "backend": "local",
                "backend_label": "Local ChromaDB",
                "collection_name": col_name,
                "count": cnt,
                "error": None
            }
        except Exception as e:
            return {
                "status": "connection_error",
                "backend": "local",
                "backend_label": "Local ChromaDB",
                "collection_name": col_name,
                "count": None,
                "error": f"Local storage access error: {type(e).__name__}"
            }


class QdrantCollectionWrapper(BaseVectorCollection):
    """Wraps Qdrant cloud client with drop-in ChromaDB API compatibility."""

    def __init__(self, client: Any, collection_name: str, vector_size: int = 384, read_only: bool = False):
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.read_only = read_only
        self._init_error: Optional[Exception] = None
        try:
            self._ensure_collection()
        except Exception as e:
            self._init_error = e

    def _ensure_collection(self):
        from qdrant_client.models import Distance, VectorParams
        try:
            exists = self.client.collection_exists(self.collection_name)
        except Exception as e:
            err_str = str(e).lower()
            if "401" in err_str or "403" in err_str or "unauthorized" in err_str or "api-key" in err_str:
                raise VectorStoreAuthError(f"Qdrant Cloud authentication failed: {e}")
            raise VectorStoreConnectionError(f"Failed to connect to Qdrant Cloud cluster: {e}")

        if not exists:
            if self.read_only:
                raise CollectionNotFoundError(
                    f"Collection '{self.collection_name}' does not exist on Qdrant Cloud. "
                    f"Automatic creation is disabled in read-only / deployment mode."
                )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )

    def add(self, ids: List[str], documents: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        if self.read_only:
            raise PermissionError("Cannot add points: Vector store is operating in read-only mode.")
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

    def get_status(self) -> Dict[str, Any]:
        """Return structured, sanitized health status for diagnostic panels."""
        if self._init_error is not None:
            if isinstance(self._init_error, CollectionNotFoundError):
                return {
                    "status": "collection_missing",
                    "backend": "qdrant",
                    "backend_label": "Qdrant Cloud (Managed)",
                    "collection_name": self.collection_name,
                    "count": None,
                    "error": str(self._init_error)
                }
            elif isinstance(self._init_error, VectorStoreAuthError):
                return {
                    "status": "auth_error",
                    "backend": "qdrant",
                    "backend_label": "Qdrant Cloud (Managed)",
                    "collection_name": self.collection_name,
                    "count": None,
                    "error": str(self._init_error)
                }
            return {
                "status": "connection_error",
                "backend": "qdrant",
                "backend_label": "Qdrant Cloud (Managed)",
                "collection_name": self.collection_name,
                "count": None,
                "error": str(self._init_error)
            }

        try:
            if not self.client.collection_exists(self.collection_name):
                return {
                    "status": "collection_missing",
                    "backend": "qdrant",
                    "backend_label": "Qdrant Cloud (Managed)",
                    "collection_name": self.collection_name,
                    "count": None,
                    "error": f"Collection '{self.collection_name}' does not exist on cluster."
                }
            cnt = self.client.count(collection_name=self.collection_name, exact=True).count
            return {
                "status": "healthy" if cnt > 0 else "empty_confirmed",
                "backend": "qdrant",
                "backend_label": "Qdrant Cloud (Managed)",
                "collection_name": self.collection_name,
                "count": cnt,
                "error": None
            }
        except Exception as e:
            err_str = str(e).lower()
            if "401" in err_str or "403" in err_str or "unauthorized" in err_str or "api-key" in err_str:
                return {
                    "status": "auth_error",
                    "backend": "qdrant",
                    "backend_label": "Qdrant Cloud (Managed)",
                    "collection_name": self.collection_name,
                    "count": None,
                    "error": "Authentication failed. Check QDRANT_API_KEY in Secrets."
                }
            return {
                "status": "connection_error",
                "backend": "qdrant",
                "backend_label": "Qdrant Cloud (Managed)",
                "collection_name": self.collection_name,
                "count": None,
                "error": f"Cluster unreachable: {type(e).__name__}"
            }

    def count(self) -> int:
        status = self.get_status()
        if status["status"] == "auth_error":
            raise VectorStoreAuthError(status["error"])
        elif status["status"] == "connection_error":
            raise VectorStoreConnectionError(status["error"])
        elif status["status"] == "collection_missing":
            raise CollectionNotFoundError(status["error"])
        return status["count"] if status["count"] is not None else 0

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
        local_path: Optional[Path] = None,
        read_only: Optional[bool] = None
    ) -> BaseVectorCollection:
        """
        Return a wrapped collection instance based on configured backend:
        'local', 'chroma_cloud', or 'qdrant'.
        If backend is None, defaults to VECTOR_DB_BACKEND secret/env or 'local'.
        If read_only is None, defaults to True if APP_MODE in ('read_only', 'hosted', 'production').
        """
        backend = (backend or get_secret("VECTOR_DB_BACKEND", "local")).lower()

        if read_only is None:
            read_only = get_secret("APP_MODE", "").lower() in ("read_only", "hosted", "production")

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
                    raise VectorStoreAuthError("QDRANT_URL and QDRANT_API_KEY must be configured.")
                client = QdrantClient(url=url, api_key=api_key, timeout=60.0)
                return QdrantCollectionWrapper(
                    client=client,
                    collection_name=collection_name,
                    read_only=read_only
                )
            except ImportError:
                raise ImportError("qdrant-client not installed. Run 'pip install qdrant-client'")

        else:
            raise ValueError(f"Unknown VECTOR_DB_BACKEND: '{backend}'. Supported: 'local', 'chroma_cloud', 'qdrant'")


if __name__ == "__main__":
    print("=" * 70)
    print("SLLIP V3 - VECTOR STORE ABSTRACTION HEALTH CHECK")
    print("=" * 70)
    backend = get_secret("VECTOR_DB_BACKEND", "local")
    print(f"Configured backend: {backend}")
    try:
        store = VectorStoreFactory.get_collection("case_law_collection")
        status = store.get_status()
        print(f"Collection status:  {status['status']}")
        print(f"Active count:       {status['count']}")
        print(f"Error (if any):     {status['error']}")
    except Exception as e:
        print(f"Exception during store check: {e}")
