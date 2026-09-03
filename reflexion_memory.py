import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from crewai import LLM



class OllamaEmbeddingWithTimeout(EmbeddingFunction[Documents]):
    

    def __init__(
        self,
        url: str = "http://localhost:11434",
        model_name: str = "nomic-embed-text",
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self._url = url
        self._model_name = model_name
        self._timeout = timeout
        self._max_retries = max_retries

        
        try:
            import ollama
            from httpx import Timeout
            self._client = ollama.Client(
                host=url,
                timeout=Timeout(timeout),
            )
        except Exception:
            
            import ollama
            self._client = ollama.Client(host=url, timeout=timeout)

    def __call__(self, input: Documents) -> Embeddings:
        """Embed documents with retry logic."""
        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.embed(
                    model=self._model_name,
                    input=input,
                )
                return response["embeddings"]
            except Exception as e:
                last_error = e
                wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                print(f"    ⚠️  Embedding attempt {attempt + 1}/{self._max_retries} "
                      f"failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)

        print(f"    ❌ All {self._max_retries} embedding attempts failed.")
        raise last_error


class ReflexionMemory:
   

    def __init__(
        self,
        persist_dir: str = "./chroma_reflexion_db",
        ollama_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "ollama/qwen2.5:7b",
        collection_name: str = "reflexion_logs",
        embed_timeout: float = 120.0,
    ):

        self._embedding_fn = OllamaEmbeddingWithTimeout(
            url=ollama_url,
            model_name=embedding_model,
            timeout=embed_timeout,
            max_retries=3,
        )

    
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
        )

        
        self._llm = LLM(model=llm_model, base_url=ollama_url)

        
        self._session_reflections: List[str] = []

    

    def reflect(
        self,
        task_description: str,
        output: str,
        eval_result: Optional[Dict] = None,
    ) -> str:
        
        
        eval_feedback = ""
        if eval_result is not None:
            accuracy = eval_result.get("accuracy", "N/A")
            passed = eval_result.get("overall_pass", "N/A")
            failures = eval_result.get("failure_reasons", [])
            checks_passed = eval_result.get("checks_passed", "?")
            checks_total = eval_result.get("checks_applicable", "?")

            eval_feedback = (
                f"\n=== EVALUATOR FEEDBACK ===\n"
                f"Overall Pass: {passed}\n"
                f"Accuracy: {accuracy} ({checks_passed}/{checks_total} checks passed)\n"
            )
            if failures:
                eval_feedback += "Failed checks:\n"
                for f in failures:
                    eval_feedback += f"  - {f}\n"
            else:
                eval_feedback += "All checks passed.\n"
            eval_feedback += "=== END EVALUATOR FEEDBACK ===\n"

        
        mem_context = ""
        if self._session_reflections:
            mem_context = (
                "\n=== PREVIOUS REFLECTIONS IN THIS SESSION ===\n"
                + "\n---\n".join(self._session_reflections[-3:])  
                + "\n=== END PREVIOUS REFLECTIONS ===\n"
            )

        prompt = (
            "You are the Self-Reflection model (M_sr) in a Reflexion system. "
            "A multi-agent trip-planning system (the Actor) just completed the "
            "task below and the Evaluator scored it. "
            "Your job is to generate a concise, actionable self-reflection "
            "that will help the Actor improve on the NEXT trial.\n\n"
            "Your reflection MUST:\n"
            "1. Acknowledge what went well (if anything)\n"
            "2. Identify the ROOT CAUSE of each failed check\n"
            "3. Provide SPECIFIC, CONCRETE fixes for the next attempt "
            "(not vague suggestions)\n"
            "4. If there were previous reflections, note whether the Actor "
            "improved or regressed\n\n"
            f"=== TASK ===\n{task_description}\n\n"
            f"=== ACTOR OUTPUT (truncated) ===\n{output[:5000]}\n"
            f"{eval_feedback}"
            f"{mem_context}\n"
            "=== YOUR REFLECTION ==="
        )
        return self._llm.call(prompt)

   

    def store(
        self,
        task_description: str,
        output: str,
        reflection: str,
        eval_result: Optional[Dict] = None,
    ) -> str:
        
        doc_id = str(uuid.uuid4())

        metadata = {
            "task_description": task_description[:1000],
            "output_preview": output[:500],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if eval_result is not None:
            metadata["accuracy"] = str(eval_result.get("accuracy", ""))
            metadata["overall_pass"] = str(eval_result.get("overall_pass", ""))

    
        try:
            self._collection.add(
                documents=[reflection],
                ids=[doc_id],
                metadatas=[metadata],
            )
        except Exception as e:
            print(f"      ChromaDB store failed: {e}")
            print(f"      Reflection saved to session memory only (not persisted).")
            

        
        self._session_reflections.append(reflection)

        return doc_id

    def retrieve_relevant(
        self,
        current_task_description: str,
        n_results: int = 3,
    ) -> List[str]:
        """Return the most relevant past reflections for a new task.

        Uses ChromaDB similarity search against stored reflection
        documents.  Returns an empty list when the collection is empty.
        """
        if self._collection.count() == 0:
            return []

        try:
            results = self._collection.query(
                query_texts=[current_task_description],
                n_results=min(n_results, self._collection.count()),
            )
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            print(f"      ChromaDB retrieval failed: {e}")
            return []

    def get_session_reflections(self) -> List[str]:
        """Return all reflections stored in the current session (mem)."""
        return list(self._session_reflections)

    def clear_session(self):
        """Clear the in-memory session reflections for a fresh run."""
        self._session_reflections.clear()

    def reflect_and_store(
        self,
        task_description: str,
        output: str,
        eval_result: Optional[Dict] = None,
    ) -> str:
        
        reflection = self.reflect(task_description, output, eval_result)
        doc_id = self.store(task_description, output, reflection, eval_result)
        return doc_id
