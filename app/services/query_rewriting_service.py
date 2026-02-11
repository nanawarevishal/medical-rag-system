"""
Query Rewriting and Expansion Service
Generates improved query variants for better retrieval.
"""

from typing import List, Dict, Any
import logging
from openai import AsyncOpenAI
from app.config import settings

logger = logging.getLogger("rag_app.query_rewriting_service")


class QueryRewritingService:
    """Service for query rewriting and expansion using OpenAI."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 256
    ):
        """
        Initialize the query rewriting service.

        Args:
            api_key: OpenAI API key (optional, uses settings if not provided)
            model: LLM model for rewriting/expansion
            temperature: Sampling temperature for controlled variation
            max_tokens: Maximum tokens for LLM output
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file.")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ==================== Public API ====================

    async def rewrite_query(self, query: str, max_rewrites: int = 3) -> Dict[str, Any]:
        """
        Generate rewritten variants of a query for improved retrieval.

        Args:
            query: Original user query
            max_rewrites: Number of rewritten variants to generate

        Returns:
            Dictionary with:
                - original: Original query
                - rewrites: List of rewritten queries
                - model: Model used
                - usage: Token usage (if available)
        """
        if not query or not query.strip():
            return {"original": query, "rewrites": [], "model": self.model, "usage": None}

        try:
            messages = self._build_rewrite_messages(query, max_rewrites)
            response = await self._call_llm(messages, self.max_tokens)
            rewrites = self._parse_lines(response)

            rewrites = self._dedupe_and_limit(
                rewrites,
                max_items=max_rewrites,
                exclude=[query]
            )

            return {
                "original": query,
                "rewrites": rewrites,
                "model": self.model,
                "usage": self._usage_from_response(response)
            }
        except Exception as e:
            raise Exception(f"Failed to rewrite query: {str(e)}")

    async def expand_query(self, query: str, max_expansions: int = 5) -> Dict[str, Any]:
        """
        Generate expansion terms/phrases to enrich the query.

        Args:
            query: Original user query
            max_expansions: Number of expansions to generate

        Returns:
            Dictionary with:
                - original: Original query
                - expansions: List of expansion phrases
                - expanded_query: Original query + expansions
                - model: Model used
                - usage: Token usage (if available)
        """
        if not query or not query.strip():
            return {
                "original": query,
                "expansions": [],
                "expanded_query": query,
                "model": self.model,
                "usage": None
            }

        try:
            messages = self._build_expansion_messages(query, max_expansions)
            response = await self._call_llm(messages, self.max_tokens)
            expansions = self._parse_lines(response)

            expansions = self._dedupe_and_limit(
                expansions,
                max_items=max_expansions,
                exclude=[query]
            )

            expanded_query = self._build_expanded_query(query, expansions)

            return {
                "original": query,
                "expansions": expansions,
                "expanded_query": expanded_query,
                "model": self.model,
                "usage": self._usage_from_response(response)
            }
        except Exception as e:
            raise Exception(f"Failed to expand query: {str(e)}")

    async def rewrite_and_expand(
        self,
        query: str,
        max_rewrites: int = 3,
        max_expansions: int = 5
    ) -> Dict[str, Any]:
        """
        Generate both rewrites and expansions for a query.

        Args:
            query: Original user query
            max_rewrites: Number of rewritten variants to generate
            max_expansions: Number of expansion phrases to generate

        Returns:
            Dictionary with:
                - original: Original query
                - rewrites: List of rewritten queries
                - expansions: List of expansion phrases
                - expanded_query: Original query + expansions
                - model: Model used
                - usage: Token usage (if available)
        """
        try:
            rewrite_result = await self.rewrite_query(query, max_rewrites=max_rewrites)
            expansion_result = await self.expand_query(query, max_expansions=max_expansions)

            return {
                "original": query,
                "rewrites": rewrite_result.get("rewrites", []),
                "expansions": expansion_result.get("expansions", []),
                "expanded_query": expansion_result.get("expanded_query", query),
                "model": self.model,
                "usage": {
                    "rewrite": rewrite_result.get("usage"),
                    "expansion": expansion_result.get("usage"),
                },
            }
        except Exception as e:
            raise Exception(f"Failed to rewrite and expand query: {str(e)}")

    # ==================== Internal Helpers ====================

    def _build_rewrite_messages(self, query: str, max_rewrites: int) -> List[Dict[str, str]]:
        system = (
            "You rewrite search queries for better retrieval. "
            "Keep the original intent. Return concise rewrites only."
        )
        user = (
            f"Rewrite the query into {max_rewrites} variants. "
            "Return one rewrite per line without numbering or bullets.\n\n"
            f"Query: {query}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _build_expansion_messages(self, query: str, max_expansions: int) -> List[Dict[str, str]]:
        system = (
            "You expand search queries with related terms and phrases to improve recall. "
            "Return only short phrases."
        )
        user = (
            f"Provide {max_expansions} expansion phrases that complement the query. "
            "Return one phrase per line without numbering or bullets.\n\n"
            f"Query: {query}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    async def _call_llm(self, messages: List[Dict[str, str]], max_tokens: int):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens
        )
        return response

    def _parse_lines(self, response) -> List[str]:
        content = response.choices[0].message.content or ""
        lines = []
        for raw_line in content.splitlines():
            line = raw_line.strip().lstrip("-•*").strip()
            if line:
                lines.append(line)
        return lines

    def _dedupe_and_limit(self, items: List[str], max_items: int, exclude: List[str]) -> List[str]:
        exclude_lower = {e.strip().lower() for e in exclude if e}
        seen = set()
        result = []
        for item in items:
            normalized = item.strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in exclude_lower or lowered in seen:
                continue
            seen.add(lowered)
            result.append(normalized)
            if len(result) >= max_items:
                break
        return result

    def _build_expanded_query(self, query: str, expansions: List[str]) -> str:
        if not expansions:
            return query
        return f"{query.strip()} " + " ".join(expansions)

    def _usage_from_response(self, response) -> Dict[str, Any] | None:
        if hasattr(response, "usage") and response.usage:
            return {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return None
