from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
import asyncio
import json
import httpx


SYSTEM_PROMPT = """You are a member of My AI Team. Be accurate and practical. Distinguish facts, inferences, and recommendations. Never imitate the other team roles.

ANSWER CONTRACT:
- Lead with the answer; no greeting, preamble, or restatement of the question.
- Write no more than 5 complete, well-constructed sentences total.
- Plan the answer internally so the fifth sentence completes the thought; never stop mid-sentence.
- Each sentence must add a specific decision, reason, example, correction, or next action.
- Prefer specific decisions, examples, and next actions over generic explanation.
- Do not add a conclusion that merely repeats the answer.
- Do not use headings or bullet fragments to evade the 5-sentence limit."""


class ProviderError(RuntimeError):
    pass


class Provider(ABC):
    name: str

    def __init__(self, api_key: str | None, model: str, timeout: float, max_output_tokens: int):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.last_usage = {"input_tokens": 0, "output_tokens": 0}

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def require_key(self) -> str:
        if not self.api_key:
            raise ProviderError(f"{self.name} is not configured. Add its API key to .env.")
        return self.api_key

    async def post(self, url: str, **kwargs) -> dict:
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, **kwargs)
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in {429, 500, 502, 503, 504} or attempt == 2:
                    detail = exc.response.text[:300]
                    raise ProviderError(f"{self.name} API returned {exc.response.status_code}: {detail}") from exc
            except httpx.HTTPError as exc:
                if attempt == 2: raise ProviderError(f"{self.name} request failed after 3 attempts: {exc}") from exc
            except ValueError as exc:
                raise ProviderError(f"{self.name} returned invalid JSON: {exc}") from exc
            await asyncio.sleep(0.5 * (2 ** attempt))
        raise ProviderError(f"{self.name} request failed.")

    @abstractmethod
    async def ask(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        raise NotImplementedError

    async def stream(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> AsyncIterator[dict]:
        raise NotImplementedError


class OpenAIProvider(Provider):
    name = "ChatGPT / OpenAI"
    provider_id = "openai"

    async def ask(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        key = self.require_key()
        data = await self.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": self.model, "instructions": system_prompt, "input": prompt, "max_output_tokens": self.max_output_tokens},
        )
        usage = data.get("usage") or {}
        self.last_usage = {"input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)}
        if data.get("output_text"):
            return data["output_text"]
        parts = [c.get("text", "") for item in data.get("output", []) for c in item.get("content", []) if c.get("type") == "output_text"]
        if not parts:
            raise ProviderError("OpenAI returned no text response.")
        return "\n".join(parts)

    async def stream(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> AsyncIterator[dict]:
        key = self.require_key()
        payload = {"model": self.model, "instructions": system_prompt, "input": prompt, "max_output_tokens": self.max_output_tokens, "stream": True}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", "https://api.openai.com/v1/responses", headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "): continue
                        data = json.loads(line[6:])
                        if data.get("type") == "response.output_text.delta" and data.get("delta"):
                            yield {"type": "delta", "text": data["delta"]}
                        elif data.get("type") == "response.completed":
                            usage = data.get("response", {}).get("usage") or {}
                            yield {"type": "usage", "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)}
                        elif data.get("type") in {"response.failed", "error"}:
                            error = data.get("error") or data.get("response", {}).get("error") or {}
                            raise ProviderError(error.get("message", "OpenAI streaming response failed."))
        except httpx.HTTPStatusError as exc:
            detail = (await exc.response.aread()).decode(errors="replace")[:300]
            raise ProviderError(f"{self.name} API returned {exc.response.status_code}: {detail}") from exc
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderError(f"{self.name} streaming request failed: {exc}") from exc


class DeepSeekProvider(Provider):
    name = "DeepSeek"
    provider_id = "deepseek"

    async def ask(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        key = self.require_key()
        data = await self.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], "max_tokens": self.max_output_tokens},
        )
        usage = data.get("usage") or {}
        self.last_usage = {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("DeepSeek returned no text response.") from exc

    async def stream(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> AsyncIterator[dict]:
        key = self.require_key()
        payload = {"model": self.model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], "max_tokens": self.max_output_tokens, "stream": True, "stream_options": {"include_usage": True}}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", "https://api.deepseek.com/chat/completions", headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]": continue
                        data = json.loads(line[6:])
                        choices = data.get("choices") or []
                        delta = choices[0].get("delta", {}).get("content") if choices else None
                        if delta: yield {"type": "delta", "text": delta}
                        usage = data.get("usage")
                        if usage: yield {"type": "usage", "input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}
        except httpx.HTTPStatusError as exc:
            detail = (await exc.response.aread()).decode(errors="replace")[:300]
            raise ProviderError(f"{self.name} API returned {exc.response.status_code}: {detail}") from exc
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderError(f"{self.name} streaming request failed: {exc}") from exc


class GeminiProvider(DeepSeekProvider):
    """Gemini through Google's documented OpenAI-compatible endpoint."""
    name = "Google Gemini"
    provider_id = "gemini"

    async def ask(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        key = self.require_key()
        data = await self.post(
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "x-goog-api-client": "my-ai-team/1.0"},
            json={"model": self.model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], "max_tokens": self.max_output_tokens},
        )
        usage = data.get("usage") or {}
        self.last_usage = {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}
        try: return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc: raise ProviderError("Gemini returned no text response.") from exc

    async def stream(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> AsyncIterator[dict]:
        compatible = OpenAICompatibleProvider(self.name, self.api_key, self.model, self.timeout, self.max_output_tokens, "https://generativelanguage.googleapis.com/v1beta/openai")
        compatible.provider_id = self.provider_id
        async for event in compatible.stream(prompt, system_prompt): yield event


class AnthropicProvider(Provider):
    name = "Anthropic Claude"
    provider_id = "anthropic"

    async def ask(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        key = self.require_key()
        data = await self.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
            json={"model": self.model, "system": system_prompt, "messages": [{"role": "user", "content": prompt}], "max_tokens": self.max_output_tokens},
        )
        usage = data.get("usage") or {}
        self.last_usage = {"input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)}
        text = "\n".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text").strip()
        if not text: raise ProviderError("Claude returned no text response.")
        return text

    async def stream(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> AsyncIterator[dict]:
        key = self.require_key(); input_tokens = 0; output_tokens = 0
        payload = {"model": self.model, "system": system_prompt, "messages": [{"role": "user", "content": prompt}], "max_tokens": self.max_output_tokens, "stream": True}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", "https://api.anthropic.com/v1/messages", headers={"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "): continue
                        data = json.loads(line[6:]); event_type = data.get("type")
                        if event_type == "message_start": input_tokens = (data.get("message", {}).get("usage") or {}).get("input_tokens", 0)
                        elif event_type == "content_block_delta" and data.get("delta", {}).get("type") == "text_delta": yield {"type": "delta", "text": data["delta"]["text"]}
                        elif event_type == "message_delta": output_tokens = (data.get("usage") or {}).get("output_tokens", output_tokens)
                        elif event_type == "error": raise ProviderError((data.get("error") or {}).get("message", "Claude streaming response failed."))
                    yield {"type": "usage", "input_tokens": input_tokens, "output_tokens": output_tokens}
        except httpx.HTTPStatusError as exc:
            detail = (await exc.response.aread()).decode(errors="replace")[:300]
            raise ProviderError(f"{self.name} API returned {exc.response.status_code}: {detail}") from exc
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc: raise ProviderError(f"{self.name} streaming request failed: {exc}") from exc


class OpenAICompatibleProvider(DeepSeekProvider):
    provider_id = "plugin"

    def __init__(self, name: str, api_key: str | None, model: str, timeout: float, max_output_tokens: int, base_url: str):
        super().__init__(api_key, model, timeout, max_output_tokens)
        self.name = name
        self.base_url = base_url.rstrip("/")

    async def ask(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        key = self.require_key()
        data = await self.post(f"{self.base_url}/chat/completions", headers={"Authorization": f"Bearer {key}", "Content-Type":"application/json"}, json={"model":self.model,"messages":[{"role":"system","content":system_prompt},{"role":"user","content":prompt}],"max_tokens":self.max_output_tokens})
        usage=data.get("usage") or {};self.last_usage={"input_tokens":usage.get("prompt_tokens",0),"output_tokens":usage.get("completion_tokens",0)}
        try:return data["choices"][0]["message"]["content"]
        except (KeyError,IndexError,TypeError) as exc:raise ProviderError(f"{self.name} returned no text response.") from exc

    async def stream(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> AsyncIterator[dict]:
        key=self.require_key();payload={"model":self.model,"messages":[{"role":"system","content":system_prompt},{"role":"user","content":prompt}],"max_tokens":self.max_output_tokens,"stream":True,"stream_options":{"include_usage":True}}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST",f"{self.base_url}/chat/completions",headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: ") or line=="data: [DONE]":continue
                        data=json.loads(line[6:]);choices=data.get("choices") or [];delta=choices[0].get("delta",{}).get("content") if choices else None
                        if delta:yield {"type":"delta","text":delta}
                        usage=data.get("usage")
                        if usage:yield {"type":"usage","input_tokens":usage.get("prompt_tokens",0),"output_tokens":usage.get("completion_tokens",0)}
        except Exception as exc:raise ProviderError(f"{self.name} streaming request failed: {exc}") from exc
