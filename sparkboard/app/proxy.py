"""
Classifying reverse proxy for an OpenAI-compatible inference server.

Written against vLLM and equally happy in front of llama.cpp's server, which
speaks the same API; "upstream" below means whichever of the two is serving.
Sits in front of that server. Every request is forwarded
untouched -- streaming or not -- so from the client's side this is just vLLM
on a different port. Separately and asynchronously, the prompt is sent back to
vLLM with an instruction to categorize it in a few words ("writing code",
"drafting an email", "debugging"). Only that short label is kept.

The privacy model, stated plainly because it is the whole point:

  * Prompt text is never written to disk and never logged.
  * It lives in memory only for the moment it takes to classify, inside one
    async task, and is dropped when that task returns.
  * What leaves this module is a category label and a token count -- never the
    text that produced them.
  * Classification is best-effort and fire-and-forget. If it fails, is slow,
    or is switched off, real traffic is unaffected: the proxy's job is to
    forward, and categorizing is a side effect it can always skip.

The categories are a fixed set. The model is asked to pick the closest one,
not to describe freely, so the output space is bounded and can't turn into a
paraphrase of the prompt.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import os
import time

import httpx

log = logging.getLogger("sparkboard.proxy")

# The bounded label set. The classifier is told to choose the nearest of
# these; anything it can't place lands in "other". Keeping the space closed is
# what stops a label from leaking prompt content.
CATEGORIES = [
    "writing code",
    "debugging code",
    "reviewing code",
    "writing or editing text",
    "drafting an email or message",
    "writing a story or creative piece",
    "summarizing or extracting",
    "translating",
    "answering a question",
    "explaining a concept",
    "planning or organizing",
    "data analysis or math",
    "role-play or conversation",
    "other",
]

_CLASSIFY_TMPL = (
    "You are labeling the intent of a request for a status dashboard. "
    "Below is a user's prompt. Reply with ONLY the single best-fitting label "
    "from this list, copied exactly, and nothing else:\n{labels}\n\n"
    "If none fit well, reply 'other'. Do not explain. Do not quote the prompt. "
    "Do not add punctuation. /no_think\n\nPrompt:\n\"\"\"\n{prompt}\n\"\"\"\n\nLabel:"
)


class ProxyConfig:
    def __init__(self):
        # Where the real inference server lives. The proxy forwards here.
        # vLLM's default is :8000, llama.cpp's server is :8080.
        self.upstream = os.environ.get(
            "SPARKBOARD_VLLM_UPSTREAM", "http://127.0.0.1:8000").rstrip("/")
        # Where classification calls go. Defaults to the main upstream (reuse
        # the same model), but can point at a separate small model on its own
        # port so labeling never competes with real inference for GPU slots.
        self.classify_upstream = os.environ.get(
            "SPARKBOARD_CLASSIFY_UPSTREAM", self.upstream).rstrip("/")
        # Model name to use for the classification call. Empty means "reuse
        # whatever model the incoming request named", which is the common case
        # for a single-model vLLM deployment. When the classify upstream is a
        # separate instance, set this to that instance's served model name.
        self.classify_model = os.environ.get("SPARKBOARD_CLASSIFY_MODEL", "")
        # Fraction of requests to classify, 0..1. A busy server doesn't need
        # every request labeled to show what it's doing, and each label costs a
        # small generation of its own.
        self.sample_rate = float(os.environ.get("SPARKBOARD_CLASSIFY_SAMPLE", "1.0"))
        # Hard ceiling on concurrent classification calls, so labeling can
        # never crowd out real traffic on the GPU.
        self.max_concurrent = int(os.environ.get("SPARKBOARD_CLASSIFY_CONCURRENCY", "2"))
        # Longest prompt slice we hand to the classifier. The tail of a very
        # long prompt rarely changes the intent and costs context.
        self.max_prompt_chars = int(os.environ.get("SPARKBOARD_CLASSIFY_MAXCHARS", "2000"))
        # Whether to classify at all. Off means pure transparent proxy.
        self.enabled = os.environ.get("SPARKBOARD_CLASSIFY", "1") == "1"
        # Request timeout for the (background) classify call.
        self.classify_timeout = float(os.environ.get("SPARKBOARD_CLASSIFY_TIMEOUT", "20"))
        # API key for the classifier's own calls. When vLLM is started with
        # --api-key, the client's passthrough requests already carry their own
        # Authorization header (the proxy forwards it), but the classification
        # call is a request the proxy generates itself and must authenticate
        # separately. Set this to the key the classify upstream expects. If the
        # classifier points at a different instance (SPARKBOARD_CLASSIFY_UPSTREAM),
        # SPARKBOARD_CLASSIFY_API_KEY overrides for that instance specifically.
        self.vllm_api_key = os.environ.get("SPARKBOARD_VLLM_API_KEY", "")
        self.classify_api_key = os.environ.get(
            "SPARKBOARD_CLASSIFY_API_KEY", self.vllm_api_key)
        # When true, the proxy supplies the vLLM key for forwarded requests that
        # arrive without an Authorization header. Off by default (transparent).
        self.inject_auth = os.environ.get("SPARKBOARD_INJECT_AUTH", "0") == "1"


class PromptFeed:
    """
    A bounded ring of recent category labels. This is the only state the
    prompt feature keeps, and it contains no prompt text -- just labels,
    timings, and token counts.
    """

    def __init__(self, maxlen=200):
        import collections

        self._events = collections.deque(maxlen=maxlen)
        self._seq = 0
        self._counts = collections.Counter()
        self._total = 0

    def add(self, label, tokens=None, model=None, latency_ms=None):
        self._seq += 1
        self._total += 1
        self._counts[label] += 1
        self._events.append({
            "seq": self._seq,
            "ts": time.time(),
            "label": label,
            "tokens": tokens,
            "model": model,
            "latency_ms": latency_ms,
        })

    def recent(self, after=0, limit=100):
        out = [e for e in self._events if e["seq"] > after]
        return out[-limit:]

    def summary(self):
        # Rank categories seen, most frequent first -- the "what is this box
        # mostly doing" view, aggregated over the ring.
        top = [{"label": k, "count": v} for k, v in self._counts.most_common(12)]
        return {
            "total": self._total,
            "seq": self._seq,
            "categories": top,
            "active": len(self._events) > 0,
        }


class ClassifyingProxy:
    def __init__(self, config: ProxyConfig, feed: PromptFeed):
        self.cfg = config
        self.feed = feed
        self._sem = asyncio.Semaphore(config.max_concurrent)
        self._rr = 0
        # Long-lived clients: one for proxying (no timeout -- generations can
        # run long), one for classification (bounded).
        self._proxy_client = httpx.AsyncClient(timeout=None)
        self._classify_client = httpx.AsyncClient(timeout=config.classify_timeout)
        self.stats = {"forwarded": 0, "classified": 0, "classify_errors": 0,
                      "classify_skipped": 0}
        # Cleared permanently the first time an upstream refuses the field.
        self._send_template_kwargs = True

    async def aclose(self):
        await self._proxy_client.aclose()
        await self._classify_client.aclose()

    # ---------------------------------------------------------- prompt pull

    @staticmethod
    def _extract_prompt(body: dict) -> str | None:
        """
        Pull the human-meaningful text out of a chat or completion request.

        For chat, that's the last user turn -- the thing actually being asked,
        not the system preamble or the assistant's prior replies. Content can
        be a plain string or the multimodal list form; only text parts are
        used, and image data is ignored entirely.
        """
        msgs = body.get("messages")
        if isinstance(msgs, list) and msgs:
            for m in reversed(msgs):
                if not isinstance(m, dict) or m.get("role") != "user":
                    continue
                content = m.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = []
                    for p in content:
                        if isinstance(p, dict) and p.get("type") == "text":
                            parts.append(p.get("text", ""))
                    if parts:
                        return "\n".join(parts)
            return None
        # Legacy completions endpoint.
        prompt = body.get("prompt")
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], str):
            return "\n".join(prompt)
        return None

    def _pick_model(self, incoming: dict) -> str | None:
        if self.cfg.classify_model:
            return self.cfg.classify_model
        m = incoming.get("model")
        return m if isinstance(m, str) and m else None

    # ------------------------------------------------------------- classify

    async def _classify(self, prompt: str, model: str | None):
        """
        Ask vLLM to label the prompt. Runs as a detached task; nothing waits
        on it and its failures never surface to the client.
        """
        if self.cfg.sample_rate < 1.0:
            # Cheap deterministic-ish sampling without pulling in random state.
            self._rr = (self._rr + 1) % 1000
            if self._rr / 1000.0 >= self.cfg.sample_rate:
                self.stats["classify_skipped"] += 1
                return

        if model is None:
            self.stats["classify_errors"] += 1
            return

        snippet = prompt.strip()[: self.cfg.max_prompt_chars]
        if not snippet:
            return

        instruction = _CLASSIFY_TMPL.format(
            labels="\n".join(f"- {c}" for c in CATEGORIES),
            prompt=snippet,
        )
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": instruction}],
            # Reasoning models can burn the whole budget thinking before they
            # emit a label, so give real headroom rather than 12 tokens.
            "max_tokens": 64,
            "temperature": 0.0,
            "stream": False,
        }
        # Ask the server to skip the thinking phase for this call. Qwen3.x
        # honors chat_template_kwargs.enable_thinking; the /no_think hint in
        # the prompt is a belt-and-braces fallback for builds that don't.
        #
        # Not every server accepts the field, though: some builds of
        # llama.cpp's server reject unknown top-level keys with a 400. So it
        # is sent once, and if that is what comes back the field is dropped
        # for the rest of the process's life rather than failing every label
        # forever. One wasted request, then it self-corrects.
        if self._send_template_kwargs:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        t0 = time.time()
        async with self._sem:
            try:
                headers = {}
                if self.cfg.classify_api_key:
                    headers["Authorization"] = f"Bearer {self.cfg.classify_api_key}"
                url = f"{self.cfg.classify_upstream}/v1/chat/completions"
                r = await self._classify_client.post(url, json=payload, headers=headers)
                if r.status_code == 400 and self._send_template_kwargs:
                    log.info("classify upstream rejected chat_template_kwargs -- "
                             "retrying without it and dropping it from now on")
                    self._send_template_kwargs = False
                    payload.pop("chat_template_kwargs", None)
                    r = await self._classify_client.post(url, json=payload,
                                                         headers=headers)
                r.raise_for_status()
                data = r.json()
            except Exception as exc:
                self.stats["classify_errors"] += 1
                log.debug("classify request failed: %s", exc)
                return

        # Parsing is separate from the request so a malformed response shape
        # (reasoning models return some surprising ones) can't crash the task.
        try:
            label = self._parse_label(data)
        except Exception as exc:
            self.stats["classify_errors"] += 1
            log.debug("classify parse failed: %s", exc)
            return

        if label:
            self.stats["classified"] += 1
            self.feed.add(
                label,
                tokens=self._prompt_tokens(data),
                model=model,
                latency_ms=int((time.time() - t0) * 1000),
            )

    @staticmethod
    def _parse_label(data: dict) -> str | None:
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return None

        # Reasoning models (Qwen3.x, etc.) can return content=None with the
        # text in reasoning_content, or emit only reasoning tokens when the
        # max_tokens budget is small. Try content, then a couple of common
        # reasoning fields, and coerce to string defensively -- content may be
        # None, a number, or a list in the multimodal shape.
        raw = msg.get("content")
        if raw is None:
            raw = msg.get("reasoning_content") or msg.get("reasoning")
        if isinstance(raw, list):
            # multimodal content parts -> join the text pieces
            raw = " ".join(p.get("text", "") for p in raw
                           if isinstance(p, dict) and p.get("type") == "text")
        if not isinstance(raw, str):
            return None

        # Strip any <think>...</think> block a reasoning model prepends.
        text = re.sub(r"<think>.*?</think>", " ", raw, flags=re.S | re.I)
        text = text.strip().strip(".").strip('"').strip().lower()
        if not text:
            return None  # empty -> no label, NOT a spurious first-category match

        # Snap to the closest known category. Exact match first, then
        # containment either way, so "code" -> "writing code" and a stray
        # "label: writing code" still resolves. Guard against empty substrings.
        for c in CATEGORIES:
            if text == c:
                return c
        for c in CATEGORIES:
            if text in c or c in text:
                return c
        return "other"

    @staticmethod
    def _prompt_tokens(data: dict):
        try:
            return data.get("usage", {}).get("prompt_tokens")
        except AttributeError:
            return None

    # ---------------------------------------------------------------- proxy

    def _maybe_classify(self, raw_body: bytes):
        """Kick off classification for a request body, if enabled. Never raises."""
        if not self.cfg.enabled:
            return
        try:
            body = json.loads(raw_body)
        except (json.JSONDecodeError, ValueError):
            return
        if not isinstance(body, dict):
            return
        prompt = self._extract_prompt(body)
        if not prompt:
            return
        model = self._pick_model(body)
        # Detached: we do not await this. The prompt string is captured by the
        # task and released when it finishes.
        asyncio.create_task(self._classify(prompt, model))

    async def forward(self, request, path: str):
        """
        Transparently forward one request to vLLM and stream the response
        back. Classification is triggered alongside, not in the path.
        """
        from fastapi.responses import Response, StreamingResponse

        raw = await request.body()

        # Fire classification before forwarding so a long generation doesn't
        # delay the label -- they run concurrently.
        if request.method == "POST" and raw:
            self._maybe_classify(raw)

        url = f"{self.cfg.upstream}/{path.lstrip('/')}"
        # Strip hop-by-hop headers; keep the rest so auth etc. passes through.
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("host", "content-length", "connection")}

        # Auth handling. By default the proxy is transparent to auth: whatever
        # Authorization header the client sent is relayed, and vLLM enforces it.
        # With inject_auth on, the proxy supplies the vLLM key for any request
        # that arrives without one -- so the proxy can hold the real key and
        # clients don't need it. A client that DOES send its own key keeps it
        # (we never overwrite), which lets both styles coexist.
        if self.cfg.inject_auth and self.cfg.vllm_api_key:
            has_auth = any(k.lower() == "authorization" for k in headers)
            if not has_auth:
                headers["Authorization"] = f"Bearer {self.cfg.vllm_api_key}"

        # Decide streaming from the request body, matching vLLM's own behavior.
        stream = False
        if raw:
            try:
                stream = bool(json.loads(raw).get("stream"))
            except (json.JSONDecodeError, ValueError):
                stream = False

        self.stats["forwarded"] += 1

        if not stream:
            try:
                upstream = await self._proxy_client.request(
                    request.method, url, content=raw, headers=headers,
                    params=request.query_params)
            except httpx.HTTPError as exc:
                return Response(
                    content=json.dumps({"error": f"upstream unreachable: {exc}"}),
                    status_code=502, media_type="application/json")
            passthru = {k: v for k, v in upstream.headers.items()
                        if k.lower() not in ("content-length", "transfer-encoding",
                                             "connection", "content-encoding")}
            return Response(content=upstream.content, status_code=upstream.status_code,
                            headers=passthru, media_type=upstream.headers.get("content-type"))

        # Streaming: relay chunks as they arrive so tokens reach the client
        # with no added latency.
        async def relay():
            req = self._proxy_client.build_request(
                request.method, url, content=raw, headers=headers,
                params=request.query_params)
            resp = await self._proxy_client.send(req, stream=True)
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()

        return StreamingResponse(
            relay(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform",
                     "X-Accel-Buffering": "no"},
        )
