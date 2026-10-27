"""Thin LLM client. Pick a provider with environment variables (see .env.example).

Supported out of the box:
  LLM_PROVIDER=ollama     free, local (https://ollama.com) - no API key needed
  LLM_PROVIDER=anthropic  needs ANTHROPIC_API_KEY and `pip install anthropic`
  LLM_PROVIDER=openai     needs OPENAI_API_KEY and `pip install openai`
  LLM_PROVIDER=openai_compatible
                          any OpenAI-style gateway (LiteLLM, vLLM, ...). Uses only the standard library.
                          LLM_BASE_URL, LLM_API_KEY, optional LLM_EXTRA_HEADERS='{"header": "value"}'

You may replace this file, add native tool calling, or use a framework
(LangGraph, LlamaIndex, etc.). Keep the provider configurable via env vars.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

# minimal .env loader (no extra dependency)
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.split("#", 1)[0].strip()
        if "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


_PS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
try {
  $bytes = [IO.File]::ReadAllBytes($env:LLM_BODY_FILE)
  $h = @{ Authorization = "Bearer $env:LLM_API_KEY" }
  $r = Invoke-RestMethod -Uri $env:LLM_URL -Method Post -Headers $h -ContentType 'application/json' -Body $bytes -TimeoutSec ([int]$env:LLM_TIMEOUT)
  $r | ConvertTo-Json -Depth 20 -Compress
} catch {
  $m = $_.Exception.Message
  try {
    $resp = $_.Exception.Response
    if ($resp) {
      $sr = New-Object IO.StreamReader($resp.GetResponseStream())
      $m = $m + " | " + $sr.ReadToEnd()
    }
  } catch {}
  if ($_.ErrorDetails -and $_.ErrorDetails.Message) { $m = $m + " | " + $_.ErrorDetails.Message }
  [Console]::Error.WriteLine($m)
  exit 2
}
"""


def _post_powershell(url: str, body: bytes, timeout: float) -> dict:
    """Fallback transport for machines where Python's urllib stalls but PowerShell works."""
    import subprocess, tempfile
    bf = tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False); bf.write(body); bf.close()
    sf = tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False, encoding="utf-8"); sf.write(_PS_SCRIPT); sf.close()
    env = dict(os.environ, LLM_BODY_FILE=bf.name, LLM_URL=url, LLM_TIMEOUT=str(int(timeout)))
    try:
        p = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", sf.name],
                           capture_output=True, env=env, timeout=timeout + 15)
    finally:
        for f in (bf.name, sf.name):
            try: os.unlink(f)
            except OSError: pass
    if p.returncode != 0:
        raise RuntimeError("LLM gateway error: " + p.stderr.decode("utf-8", "replace")[:1200])
    return json.loads(p.stdout.decode("utf-8"))


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
        self.model = model or os.getenv("LLM_MODEL", {
            "ollama": "llama3.1:8b", "anthropic": "claude-sonnet-4-5", "openai": "gpt-4o-mini",
            "openai_compatible": "gpt-4o-mini"}[self.provider])
        self.calls = 0  # simple counter you can report for cost/latency

    def chat(self, messages: list[dict], system: str | None = None, max_tokens: int = 1024,
             temperature: float = 0.0) -> str:
        """messages: [{"role": "user"|"assistant", "content": str}] -> assistant text."""
        self.calls += 1
        if self.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(model=self.model, max_tokens=max_tokens, temperature=temperature,
                                          system=system or "", messages=messages)
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        if self.provider == "openai":
            from openai import OpenAI
            client = OpenAI()
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            resp = client.chat.completions.create(model=self.model, messages=msgs, max_tokens=max_tokens,
                                                  temperature=temperature)
            return resp.choices[0].message.content or ""
        if self.provider == "openai_compatible":
            base = os.environ["LLM_BASE_URL"].rstrip("/")
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ.get('LLM_API_KEY', '')}"}
            headers.update(json.loads(os.getenv("LLM_EXTRA_HEADERS", "{}") or "{}"))
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
            max_attempts = max(1, int(os.getenv("LLM_MAX_ATTEMPTS", "2")))
            transport = os.getenv("LLM_TRANSPORT", "powershell" if os.name == "nt" else "urllib").lower()
            min_gap = float(os.getenv("LLM_MIN_INTERVAL_SECONDS", "0"))
            models = [self.model] + [m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "").split(",") if m.strip()]
            last_err = None
            for attempt in range(max_attempts):  # bounded retries; 429s wait for the time the API asks for
                model = models[min(getattr(self, "_mi", 0), len(models) - 1)]
                body = json.dumps({"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature}).encode()
                gap = min_gap - (time.time() - getattr(self, "_last", 0))  # client-side pacing for per-minute limits
                if gap > 0:
                    time.sleep(gap)
                self._last = time.time()
                try:
                    if transport == "powershell":
                        data = _post_powershell(f"{base}/chat/completions", body, timeout)
                        text = data["choices"][0]["message"]["content"] or ""
                    else:
                        req = urllib.request.Request(f"{base}/chat/completions", data=body, headers=headers)
                        with urllib.request.urlopen(req, timeout=timeout) as r:
                            text = json.loads(r.read())["choices"][0]["message"]["content"] or ""
                    self._last = time.time()
                    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # drop reasoning blocks some models emit
                except urllib.error.HTTPError as e:
                    msg = f"HTTP {e.code}: {e.read()[:300]!r}"
                    last_err = RuntimeError(f"LLM gateway returned {msg}")
                    if e.code not in (429, 500, 502, 503, 504):
                        raise last_err from None
                    err_text = msg
                except (urllib.error.URLError, RuntimeError, OSError, TimeoutError) as e:
                    last_err = e
                    err_text = str(e)
                if attempt >= max_attempts - 1:
                    break
                if "429" in err_text or "RESOURCE_EXHAUSTED" in err_text or "quota" in err_text.lower():
                    if "per day" in err_text and getattr(self, "_mi", 0) < len(models) - 1:
                        self._mi = getattr(self, "_mi", 0) + 1  # daily cap hit: move to the next model
                        print(f"[llm] daily quota hit; switching to {models[self._mi]}", flush=True)
                        continue
                    m = re.search(r"retry in ([0-9.]+)s", err_text)
                    wait = min(float(m.group(1)) + 2, 90) if m else 30
                    print(f"[llm] rate limited; waiting {wait:.0f}s", flush=True)
                    time.sleep(wait)
                else:
                    time.sleep(min(5 * 2 ** attempt, 60))
            raise last_err
        if self.provider == "ollama":
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            body = json.dumps({"model": self.model, "messages": msgs, "stream": False,
                               "options": {"temperature": temperature, "num_predict": max_tokens}}).encode()
            req = urllib.request.Request(f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())["message"]["content"]
        raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")