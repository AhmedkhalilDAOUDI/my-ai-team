import asyncio
import html
import ipaddress
import json
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

MAX_DOWNLOAD_BYTES = 2_000_000
MAX_SOURCE_CHARACTERS = 200_000
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}


class SourceImportError(ValueError):
    pass


def _validate_public_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise SourceImportError("Enter a public http:// or https:// URL.")
    host = parsed.hostname.lower().rstrip(".")
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        raise SourceImportError("Local and private network URLs are not allowed.")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)}
    except socket.gaierror as exc:
        raise SourceImportError("The website address could not be resolved.") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise SourceImportError("Local and private network URLs are not allowed.")
    return parsed.geturl()


class _ReadableHTML(HTMLParser):
    def __init__(self):
        super().__init__();self.parts=[];self.title=[];self.skip=0;self.in_title=False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg", "canvas"}:self.skip+=1
        if tag=="title":self.in_title=True
        if tag in {"p", "div", "article", "section", "main", "li", "h1", "h2", "h3", "h4", "br", "blockquote"}:self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg", "canvas"} and self.skip:self.skip-=1
        if tag=="title":self.in_title=False

    def handle_data(self, data):
        if self.skip:return
        value=html.unescape(data).strip()
        if not value:return
        if self.in_title:self.title.append(value)
        self.parts.append(value+" ")

    def result(self):
        text="".join(self.parts);text=re.sub(r"[ \t]+"," ",text);text=re.sub(r"\n\s*\n+","\n\n",text).strip()
        return " ".join(self.title).strip(),text


async def _fetch_page(url: str) -> dict:
    current=_validate_public_url(url)
    headers={"User-Agent":"My-AI-Team/1.0 (+local research tool)","Accept":"text/html,text/plain,application/xhtml+xml"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(25,connect=10),follow_redirects=False,headers=headers) as client:
        for _ in range(4):
            async with client.stream("GET",current) as response:
                if response.status_code in {301,302,303,307,308}:
                    location=response.headers.get("location")
                    if not location:raise SourceImportError("The website returned an invalid redirect.")
                    current=_validate_public_url(urljoin(current,location));continue
                response.raise_for_status()
                content_type=response.headers.get("content-type","").lower()
                if not any(kind in content_type for kind in ("text/html","text/plain","application/xhtml+xml")):
                    raise SourceImportError("This URL is not a readable webpage. Upload the file directly instead.")
                chunks=[];size=0
                async for chunk in response.aiter_bytes():
                    size+=len(chunk)
                    if size>MAX_DOWNLOAD_BYTES:raise SourceImportError("The webpage is too large to import safely.")
                    chunks.append(chunk)
                raw=b"".join(chunks).decode(response.encoding or "utf-8","replace")
                if "text/plain" in content_type:title=urlparse(current).path.rsplit("/",1)[-1] or urlparse(current).hostname;text=raw
                else:
                    parser=_ReadableHTML();parser.feed(raw);title,text=parser.result()
                text=text[:MAX_SOURCE_CHARACTERS].strip()
                if len(text)<50:raise SourceImportError("The website did not expose enough readable text.")
                return {"title":title or urlparse(current).hostname,"url":current,"source_type":"webpage","text":text}
        raise SourceImportError("The website redirected too many times.")


def _clean_vtt(value: str) -> str:
    lines=[];previous=""
    for line in value.splitlines():
        line=line.strip()
        if not line or line.startswith(("WEBVTT","Kind:","Language:","NOTE")) or "-->" in line or line.isdigit():continue
        line=html.unescape(re.sub(r"<[^>]+>","",line));line=re.sub(r"\s+"," ",line).strip()
        if line and line!=previous:lines.append(line);previous=line
    return "\n".join(lines)[:MAX_SOURCE_CHARACTERS]


def _youtube_sync(url: str) -> dict:
    executable=shutil.which("yt-dlp")
    command=([executable] if executable else [sys.executable,"-m","yt_dlp"])+["--no-playlist","--skip-download","--write-subs","--write-auto-subs","--sub-langs","en.*,en,fr.*,fr,ar.*,ar","--sub-format","vtt","--dump-single-json","--no-warnings"]
    with tempfile.TemporaryDirectory(prefix="my-ai-team-source-") as directory:
        command.extend(["-o",str(Path(directory)/"%(id)s.%(ext)s"),url])
        try:result=subprocess.run(command,capture_output=True,text=True,timeout=90,check=False)
        except (subprocess.TimeoutExpired,OSError) as exc:raise SourceImportError("YouTube caption extraction timed out or is unavailable.") from exc
        if result.returncode!=0:
            detail=(result.stderr or "YouTube could not provide this video.").strip().splitlines()[-1]
            raise SourceImportError(f"YouTube import failed: {detail[:300]}")
        try:metadata=json.loads(result.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError,IndexError) as exc:raise SourceImportError("YouTube returned invalid video metadata.") from exc
        files=sorted(Path(directory).glob("*.vtt"),key=lambda path:path.stat().st_size,reverse=True)
        if not files:raise SourceImportError("This YouTube video has no accessible captions. Add captions or upload a transcript.")
        transcript=_clean_vtt(files[0].read_text(encoding="utf-8",errors="replace"))
        if len(transcript)<50:raise SourceImportError("The available YouTube captions were empty or unreadable.")
        details=[f"Title: {metadata.get('title') or 'YouTube video'}",f"Channel: {metadata.get('uploader') or metadata.get('channel') or 'Unknown'}"]
        if metadata.get("duration") is not None:details.append(f"Duration seconds: {metadata['duration']}")
        return {"title":metadata.get("title") or "YouTube video","url":metadata.get("webpage_url") or url,"source_type":"youtube","text":"\n".join(details)+"\n\nTranscript:\n"+transcript}


async def extract_web_source(url: str) -> dict:
    safe_url=_validate_public_url(url)
    host=(urlparse(safe_url).hostname or "").lower()
    if host in YOUTUBE_HOSTS:return await asyncio.to_thread(_youtube_sync,safe_url)
    return await _fetch_page(safe_url)
