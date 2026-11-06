#!/usr/bin/env python3
"""extract_source.py — turn a paper or lecture into plain-text chunks the agent can read.

Reading a large PDF straight into the model context (50–70k tokens) broke 3 of 5 runs in
the pilot eval: the API connection dropped mid-response. This script extracts text first
and splits it into page-aligned chunks, so the agent reads only what it needs.

Usage:
  extract_source.py SOURCE [--out DIR] [--chunk-chars 24000] [--force]
                    [--max-mb 100] [--max-pages 400] [--allow-private]

SOURCE is one of:
  - a local PDF or .html file;
  - an http(s) URL to a PDF or a web page;
  - an arXiv id or URL (2211.00593, 2211.00593v2, hep-th/9901001, arxiv.org/abs/...):
    the HTML version is tried first, the PDF is the fallback.

Web pages are cleaned with `defuddle` if it is installed, otherwise with a minimal tag
stripper. Everything is downloaded once, by this script, into a private temp dir; the
cleaners only ever see local files.

Safety limits: http(s) only; hosts that resolve to private, loopback, link-local or
reserved addresses are refused on the first request and on every redirect (override with
--allow-private); downloads are capped in size and total time; external tools run with a
timeout. The extracted text is untrusted data: it is written for reading, never executed.

Writes DIR/source.txt (full text), DIR/chunk_NN.txt and DIR/index.md (chunk -> pages and
first line). Refuses to overwrite an existing extraction unless --force. Prints the path of
index.md. Depends on `pdftotext` (poppler-utils) for PDFs; stdlib otherwise.
"""
import argparse
import html.parser
import ipaddress
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "paper-lab/0.1 (+https://github.com/olyamasaeva/paper-lab)"}
ARXIV_ID = r"(?:\d{4}\.\d{4,5}|[a-z][a-z.-]*(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?"
ARXIV_BARE = re.compile(ARXIV_ID + r"$")
ARXIV_PATH = re.compile(r"/(?:abs|pdf|html)/(" + ARXIV_ID + r")(?:\.pdf)?/?$")
OUR_FILES = re.compile(r"^(source\.txt|index\.md|chunk_\d+\.txt)$")
PDFTOTEXT_TIMEOUT = 300
DEFUDDLE_TIMEOUT = 120


class Limits:
    max_bytes = 100 * 2**20
    deadline_s = 180
    allow_private = False


def check_url(url):
    """Refuse non-http(s) schemes and hosts that resolve to internal addresses."""
    u = urllib.parse.urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise SystemExit(f"refusing {url!r}: only http(s) URLs are fetched")
    if Limits.allow_private:
        return
    try:
        infos = socket.getaddrinfo(u.hostname, u.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise SystemExit(f"cannot resolve {u.hostname}: {e}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            raise SystemExit(f"refusing {url!r}: {u.hostname} resolves to {ip}, an internal "
                             "address (pass --allow-private if you mean it)")


class _CheckedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_CheckedRedirect)


def fetch(url, dst):
    """Download url to dst with a size cap and a total deadline. Returns (ctype, charset)."""
    check_url(url)
    start = time.monotonic()
    req = urllib.request.Request(url, headers=UA)
    got = 0
    try:
        with _opener.open(req, timeout=30) as r, open(dst, "wb") as f:
            ctype = r.headers.get_content_type() or ""
            charset = r.headers.get_content_charset()
            while True:
                block = r.read(1 << 16)
                if not block:
                    break
                got += len(block)
                if got > Limits.max_bytes:
                    raise SystemExit(f"{url}: larger than {Limits.max_bytes >> 20} MB (see --max-mb)")
                if time.monotonic() - start > Limits.deadline_s:
                    raise SystemExit(f"{url}: download took longer than {Limits.deadline_s} s")
                f.write(block)
        return ctype, charset
    except BaseException:
        if os.path.exists(dst):
            os.remove(dst)
        raise


def pdf_pages(path, max_pages):
    if not shutil.which("pdftotext"):
        sys.exit("pdftotext not found: install poppler-utils")
    path = os.path.abspath(path)  # never starts with "-", so it can't be read as an option
    try:
        txt = subprocess.run(["pdftotext", "-layout", "-l", str(max_pages), path, "-"],
                             check=True, capture_output=True, text=True, errors="replace",
                             timeout=PDFTOTEXT_TIMEOUT).stdout
    except subprocess.TimeoutExpired:
        sys.exit(f"pdftotext took longer than {PDFTOTEXT_TIMEOUT} s on {path}")
    pages = txt.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()  # pdftotext ends every page with a form feed
    return pages


class _Strip(html.parser.HTMLParser):
    """Minimal fallback: drop navigation and scripts, keep text and inline maths."""
    SKIP = {"script", "style", "nav", "header", "footer", "noscript"}
    BLOCK = {"p", "br", "div", "li", "h1", "h2", "h3", "h4", "h5", "tr", "section", "pre"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.stack = [], []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            kind = dict(attrs).get("type") or ""
            if tag == "script" and "math" in kind:  # MathJax: <script type="math/tex">
                self.out.append(" $")
                self.stack.append("math")
                return
            self.stack.append(tag)
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag == "script" and self.stack and self.stack[-1] == "math":
            self.stack.pop()
            self.out.append("$ ")
        elif tag in self.stack:  # tolerate bad nesting: close up to the matching tag
            while self.stack and self.stack.pop() != tag:
                pass

    def handle_data(self, data):
        if not self.stack or self.stack[-1] == "math":
            self.out.append(data)


def _sniff_charset(raw):
    m = re.search(rb"<meta[^>]+charset=[\"']?([\w-]+)", raw[:4096], re.I)
    return m.group(1).decode("ascii", "ignore") if m else None


def html_text(path, charset=None):
    """Clean a LOCAL html file. Never fetches anything itself."""
    path = os.path.abspath(path)
    if shutil.which("defuddle"):
        try:
            r = subprocess.run(["defuddle", "parse", path, "--markdown"],
                               capture_output=True, text=True, errors="replace",
                               timeout=DEFUDDLE_TIMEOUT)
            if r.returncode == 0 and len(r.stdout) > 500:
                return r.stdout
        except subprocess.TimeoutExpired:
            pass
    raw = open(path, "rb").read()
    enc = charset or _sniff_charset(raw) or "utf-8"
    try:
        text = raw.decode(enc, errors="replace")
    except LookupError:
        text = raw.decode("utf-8", errors="replace")
    p = _Strip()
    p.feed(text)
    return re.sub(r"\n{3,}", "\n\n", "".join(p.out))


def arxiv_id(source):
    s = source.strip()
    if ARXIV_BARE.match(s):
        return s
    u = urllib.parse.urlsplit(s if "://" in s else "https://" + s)
    host = (u.hostname or "").lower()
    if host == "arxiv.org" or host.endswith(".arxiv.org"):
        m = ARXIV_PATH.search(u.path)
        if m:
            return m.group(1)
    return None


def load(source, work, max_pages):
    """Return (kind, pages): kind is 'pdf' or 'html'; html is one long page."""
    aid = arxiv_id(source)
    if aid:
        dst = os.path.join(work, "arxiv.html")
        try:
            ctype, cs = fetch(f"https://arxiv.org/html/{aid}", dst)
            if "html" in ctype:
                text = html_text(dst, cs)
                if len(text) > 5000:
                    return "html", [text]
        except (urllib.error.URLError, OSError):
            pass
        source = f"https://arxiv.org/pdf/{aid}"
    if re.match(r"^[a-z][a-z0-9+.-]*://", source, re.I):
        dst = os.path.join(work, "download")
        ctype, cs = fetch(source, dst)
        with open(dst, "rb") as f:
            is_pdf = f.read(5) == b"%PDF-"
        if is_pdf or "pdf" in ctype:
            return "pdf", pdf_pages(dst, max_pages)
        return "html", [html_text(dst, cs)]
    if not os.path.isfile(source):
        sys.exit(f"{source!r} is neither a file, a URL nor an arXiv id")
    with open(source, "rb") as f:
        if f.read(5) == b"%PDF-":
            return "pdf", pdf_pages(source, max_pages)
    return "html", [html_text(source)]


def chunk(pages, limit):
    """Page-aligned chunks of about `limit` chars; long single pages split on paragraphs."""
    assert limit >= 1
    chunks, cur, first = [], [], 1
    for i, page in enumerate(pages, start=1):
        if cur and sum(len(p) for p in cur) + len(page) > limit:
            chunks.append((first, i - 1, "\f".join(cur)))
            cur, first = [], i
        cur.append(page)
    if cur:
        chunks.append((first, len(pages), "\f".join(cur)))
    out = []
    for a, b, text in chunks:
        while len(text) > limit * 1.5:
            cut = text.rfind("\n\n", 0, limit)
            cut = cut if cut > limit // 2 else limit
            out.append((a, b, text[:cut]))
            text = text[cut:]
        out.append((a, b, text))
    return out


def positive_int(s):
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return v


def prepare_out(out, force):
    os.makedirs(out, exist_ok=True)
    ours = [n for n in os.listdir(out) if OUR_FILES.match(n)]
    if ours and not force:
        sys.exit(f"{out} already holds an extraction ({len(ours)} files); pass --force to replace it")
    for n in ours:  # os.remove on a symlink removes the link, not its target
        os.remove(os.path.join(out, n))


def write_new(path, text):
    with open(path, "x", encoding="utf-8") as f:  # "x": never follows a planted file
        f.write(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source")
    ap.add_argument("--out", default=None, help="output dir (default: a new temp dir)")
    ap.add_argument("--chunk-chars", type=positive_int, default=24000)
    ap.add_argument("--force", action="store_true", help="replace an existing extraction in --out")
    ap.add_argument("--max-mb", type=positive_int, default=100, help="download size cap")
    ap.add_argument("--max-pages", type=positive_int, default=400, help="PDF pages to extract")
    ap.add_argument("--allow-private", action="store_true",
                    help="allow URLs that resolve to private or loopback addresses")
    a = ap.parse_args()
    Limits.max_bytes = a.max_mb * 2**20
    Limits.allow_private = a.allow_private

    out = a.out or tempfile.mkdtemp(prefix="paper-lab-src-")
    prepare_out(out, a.force)
    work = tempfile.mkdtemp(prefix="paper-lab-dl-")
    try:
        kind, pages = load(a.source, work, a.max_pages)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    full = "\f".join(pages)
    if len(full.strip()) < 200:
        sys.exit("extracted almost no text: scanned PDF? run OCR first")
    write_new(os.path.join(out, "source.txt"), full)
    head = f"{len(pages)} pages" if kind == "pdf" else "html"
    lines = [f"# {a.source}\n", f"{head}, {len(full)} chars\n"]
    for n, (p0, p1, text) in enumerate(chunk(pages, a.chunk_chars), start=1):
        name = f"chunk_{n:02d}.txt"
        write_new(os.path.join(out, name), text)
        first = next((l.strip() for l in text.splitlines() if len(l.strip()) > 3), "")[:80]
        span = f"pages {p0}–{p1}" if kind == "pdf" else "html"
        lines.append(f"- `{name}` · {span} · {first}")
    idx = os.path.join(out, "index.md")
    write_new(idx, "\n".join(lines) + "\n")
    print(idx)


if __name__ == "__main__":
    main()
