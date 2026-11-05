#!/usr/bin/env python3
"""extract_source.py — turn a paper or lecture into plain-text chunks the agent can read.

Reading a large PDF straight into the model context (50–70k tokens) broke 3 of 5 runs in
the pilot eval: the API connection dropped mid-response. This script extracts text first
and splits it into page-aligned chunks, so the agent reads only what it needs.

Usage:
  extract_source.py SOURCE [--out DIR] [--chunk-chars 24000]

SOURCE is one of:
  - a local PDF;
  - a URL to a PDF;
  - an arXiv id or URL (2211.00593, arxiv.org/abs/2211.00593): the HTML version is tried
    first, the PDF is the fallback;
  - any other URL or a local .html file: cleaned with `defuddle` if it is installed,
    otherwise with a minimal tag stripper.

Writes DIR/source.txt (full text), DIR/chunk_NN.txt and DIR/index.md (chunk -> pages and
first heading). Prints the path of index.md. Depends on `pdftotext` (poppler-utils) for
PDFs; stdlib otherwise.
"""
import argparse
import html.parser
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request

UA = {"User-Agent": "paper-lab/0.1 (+https://github.com/olyamasaeva/paper-lab)"}
ARXIV_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf|html)/)?(\d{4}\.\d{4,5})(v\d+)?", re.I)


def fetch(url, dst):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)
        return r.headers.get("Content-Type", "")


def pdf_pages(path):
    if not shutil.which("pdftotext"):
        sys.exit("pdftotext not found: install poppler-utils")
    txt = subprocess.run(["pdftotext", "-layout", path, "-"], check=True,
                         capture_output=True, text=True).stdout
    return txt.split("\f")


class _Strip(html.parser.HTMLParser):
    SKIP = {"script", "style", "nav", "header", "footer"}

    def __init__(self):
        super().__init__()
        self.out, self.depth = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.depth += 1
        elif tag in {"p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth:
            self.depth -= 1

    def handle_data(self, data):
        if not self.depth:
            self.out.append(data)


def html_text(path_or_url):
    if shutil.which("defuddle"):
        r = subprocess.run(["defuddle", "parse", path_or_url, "--markdown"],
                           capture_output=True, text=True)
        if r.returncode == 0 and len(r.stdout) > 500:
            return r.stdout
    if path_or_url.startswith("http"):
        tmp = tempfile.mktemp(suffix=".html")
        fetch(path_or_url, tmp)
        path_or_url = tmp
    p = _Strip()
    p.feed(open(path_or_url, encoding="utf-8", errors="replace").read())
    return re.sub(r"\n{3,}", "\n\n", "".join(p.out))


def load(source, work):
    """Return a list of 'pages' (PDF pages, or one item for HTML)."""
    m = ARXIV_RE.fullmatch(source.strip()) or (
        "arxiv.org" in source and ARXIV_RE.search(source))
    if m:
        aid = m.group(1) + (m.group(2) or "")
        try:
            text = html_text(f"https://arxiv.org/html/{aid}")
            if len(text) > 5000:
                return [text]
        except Exception:
            pass
        source = f"https://arxiv.org/pdf/{aid}"
    if source.startswith("http"):
        dst = os.path.join(work, "download")
        ctype = fetch(source, dst)
        if "pdf" in ctype or open(dst, "rb").read(5) == b"%PDF-":
            return pdf_pages(dst)
        return [html_text(source)]
    if source.lower().endswith(".pdf"):
        return pdf_pages(source)
    return [html_text(source)]


def chunk(pages, limit):
    chunks, cur, first = [], [], 1
    for i, page in enumerate(pages, start=1):
        if cur and sum(len(p) for p in cur) + len(page) > limit:
            chunks.append((first, i - 1, "\f".join(cur)))
            cur, first = [], i
        cur.append(page)
    if cur:
        chunks.append((first, len(pages), "\f".join(cur)))
    # an HTML source is one long "page": split it on paragraph boundaries
    out = []
    for a, b, text in chunks:
        while len(text) > limit * 1.5:
            cut = text.rfind("\n\n", 0, limit)
            cut = cut if cut > limit // 2 else limit
            out.append((a, b, text[:cut]))
            text = text[cut:]
        out.append((a, b, text))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source")
    ap.add_argument("--out", default=None, help="output dir (default: a temp dir)")
    ap.add_argument("--chunk-chars", type=int, default=24000)
    a = ap.parse_args()
    out = a.out or tempfile.mkdtemp(prefix="paper-lab-src-")
    os.makedirs(out, exist_ok=True)
    pages = load(a.source, out)
    full = "\f".join(pages)
    if len(full.strip()) < 200:
        sys.exit("extracted almost no text: scanned PDF? run OCR first")
    open(os.path.join(out, "source.txt"), "w", encoding="utf-8").write(full)
    kind = f"{len(pages)} pages" if len(pages) > 1 else "html"
    lines = [f"# {a.source}\n", f"{kind}, {len(full)} chars\n"]
    for n, (p0, p1, text) in enumerate(chunk(pages, a.chunk_chars), start=1):
        name = f"chunk_{n:02d}.txt"
        open(os.path.join(out, name), "w", encoding="utf-8").write(text)
        head = next((l.strip() for l in text.splitlines() if len(l.strip()) > 3), "")[:80]
        span = f"pages {p0}–{p1}" if len(pages) > 1 else "html"
        lines.append(f"- `{name}` · {span} · {head}")
    idx = os.path.join(out, "index.md")
    open(idx, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(idx)


if __name__ == "__main__":
    main()
