#!/usr/bin/env python3
"""build_notebook.py — build a valid Jupyter .ipynb from a JSON spec.

Spec (file or stdin):
{
  "title": "...",
  "outfile": "path/to/notebook.ipynb",   # --out overrides it
  "kernel": "python3",                     # optional
  "cells": [
     {"type":"md",       "src":"# heading\\ntext"},
     {"type":"code",     "src":"import torch", "tag":"setup"},
     {"type":"solution", "src":"# YOUR CODE HERE\\nraise NotImplementedError",
                          "id":"ex1", "points":2},
     {"type":"test",     "src":"assert foo()==42", "id":"ex1", "points":2}
  ]
}

Cell types:
  md        — markdown
  code      — a regular code cell (given, not an exercise)
  solution  — an exercise: the student writes code (nbgrader solution cell)
  test      — the check for an exercise (nbgrader autograder cell, locked)

nbgrader metadata is set automatically, so the notebook works both as a plain
notebook (just run the assert cells) and under `nbgrader autograde`.
"""
import argparse, json, sys


def cell_id(i):
    return f"c{i:04d}"


def _md_norm(s):
    # Jupyter renders $...$ / $$...$$, not \(...\) / \[...\]: normalise the delimiters
    s = s.replace("\\(", "$").replace("\\)", "$")
    s = s.replace("\\[", "$$").replace("\\]", "$$")
    return s


def md_cell(i, src):
    return {
        "cell_type": "markdown",
        "id": cell_id(i),
        "metadata": {},
        "source": _md_norm(src).splitlines(keepends=True) or [""],
    }


def _nbgrader(grade, solution, locked, gid, points, task=False):
    return {
        "grade": grade, "solution": solution, "locked": locked,
        "grade_id": gid, "points": points, "task": task, "schema_version": 3,
    }


def code_cell(i, src, meta=None):
    return {
        "cell_type": "code",
        "id": cell_id(i),
        "metadata": meta or {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True) or [""],
    }


def build(spec, out_override=None):
    cells = []
    for i, c in enumerate(spec.get("cells", []), start=1):
        t = c.get("type", "md")
        src = c.get("src", "")
        if t == "md":
            cells.append(md_cell(i, src))
        elif t == "code":
            meta = {}
            if c.get("tag"):
                meta["tags"] = [c["tag"]]
            cells.append(code_cell(i, src, meta))
        elif t == "solution":
            gid = c.get("id", f"sol{i}")
            pts = c.get("points", 1)
            meta = {"nbgrader": _nbgrader(False, True, False, gid, pts),
                    "tags": ["exercise"]}
            cells.append(code_cell(i, src, meta))
        elif t == "test":
            gid = c.get("id", f"sol{i}") + "_test"
            pts = c.get("points", 1)
            meta = {"nbgrader": _nbgrader(True, False, True, gid, pts),
                    "tags": ["test"]}
            cells.append(code_cell(i, src, meta))
        else:
            raise SystemExit(f"unknown cell type: {t!r}")

    kernel = spec.get("kernel", "python3")
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": kernel},
            "language_info": {"name": "python"},
            "paper_lab": {"title": spec.get("title", "")},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out = out_override or spec.get("outfile")
    if not out:
        raise SystemExit("no outfile given (spec.outfile or --out)")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    n_ex = sum(1 for c in spec.get("cells", []) if c.get("type") == "solution")
    print(f"wrote {out} — {len(cells)} cells, {n_ex} exercises")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", help="path to the JSON spec (stdin if omitted)")
    ap.add_argument("--out", help="override outfile")
    a = ap.parse_args()
    raw = open(a.spec, encoding="utf-8").read() if a.spec else sys.stdin.read()
    build(json.loads(raw), a.out)


if __name__ == "__main__":
    main()
