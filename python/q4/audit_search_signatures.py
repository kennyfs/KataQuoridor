#!/usr/bin/env python3
"""Compare parameter names and order of the function definitions in cpp/q4/search/ with their counterparts in
cpp/search/ (Q4 file `q4foo.cpp` <-> KataGo `foo.cpp`).  Only column-0 definitions are parsed, so call sites are not
picked up. An overloaded Q4 function is compared with the closest KataGo overload.  Use --calls NAME to list the call sites of a function in the Q4 tree for a manual argument-order check.

    python python/q4/audit_search_signatures.py            # table of differences
    python python/q4/audit_search_signatures.py --all      # also list functions whose signatures agree
    python python/q4/audit_search_signatures.py --calls getReducedPlaySelectionWeight
"""
import argparse
import glob
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DEF_START = re.compile(r"^(?:template\s*<[^>]*>\s*)?(?:static\s+|inline\s+)*[A-Za-z_][\w:<>,\*&\s]*?[\s\*&]((?:\w+::)*~?\w+)\(")


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def split_params(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "(<[{":
            depth += 1
        elif ch in ")>]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def param_name(p):
    p = re.sub(r"=.*$", "", p.strip(), flags=re.S)
    p = re.sub(r"\[[^\]]*\]\s*$", "", p).strip()  # array suffix
    m = re.search(r"(\w+)\s*$", p)
    return m.group(1) if m else p


def parse_file(path):
    text = strip_comments(open(path).read())
    defs = {}
    for m in DEF_START.finditer(text.replace("\r", "")):
        pass
    lines = text.split("\n")
    pos = 0
    offsets = []
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1
    for i, ln in enumerate(lines):
        m = DEF_START.match(ln)
        if not m or ln.startswith((" ", "\t", "#", "}")) or "=" in ln.split("(")[0]:
            continue
        start = offsets[i] + m.end()
        depth, j = 1, start
        while j < len(text) and depth:
            depth += {"(": 1, ")": -1}.get(text[j], 0)
            j += 1
        tail = text[j:j + 40].lstrip()
        if not re.match(r"(const\s*)?(noexcept\s*)?(override\s*)?\{", tail):
            continue
        params = [param_name(p) for p in split_params(text[start:j - 1])]
        defs.setdefault(m.group(1), []).append((params, os.path.basename(path), i + 1))
    return defs


def collect(files):
    allf = {}
    for f in files:
        for name, vs in parse_file(f).items():
            allf.setdefault(name, []).extend(vs)
    return allf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--calls")
    a = ap.parse_args()
    q4 = sorted(glob.glob(os.path.join(ROOT, "cpp/q4/search/*.cpp")))
    kg = sorted(glob.glob(os.path.join(ROOT, "cpp/search/*.cpp")))
    if a.calls:
        pat = re.compile(r"\b" + re.escape(a.calls) + r"\s*\(")
        for f in q4:
            text = strip_comments(open(f).read())
            for m in pat.finditer(text):
                ls = text.rfind("\n", 0, m.start()) + 1
                if "::" in text[ls:m.start()]:
                    continue  # the definition (return type + Search::)
                depth, j = 1, m.end()
                while j < len(text) and depth:
                    depth += {"(": 1, ")": -1}.get(text[j], 0)
                    j += 1
                line = text.count("\n", 0, m.start()) + 1
                print(f"{os.path.basename(f)}:{line}: " + " ".join(text[ls:j].split()))
        return
    dq, dk = collect(q4), collect(kg)
    same = diff = only_q4 = 0
    print("| function | KataGo parameters | Q4 parameters | note |\n|---|---|---|---|")
    for name in sorted(dq):
        for params, fn, ln in dq[name]:
            kgs = dk.get(name)
            if not kgs:
                only_q4 += 1
                continue
            best = min(kgs, key=lambda k: (k[0] != params, abs(len(k[0]) - len(params))))
            kp = best[0]
            if kp == params:
                same += 1
                if a.all:
                    print(f"| `{name}` | = | = | {fn}:{ln} |")
                continue
            diff += 1
            common_kg = [x for x in kp if x in params]
            common_q4 = [x for x in params if x in kp]
            note = "added: " + ",".join(sorted(set(params) - set(kp))) + " removed: " + ",".join(sorted(set(kp) - set(params)))
            if common_kg != common_q4:
                note = "**REORDERED** " + note
            print(f"| `{name}` | {', '.join(kp)} | {', '.join(params)} | {note} ({fn}:{ln}) |")
    print(f"\n{same} identical, {diff} different, {only_q4} Q4-only definitions", file=sys.stderr)


if __name__ == "__main__":
    main()
