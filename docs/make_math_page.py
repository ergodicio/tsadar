"""Generates docs/source/math_full.rst, the Read the Docs rendering of docs/tsadar_math.tex.

Run from the repository root after editing the LaTeX source:

    python docs/make_math_page.py            # regenerate the page
    python docs/make_math_page.py --figures  # also re-render the TikZ figures

Requires pandoc (``pip install pypandoc_binary``). ``--figures`` additionally needs the ``tectonic``
LaTeX engine on PATH (or given with ``--tectonic``) and ``pymupdf``; the rendered figures are committed in
docs/source/_elfolder, so this is only needed when a TikZ picture changes.
"""
import argparse
import os
import re
import shutil
import subprocess
import tempfile

import pypandoc

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "tsadar_math.tex")
OUT = os.path.join(HERE, "source", "math_full.rst")
FIG_DIR = os.path.join(HERE, "source", "_elfolder")

SOURCE_URL = "https://github.com/ergodicio/tsadar/blob/main/docs/tsadar_math.tex"

STANDALONE = r"""\documentclass[tikz,border=4pt]{standalone}
\usepackage{amsmath,amssymb}
\usepackage{tikz-3dplot}
\begin{document}
%s
\end{document}
"""


MACROS = ""


def to_rst(latex: str) -> str:
    """Converts a LaTeX fragment to RST, resolving the document's macros and citations."""
    rst = pypandoc.convert_text(MACROS + latex, "rst", format="latex", extra_args=["--wrap=none"])
    rst = rst.replace("\r\n", "\n").strip().replace(r"\texttt{", r"\mathtt{")
    rst = re.sub(r":raw-latex:`\\cite\{([^}]*)\}`", lambda m: " ".join(f"[{k.strip()}]_" for k in m.group(1).split(",")), rst)
    rst = re.sub(r":math:`[^`]*`", lambda m: re.sub(r"\s*\n\s*", " ", m.group(0)), rst)
    return re.sub(r"\*{1,2}(``[^*\n]*``)\*{1,2}", r"\1", rst)


def braced(text: str, start: int) -> tuple:
    """Returns (content, end) for the brace group opening at text[start] == "{"."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{" and text[i - 1] != "\\":
            depth += 1
        elif text[i] == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
    raise ValueError("unbalanced braces")


def command_args(text: str, command: str) -> list:
    """Brace-group arguments of every occurrence of \\command{...} in text."""
    out = []
    for m in re.finditer(re.escape("\\" + command) + r"\s*\{", text):
        out.append(braced(text, m.end() - 1)[0])
    return out


def label(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")


def render_tikz(picture: str, name: str, tectonic: str) -> None:
    import fitz

    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "fig.tex"), "w", encoding="utf-8") as f:
            f.write(STANDALONE % picture)
        subprocess.run([tectonic, "fig.tex"], cwd=td, check=True)
        with fitz.open(os.path.join(td, "fig.pdf")) as pdf:
            pdf[0].get_pixmap(dpi=220).save(os.path.join(FIG_DIR, name))


def extract_figures(body: str, render: bool, tectonic: str) -> tuple:
    """Replaces every figure environment with a placeholder paragraph and returns the RST for each."""
    figures = []
    labels = {}
    tikz_count = 0

    def _figure(match):
        nonlocal tikz_count
        fig = match.group(0)
        number = len(figures) + 1
        panels = re.findall(r"\\begin\{subfigure\}.*?\\end\{subfigure\}", fig, flags=re.S) or [fig]
        multi = len(panels) > 1
        remainder = fig
        images, captions = [], []
        for letter, panel in zip("abcdefgh", panels):
            remainder = remainder.replace(panel, "") if multi else ""
            graphic = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", panel)
            if graphic:
                images.append(graphic.group(1))
            else:
                tikz_count += 1
                name = f"math_tikz_{tikz_count}.png"
                if render:
                    picture = re.search(r"\\tdplotsetmaincoords.*?\\end\{tikzpicture\}", panel, flags=re.S).group(0)
                    render_tikz(picture, name, tectonic)
                images.append(name)
            caption = command_args(panel, "caption")
            if caption:
                captions.append((f"({letter}) " if multi else "") + to_rst(caption[0]).replace("\n", " "))
            for name in command_args(panel, "label"):
                labels[name] = (number, label(name) if name.startswith("fig:") else f"fig-{label(name)}")
        captions += [to_rst(c).replace("\n", " ") for c in command_args(remainder, "caption")]
        for name in command_args(remainder, "label"):
            labels[name] = (number, label(name) if name.startswith("fig:") else f"fig-{label(name)}")

        targets = sorted({target for n, target in labels.values() if n == number})
        lines = [f".. _{t}:\n" for t in targets]
        width = "48%" if multi else "60%"
        for image in images[:-1]:
            lines += [f".. image:: _elfolder/{image}", f"   :width: {width}", ""]
        lines += [f".. figure:: _elfolder/{images[-1]}", f"   :width: {width}", ""]
        lines += [f"   **Figure {number}.** " + " ".join(captions), ""]
        figures.append("\n".join(lines))
        return f"\n\nFIGUREPLACEHOLDER{number}\n\n"

    body = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", _figure, body, flags=re.S)
    return body, figures, labels


def fix_math(tex: str) -> str:
    return re.sub(r"\\si\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", lambda m: r"\mathrm{" + m.group(1).replace(".", r"\,") + "}", tex)


def number_headings(lines: list) -> list:
    levels = {"=": 0, "-": 1, "~": 2}
    counters = [0, 0, 0]
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip() and nxt and len(set(nxt)) == 1 and nxt[0] in "=-~^" and len(nxt) >= len(line):
            if nxt[0] == "^":
                out += [f".. rubric:: {line}", ""]
            else:
                level = levels[nxt[0]]
                counters[level] += 1
                counters[level + 1 :] = [0] * (2 - level)
                title = ".".join(str(c) for c in counters[: level + 1]) + " " + line
                out += [title, nxt[0] * len(title)]
            i += 2
            continue
        out.append(line)
        i += 1
    return out


def fix_math_blocks(lines: list, eq_labels: dict) -> list:
    out = []
    i = 0
    while i < len(lines):
        if lines[i].strip() != ".. math::":
            out.append(lines[i])
            i += 1
            continue
        j = i + 1
        block = []
        while j < len(lines) and (not lines[j].strip() or lines[j].startswith("   ")):
            block.append(lines[j])
            j += 1
        while block and not block[-1].strip():
            block.pop()
            j -= 1
        text = "\n".join(block)
        found = re.search(r"\\label\{([^}]*)\}", text)
        text = re.sub(r"[ \t]*\\(?:begin|end)\{equation\}[ \t]*\n?", "", text)
        text = re.sub(r"[ \t]*\\label\{[^}]*\}[ \t]*\n?", "", text)
        out.append(".. math::")
        if found:
            out.append(f"   :label: {eq_labels[found.group(1)]}")
        out.append("")
        out += [l for l in text.split("\n") if l.strip()]
        i = j
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figures", action="store_true", help="re-render the TikZ figures")
    parser.add_argument("--tectonic", default=shutil.which("tectonic") or "tectonic", help="path to tectonic")
    args = parser.parse_args()

    with open(TEX, encoding="utf-8") as f:
        tex = f.read()

    global MACROS
    MACROS = "\n".join(re.findall(r"^\\newcommand.*$", tex, flags=re.M)) + "\n"
    title = command_args(tex, "title")[0]
    body = tex.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0]
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, flags=re.S).group(1)
    bibliography = re.search(r"\\begin\{thebibliography\}\{[^}]*\}(.*?)\\end\{thebibliography\}", body, flags=re.S).group(1)
    for pattern in (
        r"\\maketitle",
        r"\\tableofcontents",
        r"\\begin\{center\}\s*\\fbox.*?\\end\{center\}",
        r"\\begin\{abstract\}.*?\\end\{abstract\}",
        r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
    ):
        body = re.sub(pattern, "", body, count=1, flags=re.S)

    eq_labels = {name: label(name) for name in re.findall(r"\\label\{([^}]*)\}", body) if not name.startswith(("sec:", "fig:"))}
    body, figures, fig_labels = extract_figures(body, args.figures, args.tectonic)
    for name in fig_labels:
        eq_labels.pop(name, None)

    rst = to_rst(fix_math(body))

    def _ref(match):
        text, target = match.group(1).strip("[]"), match.group(2)
        if target in eq_labels:
            return f":eq:`{eq_labels[target]}`"
        if target in fig_labels:
            number, anchor = fig_labels[target]
            return f":ref:`{number} <{anchor}>`"
        return f":ref:`{text} <{target}>`"

    rst = re.sub(r"`([^`<]*?) <#([^>]*)>`__", _ref, rst)
    for number, figure in enumerate(figures, start=1):
        rst = rst.replace(f"FIGUREPLACEHOLDER{number}", figure)

    lines = fix_math_blocks(number_headings(rst.split("\n")), eq_labels)

    references = []
    for entry in re.split(r"\\bibitem", bibliography)[1:]:
        key, end = braced(entry, entry.index("{"))
        references.append(f".. [{key}] " + to_rst(entry[end:]).replace("\n", " "))

    header = [
        f".. Generated from docs/tsadar_math.tex by docs/make_math_page.py. Do not edit this file directly.",
        "",
        title,
        "#" * len(title),
        "",
        ".. note::",
        "",
        "   **Work in progress.** This document is under active development. Sections may be incomplete or",
        "   lag behind the code, and it has not yet been fully reviewed; where the two disagree, the code is",
        f"   authoritative. This page is generated from the `LaTeX source <{SOURCE_URL}>`_.",
        "",
        to_rst(abstract),
        "",
    ]
    footer = ["", "References", "==========", ""] + [line for ref in references for line in (ref, "")]
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(header + lines + footer))
    print(f"wrote {os.path.relpath(OUT)}: {len(header + lines + footer)} lines, {len(figures)} figures")


if __name__ == "__main__":
    main()
