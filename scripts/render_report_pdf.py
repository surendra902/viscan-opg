"""Render REPORT.md to a four-page A4 PDF and prove the page count.

Chrome lays the document out; PyMuPDF reads the result back and prints where the
content of each page ends, so a change that pushes the report onto a fifth page
fails visibly instead of shipping quietly.

Images are rewritten to absolute file URIs before printing, because the HTML is
written to a temporary directory and a relative src would render as a broken box.

Usage:
    python scripts/render_report_pdf.py                 # writes REPORT.pdf
    python scripts/render_report_pdf.py --output /tmp/report.pdf
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
MAX_PAGES = 4

CSS = """
@page { size: A4; margin: 11mm 11mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 9pt; line-height: 1.18; color: #111; }
h1 { font-size: 15pt; margin: 0 0 2pt; }
h2 { font-size: 11pt; margin: 4pt 0 1.5pt; border-bottom: 1px solid #ccc; break-after: avoid; }
h3 { font-size: 9.5pt; margin: 6pt 0 2pt; }
p { margin: 0 0 1.5pt; }
p > em { font-size: 8.2pt; line-height: 1.15; color: #333; }
table { border-collapse: collapse; width: 100%; font-size: 7.6pt; margin: 1.5pt 0 2.5pt; }
th, td { border: 1px solid #bbb; padding: 1.2pt 3pt; text-align: left; }
th { background: #f0f0f0; }
code { font-family: Consolas, monospace; font-size: 8pt; }
pre { background: #f6f6f6; padding: 3.5pt; font-size: 8pt; line-height: 1.18; }
blockquote { margin: 2.5pt 0 4pt; padding: 2.5pt 6pt; background: #fff8e1; border-left: 3px solid #f0b400; }
ul, ol { margin: 1.5pt 0 3pt 14pt; padding: 0; }
img { max-width: 100%; height: auto; display: block; margin: 2pt auto 1pt; }
"""


def absolutise_images(body: str) -> str:
    """Point every relative img src at the file on disk so Chrome can load it from a temp page."""

    def replace(match: re.Match) -> str:
        src = match.group(2)
        if src.startswith(("http://", "https://", "data:", "file:")):
            return match.group(0)
        return f'{match.group(1)}src="{(ROOT / src).resolve().as_uri()}"'

    return re.sub(r'(<img [^>]*?)src="([^"]+)"', replace, body)



def chrome_html() -> str:
    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    raise SystemExit("Chrome not found; install Google Chrome to render the PDF.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "REPORT.md")
    parser.add_argument("--output", type=Path, default=ROOT / "REPORT.pdf")
    args = parser.parse_args()

    source = args.source.resolve()
    text = source.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    sources = re.findall(r'<img [^>]*?src="([^"]+)"', body)
    missing = [
        src
        for src in sources
        if not src.startswith(("http://", "https://", "data:", "file:")) and not (ROOT / src).is_file()
    ]
    if missing:
        print(f"FAIL: the report links images that are not on disk: {missing}")
        return 1
    body = absolutise_images(body)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style>
</head><body>{body}</body></html>"""

    import fitz

    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "report.html"
        html_path.write_text(page, encoding="utf-8")
        pdf_tmp = Path(tmp) / "report.pdf"
        subprocess.run(
            [chrome_html(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={pdf_tmp}", html_path.as_uri()],
            check=True,
            capture_output=True,
        )
        rendered = pdf_tmp.read_bytes()

    out = args.output.resolve()
    out.write_bytes(rendered)

    doc = fitz.open(out)
    print(f"words: {len(text.split())}  pages: {doc.page_count}  -> {out}")
    for i, sheet in enumerate(doc):
        blocks = [b for b in sheet.get_text("blocks") if b[4].strip()]
        bottom = max((b[3] for b in blocks), default=0)
        lines = sheet.get_text().strip().splitlines()
        tail = lines[-1][:52] if lines else "empty"
        print(f"  p{i + 1}: bottom={bottom:.0f}/{sheet.rect.height:.0f} last line: {tail!r}")
    if doc.page_count > MAX_PAGES:
        print(f"FAIL: {doc.page_count} pages, the report is capped at {MAX_PAGES}.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
