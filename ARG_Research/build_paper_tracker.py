"""
Research Paper Tracker builder
==================================
Parses `research links new.txt` (412 numbered entries: title, links, DOI,
abstract, and status annotations like "request" / "not download, recent
paper"), matches each entry to a PDF in `research papers/` (filenames are
often truncated/cryptic publisher names, so matching uses both fuzzy
filename comparison and first-page text extraction), renames matched PDFs
to readable "{year} - {title}.pdf" names (rename_log.csv records every
change), shelves exact duplicate copies (-1/-2 suffixed) into
`research papers/duplicates/`, and generates `paper_tracker.html` — a
self-contained, searchable, category-filtered page with clickable PDF /
DOI / site links.

Run (from anywhere):
    python build_paper_tracker.py            # full run: rename + build page
    python build_paper_tracker.py --dry-run  # report matches, change nothing
"""
import argparse
import csv
import html
import json
import re
import shutil
import sys
import unicodedata
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
LINKS_TXT = ROOT / "research links new.txt"
PDF_DIR = ROOT / "research papers"
DUP_DIR = PDF_DIR / "duplicates"
OUT_HTML = ROOT / "paper_tracker.html"
RENAME_LOG = ROOT / "rename_log.csv"

URL_RE = re.compile(r"https?://[^\s]+")
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>]+)")
YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")
MONTH_YEAR_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s*(19[5-9]\d|20[0-2]\d)")

TOPIC_RULES = [
    ("Acoustic / Audio", ["acoustic", "audio", "sound", "microphone", "hydrophone", "piezoelectric"]),
    ("Machine / Deep Learning", ["machine learning", "deep learning", "neural network", "lstm", "transformer",
                                  "random forest", "xgboost", "cnn", "deep-learning", "machine-learning", "ai "]),
    ("Tipping Bucket", ["tipping bucket", "tipping-bucket", "tbr"]),
    ("Disdrometer", ["disdrometer", "parsivel", "drop size", "dsd"]),
    ("Microwave Links", ["microwave link", "cellular", "cml"]),
    ("Radar", ["radar", "reflectivity"]),
    ("Satellite", ["satellite", "gpm", "trmm", "imerg", "remote sensing"]),
    ("Calibration / QC", ["calibration", "quality control", "quality assessment", "error", "uncertaint",
                           "adjustment", "correction", "bias"]),
    ("Low-cost Sensors", ["low-cost", "low cost", "arduino", "raspberry", "iot", "3d print", "printable"]),
    ("Optical / Camera", ["optical", "camera", "video", "image", "vision"]),
]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ----------------------------------------------------------------------------
# 1. Parse the links txt into entries
# ----------------------------------------------------------------------------

JUNK_TITLE_RE = re.compile(
    r"(?i)^(article|preprint|conference|chapter|thesis|poster|book|data|full-?text)"
    r"[\w\s]{0,25}(available)?\s*$")
META_LINE_RE = re.compile(
    r"(?i)^(doi|pmid|pmcid|abstract|keywords|copyright|©|\s*(january|february|march|april|may|june|"
    r"july|august|september|october|november|december)\s+(19|20)\d\d)")


def _clean_title_line(text: str) -> str:
    t = URL_RE.sub(" ", text)
    t = re.sub(r"\s*[-–—]\s*$", "", re.sub(r"^\s*[-–—]\s*", "", t.strip()))
    t = re.sub(r"\s+[-–—]\s+$", "", t).strip(" -–—\t")
    return re.sub(r"\s+", " ", t).strip()


def parse_entries(txt_path: Path):
    lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    # Entry headers are "N." lines where N has no digit right after the dot
    # (bare DOIs like "10.1016/..." and numbers inside abstracts like
    # "1009.82 g" would otherwise be mistaken for headers). Numbering in the
    # file itself has typos (155 for 115, duplicated 161, out-of-order 376),
    # so ordering is deliberately NOT enforced.
    hdr_re = re.compile(r"^\s*(\d{1,3})\s*\.(?!\d)(.*)$")
    header_idx = [(i, int(m.group(1))) for i, ln in enumerate(lines)
                  if (m := hdr_re.match(ln))]
    entries = []
    for k, (start, num) in enumerate(header_idx):
        end = header_idx[k + 1][0] if k + 1 < len(header_idx) else len(lines)
        block = lines[start:end]
        header = hdr_re.match(block[0]).group(2).strip()
        body = "\n".join(block)

        urls = [u.rstrip(".,);") for u in URL_RE.findall(body)]
        title = _clean_title_line(header)
        if not title or JUNK_TITLE_RE.match(title):
            # ResearchGate copy-paste style: header is "ArticlePDF Available"
            # etc. and the real title sits on one of the next lines.
            title = ""
            for ln in block[1:8]:
                t = ln.strip()
                if not t or META_LINE_RE.match(t):
                    continue
                cand = _clean_title_line(t)
                if len(cand) >= 15 and not JUNK_TITLE_RE.match(cand):
                    title = cand
                    break
        if not title and urls:
            slug = urls[0].rstrip("/").rsplit("/", 1)[-1]
            title = re.sub(r"[-_+]", " ", slug)[:120] or f"Entry {num}"
        title = re.sub(r"\s+", " ", title).strip()

        doi_m = DOI_RE.search(body)
        doi = doi_m.group(1).rstrip(".") if doi_m else ""

        # Status: short annotation lines only (abstract paragraphs are long)
        status = "downloaded"
        for ln in block:
            t = ln.strip().lower()
            if 0 < len(t) < 60 and not URL_RE.search(t):
                if "request" in t:
                    status = "request"
                    break
                if "not download" in t or ("recent" in t and "paper" in t):
                    status = "recent"
                    break

        # Abstract: between "Abstract" marker and Keywords/PMID/Copyright
        abstract = ""
        in_abs = False
        abs_lines = []
        for ln in block[1:]:
            t = ln.strip()
            low = t.lower()
            if not in_abs and low.startswith("abstract") and len(t) < 25:
                in_abs = True
                continue
            if in_abs:
                if (low.startswith(("keywords", "copyright", "pubmed disclaimer", "pmid", "©"))
                        or re.match(r"^\d+\s*\.", t)):
                    break
                abs_lines.append(t)
        abstract = re.sub(r"\s+", " ", " ".join(abs_lines)).strip()

        kw = ""
        for ln in block:
            if ln.strip().lower().startswith("keywords"):
                kw = ln.strip()[9:].strip(" :").rstrip(".")
                break

        # Year: month-year lines first, then year in DOI, then any year in urls
        year = ""
        m = MONTH_YEAR_RE.search(body)
        if m:
            year = m.group(1)
        if not year and doi:
            m = YEAR_RE.search(doi)
            if m:
                year = m.group(1)
        if not year:
            for u in urls:
                m = YEAR_RE.search(u)
                if m:
                    year = m.group(1)
                    break

        site = next((u for u in urls if "doi.org" not in u), urls[0] if urls else "")

        low_all = (title + " " + abstract + " " + kw).lower()
        topics = [name for name, kws in TOPIC_RULES if any(k in low_all for k in kws)]

        entries.append({
            "num": num, "title": title, "doi": doi, "year": year, "site": site,
            "urls": urls, "abstract": abstract, "keywords": kw, "status": status,
            "topics": topics, "pdf": None, "pdf_pages": None, "pdf_mb": None,
            "match": "", "renamed_from": "",
        })
    return entries


# ----------------------------------------------------------------------------
# 2. Scan PDFs, extract first-page text, group -N duplicate copies
# ----------------------------------------------------------------------------

def first_page_text(path: Path, max_pages=2) -> tuple[str, int]:
    try:
        import logging
        logging.getLogger("pypdf").setLevel(logging.CRITICAL)
        from pypdf import PdfReader
        r = PdfReader(str(path))
        n = len(r.pages)
        txt = ""
        for p in r.pages[:max_pages]:
            try:
                txt += p.extract_text() or ""
            except Exception:
                pass
            if len(txt) > 4000:
                break
        return txt[:6000], n
    except Exception:
        return "", 0


def scan_pdfs():
    pdfs = {}
    dup_re = re.compile(r"^(.*?)-(\d)$")
    groups = {}
    for f in sorted(PDF_DIR.glob("*.pdf")):
        stem = f.stem
        m = dup_re.match(stem)
        base = m.group(1) if m else stem
        groups.setdefault(base, []).append(f)
    primaries, duplicates = [], []
    for base, files in groups.items():
        files.sort(key=lambda p: (len(p.stem), p.stem))  # base name (no -N) first
        primaries.append(files[0])
        duplicates.extend(files[1:])
    return primaries, duplicates


# ----------------------------------------------------------------------------
# 3. Match PDFs to entries
# ----------------------------------------------------------------------------

def match_pdfs(entries, primaries, verbose=True):
    ntitle = {e["num"]: norm(e["title"]) for e in entries}
    by_num = {e["num"]: e for e in entries}
    unmatched_pdfs = []

    print(f"  Extracting text from {len(primaries)} PDFs (first pages)...", flush=True)
    pdf_info = {}
    for i, f in enumerate(primaries, 1):
        txt, pages = first_page_text(f)
        pdf_info[f] = (norm(txt), pages)
        if i % 40 == 0:
            print(f"    {i}/{len(primaries)}", flush=True)

    # Score every (pdf, entry) pair, then assign greedily best-first so a PDF
    # that loses its top entry to a better-scoring PDF falls back to its next
    # candidate instead of being silently dropped.
    candidates = []
    for f in primaries:
        fname_n = norm(re.sub(r"[-_]", " ", f.stem))
        page_n, _pages = pdf_info[f]
        for e in entries:
            tn = ntitle[e["num"]]
            if not tn or len(tn) < 10:
                continue
            score, how = 0.0, ""
            if len(fname_n) >= 15 and (fname_n in tn or tn in fname_n):
                score, how = 1.0, "filename"
            else:
                k = min(len(fname_n), len(tn))
                if k >= 15:
                    r = SequenceMatcher(None, fname_n[:k], tn[:k]).ratio()
                    if r > 0.80:
                        score, how = r, "filename~"
            if score < 0.9 and len(tn) >= 25 and tn in page_n:
                score, how = 0.98, "page-text"
            if score >= 0.80:
                candidates.append((score, f, e, how))

    # Fallback for cryptic publisher filenames (arxiv IDs, "document.pdf"):
    # fraction of the title's distinctive words present in the first pages.
    scored_pdfs = {f for _s, f, _e, _h in candidates}
    for f in primaries:
        if f in scored_pdfs:
            continue
        # Only the title zone (top of page 1): matching against the whole
        # first pages produced false hits from title words appearing in other
        # papers' reference lists.
        page_head = pdf_info[f][0][:1200]
        if len(page_head) < 100:
            continue  # no text layer (scanned) -- nothing to compare against
        for e in entries:
            words = [norm(w) for w in re.findall(r"[A-Za-z]{4,}", e["title"])]
            if len(words) < 6:
                continue
            frac = sum(1 for w in words if w in page_head) / len(words)
            if frac >= 0.9:
                candidates.append((0.80 + 0.1 * frac, f, e, "word-overlap"))

    candidates.sort(key=lambda c: -c[0])
    taken_pdfs, taken_entries = set(), set()
    for score, f, e, how in candidates:
        if f in taken_pdfs or id(e) in taken_entries:
            continue
        taken_pdfs.add(f)
        taken_entries.add(id(e))
        e["pdf"] = f
        e["match"] = how
        e["pdf_pages"] = pdf_info[f][1] or None
        e["pdf_mb"] = round(f.stat().st_size / 1e6, 2)

    unmatched_pdfs = [f for f in primaries if f not in taken_pdfs]
    return unmatched_pdfs


# ----------------------------------------------------------------------------
# 4. Rename matched PDFs, shelve duplicate copies
# ----------------------------------------------------------------------------

def safe_name(title, year, maxlen=110):
    t = re.sub(r'[\\/:*?"<>|]', "", title).strip()
    t = re.sub(r"\s+", " ", t)
    if len(t) > maxlen:
        t = t[:maxlen].rsplit(" ", 1)[0]
    y = year or "n.d."
    return f"{y} - {t}.pdf"


def apply_renames(entries, duplicates, dry_run):
    log_rows = []
    for e in entries:
        f = e.get("pdf")
        if not f:
            continue
        new_name = safe_name(e["title"], e["year"])
        if f.name == new_name:
            continue
        target = f.with_name(new_name)
        k = 2
        while target.exists() and target != f:
            target = f.with_name(new_name.replace(".pdf", f" ({k}).pdf"))
            k += 1
        if not dry_run:
            f.rename(target)
        log_rows.append({"old": f.name, "new": target.name})
        e["renamed_from"] = f.name
        e["pdf"] = target
    if duplicates and not dry_run:
        DUP_DIR.mkdir(exist_ok=True)
        for d in duplicates:
            try:
                shutil.move(str(d), str(DUP_DIR / d.name))
            except Exception as ex:
                print(f"    ! could not move duplicate {d.name}: {ex}")
    if log_rows and not dry_run:
        exists = RENAME_LOG.exists()
        with open(RENAME_LOG, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["old", "new"])
            if not exists:
                w.writeheader()
            w.writerows(log_rows)
    return log_rows


# ----------------------------------------------------------------------------
# 5. Generate the HTML page
# ----------------------------------------------------------------------------

def build_html(entries, unmatched_pdfs, dup_count):
    records = []
    for e in entries:
        pdf_href = ""
        if e.get("pdf"):
            pdf_href = "research papers/" + quote(e["pdf"].name)
        status = e["status"]
        if status == "downloaded" and not e.get("pdf"):
            status = "missing"
        report_bits = []
        if e.get("pdf_pages"):
            report_bits.append(f"{e['pdf_pages']} pages")
        if e.get("pdf_mb"):
            report_bits.append(f"{e['pdf_mb']} MB")
        if e.get("match"):
            report_bits.append(f"matched via {e['match']}")
        if e.get("renamed_from"):
            report_bits.append(f"renamed from “{e['renamed_from']}”")
        records.append({
            "num": e["num"], "title": e["title"], "doi": e["doi"], "year": e["year"],
            "site": e["site"], "abstract": e["abstract"], "keywords": e["keywords"],
            "status": status, "topics": e["topics"], "pdf": pdf_href,
            "report": " · ".join(report_bits),
        })
    extra = [{"num": None, "title": f.name, "doi": "", "year": "", "site": "",
              "abstract": "PDF present in the folder but not matched to any entry in the links file.",
              "keywords": "", "status": "unmatched", "topics": [],
              "pdf": "research papers/" + quote(f.name), "report": ""} for f in unmatched_pdfs]
    data_json = json.dumps(records + extra, ensure_ascii=False)

    counts = {}
    for r in records + extra:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    page = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ARG Research — Paper Tracker</title>
<style>
:root{
  --bg:#f4f2ec; --panel:#fffdf7; --sunken:#ece9df; --ink:#26221a; --soft:#5b5344;
  --faint:#8b8271; --line:#d8d2c2; --accent:#8a4b12; --accent-soft:#f3e2cd;
  --green:#3d6b35; --green-soft:#e2ecd9; --amber:#9a6b00; --amber-soft:#f5ecd0;
  --red:#963636; --red-soft:#f3dcdc; --blue:#2f5b7c; --blue-soft:#dde9f1;
  --grey:#6b6b6b; --grey-soft:#e7e7e2;
  --mono:ui-monospace,'Cascadia Code',Consolas,monospace;
  --serif:Georgia,'Iowan Old Style','Palatino Linotype',serif;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#191611; --panel:#211d16; --sunken:#14110c; --ink:#ece5d6; --soft:#bfb49d;
  --faint:#877c66; --line:#3a3427; --accent:#e0964a; --accent-soft:#3a2a17;
  --green:#8fc07f; --green-soft:#233820; --amber:#e3b84e; --amber-soft:#3a2f12;
  --red:#e08a8a; --red-soft:#3d2020; --blue:#84b3d4; --blue-soft:#1c2f3d;
  --grey:#9a9a94; --grey-soft:#2a2a26;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
header{max-width:1200px;margin:0 auto;padding:40px 24px 8px}
h1{font-family:var(--serif);font-size:2rem;margin:0 0 4px}
h1 .drop{color:var(--accent)}
.sub{color:var(--soft);margin:0 0 18px;max-width:70ch}
.statbar{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px}
.statbar .chip{font-family:var(--mono);font-size:12.5px;padding:6px 12px;border-radius:999px;
  background:var(--panel);border:1px solid var(--line);color:var(--soft)}
.statbar .chip b{color:var(--ink)}
.controls{max-width:1200px;margin:0 auto;padding:0 24px 14px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
#q{flex:1 1 260px;padding:10px 14px;border-radius:10px;border:1px solid var(--line);
  background:var(--panel);color:var(--ink);font-size:14.5px;outline:none}
#q:focus{border-color:var(--accent)}
.tabs{display:flex;flex-wrap:wrap;gap:6px}
.tab{cursor:pointer;font-size:13px;padding:7px 13px;border-radius:999px;border:1px solid var(--line);
  background:var(--panel);color:var(--soft);user-select:none;transition:.15s}
.tab:hover{border-color:var(--accent)}
.tab.on{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
select{padding:8px 10px;border-radius:10px;border:1px solid var(--line);background:var(--panel);color:var(--ink)}
main{max-width:1200px;margin:0 auto;padding:0 24px 80px}
.count-line{color:var(--faint);font-size:13px;margin:4px 2px 10px}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);
  border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.06)}
thead th{font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:.06em;
  text-align:left;color:var(--faint);background:var(--sunken);padding:10px 12px;cursor:pointer;white-space:nowrap}
thead th:hover{color:var(--accent)}
tbody td{padding:12px;border-top:1px solid var(--line);vertical-align:top}
tr.paper:hover td{background:color-mix(in srgb,var(--accent) 4%,transparent)}
.t-num{font-family:var(--mono);color:var(--faint);font-size:12px;white-space:nowrap}
.t-title{font-weight:600;max-width:420px}
.t-title .kw{display:block;font-weight:400;font-size:12px;color:var(--faint);margin-top:3px}
.t-year{font-family:var(--mono);white-space:nowrap}
.doi{font-family:var(--mono);font-size:12px;word-break:break-all;color:var(--soft);text-decoration:none;border-bottom:1px dotted var(--faint)}
.doi:hover{color:var(--accent)}
.badge{display:inline-block;font-size:11.5px;font-weight:600;padding:3px 10px;border-radius:999px;white-space:nowrap}
.s-downloaded{background:var(--green-soft);color:var(--green)}
.s-recent{background:var(--amber-soft);color:var(--amber)}
.s-request{background:var(--blue-soft);color:var(--blue)}
.s-missing{background:var(--red-soft);color:var(--red)}
.s-unmatched{background:var(--grey-soft);color:var(--grey)}
.topics{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
.topic{font-size:10.5px;padding:2px 8px;border-radius:999px;background:var(--accent-soft);color:var(--accent);font-weight:600}
.actions{display:flex;flex-direction:column;gap:6px;min-width:96px}
.btn{display:inline-flex;align-items:center;gap:6px;justify-content:center;font-size:12.5px;font-weight:600;
  padding:6px 12px;border-radius:8px;text-decoration:none;border:1px solid transparent;transition:.15s;white-space:nowrap}
.btn-pdf{background:var(--accent);color:#fff}
.btn-pdf:hover{filter:brightness(1.1)}
.btn-site{border-color:var(--line);color:var(--soft);background:transparent}
.btn-site:hover{border-color:var(--accent);color:var(--accent)}
details{margin-top:6px}
summary{cursor:pointer;font-size:12.5px;color:var(--accent);user-select:none}
details p{font-size:13px;color:var(--soft);margin:8px 0 0;max-width:75ch}
.report{font-size:12px;color:var(--faint);font-family:var(--mono);max-width:220px}
.empty{padding:40px;text-align:center;color:var(--faint)}
footer{max-width:1200px;margin:0 auto;padding:20px 24px;color:var(--faint);font-size:12.5px;border-top:1px solid var(--line)}
@media(max-width:900px){.t-abs,.t-report{display:none}}
</style>
</head>
<body>
<header>
  <h1><span class="drop">☔</span> ARG Research — Paper Tracker</h1>
  <p class="sub">Every paper from <span style="font-family:var(--mono)">research links new.txt</span>, matched to its
  PDF, categorised, and one click away. Click a column header to sort; use the tabs to filter by status.</p>
  <div class="statbar" id="statbar"></div>
</header>
<div class="controls">
  <input id="q" type="search" placeholder="Search title, abstract, DOI, keywords…">
  <div class="tabs" id="tabs"></div>
  <select id="topicSel"><option value="">All topics</option></select>
</div>
<main>
  <div class="count-line" id="countline"></div>
  <table>
    <thead><tr>
      <th data-k="num">#</th><th data-k="title">Paper</th><th data-k="year">Year</th>
      <th>DOI</th><th data-k="status">Status</th><th class="t-report">PDF report</th><th>Open</th>
    </tr></thead>
    <tbody id="rows"></tbody>
  </table>
</main>
<footer>Generated __DATE__ · __NDL__ downloaded · __NREC__ recent (no sci-hub) · __NREQ__ to request ·
__NMISS__ missing PDFs · __NUNM__ unmatched files · __NDUP__ duplicate copies shelved to “duplicates/”.
Regenerate any time with <span style="font-family:var(--mono)">python build_paper_tracker.py</span></footer>
<script>
const DATA = __DATA__;
const LABELS = {downloaded:"Downloaded", recent:"Recent (no sci-hub)", request:"To request",
                missing:"PDF missing", unmatched:"Unmatched file"};
let statusF = "", topicF = "", query = "", sortK = "num", sortAsc = true;

const tabsEl = document.getElementById('tabs');
const cnt = {};
DATA.forEach(r => cnt[r.status] = (cnt[r.status]||0)+1);
const order = ["","downloaded","recent","request","missing","unmatched"];
order.forEach(s=>{
  if(s && !cnt[s]) return;
  const b=document.createElement('div');
  b.className='tab'+(s===statusF?' on':'');
  b.textContent = s ? `${LABELS[s]} (${cnt[s]})` : `All (${DATA.length})`;
  b.onclick=()=>{statusF=s;document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));b.classList.add('on');render();};
  tabsEl.appendChild(b);
});
const topics=[...new Set(DATA.flatMap(r=>r.topics))].sort();
const sel=document.getElementById('topicSel');
topics.forEach(t=>{const o=document.createElement('option');o.value=t;o.textContent=t;sel.appendChild(o);});
sel.onchange=()=>{topicF=sel.value;render();};
document.getElementById('q').oninput=e=>{query=e.target.value.toLowerCase();render();};
document.querySelectorAll('thead th[data-k]').forEach(th=>{
  th.onclick=()=>{const k=th.dataset.k;if(sortK===k)sortAsc=!sortAsc;else{sortK=k;sortAsc=true;}render();};
});
const sb=document.getElementById('statbar');
sb.innerHTML=`<span class="chip"><b>${DATA.length}</b> papers tracked</span>`+
 order.slice(1).filter(s=>cnt[s]).map(s=>`<span class="chip">${LABELS[s]}: <b>${cnt[s]}</b></span>`).join('');

function esc(s){const d=document.createElement('div');d.textContent=s||'';return d.innerHTML;}
function render(){
  let rows=DATA.filter(r=>
    (!statusF||r.status===statusF)&&(!topicF||r.topics.includes(topicF))&&
    (!query||[r.title,r.abstract,r.doi,r.keywords].join(' ').toLowerCase().includes(query)));
  rows.sort((a,b)=>{
    let x=a[sortK],y=b[sortK];
    if(sortK==='num'){x=x??1e9;y=y??1e9;}
    if(x==null||x==='')x=sortAsc?'\\uffff':'';if(y==null||y==='')y=sortAsc?'\\uffff':'';
    return (x<y?-1:x>y?1:0)*(sortAsc?1:-1);});
  document.getElementById('countline').textContent=`${rows.length} of ${DATA.length} shown`;
  const tb=document.getElementById('rows');
  if(!rows.length){tb.innerHTML='<tr><td colspan="7" class="empty">No papers match.</td></tr>';return;}
  tb.innerHTML=rows.map(r=>{
    const doiLink=r.doi?`<a class="doi" href="https://doi.org/${encodeURIComponent(r.doi)}" target="_blank" rel="noopener">${esc(r.doi)}</a>`:'<span class="doi" style="border:none">—</span>';
    const topicChips=r.topics.length?`<div class="topics">${r.topics.map(t=>`<span class="topic">${esc(t)}</span>`).join('')}</div>`:'';
    const abs=r.abstract?`<details><summary>Abstract</summary><p>${esc(r.abstract)}</p></details>`:'';
    const kw=r.keywords?`<span class="kw">${esc(r.keywords)}</span>`:'';
    const pdfBtn=r.pdf?`<a class="btn btn-pdf" href="${r.pdf}" target="_blank">📄 PDF</a>`:'';
    const siteBtn=r.site?`<a class="btn btn-site" href="${esc(r.site)}" target="_blank" rel="noopener">🔗 Site</a>`:'';
    return `<tr class="paper">
      <td class="t-num">${r.num??'—'}</td>
      <td class="t-title">${esc(r.title)}${kw}${topicChips}${abs}</td>
      <td class="t-year">${esc(r.year)||'—'}</td>
      <td>${doiLink}</td>
      <td><span class="badge s-${r.status}">${LABELS[r.status]}</span></td>
      <td class="t-report"><div class="report">${esc(r.report)||'—'}</div></td>
      <td><div class="actions">${pdfBtn}${siteBtn}</div></td>
    </tr>`;}).join('');
}
render();
</script>
</body>
</html>"""
    page = (page.replace("__DATA__", data_json)
                .replace("__DATE__", date.today().isoformat())
                .replace("__NDL__", str(counts.get("downloaded", 0)))
                .replace("__NREC__", str(counts.get("recent", 0)))
                .replace("__NREQ__", str(counts.get("request", 0)))
                .replace("__NMISS__", str(counts.get("missing", 0)))
                .replace("__NUNM__", str(counts.get("unmatched", 0)))
                .replace("__NDUP__", str(dup_count)))
    OUT_HTML.write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report matches, change nothing")
    args = ap.parse_args()

    print("[1] Parsing links file...")
    entries = parse_entries(LINKS_TXT)
    from collections import Counter
    print(f"  {len(entries)} entries | status: {dict(Counter(e['status'] for e in entries))}")

    print("[2] Scanning PDF folder...")
    primaries, duplicates = scan_pdfs()
    print(f"  {len(primaries)} primary PDFs, {len(duplicates)} duplicate copies (-1/-2 suffixed)")

    print("[3] Matching PDFs to entries...")
    unmatched = match_pdfs(entries, primaries)
    matched_n = sum(1 for e in entries if e.get("pdf"))
    dl = [e for e in entries if e["status"] == "downloaded"]
    print(f"  Matched {matched_n} PDFs | 'downloaded' entries with a PDF: "
          f"{sum(1 for e in dl if e.get('pdf'))}/{len(dl)} | unmatched PDFs: {len(unmatched)}")

    print(f"[4] {'DRY RUN — would rename' if args.dry_run else 'Renaming'} matched PDFs...")
    log = apply_renames(entries, duplicates, args.dry_run)
    print(f"  {len(log)} renames{' (not applied)' if args.dry_run else ' applied, logged to rename_log.csv'}"
          f" | {len(duplicates)} duplicates {'would move' if args.dry_run else 'moved'} to duplicates/")

    print("[5] Building HTML page...")
    build_html(entries, unmatched, len(duplicates))
    print(f"  Saved: {OUT_HTML}")


if __name__ == "__main__":
    main()
