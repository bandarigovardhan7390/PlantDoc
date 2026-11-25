"""
PLANTDOC - check knowledge_base.json and the Hindi / Telugu translation files

Run:   python check_kb.py                       (uses the ASSETS folder below)
       python check_kb.py "D:\\some\\other\\assets"

It checks, in this order:
  1. knowledge_base.json is valid JSON (and tells you the line if it is not)
  2. every issue has the fields the app reads, and matches the detector's Tomato classes
  3. every interface text (ui) has English, Hindi and Telugu, with the same {placeholders}
  4. every English sentence in the issues has a Hindi and a Telugu translation
     in translations_hi.json / translations_te.json
  5. numbers (doses, days, percentages) are the same in English and in the translation
  6. translations that still contain no Hindi / Telugu letters (left in English)

It writes missing_hi.json and missing_te.json (English sentence -> "") for anything missing,
so you only have to fill in the empty values.
"""
import json
import re
import sys
from pathlib import Path

ASSETS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"C:\Users\ASUS\AndroidStudioProjects\Plantdoc\app\src\main\assets")
OUT = Path(__file__).resolve().parent

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
TELUGU = re.compile(r"[\u0C00-\u0C7F]")
PLACEHOLDER = re.compile(r"\{[a-z_]+\}")
NUMBER = re.compile(r"\d+(?:\.\d+)?")

problems = 0


def bad(msg):
    global problems
    problems += 1
    print("  PROBLEM:", msg)


def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        bad(f"{path.name} not found in {path.parent}")
    except json.JSONDecodeError as e:
        bad(f"{path.name} is not valid JSON: {e.msg} at line {e.lineno}, column {e.colno}")
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        for i in range(max(0, e.lineno - 3), min(len(lines), e.lineno + 2)):
            print(f"      {i + 1:5d} | {lines[i][:110]}")
    return None


def sentences(kb):
    """every English sentence in the knowledge base that the app looks up in the translation files"""
    out = []
    for lst in kb.get("common", {}).values():
        out += lst
    for issue in kb.get("issues", {}).values():
        out += [issue.get("display", ""), issue.get("summary", ""), issue.get("escalate_if", "")]
        for p in issue.get("pests_to_check", []):
            out += [p["name"], p["why"]]
        out += issue.get("effects_if_untreated", [])
        for lst in issue.get("actions", {}).values():
            out += lst
        out += list(issue.get("area_guidance", {}).values())
        out += issue.get("prevention", [])
        for c in issue.get("chemical_options", []):
            out += [c["use_for"], c["dose"]]
    seen, uniq = set(), []
    for s in out:
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


print("Folder:", ASSETS)
print("\n1. knowledge_base.json")
kb = load(ASSETS / "knowledge_base.json")
if kb is None:
    print("\nFix the JSON first, then run this again.")
    sys.exit(1)
print("  valid JSON")

print("\n2. issues")
required = ["crop", "display", "type", "summary", "pests_to_check", "effects_if_untreated", "actions",
            "area_guidance", "escalate_if", "prevention", "chemical_options", "sources"]
for key, issue in kb.get("issues", {}).items():
    for f in required:
        if f not in issue:
            bad(f"issue '{key}' has no field '{f}'")
    acts = issue.get("actions", {})
    need = ["Healthy"] if issue.get("type") == "healthy" else ["Mild", "Moderate", "Severe"]
    for n in need:
        if n not in acts:
            bad(f"issue '{key}' has no actions for '{n}'")
print("  issues in file:", ", ".join(kb.get("issues", {})))
cn = ASSETS / "class_names.json"
if cn.exists():
    classes = [c for c in json.loads(cn.read_text(encoding="utf-8-sig")) if str(c).startswith("Tomato")]
    for c in classes:
        if c not in kb["issues"]:
            bad(f"detector class '{c}' has no entry in the knowledge base")
    print("  detector Tomato classes:", ", ".join(classes))

print("\n3. interface text (ui)")
for k, v in kb.get("ui", {}).items():
    for lang in ("en", "hi", "te"):
        if not v.get(lang, "").strip():
            bad(f"ui '{k}' has no '{lang}' text")
    base = sorted(PLACEHOLDER.findall(v.get("en", "")))
    for lang in ("hi", "te"):
        if sorted(PLACEHOLDER.findall(v.get(lang, ""))) != base:
            bad(f"ui '{k}': placeholders in '{lang}' differ from English {base}")
print("  ui keys:", len(kb.get("ui", {})))

en_sentences = sentences(kb)
print(f"\n4-6. translations ({len(en_sentences)} English sentences to translate)")
for code, name, script in (("hi", "Hindi", DEVANAGARI), ("te", "Telugu", TELUGU)):
    tr = load(ASSETS / f"translations_{code}.json")
    if tr is None:
        continue
    missing = [s for s in en_sentences if s not in tr or not str(tr[s]).strip()]
    print(f"\n  {name}: {len(en_sentences) - len(missing)} of {len(en_sentences)} translated")
    if missing:
        (OUT / f"missing_{code}.json").write_text(
            json.dumps({s: "" for s in missing}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {len(missing)} missing, written to {OUT / ('missing_' + code + '.json')}:")
        for s in missing:
            print("     -", s[:100])
        problems += len(missing)
    for s in en_sentences:
        t = str(tr.get(s, ""))
        if not t:
            continue
        if not script.search(t) and re.search(r"[A-Za-z]{4,}", s) and not re.fullmatch(r"[\d\s.,%/+()-]+", s):
            bad(f"{name} text has no {name} letters (still English?): {s[:70]}")
        if sorted(NUMBER.findall(s)) != sorted(NUMBER.findall(t)):   # sorted: word order differs between languages
            bad(f"{name} numbers differ: EN {sorted(NUMBER.findall(s))} vs {sorted(NUMBER.findall(t))}  <- {s[:70]}")
    extra = [k for k in tr if k not in en_sentences]
    if extra:
        print(f"  note: {len(extra)} entries in translations_{code}.json no longer match any English sentence (harmless).")

print("\n" + "=" * 60)
print("ALL GOOD" if problems == 0 else f"{problems} problem(s) found - see above")