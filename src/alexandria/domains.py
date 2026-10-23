"""Discipline labels: clean the free-text `field_subfield` and map it onto canonical domains.

`domain_of` and its rules are copied from SCI-JEPA (src/sci_jepa/data/ku_schema.py) so WS1 statistics and the
SCI-JEPA experiments use the same 10 domains; tests/test_domains.py checks they agree when sci_jepa is installed.
"""
from __future__ import annotations

import re
import unicodedata

DOMAINS = ["Physics", "Astrophysics", "Mathematics", "Computer Science", "Statistics", "EESS & Engineering",
           "Quantitative Biology & Medicine", "Economics & Finance", "Chemistry", "Other"]
_DOMAIN_RULES = [
    ("Astrophysics", r"astro|cosmolog|galax|stellar|solar|planet|space science|celestial|gravitational wave"),
    ("Quantitative Biology & Medicine", r"biophys|bioinform|biolog|genom|neuro|medic|clinic|health|pharma|epidem|"
                                        r"physiolog|ecolog|biochem|life science|cancer|patholog"),
    ("Physics", r"statistical (mechanics|physics)|mathematical physics|computational physics|physic|quantum|optic|"
                r"photon|condensed|nuclear|plasma|particle|high energy|material|superconduct|thermo|fluid|acoustic|"
                r"spintronic|semiconductor|laser|atomic|molecular physics|nanoscience|nanotech|geophys|atmospher|"
                r"meteorolog|climat|hydrolog|earth science|geoscience|oceanograph|environmental science"),
    ("Statistics", r"statist|probability theory and statistics|econometric"),
    ("Mathematics", r"mathemat|algebra|geometr|topolog|number theory|combinator|analysis|probability|logic|"
                    r"calculus|differential equation|dynamical system|operator theory|category theory|"
                    r"operations research|optimization"),
    ("Computer Science", r"comput|artificial intelligence|machine learning|deep learning|software|robot|"
                         r"information (retrieval|theory|science|systems)|data (science|mining)|vision|"
                         r"natural language|language processing|cryptograph|security|network|database|"
                         r"human-computer|algorithm|programming"),
    ("EESS & Engineering", r"electrical|electronic|signal|control|systems|telecommunic|audio|speech|"
                           r"image processing|engineering|aerospace|mechanical|civil|energy|power"),
    ("Economics & Finance", r"econom|financ|business|management|marketing"),
    ("Chemistry", r"chemi"),
]
_DOMAIN_RE = [(d, re.compile(p, re.I)) for d, p in _DOMAIN_RULES]


def split_field(fs: str | None) -> tuple[str, str]:
    s = " ".join(str(fs or "").split())
    for sep in (" — ", "—", " – ", " - "):
        if sep in s:
            a, _ = s.split(sep, 1)
            return a.strip() or "Unknown", s
    return (s or "Unknown"), (s or "Unknown")


def domain_of(field_subfield: str | None) -> str:
    coarse, full = split_field(field_subfield)
    for text in (coarse, full):
        for d, rx in _DOMAIN_RE:
            if rx.search(text):
                return d
    return "Other"


def clean_label(label: str | None) -> str:
    """Merge spelling variants of one coarse field: 'Astro-Physics' / 'astrophysics..' / 'Accounting & Finance'
    -> 'astrophysics' / 'accounting and finance'. Used for counting distinct labels, not for display."""
    s = unicodedata.normalize("NFKC", str(label or "")).strip().lower()
    s = s.replace("&", " and ").replace("-", "").replace("_", " ")
    s = re.sub(r"[\[\]().,;:]+", " ", s)
    return " ".join(s.split()) or "unknown"


def is_non_english_label(label: str | None) -> bool:
    letters = [c for c in str(label or "") if c.isalpha()]
    return bool(letters) and sum(1 for c in letters if "LATIN" not in unicodedata.name(c, "")) / len(letters) > 0.5
