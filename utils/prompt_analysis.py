"""Prompt analysis utilities for extracting product insights from user prompts."""

import re
from collections import Counter
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

INTENT_PATTERNS: dict[str, list[str]] = {
    # Keep your core intents, but tighten a couple of high-false-positive patterns
    "Show / Visualise": [
        r"\bshow\b",
        r"\bvisuali[sz](?:e|ation|ing)\b",
        r"\bdisplay\b",
        r"\bmap\b",
        r"\bplot\b",
        r"\bchart\b",
        r"\bgraph\b",
        r"\billustrat(?:e|ion)\b",
        r"\brender\b",
        r"\boverlay\b",
        r"\bview\b",
    ],
    "Compare": [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bdifference\b",
        r"\bcontrast\b",
        # avoid bare "between" (too broad); keep "between X and Y" form
        r"\bbetween\b.+\band\b",
    ],
    "Quantify": [
        r"\bhow much\b",
        r"\bhow many\b",
        r"\bquantif\w*\b",
        r"\btotal\b",
        r"\bpercentage\b",
        r"\bproportion\b",
        r"\bshare\b",
        r"\bamount\b",
        r"\bextent\b",
        r"\barea\b",
        r"\bhectares?\b",
        r"\bkm2\b|\bkm\^2\b|\bsq(?:uare)?\s*km\b",
        r"\bsq(?:uare)?\s*m(?:etre|eter)s?\b",
    ],
    "Trend / Change": [
        r"\btrend\b",
        r"\bchange\b",
        r"\bover time\b",
        r"\bthrough time\b",
        r"\bsince\b",
        r"\bincrease\b",
        r"\bdecrease\b",
        r"\bgrowth\b",
        r"\bdecline\b",
        r"\bevolution\b",
        r"\bprogress\b",
        r"\btime\s*series\b",
        r"\bhistor\w+\b",
    ],
    "Causes / Drivers": [
        r"\bcause\b",
        r"\bdriver\b",
        r"\bdrivers\b",
        r"\breason\b",
        r"\bwhy\b",
        r"\bfactor\b",
        r"\battribut\w+\b",
        r"\bresponsible\b",
        r"\btop\s+\w*\s*cause\b",
        r"\bsource\b",
    ],
    "Locate / Where": [
        r"\bwhere\b",
        r"\blocati\w+\b",
        r"\bfind\b",
        # avoid bare "identify" here; it was over-firing in your labelled set
        r"\bwhich areas?\b",
        r"\bhot\s*spots?\b|\bhotspots?\b",
        r"\bregion\b",
        r"\bcoordinates?\b",
    ],

    # Add intents that were common in your LLM-labelled data / merge map
    "Analyse": [
        r"\banalys(?:e|es|ed|ing|is)\b",
        r"\banaly[sz](?:e|es|ed|ing)\b",
        r"\bdiagnos\w+\b",
        r"\bevaluat\w+\b",
        r"\binterpret\w+\b",
    ],
    "Clarify": [
        r"\bclarif\w+\b",
        r"\bexplain\b",
        r"\bdefine\b|\bdefinition\b",
        r"\bwhat does\b",
        r"\bhow does\b",
        r"\bcapabilit(?:y|ies)\b",
        r"\bis it possible\b|\bcan you\b",
    ],
    "Recommend Actions": [
        r"\brecommend\w*\b",
        r"\bsuggest\w*\b",
        r"\badvise\b",
        r"\bwhat should\b",
        r"\bhow can we\b",
        r"\bmitigat\w+\b",
        r"\breduc\w+\b",
        r"\bprevent\w+\b",
        r"\baction(?:s| plan)?\b",
        r"\bstrategy\b|\bintervention\b",
    ],
    "Risk assessment": [
        r"\brisk\b",
        r"\bat risk\b",
        r"\bthreat\b|\bhazard\b",
        r"\bvulnerab\w+\b",
        r"\bexposure\b",
        r"\bsusceptib\w+\b",
    ],
    "Data availability": [
        r"\bdata (?:available|availability)\b",
        r"\bavailability\b",
        r"\bcoverage\b",
        r"\bdo you have (?:data|a dataset)\b",
        r"\bwhich dataset\b",
        r"\bdata source\b",
        r"\bresolution\b",
        r"\bupdate (?:frequency|cadence)\b",
    ],
    "Download data": [
        r"\bdownload\b",
        r"\bexport\b",
        r"\bextract\b",
        r"\bget (?:the )?data\b",
        r"\bcsv\b",
        r"\bgeojson\b",
        r"\bshapefile\b|\bshp\b",
        r"\bapi\b|\bendpoint\b",
    ],
    "Identify": [
        r"\bidentify\b",
        r"\blist\b",
        r"\benumerat\w+\b",
        r"\bclassif\w+\b",
        r"\bcategoris(?:e|ation)\b|\bcategorize|categorization\b",
    ],
    # Useful when prompts are acknowledgements rather than requests
    "Acknowledge / Affirm": [
        r"^\s*(?:yes|yeah|yep|ok|okay|thanks|thank you|got it|great|perfect)\b",
    ],
}

_COMPILED_INTENTS: dict[str, list[re.Pattern]] = {
    intent: [re.compile(p, re.IGNORECASE) for p in patterns]
    for intent, patterns in INTENT_PATTERNS.items()
}

def classify_prompt_intent(prompt: str) -> list[str]:
    """Return all matching intent labels for a prompt (a prompt can have multiple)."""
    if not prompt or not prompt.strip():
        return ["Unclassified"]
    intents: list[str] = []
    for intent, patterns in _COMPILED_INTENTS.items():
        if any(p.search(prompt) for p in patterns):
            intents.append(intent)
    return intents or ["Unclassified"]


# ---------------------------------------------------------------------------
# Environmental topic extraction
# ---------------------------------------------------------------------------

TOPIC_PATTERNS: dict[str, list[str]] = {
    "Forest loss": [
        r"\bforest\s*loss\b",
        r"\bdeforest\w*\b",
        r"\btree\s*loss\b",
        r"\btree\s*cover\s*loss\b",
        r"\bforest\s*cover\s*loss\b",
    ],
    "Grassland": [
        r"\bgrasslands?\b",
        r"\bgrassland\s*extent\b",
        r"\bpasture\b",
        r"\bpastures\b",
        r"\bsavann?a\b",
        r"\bmeadow\b",
    ],
    "Cropland": [
        r"\bcroplands?\b",
        r"\bagricultur\w+\b",
        r"\bfarm(?:ing|land)?\b",
        r"\bcrops?\b",
    ],
    "Restoration": [
        r"\brestor\w+\b",
        r"\breforest\w+\b",
        r"\bafforest\w+\b",
        r"\bregenerat\w+\b",
    ],
    "Wildfire": [
        r"\bwildfires?\b",
        # drop bare "\bfire\b" (too many false positives); keep environmentally-specific fire phrasing
        r"\b(?:forest|bush|vegetation|grass)\s*fires?\b",
        r"\bactive\s*fires?\b",
        r"\bburn(?:ed|t)?\b",
        r"\bburn\s*scar(?:s)?\b",
        r"\bburn(?:ed|t)?\s*area\b",
    ],
    "Land cover change": [
        r"\bland\s*cover(?:\s*change)?\b",
        r"\bland\s*use(?:\s*change)?\b",
        r"\blulc\b",
        r"\bconversion\b",
        r"\bconverted\b",
    ],
    "Disturbance": [
        r"\bdisturbanc\w+\b",
        r"\bdegrad(?:ed|ation|ing)\b",
    ],
    "Urbanisation": [
        r"\burban(?:is|iz)\w*\b",
        r"\bcities\b|\bcity\b",
        r"\bbuilt[\s-]?up\b",
        r"\bdevelop(?:ment|ing|ed)?\b",
    ],
    "Water / Wetland": [
        r"\bwetlands?\b",
        r"\bpeatlands?\b",  # often queried with wetlands in practice
        r"\blakes?\b",
        r"\brivers?\b",
        r"\bflood(?:ing|s)?\b",
        r"\bcoastal\b",
        r"\bwatershed\b",
        r"\bsurface\s*water\b|\bwater\s*body\b",
    ],
    "Biodiversity": [
        r"\bbiodiversity\b",
        r"\bbiodiversity\s*(?:loss|risk)s?\b",
        r"\bspecies\b",
        r"\bhabitat\b",
        r"\bwildlife\b",
        r"\bprotected\s*areas?\b",
    ],
    "Carbon / Climate": [
        r"\bcarbon\b",
        r"\bcarbon\s*footprint\b",
        r"\bclimate\b",
        r"\bemission\w*\b",
        r"\bgreenhouse\b",
        r"\bco2\b",
        r"\bghg\b",
    ],
    "Mining": [
        r"\bmining\b",
        r"\bextract\w+\b",
        r"\bquarr\w+\b",
    ],
    "Natural land": [
        r"\bnatural\s*lands?\b",
        r"\bnatural\s*habitat\b",
        r"\bprimary\s*forest\b",
        r"\bintact\s*forest\b",
    ],
}

_COMPILED_TOPICS: dict[str, list[re.Pattern]] = {
    topic: [re.compile(p, re.IGNORECASE) for p in patterns]
    for topic, patterns in TOPIC_PATTERNS.items()
}

def extract_prompt_topics(prompt: str) -> list[str]:
    """Return all matching environmental topics for a prompt."""
    if not prompt or not prompt.strip():
        return []
    topics: list[str] = []
    for topic, patterns in _COMPILED_TOPICS.items():
        if any(p.search(prompt) for p in patterns):
            topics.append(topic)
    return topics


# ---------------------------------------------------------------------------
# Geographic entity extraction (lightweight regex-based)
# ---------------------------------------------------------------------------

# Prefer canonical names in the list, then add aliases that map to canonicals.
GEO_ALIASES: list[tuple[re.Pattern, str]] = [
    # Common abbreviations / variants
    (re.compile(r"\bU\.?S\.?A\.?\b", re.IGNORECASE), "United States"),
    (re.compile(r"\bU\.?S\.?\b", re.IGNORECASE), "United States"),
    (re.compile(r"\bU\.?K\.?\b", re.IGNORECASE), "United Kingdom"),
    (re.compile(r"\bU\.?A\.?E\.?\b", re.IGNORECASE), "United Arab Emirates"),
    (re.compile(r"\bDRC\b", re.IGNORECASE), "Democratic Republic of Congo"),
    (re.compile(r"\bDR\s*Congo\b", re.IGNORECASE), "Democratic Republic of Congo"),
    (re.compile(r"\bDemocratic Republic of the Congo\b", re.IGNORECASE), "Democratic Republic of Congo"),
    (re.compile(r"\bCote d'Ivoire\b", re.IGNORECASE), "Côte d'Ivoire"),
    (re.compile(r"\bIvory Coast\b", re.IGNORECASE), "Côte d'Ivoire"),
    (re.compile(r"\bAmazon basin\b", re.IGNORECASE), "Amazon Basin"),
    (re.compile(r"\bCongo basin\b", re.IGNORECASE), "Congo Basin"),
    (re.compile(r"\bworld\b", re.IGNORECASE), "Global"),
    (re.compile(r"\bglobal\b", re.IGNORECASE), "Global"),
]

# Canonical entities (you can keep expanding this list)
GEO_ENTITIES: list[str] = [
    # Countries / regions (keep your existing list, but prefer canonicals)
    "Côte d'Ivoire",
    "Democratic Republic of Congo",
    "United Arab Emirates",
    "United Kingdom",
    "United States",
    "Global",
    "Amazon Basin",
    "Congo Basin",
    # ... plus the rest of your existing GEO_ENTITIES list ...
]

# If you want to retain your full existing list, you can extend GEO_ENTITIES like this:
# GEO_ENTITIES = sorted(set(GEO_ENTITIES + [...your existing long list...]), key=str.lower)

_GEO_PATTERN_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(g) + r"\b", re.IGNORECASE), g)
    for g in sorted(GEO_ENTITIES, key=len, reverse=True)  # longest first
]

def extract_geographic_entities(prompt: str) -> list[str]:
    """Return geographic entities mentioned in the prompt, normalised to canonical names."""
    if not prompt or not prompt.strip():
        return []
    found: list[str] = []

    # 1) Aliases first (so abbreviations normalise to canonical)
    for pat, canonical in GEO_ALIASES:
        if pat.search(prompt) and canonical not in found:
            found.append(canonical)

    # 2) Direct canonical matches
    for pat, name in _GEO_PATTERN_MAP:
        if pat.search(prompt) and name not in found:
            found.append(name)

    return found


# ---------------------------------------------------------------------------
# Temporal reference extraction
# ---------------------------------------------------------------------------

# We return canonical strings (e.g. "Last year", "Last 10 years", "2015-2024", "2010-present", "2024")
# rather than coarse types, because your label set is phrased that way.
_TEMPORAL_RULES: list[tuple[re.Pattern, Callable[[re.Match], str]]] = [
    # Relative fixed windows
    (re.compile(r"\b(?:last|past|previous)\s+(?:year|12\s*months?)\b", re.IGNORECASE),
     lambda m: "Last year"),
    (re.compile(r"\b(?:last|past|previous)\s+(?:month|30\s*days?)\b", re.IGNORECASE),
     lambda m: "Last month"),
    (re.compile(r"\b(?:last|past|previous)\s+(?:week|7\s*days?)\b", re.IGNORECASE),
     lambda m: "Last week"),

    # Relative N units
    (re.compile(r"\b(?:last|past)\s+(\d+)\s*years?\b", re.IGNORECASE),
     lambda m: f"Last {int(m.group(1))} years"),
    (re.compile(r"\b(?:last|past)\s+(\d+)\s*months?\b", re.IGNORECASE),
     lambda m: f"Last {int(m.group(1))} months"),

    # Named periods
    (re.compile(r"\blast\s+decade\b|\bpast\s+decade\b", re.IGNORECASE),
     lambda m: "Last decade"),
    (re.compile(r"\blast\s+summer\b", re.IGNORECASE),
     lambda m: "Last summer"),
    (re.compile(r"\brecently\b|\bin\s+recent\s+years\b", re.IGNORECASE),
     lambda m: "Recently"),
    (re.compile(r"\bhistorical\b|\bhistoric(?:al)?\b", re.IGNORECASE),
     lambda m: "Historical"),

    # Explicit ISO dates (keep as-is)
    (re.compile(r"\b((?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01]))\b"),
     lambda m: m.group(1)),

    # "1992/3" -> "1992-1993"
    (re.compile(r"\b((?:19|20)\d{2})\s*/\s*(\d{2})\b"),
     lambda m: f"{m.group(1)}-{m.group(1)[:2]}{m.group(2)}"),

    # Year ranges (various forms)
    (re.compile(r"\bbetween\s+((?:19|20)\d{2})\s+and\s+((?:19|20)\d{2})\b", re.IGNORECASE),
     lambda m: f"{m.group(1)}-{m.group(2)}"),
    (re.compile(r"\bfrom\s+((?:19|20)\d{2})\s+to\s+((?:19|20)\d{2})\b", re.IGNORECASE),
     lambda m: f"{m.group(1)}-{m.group(2)}"),
    (re.compile(r"\b((?:19|20)\d{2})\s*[-–]\s*((?:19|20)\d{2})\b"),
     lambda m: f"{m.group(1)}-{m.group(2)}"),

    # Since / onward -> "YYYY-present"
    (re.compile(r"\bsince\s+((?:19|20)\d{2})\b", re.IGNORECASE),
     lambda m: f"{m.group(1)}-present"),
    (re.compile(r"\bfrom\s+((?:19|20)\d{2})\s+(?:to\s+)?(?:present|now|today|to\s*date|current)\b", re.IGNORECASE),
     lambda m: f"{m.group(1)}-present"),

    # Single year phrasing
    (re.compile(r"\b(?:in|for|during)\s+((?:19|20)\d{2})\b", re.IGNORECASE),
     lambda m: m.group(1)),
    (re.compile(r"\b((?:19|20)\d{2})\s+only\b", re.IGNORECASE),
     lambda m: m.group(1)),
]

def extract_temporal_references(prompt: str) -> list[str]:
    """Return canonical temporal references found in the prompt."""
    if not prompt or not prompt.strip():
        return []
    refs: list[str] = []
    for pat, labeller in _TEMPORAL_RULES:
        for m in pat.finditer(prompt):
            label = labeller(m)
            if label not in refs:
                refs.append(label)
    return refs



# ---------------------------------------------------------------------------
# Bigram extraction
# ---------------------------------------------------------------------------

BIGRAM_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "i",
    "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or",
    "our", "please", "show", "tell", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "to", "us", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "will", "with", "would",
    "you", "your", "help", "thanks", "give", "generate", "analyse", "user",
    "been", "being", "some", "any", "all", "each", "every", "both", "no",
    "not", "only", "just", "also", "about", "over", "last", "past",
}


def extract_bigrams(prompt: str) -> list[str]:
    """Extract meaningful bigrams (2-word phrases) from a prompt."""
    if not prompt or not prompt.strip():
        return []
    s = str(prompt).lower()
    words = re.findall(r"[a-z]{2,}", s)
    words = [w for w in words if w not in BIGRAM_STOPWORDS and not w.startswith("http")]
    bigrams: list[str] = []
    for i in range(len(words) - 1):
        bg = f"{words[i]} {words[i + 1]}"
        bigrams.append(bg)
    return bigrams


def top_bigrams(prompts: list[str], top_n: int = 30) -> list[tuple[str, int]]:
    """Return top N bigrams across all prompts."""
    counts: Counter[str] = Counter()
    for p in prompts:
        counts.update(extract_bigrams(p))
    return counts.most_common(top_n)


# ---------------------------------------------------------------------------
# Anti-pattern detection
# ---------------------------------------------------------------------------

def detect_anti_patterns(prompt: str) -> list[str]:
    """Detect potential anti-patterns in a user prompt.

    Returns a list of anti-pattern labels found.
    """
    if not prompt or not prompt.strip():
        return ["Empty prompt"]

    patterns: list[str] = []
    text = prompt.strip()
    words = text.split()

    # Too short / vague
    if len(words) <= 3:
        patterns.append("Too short (≤3 words)")

    # No geographic context
    geo = extract_geographic_entities(text)
    if not geo:
        # Also check for any proper-noun-like capitalized words as proxy
        has_place_hint = bool(re.search(r"\b[A-Z][a-z]{2,}", text))
        if not has_place_hint:
            patterns.append("No geographic context")

    # No temporal context
    temporal = extract_temporal_references(text)
    if not temporal:
        patterns.append("No temporal context")

    # Overly broad / generic (no specific topic)
    topics = extract_prompt_topics(text)
    if not topics:
        patterns.append("No specific topic detected")

    # Copy-paste / excessively long
    if len(text) > 500:
        patterns.append("Excessively long (>500 chars)")

    # Contains URLs (user pasting raw data?)
    if re.search(r"https?://", text):
        patterns.append("Contains URL")

    return patterns


# ---------------------------------------------------------------------------
# Bulk analysis helpers (operate on DataFrames)
# ---------------------------------------------------------------------------

def analyse_prompts_bulk(
    prompts: list[str],
) -> list[dict[str, Any]]:
    """Analyse a list of prompts and return per-prompt metadata.

    Each dict contains:
        - intents: list[str]
        - primary_intent: str
        - topics: list[str]
        - geo_entities: list[str]
        - temporal_refs: list[str]
        - anti_patterns: list[str]
        - has_anti_pattern: bool
        - bigrams: list[str]
    """
    results: list[dict[str, Any]] = []
    for p in prompts:
        intents = classify_prompt_intent(p)
        topics = extract_prompt_topics(p)
        geo = extract_geographic_entities(p)
        temporal = extract_temporal_references(p)
        anti = detect_anti_patterns(p)
        bgs = extract_bigrams(p)
        results.append({
            "intents": intents,
            "primary_intent": intents[0] if intents else "Unclassified",
            "topics": topics,
            "geo_entities": geo,
            "temporal_refs": temporal,
            "anti_patterns": anti,
            "has_anti_pattern": bool(anti),
            "bigrams": bgs,
        })
    return results
