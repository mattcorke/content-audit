import streamlit as st
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from collections import Counter
import re
import json
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# ---------------------------------------------------------------------------
# Dismissals persistence
# ---------------------------------------------------------------------------

DISMISSALS_FILE = Path(__file__).parent / "dismissed_pairs.json"


def load_dismissed_pairs() -> set[tuple[str, str]]:
    """Load dismissed URL pairs from disk."""
    if not DISMISSALS_FILE.exists():
        return set()
    try:
        data = json.loads(DISMISSALS_FILE.read_text())
        return {(p[0], p[1]) for p in data}
    except (json.JSONDecodeError, KeyError):
        return set()


def save_dismissed_pairs(pairs: set[tuple[str, str]]) -> None:
    """Save dismissed URL pairs to disk."""
    DISMISSALS_FILE.write_text(json.dumps(sorted(pairs), indent=2))


def make_pair_key(url_a: str, url_b: str) -> tuple[str, str]:
    """Canonical key for a URL pair (sorted so order doesn't matter)."""
    return tuple(sorted([url_a, url_b]))


# ---------------------------------------------------------------------------
# Page content fetching & comparison
# ---------------------------------------------------------------------------

FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ContentAuditBot/1.0)",
}
STRIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"}


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_page(url: str) -> dict:
    """Fetch a URL and extract key content elements."""
    try:
        resp = requests.get(url, headers=FETCH_HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        return {"error": str(e)}

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove boilerplate tags
    for tag in soup.find_all(STRIP_TAGS):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""
    meta_desc = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_desc = meta_tag["content"].strip()

    headings = {}
    for level in ("h1", "h2", "h3"):
        headings[level] = [h.get_text(strip=True) for h in soup.find_all(level)]

    body_el = soup.find("main") or soup.find("article") or soup.body
    body_text = body_el.get_text(separator=" ", strip=True) if body_el else ""
    # Collapse whitespace
    body_text = re.sub(r"\s+", " ", body_text).strip()

    words = re.findall(r"[a-zA-Z]{3,}", body_text.lower())
    word_count = len(words)

    # Top keyword phrases (unigrams)
    stop = {"the", "and", "for", "that", "this", "with", "are", "from", "was",
            "were", "been", "have", "has", "had", "not", "but", "can", "will",
            "your", "you", "our", "all", "more", "about", "also", "into", "than",
            "its", "which", "their", "them", "other", "some", "what", "when",
            "how", "who", "may", "most", "any", "each", "only", "over", "such"}
    filtered = [w for w in words if w not in stop and len(w) > 2]
    top_keywords = Counter(filtered).most_common(20)

    return {
        "title": title,
        "meta_desc": meta_desc,
        "headings": headings,
        "body_text": body_text,
        "word_count": word_count,
        "top_keywords": top_keywords,
    }


def compare_pages(page_a: dict, page_b: dict) -> dict:
    """Compare two fetched pages and return similarity metrics."""
    if "error" in page_a or "error" in page_b:
        return {"error": True}

    # Body text cosine similarity
    texts = [page_a["body_text"], page_b["body_text"]]
    if all(t.strip() for t in texts):
        vec = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf = vec.fit_transform(texts)
        content_sim = cosine_similarity(tfidf)[0][1]
    else:
        content_sim = 0.0

    # Title similarity
    title_sim = fuzz.token_sort_ratio(page_a["title"], page_b["title"]) / 100

    # Shared keywords
    kw_a = {kw for kw, _ in page_a["top_keywords"]}
    kw_b = {kw for kw, _ in page_b["top_keywords"]}
    shared_kw = kw_a & kw_b

    # Shared headings (H2s — most indicative of subtopic overlap)
    h2_a = {h.lower().strip() for h in page_a["headings"].get("h2", [])}
    h2_b = {h.lower().strip() for h in page_b["headings"].get("h2", [])}
    shared_h2 = h2_a & h2_b
    # Fuzzy H2 matches
    fuzzy_h2 = set()
    for ha in h2_a - shared_h2:
        for hb in h2_b - shared_h2:
            if fuzz.token_sort_ratio(ha, hb) >= 75:
                fuzzy_h2.add((ha, hb))

    return {
        "content_similarity": round(content_sim, 3),
        "title_similarity": round(title_sim, 3),
        "shared_keywords": shared_kw,
        "shared_h2_exact": shared_h2,
        "shared_h2_fuzzy": fuzzy_h2,
    }

st.set_page_config(
    page_title="Content Audit Tool",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------

REQUIRED_CONCEPTS = [
    "url",
    "clicks",
    "revenue",
]

OPTIONAL_CONCEPTS = [
    "conversions",
    "sc_impressions",
    "sc_clicks",
    "googlebot_crawls",
    "content_type",
    "niche",
    "focus_kw",
    "h1",
]

COLUMN_ALIASES = {
    "url": ["url", "page", "landing page", "address", "page path", "post url", "post irl"],
    "clicks": ["clicks", "total clicks", "ga clicks", "sessions"],
    "conversions": ["conversions", "goal completions", "transactions", "conv"],
    "revenue": ["revenue", "transaction revenue", "total revenue", "rev", "revenue (aud)"],
    "sc_impressions": [
        "search console impressions",
        "sc impressions",
        "impressions",
        "gsc impressions",
    ],
    "sc_clicks": [
        "search console clicks",
        "sc clicks",
        "gsc clicks",
        "organic clicks",
    ],
    "googlebot_crawls": [
        "googlebot crawl",
        "googlebot crawls",
        "crawl count",
        "crawls",
        "bot hits",
        "googlebot",
    ],
    "content_type": [
        "content type",
        "content_type",
        "type",
        "page type",
        "template",
    ],
    "niche": [
        "niche",
        "category",
        "vertical",
        "topic",
        "segment",
    ],
    "focus_kw": [
        "focus kw",
        "focus_kw",
        "focus keyword",
        "target keyword",
        "primary keyword",
        "main keyword",
        "kw",
    ],
    "h1": [
        "h1",
        "h1 tag",
        "heading",
        "heading 1",
        "page heading",
        "main heading",
    ],
}

EXCLUDED_CONTENT_TYPES = {"experiment", "rewards", "white-label"}


def auto_map_columns(df_columns: list[str]) -> dict[str, str | None]:
    """Try to automatically match spreadsheet columns to required concepts."""
    mapping: dict[str, str | None] = {}
    lower_cols = {c.lower().strip(): c for c in df_columns}

    for concept, aliases in COLUMN_ALIASES.items():
        matched = None
        for alias in aliases:
            if alias in lower_cols:
                matched = lower_cols[alias]
                break
        mapping[concept] = matched
    return mapping


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def extract_path(url: str) -> str:
    """Return the path portion of a URL, stripped of query strings."""
    try:
        parsed = urlparse(str(url))
        path = parsed.path.rstrip("/") or "/"
        return path
    except Exception:
        return str(url)


def slugify_path(path: str) -> str:
    """Turn a URL path into space-separated tokens for TF-IDF."""
    path = re.sub(r"[/_\-\.]+", " ", path)
    path = re.sub(r"\d{4,}", "", path)  # strip long numbers (dates, IDs)
    return path.strip().lower()


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def strip_niche_prefix(path: str) -> str:
    """Remove the first directory segment (niche) from a URL path.

    e.g. "/travel-insurance/best-policies" -> "/best-policies"
         "/credit-cards/compare"          -> "/compare"
         "/simple-page"                   -> "/simple-page" (no change)
    """
    segments = path.strip("/").split("/")
    if len(segments) > 1:
        return "/" + "/".join(segments[1:])
    return path


def compute_similarity_matrix(paths: list[str], threshold: float) -> list[tuple]:
    """Return pairs of (i, j, score) where cosine similarity >= threshold."""
    paths = [strip_niche_prefix(p) for p in paths]
    slugs = [slugify_path(p) for p in paths]

    # Filter out empty slugs
    valid = [(i, s) for i, s in enumerate(slugs) if s.strip()]
    if len(valid) < 2:
        return []

    indices, texts = zip(*valid)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    tfidf = vec.fit_transform(texts)
    cos = cosine_similarity(tfidf)

    pairs = []
    for a in range(len(indices)):
        for b in range(a + 1, len(indices)):
            score = cos[a][b]
            if score >= threshold:
                # Also check token-level fuzzy ratio for extra confidence
                fuzz_score = fuzz.token_sort_ratio(texts[a], texts[b]) / 100
                combined = (score + fuzz_score) / 2
                if combined >= threshold:
                    pairs.append((indices[a], indices[b], round(combined, 3)))
    return pairs


def compute_kw_h1_boost(row_a, row_b, has_focus_kw: bool, has_h1: bool) -> float:
    """Return a boost (0.0–0.3) based on Focus KW and H1 overlap."""
    boost = 0.0

    if has_focus_kw:
        kw_a = str(row_a.get("focus_kw", "")).strip().lower()
        kw_b = str(row_b.get("focus_kw", "")).strip().lower()
        if kw_a and kw_b and kw_a != "nan" and kw_b != "nan":
            if kw_a == kw_b:
                boost += 0.2  # exact keyword match — strong signal
            else:
                kw_sim = fuzz.token_sort_ratio(kw_a, kw_b) / 100
                if kw_sim >= 0.7:
                    boost += 0.2 * kw_sim  # partial match scaled

    if has_h1:
        h1_a = str(row_a.get("h1", "")).strip().lower()
        h1_b = str(row_b.get("h1", "")).strip().lower()
        if h1_a and h1_b and h1_a != "nan" and h1_b != "nan":
            h1_sim = fuzz.token_sort_ratio(h1_a, h1_b) / 100
            if h1_sim >= 0.6:
                boost += 0.1 * h1_sim  # H1 similarity scaled

    return boost


def find_cannibalization(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Find groups of URLs with similar paths (potential cannibalization)."""
    has_focus_kw = "focus_kw" in df.columns
    has_h1 = "h1" in df.columns

    paths = df["_path"].tolist()
    # Use a lower internal threshold so boosted pairs aren't missed
    internal_threshold = max(threshold - 0.15, 0.2) if (has_focus_kw or has_h1) else threshold
    pairs = compute_similarity_matrix(paths, internal_threshold)

    if not pairs:
        return pd.DataFrame()

    rows = []
    for i, j, url_score in pairs:
        row_a = df.iloc[i]
        row_b = df.iloc[j]
        boost = compute_kw_h1_boost(row_a, row_b, has_focus_kw, has_h1)
        combined = min(round(url_score + boost, 3), 1.0)

        if combined < threshold:
            continue

        row = {
            "URL A": row_a["url"],
            "URL B": row_b["url"],
            "Score": combined,
            "URL Similarity": url_score,
        }
        if has_focus_kw:
            row["Focus KW A"] = row_a.get("focus_kw", "")
            row["Focus KW B"] = row_b.get("focus_kw", "")
        if has_h1:
            row["H1 A"] = row_a.get("h1", "")
            row["H1 B"] = row_b.get("h1", "")
        row["Clicks A"] = row_a["clicks"]
        row["Clicks B"] = row_b["clicks"]
        if "conversions" in df.columns and df["conversions"].any():
            row["Conversions A"] = row_a["conversions"]
            row["Conversions B"] = row_b["conversions"]
        row["Revenue A"] = row_a["revenue"]
        row["Revenue B"] = row_b["revenue"]
        if "sc_impressions" in df.columns and df["sc_impressions"].any():
            row["SC Impressions A"] = row_a["sc_impressions"]
            row["SC Impressions B"] = row_b["sc_impressions"]
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).sort_values("Score", ascending=False)
    return result


def find_merge_candidates(
    df: pd.DataFrame, threshold: float, click_cap: int, impression_cap: int | None
) -> pd.DataFrame:
    """Find similar URLs where BOTH are low-performing — good merge targets."""
    paths = df["_path"].tolist()
    pairs = compute_similarity_matrix(paths, threshold)

    if not pairs:
        return pd.DataFrame()

    has_impressions = df["sc_impressions"].any()
    has_conversions = df["conversions"].any()

    rows = []
    for i, j, score in pairs:
        a = df.iloc[i]
        b = df.iloc[j]
        both_low_clicks = a["clicks"] <= click_cap and b["clicks"] <= click_cap
        both_low_impressions = (
            has_impressions
            and impression_cap is not None
            and a["sc_impressions"] <= impression_cap
            and b["sc_impressions"] <= impression_cap
        )
        if both_low_clicks or both_low_impressions:
            row = {
                "URL A": a["url"],
                "URL B": b["url"],
                "Similarity": score,
                "Combined Clicks": a["clicks"] + b["clicks"],
                "Combined Revenue": a["revenue"] + b["revenue"],
                "Clicks A": a["clicks"],
                "Clicks B": b["clicks"],
            }
            if has_conversions:
                row["Combined Conversions"] = a["conversions"] + b["conversions"]
            if has_impressions:
                row["SC Impressions A"] = a["sc_impressions"]
                row["SC Impressions B"] = b["sc_impressions"]
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Similarity", ascending=False)


def find_delete_candidates(
    df: pd.DataFrame,
    max_clicks: int,
    max_revenue: float,
    max_conversions: int | None = None,
    max_impressions: int | None = None,
    max_crawls: int | None = None,
) -> pd.DataFrame:
    """Flag pages with universally poor metrics as delete candidates."""
    mask = (df["clicks"] <= max_clicks) & (df["revenue"] <= max_revenue)

    if max_conversions is not None and df["conversions"].any():
        mask = mask & (df["conversions"] <= max_conversions)
    if max_impressions is not None and df["sc_impressions"].any():
        mask = mask & (df["sc_impressions"] <= max_impressions)
    if max_crawls is not None and df["googlebot_crawls"].any():
        mask = mask & (df["googlebot_crawls"] <= max_crawls)

    # Build display columns based on what's available
    display_cols = ["url", "clicks", "revenue"]
    if df["conversions"].any():
        display_cols.append("conversions")
    if df["sc_impressions"].any():
        display_cols.append("sc_impressions")
    if df["sc_clicks"].any():
        display_cols.append("sc_clicks")
    if df["googlebot_crawls"].any():
        display_cols.append("googlebot_crawls")

    result = df[mask][display_cols].copy()
    sort_cols = ["clicks", "revenue"]
    if "sc_impressions" in display_cols:
        sort_cols.insert(1, "sc_impressions")
    result = result.sort_values(sort_cols, ascending=True)
    return result


def score_pages(df: pd.DataFrame, mapped_concepts: set[str]) -> pd.DataFrame:
    """Add a composite health score (0-100) to every page."""
    scored = df.copy()

    def pct_rank(series):
        return series.rank(pct=True, method="average").fillna(0)

    # Build weights dynamically based on available columns
    weights: dict[str, float] = {}
    weights["clicks"] = 30
    weights["revenue"] = 30
    if "conversions" in mapped_concepts:
        weights["conversions"] = 20
    if "sc_impressions" in mapped_concepts:
        weights["sc_impressions"] = 15
    if "googlebot_crawls" in mapped_concepts:
        weights["googlebot_crawls"] = 10

    # Normalise weights to sum to 100
    total = sum(weights.values())
    weights = {k: v / total * 100 for k, v in weights.items()}

    tmp_cols = []
    score = pd.Series(0.0, index=scored.index)
    for col, weight in weights.items():
        tmp = f"_{col}_pct"
        scored[tmp] = pct_rank(scored[col])
        score += scored[tmp] * weight
        tmp_cols.append(tmp)

    scored["health_score"] = score.round(1)
    return scored.drop(columns=tmp_cols)


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.title("🔍 Content Audit Tool")
st.markdown(
    "Upload your spreadsheet to find **keyword cannibalization**, "
    "**merge opportunities**, and **pages to delete**."
)

uploaded = st.file_uploader(
    "Upload spreadsheet (CSV or Excel)", type=["csv", "xlsx", "xls"]
)

if uploaded is None:
    st.info("👆 Upload a file to get started.")
    st.stop()

# Header row selection
header_row = st.number_input(
    "Header row number",
    min_value=1,
    max_value=20,
    value=1,
    help="Which row contains the column headers? (1 = first row, 2 = second row, etc.)",
)
header_idx = header_row - 1  # pandas uses 0-based indexing

# Load data
if uploaded.name.endswith(".csv"):
    raw_df = pd.read_csv(uploaded, header=header_idx)
else:
    raw_df = pd.read_excel(uploaded, header=header_idx)

st.success(f"Loaded **{len(raw_df):,}** rows and **{len(raw_df.columns)}** columns.")

# --- Column mapping ---
st.subheader("📋 Column Mapping")

auto = auto_map_columns(raw_df.columns.tolist())
col_map: dict[str, str] = {}

st.markdown("**Required columns**")
cols = st.columns(2)
for idx, concept in enumerate(REQUIRED_CONCEPTS):
    with cols[idx % 2]:
        default_idx = 0
        options = ["— not mapped —"] + list(raw_df.columns)
        if auto.get(concept):
            try:
                default_idx = options.index(auto[concept])
            except ValueError:
                default_idx = 0
        choice = st.selectbox(
            f"**{concept.replace('_', ' ').title()}**",
            options,
            index=default_idx,
            key=f"map_{concept}",
        )
        if choice != "— not mapped —":
            col_map[concept] = choice

missing = [c for c in REQUIRED_CONCEPTS if c not in col_map]
if missing:
    st.warning(f"Please map all required columns. Missing: {', '.join(missing)}")
    st.stop()

st.markdown("**Optional columns** (for enhanced analysis)")
opt_cols = st.columns(2)
for idx, concept in enumerate(OPTIONAL_CONCEPTS):
    with opt_cols[idx % 2]:
        default_idx = 0
        options = ["— not mapped —"] + list(raw_df.columns)
        if auto.get(concept):
            try:
                default_idx = options.index(auto[concept])
            except ValueError:
                default_idx = 0
        choice = st.selectbox(
            f"**{concept.replace('_', ' ').title()}**",
            options,
            index=default_idx,
            key=f"map_{concept}",
        )
        if choice != "— not mapped —":
            col_map[concept] = choice

# Build working dataframe with canonical names
df = pd.DataFrame()
for concept, src_col in col_map.items():
    df[concept] = raw_df[src_col]

# Coerce numerics (only mapped columns)
NUMERIC_CONCEPTS = ["clicks", "conversions", "revenue", "sc_impressions", "sc_clicks", "googlebot_crawls"]
for c in NUMERIC_CONCEPTS:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    else:
        df[c] = 0  # default to 0 for unmapped numeric columns

df["url"] = df["url"].astype(str)
df["_path"] = df["url"].apply(extract_path)

# Coerce text columns
for text_col in ("focus_kw", "h1"):
    if text_col in df.columns:
        df[text_col] = df[text_col].astype(str).str.strip()

# --- Filters (sidebar) ---
st.sidebar.header("Filters")

rows_before = len(df)

# Auto-exclude content types: experiment, rewards, white-label
if "content_type" in col_map:
    df["content_type"] = df["content_type"].astype(str).str.strip()
    ct_lower = df["content_type"].str.lower()
    excluded_mask = ct_lower.isin(EXCLUDED_CONTENT_TYPES)
    n_excluded = excluded_mask.sum()
    df = df[~excluded_mask].reset_index(drop=True)
    st.sidebar.markdown(
        f"**Content Type:** auto-excluded **{n_excluded:,}** rows "
        f"(experiment, rewards, white-label)"
    )

    # Also let user exclude additional content types
    remaining_types = sorted(df["content_type"].unique())
    extra_exclude = st.sidebar.multiselect(
        "Exclude additional content types",
        remaining_types,
        key="extra_ct_exclude",
    )
    if extra_exclude:
        df = df[~df["content_type"].isin(extra_exclude)].reset_index(drop=True)
else:
    st.sidebar.info("Map **Content Type** column to enable content type filtering.")

# Niche filter
if "niche" in col_map:
    df["niche"] = df["niche"].astype(str).str.strip()
    all_niches = sorted(df["niche"].unique())
    selected_niches = st.sidebar.multiselect(
        "Filter by Niche",
        all_niches,
        default=all_niches,
        key="niche_filter",
    )
    if selected_niches and len(selected_niches) < len(all_niches):
        df = df[df["niche"].isin(selected_niches)].reset_index(drop=True)
else:
    st.sidebar.info("Map **Niche** column to enable niche filtering.")

rows_after = len(df)
if rows_after < rows_before:
    st.sidebar.markdown(f"---\n**{rows_before - rows_after:,}** rows filtered out. "
                        f"**{rows_after:,}** rows remaining.")

# --- Overview ---
st.divider()
st.subheader("📊 Overview")

has_conversions = "conversions" in col_map
has_sc_impressions = "sc_impressions" in col_map
has_sc_clicks = "sc_clicks" in col_map
has_googlebot = "googlebot_crawls" in col_map

metric_cols = st.columns(3 + int(has_conversions) + int(has_sc_impressions))
idx = 0
metric_cols[idx].metric("Total Pages", f"{len(df):,}"); idx += 1
metric_cols[idx].metric("Total Clicks", f"{int(df['clicks'].sum()):,}"); idx += 1
if has_conversions:
    metric_cols[idx].metric("Total Conversions", f"{int(df['conversions'].sum()):,}"); idx += 1
metric_cols[idx].metric("Total Revenue", f"${df['revenue'].sum():,.2f}"); idx += 1
if has_sc_impressions:
    metric_cols[idx].metric("Zero-Impression Pages", f"{(df['sc_impressions'] == 0).sum():,}"); idx += 1

# Pareto chart
scored_df = score_pages(df, set(col_map.keys()))

fig_pareto = go.Figure()
sorted_by_rev = scored_df.sort_values("revenue", ascending=False).reset_index(
    drop=True
)
sorted_by_rev["cumulative_pct"] = (
    sorted_by_rev["revenue"].cumsum() / sorted_by_rev["revenue"].sum() * 100
)
sorted_by_rev["page_pct"] = (
    (sorted_by_rev.index + 1) / len(sorted_by_rev) * 100
)

fig_pareto.add_trace(
    go.Scatter(
        x=sorted_by_rev["page_pct"],
        y=sorted_by_rev["cumulative_pct"],
        mode="lines",
        name="Cumulative Revenue %",
        line=dict(color="#636EFA", width=2),
    )
)
fig_pareto.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80%")
fig_pareto.update_layout(
    title="Revenue Pareto: % of pages generating 80% of revenue",
    xaxis_title="% of Pages (sorted by revenue desc)",
    yaxis_title="Cumulative % of Revenue",
    height=350,
)
st.plotly_chart(fig_pareto, use_container_width=True)

# Health score distribution
fig_health = px.histogram(
    scored_df,
    x="health_score",
    nbins=20,
    title="Page Health Score Distribution",
    labels={"health_score": "Health Score (0-100)"},
    color_discrete_sequence=["#636EFA"],
)
fig_health.update_layout(height=300)
st.plotly_chart(fig_health, use_container_width=True)

# --- Tabs ---
st.divider()
tab_cannibal, tab_merge, tab_delete, tab_all = st.tabs(
    ["🔴 Cannibalization", "🟡 Merge Opportunities", "🗑️ Delete Candidates", "📄 All Pages"]
)

# --- Cannibalization ---
with tab_cannibal:
    st.markdown(
        "Pages with very similar URL paths that may be **competing for the same keywords**. "
        "The stronger page cannibalizes the weaker one."
    )
    if "focus_kw" in col_map or "h1" in col_map:
        st.caption(
            "Score includes URL similarity"
            + (" + Focus KW overlap" if "focus_kw" in col_map else "")
            + (" + H1 similarity" if "h1" in col_map else "")
            + "."
        )
    sim_threshold = st.slider(
        "Similarity threshold",
        0.3,
        1.0,
        0.6,
        0.05,
        key="cannibal_thresh",
        help="Lower = more results (looser matching). Higher = stricter.",
    )

    with st.spinner("Computing URL similarity…"):
        cannibal_df = find_cannibalization(df, sim_threshold)

    if cannibal_df.empty:
        st.success("No cannibalization detected at this threshold.")
    else:
        # Filter out dismissed pairs
        dismissed = load_dismissed_pairs()
        if dismissed:
            keep_mask = cannibal_df.apply(
                lambda r: make_pair_key(r["URL A"], r["URL B"]) not in dismissed,
                axis=1,
            )
            hidden_count = (~keep_mask).sum()
            cannibal_df = cannibal_df[keep_mask].reset_index(drop=True)
        else:
            hidden_count = 0

        if cannibal_df.empty:
            st.success("All pairs have been dismissed.")
        else:
            count_msg = f"Found **{len(cannibal_df):,}** potential cannibalization pairs."
            if hidden_count:
                count_msg += f" ({hidden_count} dismissed pairs hidden)"
            st.warning(count_msg)

            # Add dismiss checkbox column
            cannibal_df.insert(0, "Dismiss", False)
            edited = st.data_editor(
                cannibal_df,
                use_container_width=True,
                height=500,
                key="cannibal_editor",
                column_config={"Dismiss": st.column_config.CheckboxColumn("Dismiss", default=False)},
            )

            to_dismiss = edited[edited["Dismiss"] == True]
            if len(to_dismiss) > 0:
                if st.button(f"Dismiss {len(to_dismiss)} selected pair(s)", type="primary"):
                    new_dismissed = dismissed.copy()
                    for _, row in to_dismiss.iterrows():
                        new_dismissed.add(make_pair_key(row["URL A"], row["URL B"]))
                    save_dismissed_pairs(new_dismissed)
                    st.rerun()

        # Restore dismissed pairs button
        if hidden_count > 0:
            if st.button(f"Restore {hidden_count} dismissed pair(s)"):
                save_dismissed_pairs(set())
                st.rerun()

    # --- Deep Dive ---
    st.divider()
    st.subheader("🔬 Deep Dive")
    st.markdown(
        "Select a pair to fetch both pages and compare their **actual content** "
        "— body text similarity, shared keywords, and overlapping headings."
    )

    if not cannibal_df.empty:
        # Build pair options from the (possibly filtered) cannibal_df
        display_cannibal = cannibal_df.drop(columns=["Dismiss"], errors="ignore")
        pair_labels = [
            f"{row['URL A']}  ↔  {row['URL B']}  (score: {row['Score']})"
            for _, row in display_cannibal.iterrows()
        ]
        selected_pair = st.selectbox("Select a pair", pair_labels, key="deep_dive_pair")

        if st.button("Fetch & Compare", type="primary", key="deep_dive_btn"):
            pair_idx = pair_labels.index(selected_pair)
            pair_row = display_cannibal.iloc[pair_idx]
            url_a, url_b = pair_row["URL A"], pair_row["URL B"]

            col_a, col_b = st.columns(2)

            with st.spinner(f"Fetching pages…"):
                page_a = fetch_page(url_a)
                page_b = fetch_page(url_b)

            # Check for errors
            if "error" in page_a:
                st.error(f"Failed to fetch **{url_a}**: {page_a['error']}")
            if "error" in page_b:
                st.error(f"Failed to fetch **{url_b}**: {page_b['error']}")

            if "error" not in page_a and "error" not in page_b:
                comparison = compare_pages(page_a, page_b)

                # Similarity scores
                s1, s2 = st.columns(2)
                s1.metric("Content Similarity", f"{comparison['content_similarity']:.0%}")
                s2.metric("Title Similarity", f"{comparison['title_similarity']:.0%}")

                # Side-by-side page details
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Page A:** `{url_a}`")
                    st.markdown(f"**Title:** {page_a['title']}")
                    st.markdown(f"**Meta:** {page_a['meta_desc'][:200]}")
                    st.markdown(f"**Word count:** {page_a['word_count']:,}")
                    if page_a["headings"]["h1"]:
                        st.markdown(f"**H1:** {', '.join(page_a['headings']['h1'])}")
                    if page_a["headings"]["h2"]:
                        st.markdown("**H2s:**")
                        for h in page_a["headings"]["h2"]:
                            st.markdown(f"- {h}")

                with col_b:
                    st.markdown(f"**Page B:** `{url_b}`")
                    st.markdown(f"**Title:** {page_b['title']}")
                    st.markdown(f"**Meta:** {page_b['meta_desc'][:200]}")
                    st.markdown(f"**Word count:** {page_b['word_count']:,}")
                    if page_b["headings"]["h1"]:
                        st.markdown(f"**H1:** {', '.join(page_b['headings']['h1'])}")
                    if page_b["headings"]["h2"]:
                        st.markdown("**H2s:**")
                        for h in page_b["headings"]["h2"]:
                            st.markdown(f"- {h}")

                # Shared analysis
                st.divider()
                if comparison["shared_keywords"]:
                    st.markdown(
                        f"**Shared top keywords ({len(comparison['shared_keywords'])}):** "
                        + ", ".join(sorted(comparison["shared_keywords"]))
                    )
                else:
                    st.markdown("**Shared top keywords:** None")

                if comparison["shared_h2_exact"]:
                    st.markdown(
                        f"**Identical H2 headings ({len(comparison['shared_h2_exact'])}):** "
                        + ", ".join(sorted(comparison["shared_h2_exact"]))
                    )
                if comparison["shared_h2_fuzzy"]:
                    st.markdown(
                        f"**Similar H2 headings ({len(comparison['shared_h2_fuzzy'])}):**"
                    )
                    for ha, hb in comparison["shared_h2_fuzzy"]:
                        st.markdown(f'- "{ha}" ↔ "{hb}"')

                # Verdict
                st.divider()
                cs = comparison["content_similarity"]
                if cs >= 0.5:
                    st.error(
                        f"**High content overlap ({cs:.0%})** — these pages are likely "
                        "cannibalizing each other. Consider merging or differentiating."
                    )
                elif cs >= 0.3:
                    st.warning(
                        f"**Moderate content overlap ({cs:.0%})** — some shared topics. "
                        "Review the shared keywords and headings to decide."
                    )
                else:
                    st.success(
                        f"**Low content overlap ({cs:.0%})** — probably a false positive. "
                        "Consider dismissing this pair."
                    )
    else:
        st.info("No cannibalization pairs to deep dive into.")

# --- Merge ---
with tab_merge:
    st.markdown(
        "Similar pages where **both are underperforming** — consolidating them "
        "into a single stronger page could improve rankings."
    )
    if has_sc_impressions:
        c1, c2, c3 = st.columns(3)
    else:
        c1, c2 = st.columns(2)
    merge_thresh = c1.slider(
        "Similarity threshold", 0.3, 1.0, 0.55, 0.05, key="merge_thresh"
    )
    merge_click_cap = c2.number_input(
        "Max clicks (each page)", value=50, min_value=0, key="merge_clicks"
    )
    merge_imp_cap = None
    if has_sc_impressions:
        merge_imp_cap = c3.number_input(
            "Max impressions (each page)", value=500, min_value=0, key="merge_imp"
        )

    with st.spinner("Finding merge candidates…"):
        merge_df = find_merge_candidates(
            df, merge_thresh, merge_click_cap, merge_imp_cap
        )

    if merge_df.empty:
        st.success("No merge opportunities found with these settings.")
    else:
        st.info(f"Found **{len(merge_df):,}** merge opportunity pairs.")
        st.dataframe(merge_df, use_container_width=True, height=500)

# --- Delete ---
with tab_delete:
    st.markdown(
        "Pages failing across **all available metrics** — strong candidates "
        "for removal or noindex."
    )
    d1, d2 = st.columns(2)
    del_clicks = d1.number_input("Max clicks", value=5, min_value=0, key="del_clicks")
    del_rev = d2.number_input(
        "Max revenue ($)", value=0.0, min_value=0.0, key="del_rev"
    )

    del_conv = None
    del_imp = None
    del_crawl = None
    extra_cols = [c for c in [
        ("conversions", has_conversions),
        ("sc_impressions", has_sc_impressions),
        ("googlebot_crawls", has_googlebot),
    ] if c[1]]
    if extra_cols:
        extra_st_cols = st.columns(len(extra_cols))
        for i, (concept, _) in enumerate(extra_cols):
            if concept == "conversions":
                del_conv = extra_st_cols[i].number_input(
                    "Max conversions", value=0, min_value=0, key="del_conv"
                )
            elif concept == "sc_impressions":
                del_imp = extra_st_cols[i].number_input(
                    "Max SC impressions", value=100, min_value=0, key="del_imp"
                )
            elif concept == "googlebot_crawls":
                del_crawl = extra_st_cols[i].number_input(
                    "Max GoogleBot crawls", value=2, min_value=0, key="del_crawl"
                )

    delete_df = find_delete_candidates(
        df, del_clicks, del_rev, del_conv, del_imp, del_crawl
    )

    if delete_df.empty:
        st.success("No delete candidates with these thresholds.")
    else:
        pct = len(delete_df) / len(df) * 100
        st.error(
            f"**{len(delete_df):,}** pages ({pct:.1f}% of site) are delete candidates."
        )

        # Impact summary
        n_impact = 2 + int(has_googlebot)
        impact_cols = st.columns(n_impact)
        impact_cols[0].metric(
            "Total clicks on these pages", f"{int(delete_df['clicks'].sum()):,}"
        )
        impact_cols[1].metric(
            "Total revenue on these pages",
            f"${delete_df['revenue'].sum():,.2f}",
        )
        if has_googlebot:
            impact_cols[2].metric(
                "Avg crawl budget used",
                f"{delete_df['googlebot_crawls'].mean():.1f} crawls/page",
            )

        st.dataframe(delete_df, use_container_width=True, height=500)

# --- All Pages ---
with tab_all:
    st.markdown("All pages with **health scores**. Sort by any column.")
    all_page_cols = ["url", "health_score", "clicks", "revenue"]
    if has_conversions:
        all_page_cols.append("conversions")
    if has_sc_impressions:
        all_page_cols.append("sc_impressions")
    if has_sc_clicks:
        all_page_cols.append("sc_clicks")
    if has_googlebot:
        all_page_cols.append("googlebot_crawls")
    display_df = scored_df[all_page_cols].sort_values("health_score", ascending=True)
    st.dataframe(display_df, use_container_width=True, height=600)

# --- Export ---
st.divider()
st.subheader("📥 Export Results")

sheets = {"All Pages": display_df}
if not cannibal_df.empty:
    sheets["Cannibalization"] = cannibal_df
if not merge_df.empty:
    sheets["Merge Opportunities"] = merge_df
if not delete_df.empty:
    sheets["Delete Candidates"] = delete_df

excel_bytes = to_excel_bytes(sheets)
st.download_button(
    "⬇️ Download Full Report (Excel)",
    data=excel_bytes,
    file_name="content_audit_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
