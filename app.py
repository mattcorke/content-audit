import streamlit as st
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

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
    "conversions",
    "revenue",
    "sc_impressions",
    "sc_clicks",
    "googlebot_crawls",
]

OPTIONAL_CONCEPTS = [
    "content_type",
    "niche",
    "focus_kw",
    "h1",
]

COLUMN_ALIASES = {
    "url": ["url", "page", "landing page", "address", "page path"],
    "clicks": ["clicks", "total clicks", "ga clicks", "sessions"],
    "conversions": ["conversions", "goal completions", "transactions", "conv"],
    "revenue": ["revenue", "transaction revenue", "total revenue", "rev"],
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


def compute_similarity_matrix(paths: list[str], threshold: float) -> list[tuple]:
    """Return pairs of (i, j, score) where cosine similarity >= threshold."""
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
        row.update(
            {
                "Clicks A": row_a["clicks"],
                "Clicks B": row_b["clicks"],
                "Conversions A": row_a["conversions"],
                "Conversions B": row_b["conversions"],
                "Revenue A": row_a["revenue"],
                "Revenue B": row_b["revenue"],
                "SC Impressions A": row_a["sc_impressions"],
                "SC Impressions B": row_b["sc_impressions"],
            }
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).sort_values("Score", ascending=False)
    return result


def find_merge_candidates(
    df: pd.DataFrame, threshold: float, click_cap: int, impression_cap: int
) -> pd.DataFrame:
    """Find similar URLs where BOTH are low-performing — good merge targets."""
    paths = df["_path"].tolist()
    pairs = compute_similarity_matrix(paths, threshold)

    if not pairs:
        return pd.DataFrame()

    rows = []
    for i, j, score in pairs:
        a = df.iloc[i]
        b = df.iloc[j]
        both_low_clicks = a["clicks"] <= click_cap and b["clicks"] <= click_cap
        both_low_impressions = (
            a["sc_impressions"] <= impression_cap
            and b["sc_impressions"] <= impression_cap
        )
        if both_low_clicks or both_low_impressions:
            combined_clicks = a["clicks"] + b["clicks"]
            combined_rev = a["revenue"] + b["revenue"]
            combined_conversions = a["conversions"] + b["conversions"]
            rows.append(
                {
                    "URL A": a["url"],
                    "URL B": b["url"],
                    "Similarity": score,
                    "Combined Clicks": combined_clicks,
                    "Combined Conversions": combined_conversions,
                    "Combined Revenue": combined_rev,
                    "Clicks A": a["clicks"],
                    "Clicks B": b["clicks"],
                    "SC Impressions A": a["sc_impressions"],
                    "SC Impressions B": b["sc_impressions"],
                }
            )

    return pd.DataFrame(rows).sort_values("Similarity", ascending=False)


def find_delete_candidates(
    df: pd.DataFrame,
    max_clicks: int,
    max_conversions: int,
    max_revenue: float,
    max_impressions: int,
    max_crawls: int,
) -> pd.DataFrame:
    """Flag pages with universally poor metrics as delete candidates."""
    mask = (
        (df["clicks"] <= max_clicks)
        & (df["conversions"] <= max_conversions)
        & (df["revenue"] <= max_revenue)
        & (df["sc_impressions"] <= max_impressions)
        & (df["googlebot_crawls"] <= max_crawls)
    )
    result = df[mask][
        [
            "url",
            "clicks",
            "conversions",
            "revenue",
            "sc_impressions",
            "sc_clicks",
            "googlebot_crawls",
        ]
    ].copy()
    result = result.sort_values(
        ["clicks", "sc_impressions", "revenue"], ascending=True
    )
    return result


def score_pages(df: pd.DataFrame) -> pd.DataFrame:
    """Add a composite health score (0-100) to every page."""
    scored = df.copy()

    def pct_rank(series):
        return series.rank(pct=True, method="average").fillna(0)

    scored["_click_pct"] = pct_rank(scored["clicks"])
    scored["_conv_pct"] = pct_rank(scored["conversions"])
    scored["_rev_pct"] = pct_rank(scored["revenue"])
    scored["_imp_pct"] = pct_rank(scored["sc_impressions"])
    scored["_crawl_pct"] = pct_rank(scored["googlebot_crawls"])

    scored["health_score"] = (
        scored["_click_pct"] * 25
        + scored["_conv_pct"] * 25
        + scored["_rev_pct"] * 25
        + scored["_imp_pct"] * 15
        + scored["_crawl_pct"] * 10
    ).round(1)

    return scored.drop(
        columns=["_click_pct", "_conv_pct", "_rev_pct", "_imp_pct", "_crawl_pct"]
    )


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

# Load data
if uploaded.name.endswith(".csv"):
    raw_df = pd.read_csv(uploaded)
else:
    raw_df = pd.read_excel(uploaded)

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
    st.warning(f"Please map all columns. Missing: {', '.join(missing)}")
    st.stop()

st.markdown("**Optional columns** (for filtering)")
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

# Coerce numerics
for c in REQUIRED_CONCEPTS[1:]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

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

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Pages", f"{len(df):,}")
m2.metric("Total Clicks", f"{int(df['clicks'].sum()):,}")
m3.metric("Total Conversions", f"{int(df['conversions'].sum()):,}")
m4.metric("Total Revenue", f"${df['revenue'].sum():,.2f}")
m5.metric("Zero-Click Pages", f"{(df['clicks'] == 0).sum():,}")

# Pareto chart
scored_df = score_pages(df)

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
        st.warning(f"Found **{len(cannibal_df):,}** potential cannibalization pairs.")
        st.dataframe(cannibal_df, use_container_width=True, height=500)

# --- Merge ---
with tab_merge:
    st.markdown(
        "Similar pages where **both are underperforming** — consolidating them "
        "into a single stronger page could improve rankings."
    )
    c1, c2, c3 = st.columns(3)
    merge_thresh = c1.slider(
        "Similarity threshold", 0.3, 1.0, 0.55, 0.05, key="merge_thresh"
    )
    merge_click_cap = c2.number_input(
        "Max clicks (each page)", value=50, min_value=0, key="merge_clicks"
    )
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
        "Pages failing across **all metrics** — low clicks, no conversions, "
        "no revenue, low impressions, and low crawl activity. Strong candidates "
        "for removal or noindex."
    )
    d1, d2, d3 = st.columns(3)
    del_clicks = d1.number_input("Max clicks", value=5, min_value=0, key="del_clicks")
    del_conv = d2.number_input(
        "Max conversions", value=0, min_value=0, key="del_conv"
    )
    del_rev = d3.number_input(
        "Max revenue ($)", value=0.0, min_value=0.0, key="del_rev"
    )
    d4, d5 = st.columns(2)
    del_imp = d4.number_input(
        "Max SC impressions", value=100, min_value=0, key="del_imp"
    )
    del_crawl = d5.number_input(
        "Max GoogleBot crawls", value=2, min_value=0, key="del_crawl"
    )

    delete_df = find_delete_candidates(
        df, del_clicks, del_conv, del_rev, del_imp, del_crawl
    )

    if delete_df.empty:
        st.success("No delete candidates with these thresholds.")
    else:
        pct = len(delete_df) / len(df) * 100
        st.error(
            f"**{len(delete_df):,}** pages ({pct:.1f}% of site) are delete candidates."
        )

        # Impact summary
        ic1, ic2, ic3 = st.columns(3)
        ic1.metric(
            "Total clicks on these pages", f"{int(delete_df['clicks'].sum()):,}"
        )
        ic2.metric(
            "Total revenue on these pages",
            f"${delete_df['revenue'].sum():,.2f}",
        )
        ic3.metric(
            "Avg crawl budget used",
            f"{delete_df['googlebot_crawls'].mean():.1f} crawls/page",
        )

        st.dataframe(delete_df, use_container_width=True, height=500)

# --- All Pages ---
with tab_all:
    st.markdown("All pages with **health scores**. Sort by any column.")
    display_df = scored_df[
        [
            "url",
            "health_score",
            "clicks",
            "conversions",
            "revenue",
            "sc_impressions",
            "sc_clicks",
            "googlebot_crawls",
        ]
    ].sort_values("health_score", ascending=True)
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
