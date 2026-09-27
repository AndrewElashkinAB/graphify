import json
from pathlib import Path
import networkx as nx
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections
from graphify.report import generate

FIXTURES = Path(__file__).parent / "fixtures"

def make_inputs():
    extraction = json.loads((FIXTURES / "extraction.json").read_text())
    G = build_from_json(extraction)
    communities = cluster(G)
    cohesion = score_all(G, communities)
    labels = {cid: f"Community {cid}" for cid in communities}
    gods = god_nodes(G)
    surprises = surprising_connections(G)
    detection = {"total_files": 4, "total_words": 62400, "needs_graph": True, "warning": None}
    tokens = {"input": extraction["input_tokens"], "output": extraction["output_tokens"]}
    return G, communities, cohesion, labels, gods, surprises, detection, tokens

def test_report_contains_header():
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    report = generate(G, communities, cohesion, labels, gods, surprises, detection, tokens, "./project")
    assert "# Graph Report" in report

def test_report_contains_corpus_check():
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    report = generate(G, communities, cohesion, labels, gods, surprises, detection, tokens, "./project")
    assert "## Corpus Check" in report


def test_report_surfaces_unclassified_files():
    """#3511: detect() already tracks files it saw but could not classify
    (no supported extension), but nothing surfaced them -- a corpus that is
    mostly an unsupported language got the same "well covered" verdict as
    one that was actually extracted."""
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    detection = {
        **detection,
        "unclassified": ["Main.lean", "Util.lean", "a.toml", "b.toml", "c.toml", "readme"],
    }
    report = generate(G, communities, cohesion, labels, gods, surprises, detection, tokens, "./project")
    assert "Unclassified: 6 file(s)" in report
    assert ".lean 2" in report
    assert ".toml 3" in report


def test_report_omits_unclassified_line_when_none():
    """Backward compatible: no unclassified files, no new line."""
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    report = generate(G, communities, cohesion, labels, gods, surprises, detection, tokens, "./project")
    assert "Unclassified:" not in report

def test_report_contains_god_nodes():
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    report = generate(G, communities, cohesion, labels, gods, surprises, detection, tokens, "./project")
    assert "## God Nodes" in report

def test_report_contains_surprising_connections():
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    report = generate(G, communities, cohesion, labels, gods, surprises, detection, tokens, "./project")
    assert "## Surprising Connections" in report

def test_report_contains_communities():
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    report = generate(G, communities, cohesion, labels, gods, surprises, detection, tokens, "./project")
    assert "## Communities" in report

def test_report_contains_ambiguous_section():
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    report = generate(G, communities, cohesion, labels, gods, surprises, detection, tokens, "./project")
    assert "## Ambiguous Edges" in report

def test_report_shows_token_cost():
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    report = generate(G, communities, cohesion, labels, gods, surprises, detection, tokens, "./project")
    assert "Token cost" in report
    assert "1,200" in report

def test_report_shows_raw_cohesion_scores():
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    report = generate(G, communities, cohesion, labels, gods, surprises, detection, tokens, "./project", min_community_size=1)
    assert "Cohesion:" in report
    assert "✓" not in report
    assert "⚠" not in report


def test_report_header_does_not_embed_host_absolute_path():
    """#2628 / #2598: the header must not bake the generator host absolute path
    into GRAPH_REPORT.md — it labels with the project directory basename so the
    same graph produces the same bytes on any machine."""
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    report = generate(G, communities, cohesion, labels, gods, surprises, detection,
                      tokens, "/Users/mike/dev/apps/secretproj")
    header = report.splitlines()[0]
    assert "/Users/mike" not in header
    assert "secretproj" in header


def test_portable_root_label():
    from graphify.report import _portable_root_label
    # Absolute paths collapse to the basename on both POSIX and Windows.
    assert _portable_root_label("/Users/mike/dev/apps/proj") == "proj"
    assert _portable_root_label(r"C:\Users\mike\dev\proj") == "proj"
    # A trailing slash still yields the directory name, not an empty label.
    assert _portable_root_label("/Users/mike/dev/proj/") == "proj"
    # Relative names pass through unchanged.
    assert _portable_root_label("./project") == "project"
    assert _portable_root_label("project") == "project"


# --- work-memory lessons section ----------------------------------------------

def test_report_work_memory_section_present_with_overlay_and_dead_ends():
    """When a work-memory overlay (preferred sources) and query-scoped dead-ends
    are supplied, the report grows a `## Work-memory lessons` section listing the
    preferred sources and, separately, the dead-ends as question -> nodes."""
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    learning = {
        "overlay": {
            "auth_login": {"status": "preferred", "uses": 3, "score": 2.4,
                           "label": "login()", "stale": False},
            "redis": {"status": "tentative", "uses": 1, "score": 0.5,
                      "label": "RedisClient", "stale": False},
        },
        "dead_ends": [
            {"question": "does it use websockets?", "nodes": ["WSServer"], "date": "2026-05-01"},
        ],
    }
    report = generate(G, communities, cohesion, labels, gods, surprises, detection,
                      tokens, "./project", learning=learning)
    assert "## Work-memory lessons" in report
    assert "**Preferred sources**" in report
    assert "`login()`" in report
    # Tentative is not listed in the report's preferred block.
    assert "RedisClient" not in report
    # Dead-ends are query-scoped: question -> nodes, NOT a node-level status.
    assert "**Known dead ends**" in report
    assert "does it use websockets?" in report
    assert "`WSServer`" in report


def test_report_work_memory_section_absent_without_overlay():
    """No learning input => no section; report identical to pre-feature."""
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    before = generate(G, communities, cohesion, labels, gods, surprises, detection,
                      tokens, "./project")
    assert "## Work-memory lessons" not in before
    # Explicit empty learning also omits the section.
    empty = generate(G, communities, cohesion, labels, gods, surprises, detection,
                     tokens, "./project", learning={"overlay": {}, "dead_ends": []})
    assert "## Work-memory lessons" not in empty
    assert before == empty


def test_import_cycles_section_present_for_code_corpus():
    # #1657: the fixture is a code corpus, so the Import Cycles section shows.
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    report = generate(G, communities, cohesion, labels, gods, surprises, detection, tokens, "./project")
    assert "## Import Cycles" in report


def test_import_cycles_section_absent_for_documents_only_corpus():
    # #1657: a documents-only corpus has no imports; the section is pure noise
    # ("None detected") and must be suppressed.
    extraction = {
        "nodes": [
            {"id": "d1", "label": "intro.md", "file_type": "document"},
            {"id": "d2", "label": "guide.md", "file_type": "document"},
        ],
        "edges": [{"source": "d1", "target": "d2", "relation": "references"}],
        "input_tokens": 0, "output_tokens": 0,
    }
    G = build_from_json(extraction)
    communities = cluster(G)
    cohesion = score_all(G, communities)
    labels = {cid: f"Community {cid}" for cid in communities}
    gods = god_nodes(G)
    surprises = surprising_connections(G)
    detection = {"total_files": 2, "total_words": 100, "needs_graph": True, "warning": None}
    tokens = {"input": 0, "output": 0}
    report = generate(G, communities, cohesion, labels, gods, surprises, detection, tokens, "./project")
    assert "## Import Cycles" not in report


def test_report_hubs_are_plain_text_by_default():
    # #1712: without --obsidian the _COMMUNITY_*.md notes don't exist, so wikilinks
    # would dangle (and pollute an Obsidian vault's graph view). Default to plain text.
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    labels = {cid: f"Widget {cid}" for cid in communities}
    report = generate(G, communities, cohesion, labels, gods, surprises, detection, tokens, "./project", min_community_size=1)
    assert "## Community Hubs (Navigation)" in report
    assert "[[_COMMUNITY_" not in report, "must not emit dangling Obsidian wikilinks by default (#1712)"
    assert any(f"- Widget {cid}" in report for cid in communities)


def test_report_hubs_use_wikilinks_when_obsidian():
    # The opt-in path keeps the vault-navigable wikilink form.
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    labels = {cid: f"Widget {cid}" for cid in communities}
    report = generate(G, communities, cohesion, labels, gods, surprises, detection, tokens, "./project", min_community_size=1, obsidian=True)
    assert "[[_COMMUNITY_" in report


# --- Ambiguous Edges direction --------------------------------------------------

def _single_community_report(G):
    communities = {0: list(G.nodes)}
    detection = {"total_files": 2, "total_words": 10, "needs_graph": True, "warning": None}
    return generate(G, communities, {0: 1.0}, {0: "Community 0"}, [], [],
                    detection, {"input": 0, "output": 0}, "./project")


def test_ambiguous_edges_print_true_direction_not_storage_order():
    """An undirected nx.Graph yields each edge from whichever endpoint was
    inserted first, not source -> target. build.py keeps the true endpoints in
    _src/_tgt (#563), which to_json, the HTML export, surprising_connections and
    the query output already honour. With the callee inserted before the
    caller, the Ambiguous Edges section printed the callee as the caller."""
    extraction = {
        "nodes": [
            {"id": "widget_refresh", "label": "Widget.refresh()", "file_type": "code",
             "source_file": "app/models.py"},
            {"id": "render_dashboard", "label": "render_dashboard()", "file_type": "code",
             "source_file": "app/views.py"},
        ],
        "edges": [
            {"source": "render_dashboard", "target": "widget_refresh", "relation": "calls",
             "confidence": "AMBIGUOUS", "source_file": "app/views.py", "weight": 1.0},
        ],
    }
    G = build_from_json(extraction)
    # Precondition: storage order is callee-first, the case that used to flip.
    assert list(G.edges()) == [("widget_refresh", "render_dashboard")]
    report = _single_community_report(G)
    assert "- `render_dashboard()` → `Widget.refresh()`  [AMBIGUOUS]" in report
    assert "`Widget.refresh()` → `render_dashboard()`" not in report


def test_ambiguous_edges_fall_back_to_storage_order_on_stale_markers():
    """A marker that does not name the edge's own endpoints (a graph mutated
    after build, e.g. a node merged away, #226) and a missing marker must
    neither crash the report nor print a node the edge does not touch: both
    fall back to storage order, like the query output's guard (#2080)."""
    G = nx.Graph()
    for nid, label in (("a", "Alpha"), ("b", "Beta"), ("c", "Gamma"), ("d", "Delta")):
        G.add_node(nid, label=label, source_file="notes.md")
    G.add_edge("a", "b", relation="uses", confidence="AMBIGUOUS", source_file="notes.md",
               _src="merged_away", _tgt="a")
    G.add_edge("c", "d", relation="uses", confidence="AMBIGUOUS", source_file="notes.md")
    report = _single_community_report(G)
    assert "- `Alpha` → `Beta`  [AMBIGUOUS]" in report
    assert "- `Gamma` → `Delta`  [AMBIGUOUS]" in report
