"""contextlab.configlint (v10.8, Chapter 48): check a context-layer configuration
against the book's catalogue of anti-patterns, then check its references against
the world it runs in.

Standard library only. A configuration is a plain dict (loaded from JSON or YAML by
the caller). `lint` reads the configuration alone and answers "is this shaped like
a design we know fails?". `resolve` reads the configuration together with a
`World` (the live instruction set, the editions in force, the registered feeds and
the tools the gateway exposes) and answers "does everything it names still exist?".
A configuration can pass the first and fail the second, which is Chapter 48's point.
"""
from dataclasses import dataclass, field

ERROR, WARN = "error", "warn"

CATALOGUE = {
    "AP01": "Stuff it all in",
    "AP02": "Similarity as truth",
    "AP03": "The unversioned prompt",
    "AP04": "The summary with no provenance",
    "AP05": "The compaction that forgot the rule",
    "AP06": "The pointer that floats",
    "AP07": "The helpful default",
    "AP08": "The filter after the scan",
    "AP09": "The clock in the prefix",
    "AP10": "The cache key that forgets an input",
    "AP11": "The judge as a unit test",
    "AP12": "Tools that read alike",
}

MAX_PACKET_TOKENS = 8_000      # stand-in ceiling for a determination packet
MAX_TOOLS = 12                 # stand-in loadout ceiling for one request
LOOKALIKE = 0.6                # word-overlap above which two descriptions collide
REQUIRED_KEY = ("query", "policy", "form_edition", "index_version")


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: str
    where: str
    message: str

    def __str__(self):
        return f"{self.rule} {self.severity:<5} {self.where}: {self.message}"


def _words(text):
    return {w.strip(".,;:()").lower() for w in text.split() if len(w) > 3}


def overlap(a, b):
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / max(1, len(wa | wb))


def lint(cfg):
    """Static checks: one per anti-pattern that a configuration can show on its face."""
    out, name = [], cfg.get("name", "?")

    def add(rule, sev, msg):
        out.append(Finding(rule, sev, name, msg))

    budget = cfg.get("budget", {})
    if not budget.get("max_tokens") or budget["max_tokens"] > MAX_PACKET_TOKENS:
        add("AP01", ERROR, "no packet budget, or a budget above the ceiling")
    rank = cfg.get("ranking", {})
    if rank.get("order", ["similarity"])[0] != "authority" or not rank.get("precedence"):
        add("AP02", ERROR, "ranking does not apply authority before similarity")
    ins = cfg.get("instructions", {})
    if not ins.get("version") or ins.get("version") == "latest" or not ins.get("sha"):
        add("AP03", ERROR, "instructions are not pinned to a version and a hash")
    if cfg.get("memory", {}).get("summaries") and not cfg["memory"].get("provenance"):
        add("AP04", ERROR, "summaries are written without provenance")
    comp = cfg.get("compaction", {})
    if comp.get("mode", "off") != "off" and not comp.get("pins"):
        add("AP05", ERROR, "compaction is on and no constraint is pinned")
    index = cfg.get("index", {})
    if index.get("version", "latest") == "latest":
        add("AP06", ERROR, "index or edition pointer floats to 'latest'")
    if index.get("fill_missing_version"):
        add("AP07", ERROR, "missing versions are filled in instead of rejected")
    if index.get("filter") == "post":
        add("AP08", ERROR, "filters run after the vector scan")
    layout = cfg.get("layout", {})
    for f in layout.get("volatile", []):
        if f in layout.get("prefix", []):
            add("AP09", WARN, f"volatile field '{f}' sits in the cached prefix")
    cache = cfg.get("cache")
    if cache:
        needed = list(REQUIRED_KEY) + [f["watermark"] for f in cfg.get("feeds", [])]
        missing = [k for k in needed if k not in cache.get("key", [])]
        if missing:
            add("AP10", ERROR, "cache key omits " + ", ".join(missing))
    if cfg.get("evals", {}).get("judge_layer", 3) < 3:
        add("AP11", WARN, "a model judge runs as a per-commit unit test")
    tools = cfg.get("tools", [])
    if len(tools) > MAX_TOOLS:
        add("AP12", WARN, f"{len(tools)} tools in one request")
    for i, a in enumerate(tools):
        for b in tools[i + 1:]:
            if overlap(a["description"], b["description"]) > LOOKALIKE:
                add("AP12", WARN, f"'{a['name']}' and '{b['name']}' read alike")
    return out


@dataclass
class World:
    """What the configuration's names point at, as of one date."""
    rule_ids: set = field(default_factory=set)       # current instruction set
    editions: set = field(default_factory=set)       # editions in force in the corpus
    feeds: dict = field(default_factory=dict)        # feed name -> watermark key
    tools: set = field(default_factory=set)          # tools the gateway exposes


def resolve(cfg, world):
    """Reference checks: does every name in the configuration still exist?"""
    out, name = [], cfg.get("name", "?")
    for pin in cfg.get("compaction", {}).get("pins", []):
        if pin not in world.rule_ids:
            out.append(Finding("AP05", ERROR, name, f"pin '{pin}' names no current rule"))
    table = set(cfg.get("ranking", {}).get("precedence", []))
    for ed in sorted(world.editions - table):
        out.append(Finding("AP02", ERROR, name, f"edition '{ed}' has no precedence entry"))
    keyed = set(cfg.get("cache", {}).get("key", []))
    for feed, mark in sorted(world.feeds.items()):
        if cfg.get("cache") and mark not in keyed:
            out.append(Finding("AP10", ERROR, name, f"feed '{feed}' is not in the cache key"))
    for t in cfg.get("tools", []):
        if t["name"] not in world.tools:
            out.append(Finding("AP12", ERROR, name, f"tool '{t['name']}' is not exposed"))
    return out


def report(findings):
    errors = sum(f.severity == ERROR for f in findings)
    warns = sum(f.severity == WARN for f in findings)
    return f"{errors} error(s), {warns} warning(s)"
