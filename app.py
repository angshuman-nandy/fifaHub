import json
import os
import time
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

load_dotenv()

app = FastAPI()
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
APP_PASSCODE = os.environ.get("APP_PASSCODE", "")
ALLOWED_TOOLS = {"web_search_20250305"}

# Provider-agnostic LLM proxy (/api/llm) — server picks Anthropic vs OpenAI so the
# frontend's prompts stay provider-neutral. Default preserves pre-existing behavior.
OPENAI_URL = "https://api.openai.com/v1/responses"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1")
OPENAI_FAST_MODEL = os.environ.get("OPENAI_FAST_MODEL", "gpt-4.1-mini")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
ANTHROPIC_FAST_MODEL = os.environ.get("ANTHROPIC_FAST_MODEL", "claude-haiku-4-5-20251001")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower()

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"
# FIFA's own public feed (powers fifa.com) — used as a fallback when ESPN is down.
# Undocumented but stable; requires a browser User-Agent. Season 285023 = World Cup 2026.
FIFA_BASE = "https://api.fifa.com/api/v3"
FIFA_SEASON = "285023"
FIFA_COMPETITION = "17"
FIFA_HEADERS = {"User-Agent": "Mozilla/5.0"}
DEFAULT_DATES = "20260611-20260719"
CACHE_TTL = 60  # seconds
_cache = {}  # key -> (timestamp, payload)


def _cache_get(key):
    hit = _cache.get(key)
    if hit and (time.time() - hit[0]) < CACHE_TTL:
        return hit[1]
    return None


def _cache_set(key, payload):
    _cache[key] = (time.time(), payload)


# Shared narrative cache: a match's LLM narrative is generated at most once globally —
# every user/device after the first gets it free. File-backed best effort (HF Spaces
# storage is ephemeral, so it resets on Space restart; per-browser caches still apply).
NARRATIVE_CACHE_FILE = "narrative_cache.json"
try:
    with open(NARRATIVE_CACHE_FILE) as f:
        _narratives = json.load(f)
except Exception:
    _narratives = {}


def _narrative_store(key, text):
    _narratives[key] = text
    try:
        with open(NARRATIVE_CACHE_FILE, "w") as f:
            json.dump(_narratives, f)
    except Exception:
        pass  # read-only/full disk — in-memory copy still serves this process


def _stage_from_text(*texts):
    t = " ".join(s for s in texts if s).lower().replace("-", " ")
    if "round of 32" in t:
        return "R32"
    if "round of 16" in t:
        return "R16"
    if "quarter" in t:
        return "QF"
    if "semi" in t:
        return "SF"
    if "third place" in t or "3rd place" in t or "bronze" in t:
        return "BRONZE"
    if "final" in t:
        return "F"
    if "group" in t:
        return "GS"
    return "GS"


def _normalize_scoreboard(data, include_upcoming=False):
    matches = []
    for ev in data.get("events", []):
        comps = ev.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        state = (((ev.get("status") or {}).get("type") or {}).get("state")) or \
            (((comp.get("status") or {}).get("type") or {}).get("state"))
        if state == "pre" and not include_upcoming:
            continue
        competitors = comp.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        ht, at = home.get("team") or {}, away.get("team") or {}

        notes = comp.get("notes") or []
        note_head = notes[0].get("headline") if notes else ""
        season_slug = ((ev.get("season") or {}).get("slug")) or ""
        stage = _stage_from_text(season_slug, note_head, ev.get("name"), ev.get("shortName"))

        if state == "pre":
            # not started yet — no score/scorers, but the espnId lets the frontend
            # fetch lineups (Squad & Formations) once ESPN publishes them
            matches.append({
                "home": (ht.get("abbreviation") or "").upper(),
                "away": (at.get("abbreviation") or "").upper(),
                "homeName": ht.get("displayName") or "",
                "awayName": at.get("displayName") or "",
                "hs": None,
                "as": None,
                "status": "UPCOMING",
                "stage": stage,
                "scorers": [],
                "espnId": str(ev.get("id") or ""),
            })
            continue

        # team.id -> abbreviation map for scorer resolution
        idmap = {}
        for c in competitors:
            tm = c.get("team") or {}
            if tm.get("id"):
                idmap[str(tm.get("id"))] = (tm.get("abbreviation") or "").upper()

        def _score(c):
            try:
                return int(c.get("score"))
            except (TypeError, ValueError):
                return 0

        scorers = []
        for d in comp.get("details") or []:
            if not d.get("scoringPlay"):
                continue
            tm = d.get("team") or {}
            abbr = idmap.get(str(tm.get("id")), "")
            athletes = d.get("athletesInvolved") or []
            nm = athletes[0].get("displayName") if athletes else ""
            clk = ((d.get("clock") or {}).get("displayValue")) or ""
            s = f"{nm} ({abbr}) {clk}".strip()
            if d.get("ownGoal"):
                s += " (OG)"
            scorers.append(s)
            if len(scorers) >= 6:
                break

        matches.append({
            "home": (ht.get("abbreviation") or "").upper(),
            "away": (at.get("abbreviation") or "").upper(),
            "homeName": ht.get("displayName") or "",
            "awayName": at.get("displayName") or "",
            "hs": _score(home),
            "as": _score(away),
            "status": "LIVE" if state == "in" else "FT",
            "stage": stage,
            "scorers": scorers,
            "espnId": str(ev.get("id") or ""),
        })
    return matches

def _normalize_fifa(data, include_upcoming=False):
    """Map FIFA calendar matches onto the same shape _normalize_scoreboard emits.
    Degraded-but-correct fallback: no scorers and no ESPN event id (lineups/breakdowns
    need ESPN anyway), but scores/status/stage are FIFA-official."""
    matches = []
    for r in data.get("Results") or []:
        home, away = r.get("Home") or {}, r.get("Away") or {}
        if not home.get("Abbreviation") or not away.get("Abbreviation"):
            continue  # pairing not known yet (TBD knockout slots)
        ms = r.get("MatchStatus")
        if ms == 0:
            status = "FT"
        elif ms == 3:
            status = "LIVE"
        elif ms == 1 and include_upcoming:
            status = "UPCOMING"
        else:
            continue

        def _name(side):
            tn = side.get("TeamName") or []
            return tn[0].get("Description", "") if tn else ""

        def _score(side):
            try:
                return int(side.get("Score"))
            except (TypeError, ValueError):
                return None

        stage_name = r.get("StageName") or []
        stage_text = stage_name[0].get("Description", "") if stage_name else ""
        matches.append({
            "home": (home.get("Abbreviation") or "").upper(),
            "away": (away.get("Abbreviation") or "").upper(),
            "homeName": _name(home),
            "awayName": _name(away),
            "hs": None if status == "UPCOMING" else _score(home),
            "as": None if status == "UPCOMING" else _score(away),
            "status": status,
            "stage": _stage_from_text(stage_text),
            "scorers": [],
            "espnId": "",
        })
    return matches


with open("static/index.html") as f:
    INDEX_HTML = f.read()


@app.get("/")
async def index():
    return HTMLResponse(INDEX_HTML.replace("__APP_PASSCODE__", APP_PASSCODE))


@app.post("/api/claude")
async def claude_proxy(request: Request):
    if not API_KEY:
        raise HTTPException(500, "Server missing ANTHROPIC_API_KEY")
    body = await request.json()
    # Hard limits so a public Space can't be abused as an open proxy
    if not str(body.get("model", "")).startswith("claude-"):
        raise HTTPException(400, "Invalid model")
    body["max_tokens"] = min(int(body.get("max_tokens", 1000)), 1024)
    body["messages"] = body.get("messages", [])[:4]
    tools = body.get("tools", [])
    if any(t.get("type") not in ALLOWED_TOOLS for t in tools):
        raise HTTPException(400, "Tool not allowed")
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            ANTHROPIC_URL,
            json=body,
            headers={
                "x-api-key": API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
    return JSONResponse(status_code=r.status_code, content=r.json())


def _extract_openai_text(data):
    """Pull concatenated assistant text out of a raw Responses API JSON payload
    (the `output_text` convenience field is SDK-only, not in the REST response)."""
    parts = []
    for item in data.get("output") or []:
        if item.get("type") == "message":
            for c in item.get("content") or []:
                if c.get("type") == "output_text" and c.get("text"):
                    parts.append(c["text"])
    return "\n".join(parts)


async def _llm_anthropic(prompt, use_search, max_tokens, model):
    if not API_KEY:
        raise HTTPException(500, "Server missing ANTHROPIC_API_KEY")
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if use_search:
        body["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            ANTHROPIC_URL,
            json=body,
            headers={
                "x-api-key": API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
    if r.status_code >= 400:
        raise HTTPException(r.status_code, f"Anthropic API error: {r.text[:300]}")
    data = r.json()
    return "\n".join(b.get("text", "") for b in data.get("content") or [] if b.get("type") == "text")


async def _llm_openai(prompt, use_search, max_tokens, model):
    if not OPENAI_API_KEY:
        raise HTTPException(500, "Server missing OPENAI_API_KEY")
    body = {"model": model, "input": prompt, "max_output_tokens": max_tokens}
    if model.startswith("gpt-5"):
        # reasoning models bill hidden reasoning tokens against max_output_tokens —
        # minimal effort (narratives need no deliberation) + headroom so the visible
        # text isn't truncated by whatever reasoning remains
        body["reasoning"] = {"effort": "minimal"}
        body["max_output_tokens"] = min(max_tokens + 512, 2048)
    if use_search:
        body["tools"] = [{"type": "web_search_preview"}]
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            OPENAI_URL,
            json=body,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "content-type": "application/json"},
        )
    if r.status_code >= 400:
        raise HTTPException(r.status_code, f"OpenAI API error: {r.text[:300]}")
    data = r.json()
    return data.get("output_text") or _extract_openai_text(data)


@app.post("/api/llm")
async def llm_proxy(request: Request):
    """Provider-agnostic completion: {prompt, useSearch?, max_tokens?, tier?, cacheKey?,
    provider?} -> {text, provider, model}. The default provider comes from LLM_PROVIDER
    (server config); an explicit body provider ('anthropic'|'openai') overrides it so the
    frontend can request both takes for a comparison view. tier='fast' (default 'quality')
    picks a cheaper/smaller model for short, low-stakes write-ups.
    cacheKey (e.g. 'bd:{espnId}') makes the response shared-cacheable per provider: the
    first caller pays for generation, everyone after gets {provider:'cache'} for free."""
    body = await request.json()
    prompt = body.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        raise HTTPException(400, "Missing prompt")
    provider = body.get("provider") if body.get("provider") in ("anthropic", "openai") else \
        (LLM_PROVIDER if LLM_PROVIDER in ("anthropic", "openai") else "anthropic")
    tier = body.get("tier") if body.get("tier") in ("fast", "quality") else "quality"
    if provider == "openai":
        model = OPENAI_FAST_MODEL if tier == "fast" else OPENAI_MODEL
    else:
        model = ANTHROPIC_FAST_MODEL if tier == "fast" else ANTHROPIC_MODEL

    cache_key = body.get("cacheKey")
    if isinstance(cache_key, str) and cache_key:
        cache_key = f"{cache_key}:{provider}"  # each provider's take caches independently
        if cache_key in _narratives:
            return {"text": _narratives[cache_key], "provider": "cache", "model": model}
    if body.get("cacheOnly"):
        # lookup-only (e.g. pre-match prediction takes shown after FT) — never generate
        raise HTTPException(404, "Not cached")
    use_search = bool(body.get("useSearch"))
    max_tokens = min(int(body.get("max_tokens", 1000)), 1024)

    if provider == "openai":
        text = await _llm_openai(prompt, use_search, max_tokens, model)
    else:
        text = await _llm_anthropic(prompt, use_search, max_tokens, model)
    if isinstance(cache_key, str) and cache_key and text:
        _narrative_store(cache_key, text)
    return {"text": text, "provider": provider, "model": model}


@app.get("/api/standings")
async def standings(dates: str = DEFAULT_DATES, upcoming: bool = False):
    key = f"standings:{dates}:{upcoming}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    # ESPN primary (rich: scorers + event ids for lineups/commentary). An empty
    # list from a successful fetch is legitimate (no finished matches yet) — only
    # a fetch error triggers the fallback.
    matches, source = None, None
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{ESPN_BASE}/scoreboard?dates={dates}")
        r.raise_for_status()
        matches, source = _normalize_scoreboard(r.json(), include_upcoming=upcoming), "espn"
    except Exception:
        pass
    # FIFA official feed fallback (scores/status only — keeps the scoreboard accurate
    # with zero LLM involvement even when ESPN is down)
    if matches is None:
        try:
            url = (f"{FIFA_BASE}/calendar/matches?idSeason={FIFA_SEASON}"
                   f"&idCompetition={FIFA_COMPETITION}&language=en&count=200")
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.get(url, headers=FIFA_HEADERS)
            r.raise_for_status()
            matches, source = _normalize_fifa(r.json(), include_upcoming=upcoming), "fifa"
        except Exception:
            pass
    if matches is None:
        raise HTTPException(502, "Scoreboard fetch failed (ESPN and FIFA)")
    payload = {"matches": matches, "source": source}
    _cache_set(key, payload)
    return payload


def _normalize_match(data):
    info = {}
    gi = data.get("gameInfo") or {}
    venue = (gi.get("venue") or {}).get("fullName") or ""
    if not venue:
        v = ((data.get("header") or {}).get("competitions") or [{}])
        venue = ((v[0].get("venue") or {}).get("fullName")) if v else ""
    info["venue"] = venue
    info["attendance"] = gi.get("attendance")
    officials = gi.get("officials") or []
    info["referee"] = next((o.get("displayName") for o in officials
                            if "referee" in (o.get("position") or {}).get("displayName", "").lower()),
                           officials[0].get("displayName") if officials else "")
    hdr = data.get("header") or {}
    comp = (hdr.get("competitions") or [{}])[0]
    info["date"] = comp.get("date") or ""

    # team id -> abbreviation map (from header competitors)
    idmap = {}
    for c in comp.get("competitors") or []:
        tm = c.get("team") or {}
        if tm.get("id"):
            idmap[str(tm.get("id"))] = (tm.get("abbreviation") or "").upper()

    def _roster_side(side):
        for ros in data.get("rosters") or []:
            if ros.get("homeAway") == side:
                tm = ros.get("team") or {}
                xi, subs = [], []
                for p in ros.get("roster") or []:
                    ath = p.get("athlete") or {}
                    entry = {
                        "name": ath.get("displayName") or "",
                        "pos": ((p.get("position") or {}).get("abbreviation")) or "",
                        "jersey": p.get("jersey") or ath.get("jersey") or "",
                        "formationPlace": p.get("formationPlace"),
                    }
                    if p.get("starter"):
                        xi.append(entry)
                    else:
                        subs.append({"name": entry["name"], "pos": entry["pos"], "jersey": entry["jersey"]})
                return {
                    "teamName": (tm.get("displayName") or ""),
                    "formation": ros.get("formation") or "",
                    "startingXI": xi,
                    "subs": subs,
                }
        return None

    home = _roster_side("home")
    away = _roster_side("away")

    # stats: matched by name across the two teams
    wanted = {
        "possessionPct": "Possession", "totalShots": "Shots",
        "shotsOnTarget": "Shots on target", "wonCorners": "Corners",
        "foulsCommitted": "Fouls", "offsides": "Offsides",
        "yellowCards": "Yellow cards", "redCards": "Red cards",
        "saves": "Saves", "expectedGoals": "xG",
    }
    box = (data.get("boxscore") or {}).get("teams") or []
    side_stats = {"home": {}, "away": {}}
    for tb in box:
        side = (tb.get("homeAway") or "")
        if side not in side_stats:
            # fall back to order if homeAway missing
            side = "home" if not side_stats["home"] else "away"
        for st in tb.get("statistics") or []:
            side_stats[side][st.get("name")] = st.get("displayValue")
    stats = []
    for name, label in wanted.items():
        h = side_stats["home"].get(name)
        a = side_stats["away"].get(name)
        if h is None and a is None:
            continue
        stats.append({"label": label, "home": h, "away": a})

    events, scorers = [], []
    for ke in data.get("keyEvents") or []:
        tpl = ((ke.get("type") or {}).get("text") or "").lower()
        if ke.get("scoringPlay"):
            etype = "goal"
        elif "yellow card" in tpl:
            etype = "yellow"
        elif "red card" in tpl:
            etype = "red"
        elif "substitution" in tpl:
            etype = "sub"
        else:
            continue  # skip kickoff/halftime/period markers
        tm = ke.get("team") or {}
        abbr = idmap.get(str(tm.get("id")), "")
        side = ""
        for c in comp.get("competitors") or []:
            if str((c.get("team") or {}).get("id")) == str(tm.get("id")):
                side = c.get("homeAway") or ""
        clk = ((ke.get("clock") or {}).get("displayValue")) or ""
        parts = ke.get("participants") or ke.get("athletesInvolved") or []
        names = [((p.get("athlete") or {}).get("displayName")) or p.get("displayName") or ""
                 for p in parts]
        names = [n for n in names if n]
        text = ke.get("shortText") or ke.get("text") or " / ".join(names)
        events.append({"min": clk, "type": etype, "side": side, "text": text})
        if etype == "goal":
            nm = names[0] if names else text
            s = f"{nm} ({abbr}) {clk}".strip()
            if "penalty" in tpl:
                s += " (P)"
            if "own goal" in tpl:
                s += " (OG)"
            scorers.append(s)

    # play-by-play commentary (chronological from ESPN; frontend shows newest first).
    # Free, real-time text for live matches — replaces any LLM call mid-match.
    commentary = []
    for c in (data.get("commentary") or [])[-40:]:
        txt = c.get("text") or ""
        mn = ((c.get("time") or {}).get("displayValue")) or ""
        if txt:
            commentary.append({"min": mn, "text": txt})

    return {
        "info": info,
        "home": home,
        "away": away,
        "stats": stats,
        "events": events,
        "scorers": scorers,
        "commentary": commentary,
    }


@app.get("/api/match")
async def match(event: str):
    if not event:
        raise HTTPException(404, "Missing event id")
    key = f"match:{event}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    url = f"{ESPN_BASE}/summary?event={event}"
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url)
        if r.status_code == 404:
            raise HTTPException(404, "Match not found")
        r.raise_for_status()
        data = r.json()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(502, "ESPN summary fetch failed")
    payload = _normalize_match(data)
    if not payload.get("home") and not payload.get("away"):
        raise HTTPException(404, "Match data unavailable")
    _cache_set(key, payload)
    return payload


@app.get("/healthz")
async def healthz():
    return {"ok": True}
