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

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"
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


def _normalize_scoreboard(data):
    matches = []
    for ev in data.get("events", []):
        comps = ev.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        state = (((ev.get("status") or {}).get("type") or {}).get("state")) or \
            (((comp.get("status") or {}).get("type") or {}).get("state"))
        if state == "pre":
            continue
        competitors = comp.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        ht, at = home.get("team") or {}, away.get("team") or {}
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

        notes = comp.get("notes") or []
        note_head = notes[0].get("headline") if notes else ""
        season_slug = ((ev.get("season") or {}).get("slug")) or ""
        stage = _stage_from_text(season_slug, note_head, ev.get("name"), ev.get("shortName"))

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


@app.get("/api/standings")
async def standings(dates: str = DEFAULT_DATES):
    key = f"standings:{dates}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    url = f"{ESPN_BASE}/scoreboard?dates={dates}"
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url)
        r.raise_for_status()
        data = r.json()
    except Exception:
        raise HTTPException(502, "ESPN scoreboard fetch failed")
    matches = _normalize_scoreboard(data)
    payload = {"matches": matches, "source": "espn"}
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

    return {
        "info": info,
        "home": home,
        "away": away,
        "stats": stats,
        "events": events,
        "scorers": scorers,
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
