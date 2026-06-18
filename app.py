import asyncio
import ipaddress
import json
import os
import time
from collections import Counter
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

load_dotenv()

app = FastAPI()
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
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
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"
# FIFA's own public feed (powers fifa.com) — used as a fallback when ESPN is down.
# Undocumented but stable; requires a browser User-Agent. Season 285023 = World Cup 2026.
FIFA_BASE = "https://api.fifa.com/api/v3"
FIFA_SEASON = "285023"
FIFA_COMPETITION = "17"
FIFA_HEADERS = {"User-Agent": "Mozilla/5.0"}
DEFAULT_DATES = "20260611-20260719"
CACHE_TTL = 60          # seconds — short-lived scoreboard/match data
FORM_CACHE_TTL = 3600   # seconds — team recent-form (changes at most hourly)
SERPER_CACHE_TTL = 3 * 3600  # seconds — web search results valid for 3h
_cache = {}  # key -> (timestamp, payload)


def _cache_get(key, ttl=None):
    hit = _cache.get(key)
    if hit and (time.time() - hit[0]) < (ttl if ttl is not None else CACHE_TTL):
        return hit[1]
    return None


def _cache_set(key, payload):
    _cache[key] = (time.time(), payload)


# Shared narrative cache: a match's LLM narrative/take is generated at most once
# globally — every user/device after the first gets it free. Backed two ways:
#  - local JSON file (fast, but HF Spaces storage is ephemeral — dies on restart)
#  - private HF dataset repo (CACHE_DATASET + HF_TOKEN): downloaded at startup,
#    re-uploaded (debounced) after new entries, so the cache survives restarts/deploys.
NARRATIVE_CACHE_FILE = "narrative_cache.json"
CACHE_DATASET = os.environ.get("CACHE_DATASET", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ANALYTICS_KEY = os.environ.get("ANALYTICS_KEY", "")
VISITOR_LOG_FILE = "visitor_log.json"


def _load_narratives():
    data = {}
    if CACHE_DATASET and HF_TOKEN:
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(repo_id=CACHE_DATASET, repo_type="dataset",
                                filename="narrative_cache.json", token=HF_TOKEN)
            with open(p) as f:
                data.update(json.load(f))
        except Exception:
            pass  # dataset missing/unreachable — start from local file only
    try:
        with open(NARRATIVE_CACHE_FILE) as f:
            data.update(json.load(f))  # local file wins (newer than last upload)
    except Exception:
        pass
    return data


_narratives = _load_narratives()
_cache_upload_task = None


async def _upload_cache_soon():
    """Debounced dataset sync — coalesces bursts of new entries into one commit."""
    global _cache_upload_task
    await asyncio.sleep(20)
    _cache_upload_task = None
    try:
        from huggingface_hub import HfApi
        await asyncio.to_thread(
            HfApi(token=HF_TOKEN).upload_file,
            path_or_fileobj=NARRATIVE_CACHE_FILE,
            path_in_repo="narrative_cache.json",
            repo_id=CACHE_DATASET, repo_type="dataset",
            commit_message="cache sync",
        )
    except Exception:
        pass  # next store schedules another attempt


def _narrative_store(key, text):
    global _cache_upload_task
    _narratives[key] = text
    try:
        with open(NARRATIVE_CACHE_FILE, "w") as f:
            json.dump(_narratives, f)
    except Exception:
        pass  # read-only/full disk — in-memory copy still serves this process
    if CACHE_DATASET and HF_TOKEN and _cache_upload_task is None:
        try:
            _cache_upload_task = asyncio.get_running_loop().create_task(_upload_cache_soon())
        except RuntimeError:
            pass  # not inside the event loop — sync will happen on a later store


def _load_visitors():
    data = []
    if CACHE_DATASET and HF_TOKEN:
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(repo_id=CACHE_DATASET, repo_type="dataset",
                                filename=VISITOR_LOG_FILE, token=HF_TOKEN)
            with open(p) as f:
                data = json.load(f)
        except Exception:
            pass
    try:
        with open(VISITOR_LOG_FILE) as f:
            local = json.load(f)
        # merge: keep entries from HF that aren't in local (older history),
        # then append local entries (more recent than last upload)
        local_ts = {v["ts"] for v in local}
        merged = [v for v in data if v["ts"] not in local_ts] + local
        merged.sort(key=lambda v: v["ts"])
        data = merged
    except Exception:
        pass
    return data


_visitors: list = _load_visitors()
_ip_geo_cache: dict = {}
_visitor_upload_task = None


async def _upload_visitors_soon():
    global _visitor_upload_task
    await asyncio.sleep(30)
    _visitor_upload_task = None
    try:
        with open(VISITOR_LOG_FILE, "w") as f:
            json.dump(_visitors[-2000:], f)
    except Exception:
        pass
    try:
        from huggingface_hub import HfApi
        await asyncio.to_thread(
            HfApi(token=HF_TOKEN).upload_file,
            path_or_fileobj=VISITOR_LOG_FILE,
            path_in_repo=VISITOR_LOG_FILE,
            repo_id=CACHE_DATASET, repo_type="dataset",
            commit_message="visitor log sync",
        )
    except Exception:
        pass


def _is_public_ip(ip: str) -> bool:
    """False for loopback/private/link-local/reserved addresses (localhost testing,
    LAN clients) and anything unparseable — keeps test traffic out of analytics."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified)


_GEO_EMPTY = {
    "city": "", "country": "", "countryCode": "", "lat": None, "lon": None,
    "regionName": "", "isp": "", "org": "",
    "mobile": False, "proxy": False, "hosting": False,
}


async def _geolocate(ip: str) -> dict:
    if ip in _ip_geo_cache:
        return _ip_geo_cache[ip]
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(
                f"http://ip-api.com/json/{ip}",
                params={"fields": "status,city,regionName,country,countryCode,lat,lon,isp,org,mobile,proxy,hosting"},
            )
        data = r.json()
        if data.get("status") == "success":
            geo = {
                "city": data.get("city", ""),
                "regionName": data.get("regionName", ""),
                "country": data.get("country", ""),
                "countryCode": data.get("countryCode", ""),
                "lat": data.get("lat"),
                "lon": data.get("lon"),
                "isp": data.get("isp", ""),
                "org": data.get("org", ""),
                "mobile": bool(data.get("mobile", False)),
                "proxy": bool(data.get("proxy", False)),
                "hosting": bool(data.get("hosting", False)),
            }
        else:
            geo = dict(_GEO_EMPTY)
    except Exception:
        geo = dict(_GEO_EMPTY)
    _ip_geo_cache[ip] = geo
    return geo


def _persist_visitors():
    global _visitor_upload_task
    try:
        with open(VISITOR_LOG_FILE, "w") as f:
            json.dump(_visitors[-2000:], f)
    except Exception:
        pass
    if CACHE_DATASET and HF_TOKEN and _visitor_upload_task is None:
        try:
            _visitor_upload_task = asyncio.get_running_loop().create_task(_upload_visitors_soon())
        except RuntimeError:
            pass


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

        date = comp.get("date") or ev.get("date") or ""
        venue = ((comp.get("venue") or {}).get("fullName")) or ""

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
                "date": date,
                "venue": venue,
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
            "date": date,
            "venue": venue,
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
        stadium_name = r.get("Stadium", {}).get("Name") or []
        venue = stadium_name[0].get("Description", "") if stadium_name else ""
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
            "date": r.get("Date") or "",
            "venue": venue,
        })
    return matches


# ESPN team-id map — verified 2026-06-15 via /sports/soccer/fifa.world/teams?limit=60
ESPN_TEAM_IDS = {
    "ALG": 624,  "ARG": 202,  "AUS": 628,  "AUT": 474,  "BEL": 459,
    "BIH": 452,  "BRA": 205,  "CAN": 206,  "CPV": 2597, "COL": 208,
    "COD": 2850, "CRO": 477,  "CUW": 11678,"CZE": 450,  "ECU": 209,
    "EGY": 2620, "ENG": 448,  "FRA": 478,  "GER": 481,  "GHA": 4469,
    "HAI": 2654, "IRN": 469,  "IRQ": 4375, "CIV": 4789, "JPN": 627,
    "JOR": 2917, "MEX": 203,  "MAR": 2869, "NED": 449,  "NZL": 2666,
    "NOR": 464,  "PAN": 2659, "PAR": 210,  "POR": 482,  "QAT": 4398,
    "KSA": 655,  "SCO": 580,  "SEN": 654,  "RSA": 467,  "KOR": 451,
    "ESP": 164,  "SWE": 466,  "SUI": 475,  "TUN": 659,  "TUR": 465,
    "USA": 660,  "URU": 212,  "UZB": 2570,
}

with open("static/index.html") as f:
    INDEX_HTML = f.read()

with open("static/analytics.html") as f:
    ANALYTICS_HTML = f.read()


@app.get("/")
async def index():
    return HTMLResponse(INDEX_HTML)


@app.get("/analytics")
async def analytics_page(key: str = ""):
    if ANALYTICS_KEY and key != ANALYTICS_KEY:
        raise HTTPException(403, "Invalid key")
    return HTMLResponse(ANALYTICS_HTML)


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


@app.post("/api/track")
async def track_visit(request: Request):
    xff = request.headers.get("x-forwarded-for", "")
    ip = xff.split(",")[0].strip() if xff else (request.client.host if request.client else "unknown")
    if not _is_public_ip(ip):
        return {"ok": True, "tracked": False}  # local/private testing traffic — don't log
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    geo = await _geolocate(ip)
    entry = {
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ip": ip,
        **geo,
        "ua": request.headers.get("user-agent", "")[:200],
        "ref": str(body.get("ref", ""))[:200],
        "page": str(body.get("page", "/"))[:100],
    }
    _visitors.append(entry)
    if len(_visitors) > 5000:
        _visitors.pop(0)
    _persist_visitors()
    return {"ok": True}


@app.get("/api/analytics")
async def analytics(key: str = ""):
    if ANALYTICS_KEY and key != ANALYTICS_KEY:
        raise HTTPException(403, "Invalid key")
    # drop local/private-IP noise from earlier testing (already-recorded entries),
    # on top of track_visit no longer recording new ones
    visitors = [v for v in _visitors if _is_public_ip(v.get("ip", ""))]
    recent = visitors[-200:][::-1]  # newest first
    # aggregate by country
    countries = Counter(v["countryCode"] for v in visitors if v.get("countryCode"))
    cities = Counter(
        f"{v['city']}, {v['countryCode']}" for v in visitors
        if v.get("city") and v.get("countryCode")
    )
    unique_ips = len({v["ip"] for v in visitors})
    bot_like = sum(1 for v in visitors if v.get("hosting") or v.get("proxy"))
    return {
        "total": len(visitors),
        "uniqueIPs": unique_ips,
        "botLike": bot_like,
        "topCountries": countries.most_common(10),
        "topCities": cities.most_common(10),
        "recent": recent[:100],
    }


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


@app.get("/api/team-form")
async def team_form(team: str):
    """Zero-token recent form from ESPN's team schedule.
    Returns {code, summary, recent[6]} — each entry {date,opp,ha,gf,ga,res}.
    summary e.g. 'W3 D1 L1 · GF9 GA4 (last 6)'. Cached 1h (form changes slowly)."""
    code = team.upper().strip()
    espn_id = ESPN_TEAM_IDS.get(code)
    if not espn_id:
        raise HTTPException(404, f"Unknown team code: {code}")
    key = f"team-form:{code}"
    cached = _cache_get(key, ttl=FORM_CACHE_TTL)
    if cached is not None:
        return cached
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/all/teams/{espn_id}/schedule"
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise HTTPException(502, f"ESPN team schedule fetch failed: {e}")
    recent, w, d, l, gf, ga = [], 0, 0, 0, 0, 0
    for e in data.get("events", []):
        comp = (e.get("competitions") or [{}])[0]
        if not (((comp.get("status") or {}).get("type") or {}).get("completed")):
            continue
        competitors = comp.get("competitors") or []
        mine = next((x for x in competitors
                     if (x.get("team") or {}).get("abbreviation", "").upper() == code), None)
        opp = next((x for x in competitors if x is not mine), None)
        if not mine or not opp:
            continue
        ms_str = (mine.get("score") or {}).get("displayValue")
        os_str = (opp.get("score") or {}).get("displayValue")
        if ms_str is None or os_str is None:
            continue
        ms, os_ = int(float(ms_str)), int(float(os_str))
        res = "W" if ms > os_ else ("L" if ms < os_ else "D")
        recent.append({
            "date": e.get("date", "")[:10],
            "opp": (opp.get("team") or {}).get("abbreviation", ""),
            "ha": mine.get("homeAway", ""),
            "gf": ms, "ga": os_, "res": res,
        })
        if len(recent) <= 6:
            if res == "W": w += 1
            elif res == "D": d += 1
            else: l += 1
            gf += ms; ga += os_
    recent = recent[:6]
    summary = (f"W{w} D{d} L{l} · GF{gf} GA{ga} (last {len(recent)})"
               if recent else "no recent results")
    payload = {"code": code, "summary": summary, "recent": recent}
    _cache_set(key, payload)
    return payload


@app.post("/api/serper")
async def serper_search(request: Request):
    """Web search via Serper.dev — returns trimmed organic results + answer box.
    If SERPER_API_KEY is unset returns {organic:[]} so callers degrade gracefully.
    Cached 3h per query string."""
    body = await request.json()
    q = str(body.get("q", "")).strip()
    if not q:
        raise HTTPException(400, "Missing query q")
    if not SERPER_API_KEY:
        return {"organic": []}  # key not configured — degrade silently, never 500
    key = f"serper:{q}"
    cached = _cache_get(key, ttl=SERPER_CACHE_TTL)
    if cached is not None:
        return cached
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                "https://google.serper.dev/search",
                json={"q": q, "num": 10, "gl": "us", "hl": "en"},
                headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise HTTPException(502, f"Serper search failed: {e}")
    # Trim to token-bounded shape: top 6 organic + optional answer/knowledge
    organic = [
        {"title": o.get("title", ""), "snippet": o.get("snippet", ""),
         "date": o.get("date", ""), "link": o.get("link", "")}
        for o in (data.get("organic") or [])[:6]
    ]
    payload: dict = {"organic": organic}
    ab = data.get("answerBox") or {}
    if ab.get("answer") or ab.get("snippet"):
        payload["answerBox"] = {"answer": ab.get("answer", ""), "snippet": ab.get("snippet", "")}
    kg = data.get("knowledgeGraph") or {}
    if kg.get("description"):
        payload["knowledgeGraph"] = {"description": kg["description"]}
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

    # team id -> abbreviation map (from header competitors); also extract live score
    idmap = {}
    live_hs, live_as = None, None
    for c in comp.get("competitors") or []:
        tm = c.get("team") or {}
        if tm.get("id"):
            idmap[str(tm.get("id"))] = (tm.get("abbreviation") or "").upper()
        raw_score = c.get("score")
        try:
            score_val = int(raw_score) if raw_score is not None else None
        except (ValueError, TypeError):
            score_val = None
        if c.get("homeAway") == "home":
            live_hs = score_val
        elif c.get("homeAway") == "away":
            live_as = score_val

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

    # live score from header (None when match hasn't started)
    comp_status = comp.get("status") or {}
    status_type = (comp_status.get("type") or {}).get("name") or ""
    live_status = "LIVE" if status_type == "STATUS_IN_PROGRESS" else (
        "FT" if status_type in ("STATUS_FINAL", "STATUS_FULL_TIME") else "")

    return {
        "info": info,
        "home": home,
        "away": away,
        "stats": stats,
        "events": events,
        "scorers": scorers,
        "commentary": commentary,
        "hs": live_hs,
        "as": live_as,
        "liveStatus": live_status,
    }


@app.get("/api/match")
async def match(event: str, fresh: bool = False):
    if not event:
        raise HTTPException(404, "Missing event id")
    key = f"match:{event}"
    if not fresh:
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
