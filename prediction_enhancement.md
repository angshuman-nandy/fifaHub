# Prediction enhancement: zero-token recent-form context + on-demand deep prediction

## Status

Planning complete, fully verified against live ESPN endpoints. **No code has been
written yet.** This doc is a self-contained handoff — implement directly from it.

## Goal / context

The Predictions tab (`static/index.html`, "TAB 02") already has:
- A Poisson/Elo model (`matchProbs`) and a 3,000-run Monte Carlo simulation (`runSims`/`SIM`)
  for tournament-wide odds.
- Per-match AI "takes" (`state.matchPreds`) — `genMatchPred()` sends `buildMatchPredPrompt()`
  to **both** LLM providers (Claude + GPT via `/api/llm`) for every match card that scrolls
  into view, producing the ⚙ STAT / ◆ CLAUDE / ● GPT picks row (`picksHTML`) plus a 1-2
  sentence "take" (`aitakeHTML`/`aitakeFTHTML`).

That existing prompt (`buildMatchPredPrompt`, `static/index.html` ~line 1051) only includes:
static `TEAMS[code].note` blurbs, Elo/FIFA rank, the Poisson odds, and each team's
**in-tournament** FT results so far (`form()` helper inside `buildMatchPredPrompt`).

**The user wants predictions enriched with:**
1. Matches the teams have already played (beyond just in-tournament — last few months).
2. The quality of players in the squad.
3. Player/team performance over the last few months.
...fed to an LLM for a prediction.

**Hard constraint from the user: minimize tokens.** Decisions already made with the user:

- **Recency data comes from ESPN, with ZERO LLM tokens** — a new server endpoint fetches
  each team's recent cross-competition results (qualifiers + friendlies + WC group games)
  directly from ESPN's JSON API. No model call involved in gathering this.
- **Player quality comes from ESPN's published starting XI/formation** (already partially
  wired for lineups elsewhere in the app) — also zero tokens.
- **The new "deep prediction" is on-demand** (a button per match), **single provider**
  (not `callBothLLMs` — pick one), and **server-cached** via the existing shared `cacheKey`
  mechanism so it's generated at most once globally per match per "form fingerprint."
- **Output stays concise**: a predicted scoreline + 1-2 sentence reason — same shape as the
  existing takes, not a long structured writeup.
- This is **additive** — it does NOT replace `state.matchPreds` / the existing ⚙/◆/● row,
  Monte Carlo sim, or `predAnalysis`. Finished (FT) matches get no deep-prediction button
  (predicting a decided match is meaningless), same convention as the existing
  `aitakeFTHTML` vs `aitakeHTML` split.

A noted-but-deferred observation (do NOT act on this unless separately asked): the
*existing* auto-takes (`genMatchPred`) already call **both** providers for every card
scrolled into view — that's the larger ongoing token sink in the app. Out of scope here.

---

## Part 1 — Server: `app.py`

### 1a. New endpoint `GET /api/team-form?team=<CODE>`

Add near the other `/api/...` endpoints (e.g. after `/api/standings`, ~line 436).

**Verified live** (today is 2026-06-15, WC26 underway):

```
GET https://site.api.espn.com/apis/site/v2/sports/soccer/all/teams/{espn_id}/schedule
```
(no `season` query param needed — it already returns the most recent 25 events,
descending by date, including in-progress-tournament fixtures, qualifiers, and friendlies,
each with `competitions[0].status.type.completed`).

Response shape per event (`e = events[i]`, `c = e.competitions[0]`):
```jsonc
{
  "date": "2026-06-13T22:00Z",
  "name": "Morocco at Brazil",
  "competitions": [{
    "status": {"type": {"completed": true}},
    "competitors": [
      {"homeAway": "home", "team": {"abbreviation": "BRA"}, "score": {"value": 0.0, "displayValue": "0", "winner": false}},
      {"homeAway": "away", "team": {"abbreviation": "MAR"}, "score": {"value": 0.0, "displayValue": "0", "winner": false}}
    ]
  }]
}
```

**Team code -> ESPN numeric id map** (verified via
`GET https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/teams?limit=60`
→ `sports[0].leagues[0].teams[].team.{abbreviation,id}`). All 48 of our `TEAMS` codes
match ESPN's `abbreviation` exactly — **no alias table needed**. Full map (code -> id):

```
ALG:624  ARG:202  AUS:628  AUT:474  BEL:459  BIH:452  BRA:205  CAN:206
CPV:2597 COL:208  COD:2850 CRO:477  CUW:11678 CZE:450 ECU:209  EGY:2620
ENG:448  FRA:478  GER:481  GHA:4469 HAI:2654 IRN:469 IRQ:4375 CIV:4789
JPN:627  JOR:2917 MEX:203  MAR:2869 NED:449  NZL:2666 NOR:464 PAN:2659
PAR:210  POR:482  QAT:4398 KSA:655  SCO:580  SEN:654 RSA:467 KOR:451
ESP:164  SWE:466  SUI:475  TUN:659  TUR:465  USA:660 URU:212 UZB:2570
```

Implementation:

1. Add a constant, e.g. `ESPN_TEAM_IDS = {...}` (the map above), hardcoded — fetching
   `fifa.world/teams` at runtime to build it adds an extra request for no benefit since the
   48 codes/ids are fixed for this tournament.
2. New endpoint:
   ```python
   @app.get("/api/team-form")
   async def team_form(team: str):
       code = team.upper().strip()
       espn_id = ESPN_TEAM_IDS.get(code)
       if not espn_id:
           raise HTTPException(404, "Unknown team code")
       key = f"team-form:{code}"
       cached = _cache_get(key, ttl=FORM_CACHE_TTL)  # see note below on TTL
       if cached is not None:
           return cached
       url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/all/teams/{espn_id}/schedule"
       async with httpx.AsyncClient(timeout=20) as client:
           r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
       r.raise_for_status()
       data = r.json()
       recent = []
       w = d = l = gf = ga = 0
       for e in data.get("events", []):
           comp = (e.get("competitions") or [{}])[0]
           if not (((comp.get("status") or {}).get("type") or {}).get("completed")):
               continue
           competitors = comp.get("competitors") or []
           mine = next((x for x in competitors if (x.get("team") or {}).get("abbreviation") == code), None)
           opp = next((x for x in competitors if x is not mine), None)
           if not mine or not opp:
               continue
           my_score = (mine.get("score") or {}).get("displayValue")
           opp_score = (opp.get("score") or {}).get("displayValue")
           if my_score is None or opp_score is None:
               continue
           ms, os_ = int(float(my_score)), int(float(opp_score))
           res = "W" if ms > os_ else ("L" if ms < os_ else "D")
           recent.append({
               "date": e.get("date", "")[:10],
               "opp": (opp.get("team") or {}).get("abbreviation", ""),
               "ha": mine.get("homeAway"),
               "gf": ms, "ga": os_, "res": res,
           })
           if len(recent) <= 6:  # only the most-recent 6 feed the W/D/L/GF/GA summary
               if res == "W": w += 1
               elif res == "D": d += 1
               else: l += 1
               gf += ms; ga += os_
       recent = recent[:6]
       summary = f"W{w} D{d} L{l} · GF{gf} GA{ga} (last {len(recent)})" if recent else "no recent results"
       payload = {"code": code, "summary": summary, "recent": recent}
       _cache_set(key, payload)
       return payload
   ```

3. **Cache TTL**: `_cache_get`/`_cache_set` (app.py ~line 39-47) currently use a single
   module-level `CACHE_TTL = 60` (line 35). Recent form changes slowly (hours, not
   seconds) — either:
   - add an optional `ttl` param to `_cache_get` (default `CACHE_TTL`) so this endpoint
     can pass a longer value (e.g. `FORM_CACHE_TTL = 3600`), **or**
   - store team-form entries with their own timestamp check inline.
   Either is fine; prefer the smallest diff to `_cache_get`/`_cache_set`.

### 1b. Reuse existing endpoints — no other server changes needed

- `/api/match?event=<espnId>` (app.py ~line 570, `_normalize_match` ~line 438) already
  returns `home`/`away` with `startingXI` (name/pos/jersey/formationPlace), `formation`,
  and `subs`. This is the "player quality" / lineup signal — call it from the frontend,
  same as the existing Squad & Formations / breakdown code paths (`fetchFacts`).
- `/api/llm` (app.py ~line 361) already supports `provider`, `tier`, `cacheKey`,
  `cacheOnly`, `useSearch` — no changes needed. The new deep-prediction call should pass
  `useSearch: false` (default), a single explicit `provider` (or omit to use server
  default `LLM_PROVIDER`), `tier: 'fast'` to use the cheaper model
  (`ANTHROPIC_FAST_MODEL`/`OPENAI_FAST_MODEL`), and a `cacheKey`.

---

## Part 2 — Frontend: `static/index.html`

### 2a. State

`state` object is defined ~line 511-521. Add a new key:
```js
deepPreds:{},  /* mid -> {score, take, model, ts, fp} — on-demand deep prediction (form+XI based) */
```

`saveState()` (line 529) and `loadState()` (line 530) serialize/restore specific keys —
add `deepPreds` to both, mirroring how `matchPreds` is handled (same pattern: include in
the `JSON.stringify({...})` in `saveState`, and `Object.assign(state.deepPreds, s.deepPreds||{})`
in `loadState`).

### 2b. Form fetch helper

New small helper near `fetchFacts` (~line 727):
```js
async function fetchTeamForm(code){
  try{
    const r=await fetch('/api/team-form?team='+encodeURIComponent(code));
    if(r.ok)return await r.json();
  }catch(e){}
  return null;
}
```

### 2c. Prompt builder

New function near `buildMatchPredPrompt` (~line 1051), e.g. `buildDeepPredPrompt(m, formH, formA, xiH, xiA)`:

- Reuse `matchProbs(m.h,m.a)` for the Poisson odds (same as `buildMatchPredPrompt`).
- Include `TEAMS[m.h].note` / `TEAMS[m.a].note`, Elo, FIFA rank — same as existing prompt.
- Add `formH.summary` / `formA.summary` (e.g. `"W3 D1 L1 · GF9 GA4 (last 6)"`) and a short
  one-line list of `formH.recent` / `formA.recent` (e.g.
  `"BRA: W vs CRO(h) 1-0, W vs PAN(h) 3-0, L vs FRA(h) 1-2, ..."` — keep it to ~6 results,
  it's already compact).
- If `xiH`/`xiA` available (from `/api/match` via `state.espnIds[mid]`, guarded by
  `factsHaveXI`, ~line 736), append `formation` + starting XI player names for each side
  (these are the "player quality" signal — named players let the model reason about
  individual quality without extra tokens since no search is needed).
- Ask for the **same compact JSON shape** as `buildMatchPredPrompt`:
  `{"score":"2-1","take":"1-2 punchy sentences..."}` — reuse the existing `extractJSON()`
  (~line 696) to parse it.
- Keep the whole prompt tight — this is the token-minimization lever that matters most
  on the input side.

### 2d. Cache key / fingerprint

```js
function deepPredKey(mid, formH, formA){
  return 'dpred:'+mid+':'+(formH?formH.summary:'')+'|'+(formA?formA.summary:'');
}
```
Pass this as `cacheKey` to `/api/llm` so the shared server cache (app.py `_narrative_store`)
regenerates only when either team's recent-form summary actually changes, and is reused
across all users/devices for that match.

### 2e. `genDeepPred(mid)`

New async function near `genMatchPred` (~line 1125):

1. Look up `m = allMatches().find(x=>x.id===mid)`; bail if FT (mirror the guard in
   `genMatchPred`, ~line 1128).
2. `Promise.all([fetchTeamForm(m.h), fetchTeamForm(m.a), fetchFacts(state.espnIds[mid])])`
   — all zero-LLM-token fetches, run in parallel.
3. Build the prompt via `buildDeepPredPrompt`, compute `deepPredKey`.
4. **Single** `callLLM(prompt, false, {max_tokens:160, tier:'fast', cacheKey})` — do NOT
   use `callBothLLMs`.
5. Parse with `extractJSON`; on success, `state.deepPreds[mid] = {score, take, model:r.model, ts:new Date().toLocaleString(), key: deepPredKey(...)}`; `await saveState()`.
6. Update the DOM for that card in place (swap the button/output container's innerHTML),
   mirroring how `genMatchPred` does `document.getElementById('aitake-'+mid).outerHTML=...`
   and the `[data-aipicks="${mid}"]` refresh (~lines 1152-1155).
7. Handle errors with `toast(...)` (existing helper, ~line 526) and restore the button to
   its clickable state — same pattern as `genPredAnalysis` (~line 1011-1037) button
   disable/restore.

### 2f. Rendering / UI hook

In `renderPredictions()`, the **upcoming** match `.pcard` branches (the `data-mpred`
branches — group-stage ~lines 941-949 and knockout ~lines 965-973) currently end with:
```js
${picksHTML(m[0]/m.id)}
<div class="rat">${rationale(...)}</div>${aitakeHTML(m[0]/m.id)}
```
After the `aitakeHTML(...)` output, add a small new block, e.g. a function
`deepPredHTML(mid)`:
- If `state.deepPreds[mid]` exists: render the scoreline (reuse `_pickParse`, ~line 1072,
  for the `"2-1"` -> `"2–1 BRA"` formatting) + the `take` text + a small "🔍 Deep
  Prediction · regenerate" affordance (optional — on-demand regen isn't required by the
  user's ask, but cheap to add given the cache key already scopes by form fingerprint).
- Else: render a `<button class="btn" data-deeppred="${mid}">🔍 Deep Prediction</button>`.

Both `.pcard` branches (GS and KO) call this the same way — same as how `picksHTML`/
`aitakeHTML` are already called identically in both places.

**Do NOT** touch the FT branches (`data-mpredft`, ~lines 936 and 960) — no deep-prediction
affordance for finished matches.

### 2g. Event wiring

The delegated click handler on `#predOut` (~line 1444-1447) already handles
`predAnalysisBtn` and the `.lmchip` toggle. Add a branch:
```js
const dp=e.target.closest('[data-deeppred]');
if(dp){genDeepPred(dp.dataset.deeppred);return}
```

### 2h. Styling

Reuse existing classes — `.aitake`, `.pk`/`.aipicks`, `.btn`, `.ldg`, `.lmmodel` (CSS
lines ~237-256) already cover "small AI take with a model credit line" styling. A new
class is likely unnecessary; if one is needed for the button/idle state, keep it minimal
and colocate with the `.aitake`/`.aipicks` rules (~line 247-256).

---

## Verification

1. **Syntax**: `python3 -c "import ast; ast.parse(open('app.py').read())"`; extract the
   `<script>` block from `static/index.html` and run `node --check`.
2. **Server**: run `uvicorn app:app --port 8000`, then:
   - `curl 'localhost:8000/api/team-form?team=BRA'` and `?team=MAR` — confirm `summary`
     and `recent[]` populate (6 entries each), and a second call is fast (cache hit).
   - `curl 'localhost:8000/api/team-form?team=ZZZ'` -> 404.
3. **Browser**: open the Predictions tab, scroll to an **upcoming** group/knockout match.
   - Confirm a `🔍 Deep Prediction` button appears (and only for upcoming matches — FT
     cards unchanged).
   - Click it: Network tab shows `/api/team-form` ×2 + `/api/match` (zero-token), then
     **exactly one** `/api/llm` call. Result renders as scoreline + 1-2 sentence take.
   - Reload the page: the deep prediction persists (via `state.deepPreds` /
     `saveState`/`loadState`).
4. Confirm existing features unaffected: ⚙/◆/● picks row, AI takes (`aitakeHTML`/
   `aitakeFTHTML`), Monte Carlo champion card, and "AI analysis" section all render as
   before.

## Notes / things to double check during implementation

- `_cache_get`/`_cache_set` signature change (if you add a `ttl` param) — check all
  existing call sites (`/api/standings`, `/api/match`) still pass correctly with the
  default.
- ESPN's `all/teams/{id}/schedule` endpoint returns a flat `events[]` list already sorted
  most-recent-first with no `season` param — verified live for Brazil (205) on
  2026-06-15, returning 25 events back to mid-2024, all `completed:true` for past games
  (in-progress/future games would have `completed:false`, which the endpoint code above
  already filters out).
- If ESPN occasionally returns a team's own most-recent WC26 group match (which
  `state.results` may also have, possibly more current/accurate for live matches) — that's
  fine, harmless duplication in the context fed to the model; do not special-case it.
