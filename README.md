# Swarm Signal — Live Swarm Forecast Console

Swarm Signal is a local-first console for exploring a real-world question with a
bounded live provider panel. The normal workflow is question-first: a user enters
one question and chooses 2–12 live agents. The server performs bounded public
source discovery, sends the same evidence packet to independent role agents, asks
a synthesis lead to compare their claims and counterclaims, and returns the full
agent reports, debate trace, locally generated argument map, and deterministic
scores instead of hiding the signal behind one number. Context, domains,
objectives, and source notes remain optional advanced controls.

The LIVE mode calls the provider selected in Settings. It does not claim that a
requested virtual size equals the number of billable provider calls: live runs
use exactly the selected 2–12 role calls plus one synthesis call. Role calls are
real provider requests; the client waits for the research and provider response
instead of simulating completion with fake result data. DEMO mode is a separate
bundled playback gallery and makes no API calls.

The argument map and scores are generated locally from the returned question,
evidence citations, and structured role reports. Provider-supplied graph fields
are intentionally ignored by the local analysis engine.

## Modes

### LIVE

Open **Settings**, choose a provider, model, API key, and optional endpoint, then
use **Use for live runs**. The browser keeps the key in JavaScript memory for the
current page session only. It is sent to the same-origin server for provider
calls and is not written to localStorage, history, URLs, or result payloads.

Supported provider routes:

- OpenAI
- Anthropic
- Google Gemini
- OpenRouter
- Custom OpenAI-compatible endpoints

The provider catalog is exposed at `GET /api/providers`. Connection checks use
`POST /api/providers/test`; live runs use `POST /api/live-simulate`.

### DEMO

Open **Demo** to play three precomputed examples from policy/energy,
technology/operations, and climate/finance. Demo playback is bundled in
`static/app.js`, does not need an API key, and does not call `/api/*` for the
playback itself.

## Question-first workflow

1. Open **Settings**, configure and test a provider, and choose **Use for live
   runs**.
2. Return to **Live**, enter a question, and select the agent count. No URLs,
   prompt templates, or research packet are required.
3. Run the research swarm. The UI shows the actual stages for discovery, role
   responses, debate/synthesis, local scoring, and map rendering.
4. Inspect the complete expandable report for every selected agent, including its
   question, answer, stance, confidence, claim, counterclaim, rebuttal, cited
   source IDs, and open questions. The full transcript also includes each role
   report and any synthesis debate rounds returned by the provider.

Automatic discovery is deliberately bounded and inspectable. It derives up to
four searches from the question (the original wording plus evidence/data,
risks/criticism, and policy/impact variants), deduplicates results, and fetches
at most five discovered URLs within the eight-source collection limit. This is a
research assist, not comprehensive internet coverage. A discovery failure is
reported as an evidence limitation and the live panel may continue with an
explicitly evidence-limited packet.

## Requirements

- Python 3.12 (pinned in `.python-version`; supported by Vercel's Python runtime)
- No third-party Python packages; `requirements.lock` is intentionally empty
- `make` is convenient but not required

## Setup

From the project directory:

```bash
python3 --version
```

Use a Python 3.12 interpreter. The backend has no package installation step
because it uses only the standard library.

## Test

Run the single offline test command:

```bash
make test
```

The tests cover request validation, deterministic demo results, provider request
and response shapes, unsafe endpoint rejection, bounded live orchestration,
automatic multi-query discovery, exact agent counts, structured role reports,
local argument-map/scoring behavior, credential-safe API errors, the custom
thousands-scale, and the local HTTP/static console contract.

## Run locally

```bash
make run
```

The server binds to `127.0.0.1:8000` by default. It serves files from
`static/`, returns the static `index.html` when one is present, and keeps API
responses same-origin by sending no CORS headers.

Open http://127.0.0.1:8000/ in a browser after starting the server.

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

Simulation request:

```bash
curl -X POST http://127.0.0.1:8000/api/simulate \
  -H 'Content-Type: application/json' \
  -d '{
    "scenario": "Will the neighborhood solar project receive approval this year?",
    "context": "Funding is secured, permits are filed, and the decision is expected by December.",
    "swarm_size": 128,
    "domains": ["policy", "energy", "community"],
    "debate_rounds": 3
  }'
```

The request fields are:

- `scenario`: required non-empty string, up to 2,000 characters
- `context`: optional conditions or assumptions, up to 1,200 characters; these
  are included as directional input but are not independently verified
- `swarm_size`: an integer from `24` to `5000` (default `128`); the UI includes
  quick presets and a continuous thousands-scale slider
- `domains`: non-empty list of up to eight unique strings (default `["general"]`)
- `debate_rounds`: integer from `1` to `5` (default `2`)

The response includes `probability` and `confidence` values in `[0, 1]`,
percentage conveniences, `key_drivers`, `uncertainties`, a
`representative_roster`, `debate_messages`, and a `method` object that marks
the result as non-live and external-data-free. This endpoint is retained for
local tooling; the UI's DEMO gallery uses bundled data and does not call it.
Invalid JSON or fields produce a JSON error with a stable `code`, human-readable
`message`, and (when applicable) `field`.

## Research API

`POST /api/research` accepts a forecasting `statement` (or `question`), an
optional array of public HTTPS `urls`, an optional `discover` boolean, and an
optional per-source `timeout` in seconds. Discovery defaults to **enabled** when
the flag is omitted: the server queries one fixed public search endpoint using
up to four bounded query variants, then fetches a small deduplicated set of
public HTTPS pages. This is a discovery assist, not unrestricted browsing or a
claim to collect all information on the internet:

```bash
curl -X POST http://127.0.0.1:8000/api/research \
  -H 'Content-Type: application/json' \
  -d '{
    "statement": "Will the neighborhood solar project receive approval this year?",
    "urls": [],
    "discover": true,
    "timeout": 8
  }'
```

The response contains `statement`, bounded `sources`, `evidence_count`,
`discovery`, and `limitations`. Each source includes its title, safe
final/canonical URL when
available, readable text and excerpt, content type, fetch time, status, and a
fixed per-source error when collection failed. Only the submitted URLs and,
when `discover` is enabled, a small result set from the fixed public search
endpoint are fetched within size and timeout limits; the packet is not
comprehensive internet research. Redirects, credentials, queries/fragments,
unsafe hosts, and non-HTTPS URLs are rejected. `GET /api/research/limits` exposes the active
collection bounds. A partial collection returns HTTP 200 with source errors;
an all-failed collection returns HTTP 502, or HTTP 504 when every source timed
out.

## Live API request

The browser normally sends this request after Settings has validated a provider.
For a manual test, replace the placeholder key locally and never commit it. The
smallest live request is a question, an agent count, and a provider; automatic
research runs when no `research_packet` is supplied:

```bash
curl -X POST http://127.0.0.1:8000/api/live-simulate \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "Will the neighborhood solar project receive approval this year?",
    "agent_count": 6,
    "provider": {
      "provider_id": "openai",
      "model": "gpt-4o-mini",
      "api_key": "replace-with-your-key",
      "timeout": 20
    }
  }'
```

The live endpoint automatically performs the same bounded discovery when no
research packet is supplied. Set `research_enabled: false` or `auto_research:
false` to explicitly skip it. The request may additionally include `context`,
`domains`, `debate_rounds`, `swarm_size`, and an `objectives` array of up to
twelve bounded strings. `agent_count`, `number_of_agents`, and `num_agents` are
accepted aliases and must agree when more than one is supplied.

Source titles and excerpts are passed to the bounded role panel and synthesis
lead as untrusted evidence; prompts instruct agents to cite source IDs, identify
conflicts, compare claims, and avoid fabricated facts. The live response
includes `agent_reports`, `debate_transcript`, `evidence_items`,
`requested_agent_count`, `actual_live_agent_count`, `total_api_calls`,
`research_summary`, `source_count`, `research_limitations`, `progress_stages`,
`timing`, timestamps, and `debate_rounds_completed`. The completed-round count
reflects explicit synthesis rounds actually returned; role reports remain in the
transcript even when a provider returns no separate round records.

`argument_map.generated_locally` is true for the deterministic map built from
the question, successful evidence, and structured role reports. `scoring` and
`agent_scores` use the same local boundary and expose evidence coverage,
support, opposition, consensus, disagreement, and per-agent argument quality.
Provider graph fields are not used for those values. Optional provider fields
(`impact_nodes`, `connections`, `decision_paths`, and `agent_objectives`) remain
backward-compatible response fields but do not control local graph generation.
Provider failures are mapped to safe 4xx/502/504 responses; provider response
bodies and credentials are not returned.

## Production deployment and security

This repository is production-oriented for a controlled deployment, but it is
not a complete multi-tenant service. Before exposing it publicly:

- Put it behind authenticated HTTPS and a reverse proxy.
- Add user authentication, authorization, rate limiting, request quotas, and
  billing controls at the deployment boundary.
- Keep provider keys session-only unless an authenticated secret manager is
  added. Do not put keys in source, Docker images, logs, browser localStorage,
  URLs, or analytics payloads.
- Keep `SWARM_SIGNAL_ALLOW_INSECURE_HTTP` unset in production. It exists only to
  let offline tests call a loopback fake provider over HTTP.
- Review provider terms, model availability, privacy requirements, and data
  retention before sending sensitive scenarios to a third-party API.

## Deploy from GitHub to Vercel

This repository includes a Vercel-ready `api/` directory, `vercel.json`,
`requirements.txt`, and `.vercelignore`. No Node package or third-party Python
package is required.

1. Push this folder to a GitHub repository.
2. In Vercel, choose **Add New… → Project → Import Git Repository** and select
   the repository.
3. Keep the repository root as the project root. Leave the build command and
   output directory blank; `vercel.json` routes the static console and the
   Python functions automatically.
4. Choose **Deploy**. The home page is served from `static/index.html`; the
   API routes are backed by the handlers in `api/` and reuse the same code as
   the local server.
5. Open the deployed URL, go to **Settings**, and configure a provider for the
   current browser session. Do not commit an API key or put one in a public
   GitHub repository.

The Vercel function is stateless and subject to the duration limit of the
selected Vercel plan. Each live run performs at most twelve role calls plus one
synthesis call, so a large agent count or a slow provider can still time out on
short function plans. The local Docker/`make run` deployment remains available
when a longer request window is required.

## Deploy with Docker

The image uses the pinned Python 3.12 runtime and has no external dependency
installation step:

```bash
docker build -t local-swarm-forecast .
docker run --rm -p 8000:8000 local-swarm-forecast
```

The container listens on `0.0.0.0:8000`; open the same health and simulation
URLs through the published host port.
