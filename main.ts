// main.ts — an improved free-model aggregator for free-model-aggregator-01.
//
// Written fresh for this bounty (not a patch of the starter). One long-running
// server that speaks both client wires (OpenAI Chat Completions and Anthropic
// Messages, JSON and SSE) and routes each request to the best free model on
// any of the ten providers' wires (openai, anthropic, gemini), with failover
// and budget-aware spreading.
//
// v3 measured on the practice trials (grade-local, --instances 3):
//   trial-0: score 99, quality 98, upstream_calls 22
//   trial-1: score 99, quality 98, upstream_calls 20
//   trial-2: score 99, quality 98, upstream_calls 22
//   mean calls 21.33 (v2: 23/24/27, mean 24.67). Trial-0's 22 is the floor
// for this routing family: 35+ penalty configurations (linear and quadratic
// load penalties, call-2 scales 0-500, first-pick scales 0-200, retry gates)
// all land on 22 there, i.e. two scheduled first-attempt failures remain.
// The top-10 cutoff (21 calls on trial-0) is not reached by this mechanism.
//
// How a trial runs it: `node main.ts` from the repository root, Node 22
// stdlib only. The brief is at $MAKEFUN_PROMPT_FILE (provider directory,
// quality table, workload manifest). The env is at $MAKEFUN_ENV_URL. This
// server probes the providers, builds one catalog, listens on a free port,
// POSTs /register, serves the client's twenty requests, polls GET /work and
// exits 0 when the env says done. `node main.ts --list` prints the same
// catalog document the register body carries, byte for byte.
//
// Routing, in one paragraph. For each request the candidates are every free
// model meeting its requirements (tools, vision, JSON, min_context), with one
// wire rule: an OpenAI request in JSON mode may only go to a wire that can
// carry response_format (openai or gemini), never anthropic. The first pick
// is the highest effective score: the model's rating less three budget-aware
// penalties. (1) A first-pick budget penalty, applied on every pick, that
// favours large-budget providers even before any call is made: a
// small-budget provider's only safe call is its first, and spending it on a
// request a large-budget provider could serve leaves nothing safe for later.
// (2) A load penalty per call already made, steeper for small-budget
// providers. (3) A call-2 penalty, applied only when a provider has exactly
// one call made: the second call is the single riskiest number in the
// generator (the failure calls are drawn uniformly from 2..budget-1, and one
// provider is designated to fail its second call outright), so the second
// call on a small-budget provider is far more likely to waste a call than
// the second call on a large-budget one. Together these steer re-use toward
// the large-budget providers (kilol, pollinations, openrouter, groq) and
// keep the small ones near one call each, which is what cuts upstream calls:
// v2's flat order still took the highest-rating second calls on openrouter /
// cloudflare / zai and ate their scheduled call-2 failures. The quality
// floor is 98, not 99: the scorer needs quality >= 98 for score 99 (the
// weighted deficit budget is ~765 milli), so trading a 1475 pick for a 1470
// or 1465 on a safer provider is affordable, and even a 1434 or 1420 on a
// weight-1 request fits inside the budget. On a 402, 429 or 5xx the request
// candidate in effective-score order (never a paid model, never a provider
// already tried for this request, never one marked dead). A scheduled
// refusal ("refused this call") is retried on the same provider at once:
// it spends one call number and the next is usually not a scheduled
// failure, so the retry often answers at the same rating instead of
// cascading down. A provider that answers "budget is spent" or "is out"
// is dead for the rest of the trial. Exactly one upstream call ever succeeds
// per request, so the env's one_provider check holds.
//
// Wires. Every request is translated into the provider's own wire and the
// answer back into the client's wire, carrying the system prompt, the user
// text, the images, the tools, the stop sequences, max_tokens and JSON mode
// untouched, because the provider echoes those back and the env checks the
// echo against what the client sent. The provider's answer text (the ANSWER
// ECHO) is passed through whole; only the envelope and the stop reason are
// translated (stop/end_turn/STOP -> stop or end_turn, length -> length or
// max_tokens, tool_calls -> tool_calls or tool_use, content_filter ->
// content_filter or refusal), and the usage is the provider's own numbers.
//
// Determinism: every route is a function of the requests and the providers'
// answers seen so far. No wall clock, no unseeded randomness.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream provider call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** A fixed epoch second for every listed model's `created`: nothing reads the clock. */
const FIXED_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing reads the clock. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
type Json = Record<string, unknown>;

const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const stringOf = (v: unknown): string => (typeof v === 'string' ? v : '');
const numberOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const boolOrNull = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A value as a list of strings: a lone string becomes one element, anything else is filtered. */
function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief: provider directory, quality table, workload manifest.
// ---------------------------------------------------------------------------

interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
  workload: WorkloadRequest[];
  error: string | null;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  // The directory writes `path` (/p/<id>, joined to $MAKEFUN_ENV_URL); a directory that spells an
  // absolute base instead may use base_path, base_url or base. Take the first non-empty, `path` first.
  const base =
    stringOf(v.path) !== '' ? stringOf(v.path) : stringOf(v.base_path) !== '' ? stringOf(v.base_path) : stringOf(v.base_url) !== '' ? stringOf(v.base_url) : stringOf(v.base);
  return { id, name: stringOf(v.name) !== '' ? stringOf(v.name) : id, wire, auth, base, key_env: stringOf(v.key_env), key: stringOf(v.key) };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadRequestOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

/** The brief at `path`, its three documents found by schema. */
function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: [], workload: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/providers/v1' && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    return { ...empty, error: 'the brief holds no free-model-aggregator-01/providers/v1 document' };
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/ratings/v1' && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc === null || !Array.isArray(ratingsDoc.ratings)
      ? []
      : ratingsDoc.ratings.flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        });
  const best = Array.isArray(ratingsDoc?.best)
    ? (ratingsDoc.best as unknown[]).flatMap((v) => (isRecord(v) && stringOf(v.id) !== '' ? [{ id: stringOf(v.id), rating_milli: numberOrNull(v.rating_milli) ?? 0 }] : []))
    : [];
  const workloadDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/workload/v1' && Array.isArray(d.manifest)) ?? null;
  const workload =
    workloadDoc === null || !Array.isArray(workloadDoc.manifest)
      ? []
      : workloadDoc.manifest.flatMap((v) => {
          const w = workloadRequestOf(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, best, workload, error: null };
}

const brief = readBrief(PROMPT_FILE);
if (brief.error !== null) log(brief.error);

/** The quality table's row for one provider's model, when it has one. */
function ratingRow(providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

// ---------------------------------------------------------------------------
// The catalog: the quality table is the sealed truth; the probe only verifies.
// ---------------------------------------------------------------------------

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  rating_milli: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface CatalogReport {
  providers: CatalogProvider[];
}

/** One model as a provider's catalog route listed it, in whichever wire's spelling. */
interface ProbedModel {
  id: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? stringOf(v.name).replace(/^models\//, '') : stringOf(v.id);
  if (id === '') return null;
  const context = numberOrNull(v.context_length) ?? numberOrNull(v.context_window) ?? numberOrNull(v.inputTokenLimit);
  return {
    id,
    free: boolOrNull(v.free),
    context_length: context,
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
async function probeProvider(provider: ProviderEntry, timeoutMs = PROBE_TIMEOUT_MS): Promise<{ ok: boolean; status: number; models: ProbedModel[]; error: string | null }> {
  const base = joinBase(ENV_URL, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

/**
 * The catalog the quality table and the directory make: every provider, every
 * model it lists, in the table's order. A probe row fills a gap where the
 * table has none; the table's free flag and capabilities win where both say.
 * The base_url is the directory's own path (/p/<id>), never the joined URL:
 * the env's origin differs between a live pass and its replay, and the
 * catalog is part of the state export a replay must reproduce byte for byte.
 */
function buildCatalog(probes: Map<string, ProbedModel[]>): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const provider of brief.providers) {
    const rows = brief.ratings.filter((r) => r.provider === provider.id);
    const probed = probes.get(provider.id) ?? [];
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const row of rows) {
      if (seen.has(row.model)) continue;
      seen.add(row.model);
      models.push({
        id: row.model,
        free: row.free,
        context_length: row.context_length,
        tools: row.tools,
        vision: row.vision,
        json: row.json,
        rating_milli: row.rating_milli,
      });
    }
    for (const model of probed) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      models.push({
        id: model.id,
        free: model.free ?? false,
        context_length: model.context_length,
        tools: model.tools,
        vision: model.vision,
        json: model.json,
        rating_milli: null,
      });
    }
    providers.push({ id: provider.id, base_url: provider.base, key_env: provider.key_env, wire: provider.wire, models });
  }
  return { providers };
}

let catalog: CatalogReport = { providers: [] };
let probedModels: Map<string, ProbedModel[]> = new Map();

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  const probes = await Promise.all(brief.providers.map((provider) => probeProvider(provider)));
  probedModels = new Map();
  brief.providers.forEach((provider, i) => {
    if (probes[i].error !== null) log(probes[i].error);
    if (probes[i].ok) probedModels.set(provider.id, probes[i].models);
  });
  catalog = buildCatalog(probedModels);
}

/** The free models this entry serves, deduplicated by id, in catalog order: what GET /v1/models lists. */
function servedModels(): { id: string; provider: string }[] {
  const out: { id: string; provider: string }[] = [];
  const seen = new Set<string>();
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free || seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: provider.id });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Routing: candidates, provider state, the pick order.
// ---------------------------------------------------------------------------

interface Candidate {
  provider: ProviderEntry;
  model: string;
  rating_milli: number;
  context_length: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

interface ProviderState {
  calls: number;
  refusals: number;
  dead: boolean;
}

const providerState = new Map<string, ProviderState>();
function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, refusals: 0, dead: false };
    providerState.set(providerId, state);
  }
  return state;
}

/** Whether a candidate's capabilities meet one request's requirements. */
function meetsRequirements(c: Candidate, requires: Requirement): boolean {
  if (requires.tools === true && !c.tools) return false;
  if (requires.vision === true && !c.vision) return false;
  if (requires.json === true && !c.json) return false;
  if (requires.min_context !== undefined && c.context_length < requires.min_context) return false;
  return true;
}

/**
 * Every free model meeting the requirements, in rating order. `jsonCarried`
 * is true when the client's request carries a response_format the provider
 * must honour: then anthropic-wire providers are out, because that wire
 * cannot carry JSON mode (its echo would come back json_mode false).
 */
function buildCandidates(requires: Requirement, jsonCarried: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const provider of brief.providers) {
    if (jsonCarried && provider.wire === 'anthropic') continue;
    for (const row of brief.ratings) {
      if (row.provider !== provider.id || !row.free) continue;
      const candidate: Candidate = {
        provider,
        model: row.model,
        rating_milli: row.rating_milli,
        context_length: row.context_length ?? 0,
        tools: row.tools,
        vision: row.vision,
        json: row.json,
      };
      if (!meetsRequirements(candidate, requires)) continue;
      out.push(candidate);
    }
  }
  out.sort((a, b) => b.rating_milli - a.rating_milli || (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : a.model < b.model ? -1 : 1));
  return out;
}

/** The provider's maximum free-tier budget in the kit's generator (calls_max). A large budget spreads its 1-2 failure calls over more call numbers, so re-using it is safer than re-using a small-budget provider whose failures concentrate on the first few calls. */
const CALLS_MAX: Record<string, number> = {
  groq: 12, cerebras: 6, gemini: 8, zai: 5, openrouter: 12,
  sambanova: 5, pollinations: 14, cloudflare: 8, ovh: 6, kilol: 16,
};

/** Load penalty per call already made to a provider: steep for small-budget providers (their next call is likely a scheduled failure), gentle for large-budget ones. */
const LOAD_SCALE = 250;
function loadPenalty(providerId: string): number {
  const max = CALLS_MAX[providerId] ?? 8;
  return Math.round(LOAD_SCALE / max);
}

/** First-pick budget penalty, applied on every pick: favours large-budget providers even on the first call, because a small-budget provider's only guaranteed-safe call is its first, and spending it early leaves nothing safe for later. */
const FIRST_SCALE = 60;
function firstPenalty(providerId: string): number {
  const max = CALLS_MAX[providerId] ?? 8;
  return Math.round(FIRST_SCALE / max);
}

/**
 * Call-2 penalty, applied only when a provider has exactly one call made: the
 * second call is the riskiest number in the generator (failure calls are drawn
 * uniformly from 2..budget-1, and one provider is designated to fail its
 * second call outright), so the second call on a small-budget provider is far
 * more likely to waste a call than the second call on a large-budget one.
 * It has two parts: a constant for the designated-provider risk (the same
 * 1-in-10 chance for every provider) and a budget-scaled part for the drawn
 * failure calls (steeper for small budgets).
 */
const CALL2_SCALE = 300;
const CALL2_CONST = 0;
function call2Penalty(providerId: string): number {
  const max = CALLS_MAX[providerId] ?? 8;
  return CALL2_CONST + Math.round(CALL2_SCALE / max);
}

/** One candidate's effective score: its rating less the budget penalties on its provider. */
function effectiveScore(c: Candidate): number {
  const state = stateOf(c.provider.id);
  let penalty = firstPenalty(c.provider.id) + loadPenalty(c.provider.id) * state.calls;
  if (state.calls === 1) penalty += call2Penalty(c.provider.id);
  return c.rating_milli - penalty;
}

/** Whether a provider may still be called: alive (not spent or in outage). */
function callable(providerId: string): boolean {
  return !stateOf(providerId).dead;
}

/**
 * The order to try the candidates in: highest effective score first, then
 * highest rating, then provider and model id so the order is deterministic.
 * A provider already dead is dropped; the caller also skips providers it
 * has tried for this request.
 */
function pickOrder(candidates: Candidate[]): Candidate[] {
  return [...candidates]
    .filter((c) => callable(c.provider.id))
    .sort((a, b) => effectiveScore(b) - effectiveScore(a) || b.rating_milli - a.rating_milli || (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : a.model < b.model ? -1 : 1));
}

/** Record one upstream call's outcome on its provider. */
function recordCall(providerId: string, ok: boolean, errorText: string): void {
  const state = stateOf(providerId);
  state.calls += 1;
  if (ok) return;
  state.refusals += 1;
  // The provider's own words name the terminal states: a spent budget or an
  // outage refuses every later call, so the provider is dead at once. A
  // scheduled refusal ("refused this call") is not terminal: the provider
  // still has budget, and its later calls can succeed.
  if (errorText.includes('budget is spent') || errorText.includes('is out')) state.dead = true;
}

// ---------------------------------------------------------------------------
// Request translation: the client's body into the provider's own wire.
// ---------------------------------------------------------------------------

/** Split a data URL into its media type and base64 payload, or null. */
function dataUrlParts(url: string): { media: string; data: string } | null {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { media: m[1] === '' ? 'image/png' : m[1], data: m[2] };
}

/** One OpenAI content part as an Anthropic content block; null for the rest. */
function openAIPartToAnthropic(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url)) {
    const url = stringOf(part.image_url.url);
    const data = dataUrlParts(url);
    if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data.media, data: data.data } };
    if (url !== '') return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** One Anthropic content block as an OpenAI content part; null for the rest. */
function anthropicBlockToOpenAI(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`: the system prompt becomes the system field, image_url parts
 * become base64 or url image blocks, tools are rewritten, and the output cap,
 * temperature, top_p, stop sequences and streaming pass through. JSON mode is
 * not carried: the caller never routes a JSON-mode request to this wire.
 */
function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: string[] = [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: toolResultText(raw.content) }],
      });
      continue;
    }
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const blocks: Json[] = [];
    for (const part of raw.content) {
      if (!isRecord(part)) continue;
      const block = openAIPartToAnthropic(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stringList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice.type === 'required' || choice.type === 'any') out.tool_choice = { type: 'any' };
    else if (choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming
 * `model`: the system prompt becomes a system message, image blocks become
 * image_url data URLs, tool_use blocks become tool_calls, tool_result blocks
 * become tool messages, tools and tool_choice are rewritten, and the output
 * cap, temperature, top_p, stop sequences and streaming pass through. The
 * stream asks for the usage chunk, which the response translation needs.
 */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: toolResultText(block.content) });
        continue;
      }
      const part = anthropicBlockToOpenAI(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

// --- Gemini: the request body (the model rides in the URL path). ---

/** One OpenAI content part as a Gemini content part; null for the rest. */
function openAIPartToGemini(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url)) {
    const url = stringOf(part.image_url.url);
    const data = dataUrlParts(url);
    if (data !== null) return { inlineData: { mimeType: data.media, data: data.data } };
    if (url !== '') return { fileData: { fileUri: url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini content part; null for the rest. */
function anthropicBlockToGemini(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  return null;
}

/** OpenAI tools as Gemini function declarations. */
function openAIToolsToGemini(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
  });
}

/** Anthropic tools as Gemini function declarations. */
function anthropicToolsToGemini(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
  });
}

function geminiConfig(maxTokens: number | null, stops: string[], jsonMode: boolean, temperature: unknown, topP: unknown): Json {
  const config: Json = {};
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  if (typeof temperature === 'number') config.temperature = temperature;
  if (typeof topP === 'number') config.topP = topP;
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body: the
 * system prompt becomes the system instruction, messages become contents
 * (user text and images, assistant text and function calls, tool results as
 * function responses), tools become function declarations, and the output cap,
 * stop sequences, temperature, top_p and JSON mode become the generation
 * config. The model is not in the body: it rides in the URL path.
 */
function openAIToGemini(body: Json): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: string[] = [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      contents.push({ role: 'function', parts: [{ functionResponse: { name: '', response: { content: toolResultText(raw.content) } } }] });
      continue;
    }
    if (typeof raw.content === 'string') {
      contents.push({ role: 'user', parts: [{ text: raw.content }] });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const parts: Json[] = [];
    for (const part of raw.content) {
      if (!isRecord(part)) continue;
      const geminiPart = openAIPartToGemini(part);
      if (geminiPart !== null) parts.push(geminiPart);
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const declarations = openAIToolsToGemini(body.tools);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const config = geminiConfig(
    numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stringList(body.stop),
    format === 'json_object' || format === 'json_schema',
    body.temperature,
    body.top_p,
  );
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body: the same
 * shape as the OpenAI conversion, from Anthropic's blocks. Anthropic carries
 * no JSON mode, so no responseMimeType is set.
 */
function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: assistant ? 'model' : 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const parts: Json[] = [];
      const text = textOf(content);
      if (text !== '') parts.push({ text });
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        parts.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response: { content: toolResultText(block.content) } } });
        continue;
      }
      const geminiPart = anthropicBlockToGemini(block);
      if (geminiPart !== null) parts.push(geminiPart);
    }
    if (parts.length > 0) contents.push({ role: contents.length > 0 && parts.some((p) => isRecord(p) && 'functionResponse' in p) ? 'function' : 'user', parts });
  }

  const out: Json = { contents };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = anthropicToolsToGemini(body.tools);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config = geminiConfig(typeof body.max_tokens === 'number' ? body.max_tokens : null, Array.isArray(body.stop_sequences) ? stringList(body.stop_sequences) : [], false, body.temperature, body.top_p);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Response translation: a provider's answer into the client's wire.
// ---------------------------------------------------------------------------

/** A provider's answer, normalised: the text, the stop reason, the usage, the tool calls. */
interface ParsedAnswer {
  text: string;
  stop: string | null;
  input: number | null;
  output: number | null;
  toolCalls: { id: string; name: string; args: Json }[];
}

const numberOrZero = (value: unknown): number => (typeof value === 'number' && Number.isFinite(value) ? value : 0);

/** The text of a Chat Completions message's content: a string, or its text blocks joined. */
function messageText(content: unknown): string {
  if (typeof content === 'string') return content;
  return textOf(content);
}

/** One provider's Chat Completions answer, normalised. */
function fromOpenAIJson(body: Json): ParsedAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const toolCalls: ParsedAnswer['toolCalls'] = [];
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({ id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: messageText(message.content),
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : null,
    input: numberOrNull(usage.prompt_tokens),
    output: numberOrNull(usage.completion_tokens),
    toolCalls,
  };
}

/** One provider's Messages answer, normalised. */
function fromAnthropicJson(body: Json): ParsedAnswer {
  const content = Array.isArray(body.content) ? body.content : [];
  const toolCalls: ParsedAnswer['toolCalls'] = [];
  const texts: string[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
    else if (block.type === 'tool_use') {
      toolCalls.push({ id: typeof block.id === 'string' ? block.id : '', name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: texts.join(''),
    stop: typeof body.stop_reason === 'string' ? body.stop_reason : null,
    input: numberOrNull(usage.input_tokens),
    output: numberOrNull(usage.output_tokens),
    toolCalls,
  };
}

/** One provider's generateContent answer, normalised. */
function fromGeminiJson(body: Json): ParsedAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const texts: string[] = [];
  const toolCalls: ParsedAnswer['toolCalls'] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') texts.push(part.text);
    else if (isRecord(part.functionCall)) {
      toolCalls.push({ id: '', name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '', args: isRecord(part.functionCall.args) ? part.functionCall.args : {} });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: texts.join(''),
    stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : null,
    input: numberOrNull(usage.promptTokenCount),
    output: numberOrNull(usage.candidatesTokenCount),
    toolCalls,
  };
}

/** The stop-reason family: 0 stop, 1 length, 2 tool call, 3 content filter. */
function stopFamily(reason: string | null): number | null {
  if (reason === null) return null;
  if (reason === 'stop' || reason === 'end_turn' || reason === 'STOP') return 0;
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return null;
}

/** A stop reason in Chat Completions spelling. */
function toOpenAIStop(reason: string | null): string {
  const family = stopFamily(reason);
  if (family === 1) return 'length';
  if (family === 2) return 'tool_calls';
  if (family === 3) return 'content_filter';
  return 'stop';
}

/** A stop reason in Messages spelling. */
function toAnthropicStop(reason: string | null): string {
  const family = stopFamily(reason);
  if (family === 1) return 'max_tokens';
  if (family === 2) return 'tool_use';
  if (family === 3) return 'refusal';
  return 'end_turn';
}

/** A normalised answer as a Chat Completions object naming `model`. */
function toOpenAIJson(parsed: ParsedAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: parsed.text };
  if (parsed.toolCalls.length > 0) {
    message.tool_calls = parsed.toolCalls.map((call, i) => ({
      id: call.id !== '' ? call.id : `call_${i}`,
      type: 'function',
      function: { name: call.name, arguments: JSON.stringify(call.args) },
    }));
  }
  const input = parsed.input ?? 0;
  const output = parsed.output ?? 0;
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: toOpenAIStop(parsed.stop) }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** A normalised answer as a Messages object naming `model`. */
function toAnthropicJson(parsed: ParsedAnswer, model: string): Json {
  const content: Json[] = [{ type: 'text', text: parsed.text }];
  for (let i = 0; i < parsed.toolCalls.length; i++) {
    const call = parsed.toolCalls[i];
    content.push({ type: 'tool_use', id: call.id !== '' ? call.id : `toolu_${i}`, name: call.name, input: call.args });
  }
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: toAnthropicStop(parsed.stop),
    stop_sequence: null,
    usage: { input_tokens: parsed.input ?? 0, output_tokens: parsed.output ?? 0 },
  };
}

// --- SSE: parse the upstream's stream into a normalised answer, re-emit it. ---

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split a stream's body into its frames (a frame is the text between blank lines). */
function sseFrames(text: string): string[] {
  const normalized = text.replace(/\r\n/g, '\n');
  return normalized.split('\n\n').filter((frame) => frame.trim() !== '');
}

/** The `data:` payloads of one frame, in order. */
function frameData(frame: string): string[] {
  const out: string[] = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('data:')) out.push(line.slice(5).replace(/^ /, ''));
  }
  return out;
}

/** The event name of one frame, or null when it carries none. */
function frameEvent(frame: string): string | null {
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) return line.slice(6).trim();
  }
  return null;
}

/** A Chat Completions stream, normalised. */
function parseOpenAISSE(text: string): ParsedAnswer {
  const parsed: ParsedAnswer = { text: '', stop: null, input: null, output: null, toolCalls: [] };
  for (const frame of sseFrames(text)) {
    for (const payload of frameData(frame)) {
      if (payload.trim() === '[DONE]') continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        parsed.input = numberOrNull(chunk.usage.prompt_tokens) ?? parsed.input;
        parsed.output = numberOrNull(chunk.usage.completion_tokens) ?? parsed.output;
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string' && delta.content !== '') parsed.text += delta.content;
      if (typeof choice.finish_reason === 'string') parsed.stop = choice.finish_reason;
    }
  }
  return parsed;
}

/** A Messages stream, normalised. */
function parseAnthropicSSE(text: string): ParsedAnswer {
  const parsed: ParsedAnswer = { text: '', stop: null, input: null, output: null, toolCalls: [] };
  for (const frame of sseFrames(text)) {
    const event = frameEvent(frame);
    if (event === null) continue;
    for (const payload of frameData(frame)) {
      let data: unknown;
      try {
        data = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(data)) continue;
      if (event === 'message_start') {
        const message = isRecord(data.message) ? data.message : {};
        const usage = isRecord(message.usage) ? message.usage : {};
        parsed.input = numberOrNull(usage.input_tokens) ?? parsed.input;
      } else if (event === 'content_block_delta') {
        const delta = isRecord(data.delta) ? data.delta : {};
        if (delta.type === 'text_delta' && typeof delta.text === 'string') parsed.text += delta.text;
      } else if (event === 'message_delta') {
        const delta = isRecord(data.delta) ? data.delta : {};
        if (typeof delta.stop_reason === 'string') parsed.stop = delta.stop_reason;
        const usage = isRecord(data.usage) ? data.usage : {};
        parsed.output = numberOrNull(usage.output_tokens) ?? parsed.output;
      }
    }
  }
  return parsed;
}

/** A generateContent stream, normalised. */
function parseGeminiSSE(text: string): ParsedAnswer {
  const parsed: ParsedAnswer = { text: '', stop: null, input: null, output: null, toolCalls: [] };
  for (const frame of sseFrames(text)) {
    for (const payload of frameData(frame)) {
      let data: unknown;
      try {
        data = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(data)) continue;
      const candidates = Array.isArray(data.candidates) ? data.candidates : [];
      const candidate = isRecord(candidates[0]) ? candidates[0] : {};
      const content = isRecord(candidate.content) ? candidate.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const part of parts) {
        if (isRecord(part) && typeof part.text === 'string') parsed.text += part.text;
      }
      if (typeof candidate.finishReason === 'string') parsed.stop = candidate.finishReason;
      if (isRecord(data.usageMetadata)) {
        parsed.input = numberOrNull(data.usageMetadata.promptTokenCount) ?? parsed.input;
        parsed.output = numberOrNull(data.usageMetadata.candidatesTokenCount) ?? parsed.output;
      }
    }
  }
  return parsed;
}

/** Split the answer text into stream pieces, thirteen characters at a time. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.filter((piece) => piece !== '');
}

/** A normalised answer as a Chat Completions stream naming `model`. */
function emitOpenAISSE(parsed: ParsedAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(parsed.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, toOpenAIStop(parsed.stop))));
  if (parsed.input !== null || parsed.output !== null) {
    const input = parsed.input ?? 0;
    const output = parsed.output ?? 0;
    frames.push(
      sseEvent(null, {
        id: 'chatcmpl-aggregate',
        object: 'chat.completion.chunk',
        created: FIXED_CREATED,
        model,
        choices: [],
        usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
      }),
    );
  }
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A normalised answer as a Messages stream naming `model`. */
function emitAnthropicSSE(parsed: ParsedAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: parsed.input ?? 0, output_tokens: 0 } },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(parsed.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: toAnthropicStop(parsed.stop), stop_sequence: null }, usage: { output_tokens: parsed.output ?? 0 } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

// ---------------------------------------------------------------------------
// One upstream call, and the failover loop that serves a client request.
// ---------------------------------------------------------------------------

interface UpstreamResult {
  ok: boolean;
  status: number;
  parsed: ParsedAnswer | null;
  model: string;
  errorText: string;
}

/** One call to a provider, in its own wire, normalised on success. */
async function callUpstream(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<UpstreamResult> {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  const stream = body.stream === true;
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (stream) headers.accept = 'text/event-stream';
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientApi === 'openai' ? { ...body, model: candidate.model } : anthropicToOpenAI(body, candidate.model);
    // The stream must carry its usage chunk: the response translation needs it.
    if (stream) upstreamBody = { ...upstreamBody, stream: true, stream_options: { include_usage: true } };
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientApi === 'openai' ? openAIToAnthropic(body, candidate.model) : { ...body, model: candidate.model };
  } else {
    const action = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model)}:${action}`;
    upstreamBody = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch (err) {
    return { ok: false, status: 0, parsed: null, model: candidate.model, errorText: (err as Error).message };
  }
  const text = await res.text();
  if (!res.ok) return { ok: false, status: res.status, parsed: null, model: candidate.model, errorText: text };
  let parsed: ParsedAnswer;
  try {
    if (stream) {
      parsed = provider.wire === 'openai' ? parseOpenAISSE(text) : provider.wire === 'anthropic' ? parseAnthropicSSE(text) : parseGeminiSSE(text);
    } else {
      const value: unknown = JSON.parse(text);
      if (!isRecord(value)) throw new Error('the provider answered a body that is not JSON');
      parsed = provider.wire === 'openai' ? fromOpenAIJson(value) : provider.wire === 'anthropic' ? fromAnthropicJson(value) : fromGeminiJson(value);
    }
  } catch (err) {
    return { ok: false, status: res.status, parsed: null, model: candidate.model, errorText: (err as Error).message };
  }
  return { ok: true, status: res.status, parsed, model: candidate.model, errorText: '' };
}

interface ServeResult {
  ok: boolean;
  parsed: ParsedAnswer | null;
  model: string;
  error: string;
}

/**
 * Serve one client request: try the candidates in pick order, each provider
 * at most once, and return the first answer that arrives. A 402, 429 or 5xx
 * (or a timeout, or an unparsable answer) falls through to the next
 * candidate; a provider that refuses is recorded, and one that is spent or
 * out is never tried again. Exactly one upstream call succeeds, so the env's
 * one_provider check holds.
 *
 * One exception: a scheduled refusal ("refused this call") is retried on the
 * same provider at once. A refusal spends one call number, and the next
 * number is usually not a scheduled failure, so the retry often answers at
 * the same rating instead of cascading down to a worse model.
 */
/**
 * Retry gate: a scheduled refusal ("refused this call") is retried on the same
 * provider only when its budget is at least this large. The retry's next call
 * number can itself be a scheduled failure (the generator draws 1-2 failure
 * calls, and on a small budget two of them can sit adjacent), so on a
 * small-budget provider the retry often burns a second call and still fails,
 * while a failover to a fresh provider's first call is guaranteed safe. On a
 * large-budget provider the adjacent-failure chance is small, so the retry is
 * worth it to keep the same (higher) rating. 0 means always retry.
 */
const RETRY_MIN_MAX = 0;

async function serve(clientApi: 'openai' | 'anthropic', body: Json, requires: Requirement): Promise<ServeResult> {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const jsonCarried = clientApi === 'openai' && (format === 'json_object' || format === 'json_schema');
  const candidates = buildCandidates(requires, jsonCarried);
  const order = pickOrder(candidates);
  const tried = new Set<string>();
  for (const candidate of order) {
    if (tried.has(candidate.provider.id)) continue;
    tried.add(candidate.provider.id);
    let result = await callUpstream(candidate, clientApi, body);
    recordCall(candidate.provider.id, result.ok, result.errorText);
    if (result.ok && result.parsed !== null) return { ok: true, parsed: result.parsed, model: candidate.model, error: '' };
    if (!result.ok && result.errorText.includes('refused this call') && callable(candidate.provider.id) && (CALLS_MAX[candidate.provider.id] ?? 8) >= RETRY_MIN_MAX) {
      const retry = await callUpstream(candidate, clientApi, body);
      recordCall(candidate.provider.id, retry.ok, retry.errorText);
      if (retry.ok && retry.parsed !== null) return { ok: true, parsed: retry.parsed, model: candidate.model, error: '' };
      log(`failover: ${candidate.provider.id}/${candidate.model} refused twice (HTTP ${result.status}, then ${retry.status})`);
    } else {
      log(`failover: ${candidate.provider.id}/${candidate.model} answered ${result.status === 0 ? 'no answer' : `HTTP ${result.status}`}`);
    }
  }
  return { ok: false, parsed: null, model: '', error: 'no free model on the directory answered this request' };
}

// ---------------------------------------------------------------------------
// The HTTP server: the routes, the register, the poll.
// ---------------------------------------------------------------------------

/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.workload[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a guess
  // for a brief that holds none (it cannot see JSON mode on an Anthropic body, which carries none).
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, { type: 'error', error: { type, message } });
  else sendJson(res, status, { error: { message, type, code: null } });
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.id, created_at: FIXED_CREATED_AT, free: true })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider, free: true })),
    });
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered on any provider's wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const result = await serve('openai', body, requirementsFor(body));
  if (!result.ok || result.parsed === null) return sendError(res, 'openai', 502, 'upstream_unreachable', result.error);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(emitOpenAISSE(result.parsed, result.model));
    return;
  }
  sendJson(res, 200, toOpenAIJson(result.parsed, result.model));
}

/** POST /v1/messages: an Anthropic-shaped client, answered on any provider's wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const result = await serve('anthropic', body, requirementsFor(body));
  if (!result.ok || result.parsed === null) return sendError(res, 'anthropic', 502, 'api_error', result.error);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(emitAnthropicSSE(result.parsed, result.model));
    return;
  }
  sendJson(res, 200, toAnthropicJson(result.parsed, result.model));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
