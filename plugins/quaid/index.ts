/**
 * quaid - Total Recall memory system plugin for Clawdbot
 *
 * Uses SQLite + Ollama embeddings for fully local memory storage.
 * Replaces memory-lancedb with no external API dependencies.
 */

import type { ClawdbotPluginApi } from "openclaw/plugin-sdk";
import { Type } from "@sinclair/typebox";
import { execSync, spawn } from "node:child_process";
import * as path from "node:path";
import * as fs from "node:fs";
import * as os from "node:os";
import { registerPluginHttpRoute } from "openclaw/plugin-sdk";

// Configuration
const PLUGIN_DIR = __dirname;
const WORKSPACE = process.env.CLAWDBOT_WORKSPACE || "${QUAID_WORKSPACE}";
const PYTHON_SCRIPT = path.join(WORKSPACE, "plugins/quaid/memory_graph.py");
const DB_PATH = path.join(WORKSPACE, "data/memory.db");

// Model resolution — reads from config/memory.json, no hardcoded model IDs
let _memoryConfig: any = null;
function getMemoryConfig(): any {
  if (_memoryConfig) { return _memoryConfig; }
  try {
    _memoryConfig = JSON.parse(fs.readFileSync(path.join(WORKSPACE, "config/memory.json"), "utf8"));
  } catch {
    _memoryConfig = {};
  }
  return _memoryConfig;
}

// System gates — check if a subsystem is enabled in config
function isSystemEnabled(system: "memory" | "journal" | "projects" | "workspace"): boolean {
  const config = getMemoryConfig();
  const systems = config.systems || {};
  // Default to true if not specified
  return systems[system] !== false;
}
// Model defaults — single source of truth for fallbacks
const DEFAULT_LOW_REASONING_MODEL = "claude-haiku-4-5";
const DEFAULT_HIGH_REASONING_MODEL = "claude-opus-4-6";

function getLowReasoningModel(): string {
  return getMemoryConfig().models?.lowReasoning || DEFAULT_LOW_REASONING_MODEL;
}
function getHighReasoningModel(): string {
  const configured = getMemoryConfig().models?.highReasoning;
  return (configured && configured !== "default") ? configured : DEFAULT_HIGH_REASONING_MODEL;
}

// Config schema
const configSchema = Type.Object({
  autoCapture: Type.Optional(Type.Boolean({ default: false })),
  autoRecall: Type.Optional(Type.Boolean({ default: true })),
});

type PluginConfig = {
  autoCapture?: boolean;
  autoRecall?: boolean;
};

// User identity mapping (loaded from config/memory.json)
type UsersConfig = {
  defaultOwner: string;
  identities: Record<string, {
    channels: Record<string, string[]>;
    speakers: string[];
  }>;
};

let _usersConfig: UsersConfig | null = null;

function getUsersConfig(): UsersConfig {
  if (_usersConfig) { return _usersConfig; }
  try {
    const configPath = path.join(WORKSPACE, "config/memory.json");
    const raw = JSON.parse(fs.readFileSync(configPath, "utf8"));
    _usersConfig = raw.users || { defaultOwner: "default", identities: {} };
  } catch {
    _usersConfig = { defaultOwner: "default", identities: {} };
  }
  return _usersConfig!;
}

function resolveOwner(speaker?: string, channel?: string): string {
  const config = getUsersConfig();
  for (const [userId, identity] of Object.entries(config.identities)) {
    // Match by speaker name
    if (speaker && identity.speakers.some(s =>
      s.toLowerCase() === speaker.toLowerCase()
    )) {
      return userId;
    }
    // Match by channel
    if (channel && identity.channels[channel]) {
      const allowed = identity.channels[channel];
      if (allowed.includes("*")) { return userId; }
      if (speaker && allowed.some(a => a.toLowerCase() === speaker.toLowerCase())) { return userId; }
    }
  }
  return config.defaultOwner;
}

// ============================================================================
// Python Bridge
// ============================================================================

const PYTHON_BRIDGE_TIMEOUT_MS = 120_000; // 2 minutes

async function callPython(command: string, args: string[] = []): Promise<string> {
  return new Promise((resolve, reject) => {
    const proc = spawn("python3", [PYTHON_SCRIPT, command, ...args], {
      env: { ...process.env, MEMORY_DB_PATH: DB_PATH, CLAWDBOT_WORKSPACE: WORKSPACE },
    });

    let stdout = "";
    let stderr = "";
    let settled = false;

    const timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        proc.kill("SIGTERM");
        reject(new Error(`Python bridge timeout after ${PYTHON_BRIDGE_TIMEOUT_MS}ms: ${command} ${args.join(" ")}`));
      }
    }, PYTHON_BRIDGE_TIMEOUT_MS);

    proc.stdout.on("data", (data) => {
      stdout += data;
    });
    proc.stderr.on("data", (data) => {
      stderr += data;
    });

    proc.on("close", (code) => {
      if (settled) { return; }
      settled = true;
      clearTimeout(timer);
      if (code === 0) {
        resolve(stdout.trim());
      } else {
        reject(new Error(`Python error: ${stderr || stdout}`));
      }
    });

    proc.on("error", (err) => {
      if (settled) { return; }
      settled = true;
      clearTimeout(timer);
      reject(err);
    });
  });
}

// ============================================================================
// Memory Notes — queued for extraction at compaction/reset
// ============================================================================

// Session-scoped notes: memory_store writes here instead of directly to DB.
// At compaction/reset, these are prepended to the transcript so Opus extracts
// them with full context, edges, and quality review.
const _memoryNotes = new Map<string, string[]>();
const NOTES_DIR = "/tmp";

function getNotesPath(sessionId: string): string {
  return path.join(NOTES_DIR, `memory-notes-${sessionId}.json`);
}

function addMemoryNote(sessionId: string, text: string, category: string): void {
  // In-memory
  if (!_memoryNotes.has(sessionId)) {
    _memoryNotes.set(sessionId, []);
  }
  _memoryNotes.get(sessionId)!.push(`[${category}] ${text}`);

  // Persist to disk (survives gateway restart)
  try {
    const notesPath = getNotesPath(sessionId);
    let existing: string[] = [];
    try { existing = JSON.parse(fs.readFileSync(notesPath, "utf8")); } catch {}
    existing.push(`[${category}] ${text}`);
    fs.writeFileSync(notesPath, JSON.stringify(existing), { mode: 0o600 });
  } catch {}
}

function getAndClearMemoryNotes(sessionId: string): string[] {
  // Merge in-memory + disk
  const inMemory = _memoryNotes.get(sessionId) || [];
  let onDisk: string[] = [];
  const notesPath = getNotesPath(sessionId);
  try { onDisk = JSON.parse(fs.readFileSync(notesPath, "utf8")); } catch {}

  // Deduplicate (in case both sources have the same note)
  const all = Array.from(new Set([...inMemory, ...onDisk]));

  // Clear both
  _memoryNotes.delete(sessionId);
  try { fs.unlinkSync(notesPath); } catch {}

  return all;
}

// Also collect notes from ALL sessions (for cases where session ID isn't matched)
function getAndClearAllMemoryNotes(): string[] {
  const all: string[] = [];

  // In-memory
  for (const [sid, notes] of _memoryNotes.entries()) {
    all.push(...notes);
    _memoryNotes.delete(sid);
  }

  // Disk — glob for memory-notes-*.json
  try {
    const files = fs.readdirSync(NOTES_DIR).filter(f => f.startsWith("memory-notes-") && f.endsWith(".json"));
    for (const f of files) {
      const fp = path.join(NOTES_DIR, f);
      try {
        const notes: string[] = JSON.parse(fs.readFileSync(fp, "utf8"));
        all.push(...notes);
        fs.unlinkSync(fp);
      } catch {}
    }
  } catch {}

  return Array.from(new Set(all));
}

// ============================================================================
// Session ID Helper
// ============================================================================

function extractSessionId(messages: any[], ctx?: any): string {
  // Prefer Pi SDK session UUID if provided by hook context
  if (ctx?.sessionId) {
    return ctx.sessionId;
  }

  // Fallback: hash-based session ID (for backward compat with older source)
  // Find first user message with timestamp
  let firstTimestamp = "";
  const filteredMessages = messages.filter((m: any) => {
    if (m.role !== "user") { return false; }

    // Skip system-injected messages even if they have timestamps
    let content = '';
    if (typeof m.content === 'string') {
      content = m.content;
    } else if (Array.isArray(m.content)) {
      content = m.content.map((c: any) => c.text || '').join(' ');
    }
    if (content.startsWith("GatewayRestart:")) { return false; }
    if (content.startsWith("System:")) { return false; }
    if (content.includes('"kind": "restart"')) { return false; }

    return true;
  });
  
  if (filteredMessages.length > 0) {
    const firstMessage = filteredMessages[0];
    if (firstMessage.timestamp) {
      firstTimestamp = String(firstMessage.timestamp);
    } else {
      firstTimestamp = Date.now().toString();
    }
  } else {
    firstTimestamp = Date.now().toString();
  }
  
  // Create session identifier: timestamp hash only (channel agnostic)
  const timestampHash = require('crypto').createHash('md5').update(firstTimestamp).digest('hex').substring(0, 12);
  return timestampHash;
}

// ============================================================================
// Python Bridges (docs_updater, docs_rag)
// ============================================================================

const DOCS_UPDATER = path.join(WORKSPACE, "plugins/quaid/docs_updater.py");
const DOCS_RAG = path.join(WORKSPACE, "plugins/quaid/docs_rag.py");
const DOCS_REGISTRY = path.join(WORKSPACE, "plugins/quaid/docs_registry.py");
const PROJECT_UPDATER = path.join(WORKSPACE, "plugins/quaid/project_updater.py");

function _getApiKey(): string | undefined {
  let apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    try {
      apiKey = process.env.ANTHROPIC_API_KEY || "";
    } catch { /* env var not available */ }
  }
  return apiKey;
}

function _spawnWithTimeout(
  script: string, command: string, args: string[],
  label: string, env: Record<string, string | undefined>,
  timeoutMs: number = PYTHON_BRIDGE_TIMEOUT_MS
): Promise<string> {
  return new Promise((resolve, reject) => {
    const proc = spawn("python3", [script, command, ...args], {
      cwd: WORKSPACE,
      env: { ...process.env, ...env },
    });
    let stdout = "";
    let stderr = "";
    let settled = false;

    const timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        proc.kill("SIGTERM");
        reject(new Error(`${label} timeout after ${timeoutMs}ms: ${command} ${args.join(" ")}`));
      }
    }, timeoutMs);

    proc.stdout.on("data", (data: Buffer) => { stdout += data; });
    proc.stderr.on("data", (data: Buffer) => { stderr += data; });
    proc.on("close", (code: number | null) => {
      if (settled) { return; }
      settled = true;
      clearTimeout(timer);
      if (code === 0) { resolve(stdout.trim()); }
      else { reject(new Error(`${label} error: ${stderr || stdout}`)); }
    });
    proc.on("error", (err: Error) => {
      if (settled) { return; }
      settled = true;
      clearTimeout(timer);
      reject(err);
    });
  });
}

async function callDocsUpdater(command: string, args: string[] = []): Promise<string> {
  const apiKey = _getApiKey();
  return _spawnWithTimeout(DOCS_UPDATER, command, args, "docs_updater", {
    CLAWDBOT_WORKSPACE: WORKSPACE, ...(apiKey ? { ANTHROPIC_API_KEY: apiKey } : {}),
  });
}

async function callDocsRag(command: string, args: string[] = []): Promise<string> {
  return _spawnWithTimeout(DOCS_RAG, command, args, "docs_rag", {
    CLAWDBOT_WORKSPACE: WORKSPACE,
  });
}

async function callDocsRegistry(command: string, args: string[] = []): Promise<string> {
  return _spawnWithTimeout(DOCS_REGISTRY, command, args, "docs_registry", {
    CLAWDBOT_WORKSPACE: WORKSPACE,
  });
}

async function callProjectUpdater(command: string, args: string[] = []): Promise<string> {
  const apiKey = _getApiKey();
  return _spawnWithTimeout(PROJECT_UPDATER, command, args, "project_updater", {
    CLAWDBOT_WORKSPACE: WORKSPACE, ...(apiKey ? { ANTHROPIC_API_KEY: apiKey } : {}),
  }, 300_000); // 5 min for Opus calls
}

// ============================================================================
// Project Helpers
// ============================================================================

function getProjectNames(): string[] {
  try {
    const output = execSync(
      `python3 "${DOCS_REGISTRY}" list-projects --names-only`,
      { encoding: "utf-8", env: { ...process.env, MEMORY_DB_PATH: DB_PATH, CLAWDBOT_WORKSPACE: WORKSPACE }, timeout: 10_000 }
    ).trim();
    return output.split("\n").filter(Boolean);
  } catch {
    // Fallback: read from JSON (backward compat / fresh install)
    try {
      const configPath = path.join(WORKSPACE, "config/memory.json");
      const configData = JSON.parse(fs.readFileSync(configPath, "utf-8"));
      return Object.keys(configData?.projects?.definitions || {});
    } catch {
      return [];
    }
  }
}

/**
 * Spawn a fire-and-forget Python notification script safely.
 * Writes code to a temp file to avoid shell injection via inline -c strings.
 * The script auto-deletes its temp file on completion.
 */
function spawnNotifyScript(scriptBody: string): void {
  const tmpFile = `/tmp/notify-${Date.now()}-${Math.random().toString(36).slice(2)}.py`;
  const preamble = `import sys, os\nsys.path.insert(0, ${JSON.stringify(path.join(WORKSPACE, "plugins/quaid"))})\n`;
  const cleanup = `\nos.unlink(${JSON.stringify(tmpFile)})\n`;
  fs.writeFileSync(tmpFile, preamble + scriptBody + cleanup, { mode: 0o600 });
  const proc = spawn('python3', [tmpFile], {
    detached: true,
    stdio: 'ignore',
    env: { ...process.env, MEMORY_DB_PATH: DB_PATH },
  });
  proc.unref();
}

// ============================================================================
// Project Event Emitter (compact/reset → background processor)
// ============================================================================

function extractFilePaths(messages: any[]): string[] {
  const paths = new Set<string>();
  for (const msg of messages) {
    const text = typeof msg.content === "string"
      ? msg.content
      : (msg.content as any[])?.map((c: any) => c.text || "").join(" ") || "";
    // Match common file path patterns
    const matches = text.match(/(?:^|\s)((?:\/[\w.-]+)+|(?:[\w.-]+\/)+[\w.-]+)/gm);
    if (matches) {
      for (const m of matches) {
        const p = m.trim();
        if (p.includes("/") && !p.startsWith("http") && p.length < 200) {
          paths.add(p);
        }
      }
    }
  }
  return Array.from(paths);
}

async function getQuickProjectSummary(messages: any[]): Promise<{project_name: string | null, text: string}> {
  const transcript = buildTranscript(messages);
  if (!transcript || transcript.length < 20) {
    return { project_name: null, text: "" };
  }

  // Get API key
  let apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    try {
      apiKey = process.env.ANTHROPIC_API_KEY || "";
    } catch {
      return { project_name: null, text: transcript.slice(0, 500) };
    }
  }

  try {
    const response = await fetch(ANTHROPIC_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey!,
        "anthropic-version": "2023-06-01"
      },
      body: JSON.stringify({
        model: getLowReasoningModel(),
        max_tokens: 300,
        system: `You summarize coding sessions. Given a conversation, identify: 1) What project was being worked on (use one of the available project names, or null if unclear), 2) Brief summary of what changed/was discussed. Available projects: ${getProjectNames().join(", ")}. Use these EXACT names. Respond with JSON only: {"project_name": "name-or-null", "text": "brief summary"}`,
        messages: [{ role: "user", content: `Summarize this session:\n\n${transcript.slice(0, 4000)}` }]
      })
    });

    if (response.ok) {
      const data = await response.json() as { content?: Array<{ text?: string }> };
      const output = data.content?.[0]?.text?.trim() || "";
      try {
        const jsonMatch = output.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          const parsed = JSON.parse(jsonMatch[0]);
          return {
            project_name: typeof parsed.project_name === "string" ? parsed.project_name : null,
            text: typeof parsed.text === "string" ? parsed.text : "",
          };
        }
      } catch {}
    }
  } catch (err: unknown) {
    console.error("[quaid] Quick project summary failed:", (err as Error).message);
  }

  return { project_name: null, text: transcript.slice(0, 500) };
}

async function emitProjectEvent(messages: any[], trigger: string, sessionId?: string): Promise<void> {
  const memConfig = getMemoryConfig();
  if (!memConfig.projects?.enabled) {
    return;
  }

  try {
    // 1. Quick LLM summary
    const summary = await getQuickProjectSummary(messages);

    // 2. Write event file
    const event = {
      project_hint: summary.project_name || null,
      files_touched: extractFilePaths(messages),
      summary: summary.text,
      trigger,
      session_id: sessionId,
      timestamp: new Date().toISOString(),
    };

    const stagingDir = path.join(WORKSPACE, memConfig.projects.stagingDir || "projects/staging/");
    if (!fs.existsSync(stagingDir)) {
      fs.mkdirSync(stagingDir, { recursive: true });
    }

    const eventPath = path.join(stagingDir, `${Date.now()}-${trigger}.json`);
    fs.writeFileSync(eventPath, JSON.stringify(event, null, 2));

    // 3. Spawn background processor (detached) — pass API key for Opus calls
    let bgApiKey = process.env.ANTHROPIC_API_KEY;
    if (!bgApiKey) {
      try {
        bgApiKey = process.env.ANTHROPIC_API_KEY || "";
      } catch { /* env var not available */ }
    }
    const logFile = path.join(WORKSPACE, "logs/project-updater.log");
    const logDir = path.dirname(logFile);
    if (!fs.existsSync(logDir)) { fs.mkdirSync(logDir, { recursive: true }); }
    const logFd = fs.openSync(logFile, "a");
    try {
      const proc = spawn("python3", [PROJECT_UPDATER, "process-event", eventPath], {
        detached: true,
        stdio: ["ignore", logFd, logFd],
        cwd: WORKSPACE,
        env: { ...process.env, CLAWDBOT_WORKSPACE: WORKSPACE, ...(bgApiKey ? { ANTHROPIC_API_KEY: bgApiKey } : {}) },
      });
      proc.unref();
    } finally {
      // Close FD in parent process — child has its own copy
      fs.closeSync(logFd);
    }

    console.log(`[quaid] Emitted project event: ${trigger} -> ${summary.project_name || "unknown"}`);
  } catch (err: unknown) {
    console.error("[quaid] Failed to emit project event:", (err as Error).message);
  }
}

// ============================================================================
// Transcript Builder (shared by memory extraction + doc update)
// ============================================================================

function buildTranscript(messages: any[]): string {
  const transcript: string[] = [];
  for (const msg of messages) {
    if (msg.role !== "user" && msg.role !== "assistant") { continue; }
    let text = typeof msg.content === "string"
      ? msg.content
      : (msg.content as any[])?.map((c: any) => c.text || "").join(" ");
    if (!text) { continue; }
    text = text.replace(/^\[(?:Telegram|WhatsApp|Discord|Signal|Slack)\s+[^\]]+\]\s*/i, "");
    text = text.replace(/\n?\[message_id:\s*\d+\]/gi, "").trim();
    if (text.startsWith("GatewayRestart:") || text.startsWith("System:")) { continue; }
    if (text.includes('"kind": "restart"')) { continue; }
    if (text.includes("HEARTBEAT") && text.includes("HEARTBEAT_OK")) { continue; }
    if (text.replace(/[*_<>\/b\s]/g, '').startsWith("HEARTBEAT_OK")) { continue; }
    if (!text) { continue; }
    transcript.push(`${msg.role === "user" ? "User" : "Assistant"}: ${text}`);
  }
  return transcript.join("\n\n");
}

/**
 * Split messages into chunks by accumulating whole messages until size threshold.
 * Each chunk is an array of messages. Never splits a single message.
 */
function chunkMessages(messages: any[], maxChunkChars: number): any[][] {
  if (!messages.length) return [];

  const chunks: any[][] = [];
  let currentChunk: any[] = [];
  let currentSize = 0;

  for (const msg of messages) {
    // Estimate message size (role + content text)
    const content = typeof msg.content === 'string'
      ? msg.content
      : Array.isArray(msg.content)
        ? msg.content.map((b: any) => b.text || '').join('')
        : JSON.stringify(msg.content || '');
    const msgSize = (msg.role || '').length + content.length + 10; // overhead

    // If adding this message exceeds threshold AND chunk isn't empty, start new chunk
    if (currentSize + msgSize > maxChunkChars && currentChunk.length > 0) {
      chunks.push(currentChunk);
      currentChunk = [];
      currentSize = 0;
    }

    currentChunk.push(msg);
    currentSize += msgSize;
  }

  if (currentChunk.length > 0) {
    chunks.push(currentChunk);
  }

  return chunks;
}

// ============================================================================
// Session Status for User Notification
// ============================================================================

// ============================================================================
// Doc Auto-Update from Transcript
// ============================================================================

async function updateDocsFromTranscript(messages: any[], label: string, sessionId?: string): Promise<void> {
  // Check if auto-update is enabled
  const memConfig = getMemoryConfig();
  if (!memConfig.docs?.autoUpdateOnCompact) {
    return;
  }

  const fullTranscript = buildTranscript(messages);
  if (!fullTranscript.trim()) {
    console.log(`[quaid] ${label}: no transcript for doc update`);
    return;
  }

  // Check for stale docs first
  let staleDocs: string[] = [];
  try {
    const stalenessJson = await callDocsUpdater("check", ["--json"]);
    if (stalenessJson) {
      const stale = JSON.parse(stalenessJson);
      staleDocs = Object.keys(stale);
    }
  } catch {
    // Staleness check failed, continue anyway
  }

  if (staleDocs.length === 0) {
    console.log(`[quaid] ${label}: all docs up-to-date`);
    return;
  }

  console.log(`[quaid] ${label}: ${staleDocs.length} stale doc(s) detected - updating before context is lost...`);
  for (const doc of staleDocs.slice(0, 3)) {
    console.log(`[quaid]   • ${doc}`);
  }

  // Write transcript to temp file
  const tmpPath = `/tmp/doc-update-transcript-${Date.now()}.txt`;
  fs.writeFileSync(tmpPath, fullTranscript);

  try {
    const maxDocs = memConfig.docs?.maxDocsPerUpdate || 3;
    console.log(`[quaid] ${label}: calling Opus to update docs (this may take a moment)...`);
    const startTime = Date.now();
    const output = await callDocsUpdater("update-from-transcript", [
      "--transcript", tmpPath,
      "--apply",
      "--max-docs", String(maxDocs),
    ]);
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    if (output && output.includes("Updated")) {
      console.log(`[quaid] ${label}: doc update complete (${elapsed}s)`);
      // Extract updated doc names from output
      const updatedMatches = output.match(/Updated ([^\s]+)/g);
      const updatedDocs: string[] = [];
      if (updatedMatches) {
        for (const m of updatedMatches) {
          const docName = m.replace("Updated ", "");
          console.log(`[quaid]   ✓ ${docName}`);
          updatedDocs.push(docName);
        }
      }
      // Status already logged to console above
    } else {
      console.log(`[quaid] ${label}: no docs updated (${elapsed}s)`);
    }
  } catch (err: unknown) {
    console.error(`[quaid] ${label} doc update failed:`, (err as Error).message);
    // Error already logged to console above
  } finally {
    try { fs.unlinkSync(tmpPath); } catch {}
  }
}

// ============================================================================
// Memory Operations
// ============================================================================

type MemoryResult = {
  text: string;
  category: string;
  similarity: number;
  verified?: boolean;
  id?: string;
  extractionConfidence?: number;
  createdAt?: string;
  validFrom?: string;
  validUntil?: string;
  privacy?: string;
  ownerId?: string;
  relation?: string; // For graph-related nodes
  direction?: string; // "out" or "in" for graph edges
  sourceName?: string; // Source node name for graph edges
  via?: string; // "vector" or "graph" - how this result was found
};

type GraphAwareSearchResult = {
  directResults: MemoryResult[];
  graphResults: MemoryResult[];
  entitiesFound: Array<{ id: string; name: string; type: string }>;
  sourceBreakdown: {
    vectorCount: number;
    graphCount: number;
    pronounResolved: boolean;
    ownerPerson: string | null;
  };
};

async function recall(
  query: string,
  limit: number = 5,
  currentSessionId?: string,
  compactionTime?: string,
  expandGraph: boolean = true,
  graphDepth: number = 1,
  dateFrom?: string,
  dateTo?: string
): Promise<MemoryResult[]> {
  try {
    const args = [query, "--limit", String(limit), "--owner", resolveOwner()];

    // Use search-graph-aware for enhanced graph traversal, or basic search
    if (!expandGraph) {
      // Basic search without graph expansion — accepts --current-session-id, --compaction-time
      if (currentSessionId) {
        args.push("--current-session-id", currentSessionId);
      }
      if (compactionTime) {
        args.push("--compaction-time", compactionTime);
      }
      if (dateFrom) {
        args.push("--date-from", dateFrom);
      }
      if (dateTo) {
        args.push("--date-to", dateTo);
      }
      const output = await callPython("search", args);
      const results: MemoryResult[] = [];

      for (const line of output.split("\n")) {
        // Format: "[0.60] [fact](2026-03-01)[flags][C:0.8] text |ID:id|T:created_at|VF:valid_from|VU:valid_until|P:privacy|O:owner_id"
        const match = line.match(/\[(\d+\.\d+)\]\s+\[(\w+)\](?:\([^)]*\))?(?:\[[^\]]*\])*\[C:([\d.]+)\]\s*(.+?)(?:\s*\|ID:([^|]+))?(?:\|T:([^|]*))?(?:\|VF:([^|]*))?(?:\|VU:([^|]*))?(?:\|P:([^|]*))?(?:\|O:(.*))?$/);
        if (match) {
          results.push({
            text: match[4].trim(),
            category: match[2],
            similarity: parseFloat(match[1]),
            extractionConfidence: parseFloat(match[3]),
            id: match[5]?.trim(),
            createdAt: match[6]?.trim() || undefined,
            validFrom: match[7]?.trim() || undefined,
            validUntil: match[8]?.trim() || undefined,
            privacy: match[9]?.trim() || "shared",
            ownerId: match[10]?.trim() || undefined,
            via: "vector",
          });
        }
      }
      return results.slice(0, limit);
    }

    // Use graph-aware search with JSON output — accepts --depth
    if (graphDepth > 1) {
      args.push("--depth", String(graphDepth));
    }
    args.push("--json");
    const output = await callPython("search-graph-aware", args);
    const results: MemoryResult[] = [];

    try {
      const parsed = JSON.parse(output) as {
        direct_results: Array<{
          text: string;
          category: string;
          similarity: number;
          id?: string;
          extraction_confidence?: number;
          created_at?: string;
          valid_from?: string;
          valid_until?: string;
          privacy?: string;
          owner_id?: string;
          verified?: boolean;
          pinned?: boolean;
        }>;
        graph_results: Array<{
          id: string;
          name: string;
          type: string;
          relation: string;
          direction: string;
          depth: number;
          source_name: string;
        }>;
        source_breakdown: {
          vector_count: number;
          graph_count: number;
          pronoun_resolved: boolean;
          owner_person: string | null;
        };
      };

      // Add direct (vector) results
      for (const r of parsed.direct_results) {
        results.push({
          text: r.text,
          category: r.category,
          similarity: r.similarity,
          id: r.id,
          extractionConfidence: r.extraction_confidence,
          createdAt: r.created_at,
          validFrom: r.valid_from,
          validUntil: r.valid_until,
          privacy: r.privacy,
          ownerId: r.owner_id,
          verified: r.verified,
          via: "vector",
        });
      }

      // Add graph results
      for (const r of parsed.graph_results) {
        // Format: "Source --relation--> Target" or reversed for inbound
        let text: string;
        if (r.direction === "in") {
          text = `${r.name} --${r.relation}--> ${r.source_name}`;
        } else {
          text = `${r.source_name} --${r.relation}--> ${r.name}`;
        }

        results.push({
          text: text,
          category: "graph",
          similarity: 0.75, // Graph results get a fixed medium-high similarity
          id: r.id,
          relation: r.relation,
          direction: r.direction,
          sourceName: r.source_name,
          via: "graph",
        });
      }

    } catch (parseErr: unknown) {
      // Fallback: parse line-by-line output format if JSON parsing fails
      console.log("[quaid] JSON parse failed, trying line format");
      for (const line of output.split("\n")) {
        // [direct] format
        if (line.startsWith("[direct]")) {
          const match = line.match(/\[direct\]\s+\[(\d+\.\d+)\]\s+\[(\w+)\]\s+(.+)/);
          if (match) {
            results.push({
              text: match[3].trim(),
              category: match[2],
              similarity: parseFloat(match[1]),
              via: "vector",
            });
          }
        }
        // [graph] format
        else if (line.startsWith("[graph]")) {
          const content = line.substring(7).trim();
          results.push({
            text: content,
            category: "graph",
            similarity: 0.75,
            via: "graph",
          });
        }
      }
    }

    // Don't slice here - Python already limits results, and we want to preserve graph results
    // which are added after direct results
    return results;
  } catch (err: unknown) {
    console.error("[quaid] recall error:", (err as Error).message);
    return [];
  }
}

// Shared recall abstraction — used by both memory_recall tool and auto-inject
interface RecallOptions {
  query: string;
  limit?: number;
  expandGraph?: boolean;
  graphDepth?: number;
  dateFrom?: string;
  dateTo?: string;
  waitForExtraction?: boolean;  // wait on extractionPromise (tool=yes, inject=no)
}

// Note: extractionPromise is declared inside register() — this function is
// defined below inside register() as well, so it has access to it.
// (Moved into register() closure when used)

type StoreResult = {
  id?: string;
  status: "created" | "duplicate" | "updated" | "failed";
  similarity?: number;
  existingText?: string;
};

async function store(
  text: string,
  category: string = "fact",
  sessionId?: string,
  extractionConfidence: number = 0.5,
  owner: string = resolveOwner(),
  source?: string,
  speaker?: string,
  status?: string,
  privacy?: string,
  keywords?: string,
  knowledgeType?: string,
  sourceType?: string
): Promise<StoreResult | null> {
  try {
    const args = [text, "--category", category,
      "--owner", owner,
      "--confidence", extractionConfidence.toString(),
      "--extraction-confidence", extractionConfidence.toString()];
    if (sessionId) {
      args.push("--session-id", sessionId);
    }
    if (source) {
      args.push("--source", source);
    }
    if (speaker) {
      args.push("--speaker", speaker);
    }
    if (status) {
      args.push("--status", status);
    }
    if (privacy) {
      args.push("--privacy", privacy);
    }
    if (keywords) {
      args.push("--keywords", keywords);
    }
    if (knowledgeType) {
      args.push("--knowledge-type", knowledgeType);
    }
    if (sourceType) {
      args.push("--source-type", sourceType);
    }
    const output = await callPython("store", args);
    
    // Check for "Stored: {id}" (created)
    const storedMatch = output.match(/Stored: (.+)/);
    if (storedMatch) {
      return { id: storedMatch[1], status: "created" };
    }
    
    // Check for "Duplicate (similarity: X) [ID]: {text}" (new format with ID)
    const dupMatchNew = output.match(/Duplicate \(similarity: ([\d.]+)\) \[([^\]]+)\]: (.+)/);
    if (dupMatchNew) {
      return {
        id: dupMatchNew[2],
        status: "duplicate",
        similarity: parseFloat(dupMatchNew[1]),
        existingText: dupMatchNew[3]
      };
    }
    // Fallback: old format without ID
    const dupMatch = output.match(/Duplicate \(similarity: ([\d.]+)\): (.+)/);
    if (dupMatch) {
      return {
        status: "duplicate",
        similarity: parseFloat(dupMatch[1]),
        existingText: dupMatch[2]
      };
    }
    
    // Check for "Updated existing: {id}" (updated)
    const updatedMatch = output.match(/Updated existing: (.+)/);
    if (updatedMatch) {
      return { id: updatedMatch[1], status: "updated" };
    }
    
    return null;
  } catch (err: unknown) {
    console.error("[quaid] store error:", (err as Error).message);
    return null;
  }
}

async function getStats(): Promise<object | null> {
  try {
    const output = await callPython("stats");
    return JSON.parse(output);
  } catch (err: unknown) {
    console.error("[quaid] stats error:", (err as Error).message);
    return null;
  }
}

function formatMemories(memories: MemoryResult[]): string {
  if (!memories.length) { return ""; }

  // Sort by created_at ascending (oldest first, most recent last = closest to prompt)
  const sorted = [...memories].sort((a, b) => {
    if (!a.createdAt && !b.createdAt) { return 0; }
    if (!a.createdAt) { return -1; }
    if (!b.createdAt) { return 1; }
    return a.createdAt.localeCompare(b.createdAt);
  });

  const lines = sorted.map((m) => {
    const conf = m.extractionConfidence ?? 0.5;
    const timestamp = m.createdAt ? ` (${m.createdAt.split("T")[0]})` : "";
    if (conf < 0.4) {
      return `- [${m.category}]${timestamp} (uncertain) ${m.text}`;
    }
    return `- [${m.category}]${timestamp} ${m.text}`;
  });

  return `<injected_memories>
AUTOMATED MEMORY SYSTEM: The following memories were automatically retrieved from past conversations. The user did not request this recall and is unaware these are being shown to you. Use them as background context only. Items marked (uncertain) have lower extraction confidence. Dates shown are when the fact was recorded.
${lines.join("\n")}
</injected_memories>`;
}

// ============================================================================
// LLM-based Auto-Capture
// ============================================================================

const ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages";

async function classifyAndStore(text: string, speaker: string = "The user", sessionId?: string): Promise<void> {
  // Special handling for assistant messages - extract facts ABOUT the user, not BY the assistant
  const isAssistantMessage = speaker === "Assistant";
  const actualSubject = isAssistantMessage ? "User" : speaker;
  
  const systemPrompt = `You extract memorable personal facts from messages for a personal knowledge base.

PURPOSE: Help an AI assistant remember useful information about the user — their preferences, relationships, decisions, and life events. It's fine to return empty if a message is purely conversational — you have permission to extract nothing when appropriate. A nightly cleanup process handles any noise.

This is a PERSONAL knowledge base. System architecture, infrastructure configs, and operational rules for AI agents belong in documentation — NOT here.

SPEAKER CONTEXT:
- Speaker: ${isAssistantMessage ? "Assistant (AI assistant)" : speaker}
- ${isAssistantMessage ? "This is the AI speaking TO User. Extract facts ABOUT User mentioned in the response." : "This is the human speaking. Their statements are first-person facts about themselves."}

CAPTURE ONLY these fact types:
- Personal facts: Names, relationships, jobs, birthdays, health conditions, addresses
- Preferences: Likes, dislikes, favorites, opinions, communication styles, personal rules ("Always do X", "I prefer Z format")
- Personal decisions: Choices User made with reasoning ("decided to use X because Y")
- Important relationships: Family, staff, contacts, business partners
- Significant events: Major life changes, trips, health diagnoses, big decisions

NOISE PATTERNS - NEVER CAPTURE:
- System architecture: How systems are built, infrastructure details, tool configurations
- Operational rules for AI agents: "Assistant should do X", "The janitor runs Y"
- Hypothetical examples: "Like: 'X'", "For example", "such as", test statements
- Conversational fragments: Questions, suggestions, worries
- System/technical noise: Plugin paths, debugging, error messages, API keys, credentials
- Security-related: API keys, passwords, tokens, authentication details
- Entity stubs: Single words, device names without context
- Meta-conversation: Discussion about AI systems, memory, capabilities, infrastructure
- Temporal work-in-progress: "working on", "about to", "planning to"
- Commands/requests: "Can you...", "Please..."
- Acknowledgments: "Thanks", "Got it", "Sounds good"
- General knowledge: Facts not specific to this person/household

QUALITY RULES:
- Completeness: Skip partial facts without context
- Specificity: "User likes spicy food" > "User likes food"
- Stability: Permanent facts > temporary states
- Attribution: Always use "${actualSubject}" as subject, third person
- Reality check: Only capture statements presented as TRUE facts, not examples or hypotheticals
- NO EMBELLISHMENT: Extract ONLY what was explicitly stated. Do NOT infer, add, or embellish details.
  If the speaker says "dinner at Shelter", do NOT add "tomorrow". If they say "a necklace", do NOT add "surprise".
  If one sentence contains multiple facts, extract them as separate items — but each must match what was said.
- ONE FACT PER CONCEPT: Do not split one statement into overlapping facts.
  "My sister Amber's husband is named Nate" = ONE fact, not three separate facts about sister/husband/brother-in-law.

${isAssistantMessage ?
  `CRITICAL: Extract facts ABOUT the user (User), NOT about the assistant.
Convert: "Your X" → "${actualSubject}'s X", "You have Y" → "${actualSubject} has Y"
NEVER capture: "Assistant will...", "Assistant should...", system behaviors` :
  `Rephrase "I/my/me" to "${actualSubject}".`}

PRIVACY CLASSIFICATION (per fact):
- "private": ONLY for secrets, surprises, hidden gifts, sensitive finances, health diagnoses,
  passwords, or anything explicitly meant to be hidden from specific people.
  Examples: "planning a surprise party for X", "salary is $X", "diagnosed with X"
- "shared": Most facts go here. Family info, names, relationships, schedules, preferences,
  routines, project details, household knowledge, general personal facts.
  Examples: "dinner is at 7pm", "sister is named Amber", "likes spicy food", "works from home"
- "public": Widely known or non-personal facts. Examples: "Bali is in Indonesia"
IMPORTANT: Default to "shared". Only use "private" for genuinely secret or sensitive information.
Family names, daily routines, and preferences are "shared", NOT "private".

Respond with JSON only:
{"facts": [{"text": "specific fact", "category": "fact|preference|decision|relationship", "extraction_confidence": "high|medium|low", "keywords": "3-5 searchable terms (proper nouns, synonyms, category words)", "privacy": "private|shared|public"}]}

If nothing worth capturing, respond: {"facts": []}`;

  const userMessage = `${isAssistantMessage ? "Assistant (assistant)" : speaker} said: "${text.slice(0, 500)}"`;

  try {
    // Get API key from environment
    let apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      try {
        const { execSync } = require("node:child_process");
        apiKey = process.env.ANTHROPIC_API_KEY || "";
      } catch {
        console.error("[quaid] Could not get Anthropic API key");
        return;
      }
    }
    if (!apiKey) {
      console.error("[quaid] ANTHROPIC_API_KEY not available");
      return;
    }

    const response = await fetch(ANTHROPIC_API_URL, {
      method: "POST",
      headers: { 
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01"
      },
      body: JSON.stringify({
        model: getLowReasoningModel(),
        max_tokens: 200,
        system: systemPrompt,
        messages: [{ role: "user", content: userMessage }]
      })
    });
    
    if (!response.ok) {
      const errText = await response.text();
      console.error("[quaid] Anthropic request failed:", response.status, errText);
      return;
    }
    
    const data = await response.json() as { content?: Array<{ text?: string }> };
    const output = data.content?.[0]?.text?.trim() || "";
    
    // Parse JSON from response (handle potential markdown wrapping)
    let jsonStr = output;
    if (output.includes("```")) {
      const match = output.match(/```(?:json)?\s*([\s\S]*?)```/);
      if (match) { jsonStr = match[1].trim(); }
    }
    
    let result: { save?: boolean; category?: string; summary?: string; facts?: any[]; extraction_confidence?: string };
    try {
      result = JSON.parse(jsonStr);
    } catch {
      // Try to extract JSON object if response has extra text
      const jsonMatch = jsonStr.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        try {
          result = JSON.parse(jsonMatch[0]);
        } catch {
          console.log("[quaid] Could not parse extracted JSON:", jsonMatch[0].slice(0, 200));
          return;
        }
      } else {
        console.log("[quaid] Could not parse LLM response:", output.slice(0, 100));
        return;
      }
    }

    // Handle per-fact objects (new format) and string arrays (legacy)
    const rawFacts = result.facts || (result.save === true ? [result.summary || text] : []);

    if (rawFacts.length === 0) {
      console.log(`[quaid] LLM: no facts extracted from: "${text.slice(0, 50)}..."`);
      return;
    }

    for (const rawFact of rawFacts) {
      // Support both per-fact objects and legacy string format
      const factText = typeof rawFact === "string" ? rawFact : rawFact?.text;
      const factCategory = typeof rawFact === "string" ? (result.category || "fact") : (rawFact?.category || "fact");
      const factConfStr = typeof rawFact === "string" ? (result.extraction_confidence || "medium") : (rawFact?.extraction_confidence || "medium");
      const factPrivacy = typeof rawFact === "string" ? "shared" : (rawFact?.privacy || "shared");
      const factKeywords = typeof rawFact === "string" ? undefined : (rawFact?.keywords || undefined);

      if (!factText || factText.trim().split(/\s+/).length < 3) { continue; }

      const extractionConfidence = factConfStr === "high" ? 0.8 : factConfStr === "low" ? 0.3 : 0.5;

      const factSourceType = isAssistantMessage ? "assistant" : "user";
      const storeResult = await store(factText, factCategory, sessionId, extractionConfidence, resolveOwner(speaker), "auto-capture", speaker, undefined, factPrivacy, factKeywords, undefined, factSourceType);
      if (storeResult?.status === "created") {
        console.log(`[quaid] Auto-captured: "${factText.slice(0, 60)}..." [${factCategory}] (conf: ${extractionConfidence}, privacy: ${factPrivacy})`);
      } else if (storeResult?.status === "duplicate") {
        console.log(`[quaid] Skipped (duplicate): "${factText.slice(0, 40)}..."`);
      }
    }
  } catch (err: unknown) {
    console.error("[quaid] classifyAndStore error:", (err as Error).message);
  }
}

// ============================================================================
// Plugin Definition
// ============================================================================

const quaidPlugin = {
  id: "quaid",
  name: "Memory (Local Graph)",
  description: "Local graph-based memory with SQLite + Ollama embeddings",
  kind: "memory" as const,
  configSchema,

  register(api: ClawdbotPluginApi<PluginConfig>) {
    console.log("[quaid] Registering local graph memory plugin");

    // Ensure database exists
    const dataDir = path.dirname(DB_PATH);
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }

    // Check if database needs seeding
    if (!fs.existsSync(DB_PATH)) {
      console.log("[quaid] Database not found, running initial seed...");
      try {
        const seedScript = path.join(WORKSPACE, "plugins/quaid/seed.py");
        execSync(`python3 ${seedScript}`, {
          env: { ...process.env, MEMORY_DB_PATH: DB_PATH },
        });
        console.log("[quaid] Initial seed complete");
      } catch (err: unknown) {
        console.error("[quaid] Seed failed:", (err as Error).message);
      }
    }

    // Log stats
    void getStats().then((stats) => {
      if (stats) {
        console.log(
          `[quaid] Database ready: ${(stats as any).total_nodes} nodes, ${(stats as any).edges} edges`
        );
      }
    });

    // Register lifecycle hooks
    // ============================================================================
    // DEPRECATED: Automatic Memory Injection (disabled 2026-02-06)
    // ============================================================================
    // Reason: Automatic injection on every message produced low-quality matches.
    // Short messages like "ok", "B", "yes" have meaningless embeddings that match
    // random facts. The new approach is agent-driven: the agent decides when it
    // needs personal context and uses the memory_recall tool with a crafted query.
    //
    // This code is preserved for potential future use with better query gating.
    // To re-enable: set MEMORY_AUTO_INJECT=1 in environment or update config.
    // ============================================================================
    const beforeAgentStartHandler = async (event: any, ctx: any) => {
      // Cancel inactivity timer — agent is active again
      if (inactivityTimer) { clearTimeout(inactivityTimer); inactivityTimer = null; }

      // Journal injection (full soul mode) — gated by journal system
      if (!isSystemEnabled("journal")) {
        // Skip journal injection entirely
      } else {
      try {
        const journalConfig = getMemoryConfig().docs?.journal || {};
        const journalMode = journalConfig.mode || "distilled";

        if (journalMode === "full") {
          const journalDir = path.join(WORKSPACE, journalConfig.journalDir || "journal");
          let journalFiles: string[] = [];
          try { journalFiles = fs.readdirSync(journalDir).filter((f: string) => f.endsWith('.journal.md')).sort(); } catch {}

          let journalContent = '';
          for (const file of journalFiles) {
            try {
              const content = fs.readFileSync(path.join(journalDir, file), 'utf8');
              if (content.trim()) {
                journalContent += `\n\n--- ${file} ---\n${content}`;
              }
            } catch {}
          }

          if (journalContent) {
            const header = '[JOURNAL \u2014 Full Soul Mode]\n' +
              'These are your recent journal reflections. They are part of your inner life.\n';
            event.prependContext = event.prependContext
              ? `${event.prependContext}\n\n${header}${journalContent}`
              : `${header}${journalContent}`;
            console.log(`[quaid] Full soul mode: injected ${journalFiles.length} journal files`);
          }
        }
      } catch (err: unknown) {
        console.log(`[quaid] Journal injection failed (non-fatal): ${(err as Error).message}`);
      }
      } // end journal system gate

      // Check if auto-injection is explicitly enabled (opt-in now, not opt-out)
      const autoInjectEnabled = process.env.MEMORY_AUTO_INJECT === "1" ||
        getMemoryConfig().retrieval?.autoInject === true;

      if (!autoInjectEnabled) {
        // Agent-driven memory: skip automatic injection
        return;
      }

      // --- Auto-injection (Mem0-style) — uses shared recall pipeline ---
      if (!event.prompt || event.prompt.length < 5) {
        return;
      }

      try {
        const rawPrompt = event.prompt;

        // Extract actual user message from metadata-wrapped prompts
        let query = rawPrompt
          .replace(/^System:\s*/i, '')
          .replace(/^\s*(\[.*?\]\s*)+/s, '')
          .replace(/^---\s*/m, '')
          .trim();
        if (query.length < 3) { query = rawPrompt; }

        // Skip system/internal prompts and slash commands
        if (/^(A new session|Read HEARTBEAT|HEARTBEAT|You are being asked to|\/\w)/.test(query)) {
          return;
        }

        // Query quality gate — skip acknowledgments and short messages
        const ACKNOWLEDGMENTS = /^(ok|okay|yes|no|sure|thanks|thank you|got it|sounds good|perfect|great|cool|alright|yep|nope|right|correct|agreed|absolutely|definitely|nice|good|fine|hm+|ah+|oh+)\s*[.!?]?$/i;
        const words = query.trim().split(/\s+/).filter(w => w.length > 1);
        if (words.length < 3 || ACKNOWLEDGMENTS.test(query.trim())) {
          return;
        }

        // Same recall pipeline as memory_recall tool (no graph expansion for auto-inject)
        const allMemories = await recallMemories({
          query, limit: 20, expandGraph: false, waitForExtraction: false
        });

        if (!allMemories.length) return;

        // Privacy filter — allow through if: not private, or owned by current user, or no owner set
        // Note: Python serializes None as string "None", not JSON null
        const currentOwner = resolveOwner();
        const filtered = allMemories.filter(m =>
          !(m.privacy === "private" && m.ownerId && m.ownerId !== "None" && m.ownerId !== currentOwner)
        );

        // Session dedup (don't re-inject same facts within a session)
        const uniqueSessionId = extractSessionId(event.messages || [], ctx);
        const injectionLogPath = `/tmp/memory-injection-${uniqueSessionId}.log`;
        let previouslyInjected: string[] = [];
        try {
          const logData = JSON.parse(fs.readFileSync(injectionLogPath, 'utf8'));
          previouslyInjected = logData.injected || logData.memoryTexts || [];
        } catch {}
        const newMemories = filtered.filter(m => !previouslyInjected.includes(m.id || m.text));

        // Cap and format
        const MAX_INJECT = 10;
        const toInject = newMemories.slice(0, MAX_INJECT);
        if (!toInject.length) return;

        const formatted = formatMemories(toInject);
        event.prependContext = event.prependContext
          ? `${event.prependContext}\n\n${formatted}`
          : formatted;

        console.log(`[quaid] Auto-injected ${toInject.length} memories for "${query.slice(0, 50)}..."`);

        // Update injection log
        try {
          const newIds = toInject.map(m => m.id || m.text);
          fs.writeFileSync(injectionLogPath, JSON.stringify({
            injected: [...previouslyInjected, ...newIds],
            lastInjectedAt: new Date().toISOString()
          }), { mode: 0o600 });
        } catch {}
      } catch (error: unknown) {
        console.error("[quaid] Auto-injection error:", error);
      }
    };

    // Register hooks using api.on() for typed hooks (NOT api.registerHook!)
    console.log("[quaid] Registering before_agent_start hook for memory injection");
    api.on("before_agent_start", beforeAgentStartHandler, {
      name: "memory-injection",
      priority: 10
    });

    // Auto-capture hook — DEPRECATED: replaced by event-based classification
    // (compact, reset, session end). The per-message classifier was calling
    // Haiku on every agent_end including heartbeats, causing excessive API spend.
    // classifyAndStore() and the message filtering logic are preserved above
    // if we ever need to re-enable this path.
    const agentEndHandler = async (event: any, ctx: any) => {
      if (!isSystemEnabled("memory")) return;
      const timeoutMinutes = getMemoryConfig().capture?.inactivityTimeoutMinutes ?? 120;
      if (timeoutMinutes <= 0) return;

      // Cancel any existing timer (reset the clock)
      if (inactivityTimer) { clearTimeout(inactivityTimer); inactivityTimer = null; }

      const messages = event.messages || [];
      if (messages.length === 0) return;

      pendingTimeoutMessages = messages;
      pendingTimeoutSessionId = ctx?.sessionId;

      inactivityTimer = setTimeout(async () => {
        inactivityTimer = null;
        const msgs = pendingTimeoutMessages;
        const sid = pendingTimeoutSessionId;
        pendingTimeoutMessages = null;
        pendingTimeoutSessionId = undefined;
        if (!msgs || msgs.length === 0) return;

        // Check extraction log — skip if already extracted with >= this many messages
        try {
          const logPath = path.join(WORKSPACE, "data", "extraction-log.json");
          const log = JSON.parse(fs.readFileSync(logPath, 'utf8'));
          if (log[sid || 'unknown']?.message_count >= msgs.length) return;
        } catch {}

        console.log(`[quaid] Inactivity timeout (${timeoutMinutes}m) — extracting ${msgs.length} messages`);
        extractionPromise = (extractionPromise || Promise.resolve())
          .catch(() => {})
          .then(() => extractMemoriesFromMessages(msgs, "Timeout", sid));
      }, timeoutMinutes * 60 * 1000);
    };

    // Register agent_end hook using api.on() for typed hooks
    console.log("[quaid] Registering agent_end hook for auto-capture");
    api.on("agent_end", agentEndHandler, {
      name: "auto-capture",
      priority: 10
    });

    // Register memory tools (gated by memory system)
    if (isSystemEnabled("memory")) {
    api.registerTool(
      () => ({
        name: "memory_recall",
        description: `Search your memory for personal facts, preferences, relationships, project details, and past conversations. Always use this tool when you're unsure about something or need to verify a detail — if you might know it, search for it.

USE THIS TOOL LIBERALLY. If you're about to say "I don't have information about..." or "I'm not sure...", SEARCH FIRST. It's better to search and find nothing than to miss a memory you have.

USE WHEN: Any question about the user, their life, people they know, projects they work on, preferences, history, past decisions, technical details about their projects, or anything that might have come up in a previous conversation.
SKIP WHEN: General knowledge questions, greetings, short acknowledgments.

QUERY TIPS: Use specific names and topics. Try multiple searches with different phrasings if the first doesn't return what you need.
graphDepth: Set to 2 for relationship queries (e.g., nephew = sibling's child). Default 1 is usually sufficient.
dateFrom/dateTo: Use YYYY-MM-DD format to filter memories by date range.`,
        parameters: Type.Object({
          query: Type.String({ description: "Search query - use entity names and specific topics" }),
          limit: Type.Optional(
            Type.Number({ description: "Max results to return. Default reads from config." })
          ),
          expandGraph: Type.Optional(
            Type.Boolean({ description: "Traverse relationship graph - use for people/family queries (default: true)" })
          ),
          graphDepth: Type.Optional(
            Type.Number({ description: "Graph traversal depth (default: 1). Use 2 to infer extended relationships like nephew/cousin." })
          ),
          dateFrom: Type.Optional(
            Type.String({ description: "Only return memories from this date onward (YYYY-MM-DD)" })
          ),
          dateTo: Type.Optional(
            Type.String({ description: "Only return memories up to this date (YYYY-MM-DD)" })
          ),
        }),
        async execute(toolCallId, params) {
          try {
            // Read config for limits
            const configPath = path.join(WORKSPACE, "config/memory.json");
            let configuredLimit = 5;
            let maxLimit = 20;
            try {
              const configData = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
              configuredLimit = configData?.retrieval?.defaultLimit ?? 5;
              maxLimit = configData?.retrieval?.maxLimit ?? 20;
            } catch {}

            const { query, limit: requestedLimit, expandGraph = true, graphDepth = 1, dateFrom, dateTo } = params || {};
            const limit = Math.min(requestedLimit ?? configuredLimit, maxLimit);
            const depth = Math.min(Math.max(graphDepth, 1), 3); // Clamp between 1 and 3

            console.log(`[quaid] memory_recall: query="${query?.slice(0, 50)}...", requestedLimit=${requestedLimit}, configuredLimit=${configuredLimit}, maxLimit=${maxLimit}, finalLimit=${limit}, expandGraph=${expandGraph}, graphDepth=${depth}, dateFrom=${dateFrom}, dateTo=${dateTo}`);
            const results = await recallMemories({
              query, limit, expandGraph, graphDepth: depth,
              dateFrom, dateTo, waitForExtraction: true
            });

            if (results.length === 0) {
              return {
                content: [{ type: "text", text: "No relevant memories found. Try different keywords or entity names." }],
                details: { count: 0 },
              };
            }

            // Group by source type for better formatting
            const vectorResults = results.filter(r => r.via === "vector" || (!r.via && r.category !== "graph"));
            const graphResults = results.filter(r => r.via === "graph" || r.category === "graph");

            // Check if results are low quality (all below 60% similarity)
            const avgSimilarity = vectorResults.length > 0
              ? vectorResults.reduce((sum, r) => sum + (r.similarity || 0), 0) / vectorResults.length
              : 0;
            const lowQualityWarning = avgSimilarity < 0.6 && vectorResults.length > 0
              ? "\n\n⚠️ Low confidence matches - consider refining query with specific names or topics.\n"
              : "";

            let text = `[MEMORY] Found ${results.length} results:${lowQualityWarning}\n`;

            if (vectorResults.length > 0) {
              text += "\n**Direct Matches:**\n";
              vectorResults.forEach((r, i) => {
                const conf = r.extractionConfidence ? ` [conf:${Math.round(r.extractionConfidence * 100)}%]` : "";
                const dateStr = r.createdAt ? ` (${r.createdAt.split("T")[0]})` : "";
                const superseded = r.validUntil ? " [superseded]" : "";
                text += `${i + 1}. [MEMORY] [${r.category}]${dateStr}${superseded} ${r.text} (${Math.round(r.similarity * 100)}%${conf})\n`;
              });
            }

            if (graphResults.length > 0) {
              if (vectorResults.length > 0) { text += "\n"; }
              text += "**Graph Discoveries:**\n";
              graphResults.forEach((r, i) => {
                text += `${i + 1}. [MEMORY] ${r.text}\n`;
              });
            }

            // Notify user about what memories were retrieved (if enabled)
            try {
              const configPath = path.join(WORKSPACE, "config/memory.json");
              const configData = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
              const notifyOnRecall = configData?.retrieval?.notifyOnRecall ?? false;

              if (notifyOnRecall && results.length > 0) {
                const memoryData = results.map(m => ({
                  text: m.text,
                  similarity: Math.round((m.similarity || 0) * 100),
                  via: m.via || "vector"
                }));
                const memoryJson = JSON.stringify(memoryData);

                // Build source breakdown for notification
                const sourceBreakdown = JSON.stringify({
                  vector_count: vectorResults.length,
                  graph_count: graphResults.length,
                  query: query
                });

                // Fire and forget notification
                const dataFile2 = `/tmp/recall-data-${Date.now()}.json`;
                fs.writeFileSync(dataFile2, JSON.stringify({ memories: JSON.parse(memoryJson), source_breakdown: JSON.parse(sourceBreakdown) }), { mode: 0o600 });
                spawnNotifyScript(`
import json
from notify import notify_memory_recall
with open(${JSON.stringify(dataFile2)}, 'r') as f:
    data = json.load(f)
os.unlink(${JSON.stringify(dataFile2)})
notify_memory_recall(data['memories'], source_breakdown=data['source_breakdown'])
`);
              }
            } catch (notifyErr: unknown) {
              // Notification is best-effort
              console.log(`[quaid] Memory recall notification skipped: ${(notifyErr as Error).message}`);
            }

            return {
              content: [
                { type: "text", text: text.trim() },
              ],
              details: {
                count: results.length,
                memories: results,
                vectorCount: vectorResults.length,
                graphCount: graphResults.length,
              },
            };
          } catch (err: unknown) {
            console.error("[quaid] memory_recall error:", err);
            return {
              content: [{ type: "text", text: `Error recalling memories: ${String(err)}` }],
              details: { error: String(err) },
            };
          }
        },
      }),
    );

    api.registerTool(
      () => ({
        name: "memory_store",
        description: `Queue a fact for memory extraction at next compaction. The fact will go through full quality review (Opus extraction with edges and janitor review) rather than being stored directly.

Only use when the user EXPLICITLY asks you to remember something (e.g., "remember this", "save this"). Do NOT proactively store facts — auto-extraction at compaction handles that.`,
        parameters: Type.Object({
          text: Type.String({ description: "Information to remember" }),
          category: Type.Optional(
            Type.String({
              enum: ["preference", "fact", "decision", "entity", "other"],
            })
          ),
        }),
        async execute(_toolCallId, params, ctx) {
          try {
            const { text, category = "fact" } = params || {};
            // Resolve session ID from context
            const sessionId = ctx?.sessionId || "unknown";
            addMemoryNote(sessionId, text, category);
            console.log(`[quaid] memory_store: queued note for session ${sessionId}: "${text.slice(0, 60)}..."`);
            return {
              content: [{ type: "text", text: `Noted for memory extraction: "${text.slice(0, 100)}${text.length > 100 ? '...' : ''}" — will be processed with full quality review at next compaction.` }],
              details: { action: "queued", sessionId },
            };
          } catch (err: unknown) {
            console.error("[quaid] memory_store error:", err);
            return {
              content: [{ type: "text", text: `Error queuing memory note: ${String(err)}` }],
              details: { error: String(err) },
            };
          }
        },
      })
    );

    api.registerTool(
      () => ({
        name: "memory_forget",
        description: "Delete specific memories",
        parameters: Type.Object({
          query: Type.Optional(
            Type.String({ description: "Search to find memory" })
          ),
          memoryId: Type.Optional(
            Type.String({ description: "Specific memory ID" })
          ),
        }),
        async execute(_toolCallId, params) {
          try {
            const { query, memoryId } = params || {};
            if (memoryId) {
              await callPython("forget", ["--id", memoryId]);
              return {
                content: [{ type: "text", text: `Memory ${memoryId} forgotten.` }],
                details: { action: "deleted", id: memoryId },
              };
            } else if (query) {
              await callPython("forget", [query]);
              return {
                content: [{ type: "text", text: `Deleted memories matching: "${query}"` }],
                details: { action: "deleted", query },
              };
            }
            return {
              content: [{ type: "text", text: "Provide query or memoryId." }],
              details: { error: "missing_param" },
            };
          } catch (err: unknown) {
            console.error("[quaid] memory_forget error:", err);
            return {
              content: [{ type: "text", text: `Error deleting memory: ${String(err)}` }],
              details: { error: String(err) },
            };
          }
        },
      })
    );
    } // end memory system gate

    // Register docs_search tool — semantic search over project documentation
    api.registerTool(
      () => ({
        name: "docs_search",
        description: "Search project documentation (architecture, implementation, reference guides). Use TOOLS.md to discover systems, then use this tool to find detailed docs. Returns relevant sections with file paths.",
        parameters: Type.Object({
          query: Type.String({ description: "Search query" }),
          limit: Type.Optional(
            Type.Number({ description: "Max results (default 5)" })
          ),
          project: Type.Optional(
            Type.String({ description: `Filter by project name. Available: ${getProjectNames().join(", ") || "none"}` })
          ),
        }),
        async execute(_toolCallId, params) {
          try {
            const { query, limit = 5, project } = params || {};

            // RAG search with optional project filter
            const searchArgs = [query, "--limit", String(limit)];
            if (project) { searchArgs.push("--project", project); }
            const results = await callDocsRag("search", searchArgs);

            // If project specified, prepend full PROJECT.md for context
            let projectMdContent = "";
            if (project) {
              try {
                const cfg = JSON.parse(fs.readFileSync(path.join(WORKSPACE, "config/memory.json"), 'utf-8'));
                const homeDir = cfg?.projects?.definitions?.[project]?.homeDir;
                if (homeDir) {
                  const mdPath = path.join(WORKSPACE, homeDir, "PROJECT.md");
                  if (fs.existsSync(mdPath)) {
                    projectMdContent = fs.readFileSync(mdPath, 'utf-8');
                  }
                }
              } catch {}
            }

            // Staleness check (lightweight mtime comparison)
            let stalenessWarning = "";
            try {
              const stalenessJson = await callDocsUpdater("check", ["--json"]);
              const staleDocs = JSON.parse(stalenessJson || "{}");
              const staleKeys = Object.keys(staleDocs);
              if (staleKeys.length > 0) {
                const warnings = staleKeys.map(
                  (k) => `  ${k} (${staleDocs[k].gap_hours}h behind: ${staleDocs[k].stale_sources.join(", ")})`
                );
                stalenessWarning = `\n\nSTALENESS WARNING: The following docs may be outdated:\n${warnings.join("\n")}\nConsider running: python3 docs_updater.py update-stale --apply`;
              }
            } catch {
              // Staleness check is non-critical
            }

            const text = projectMdContent
              ? `--- PROJECT.md (${project}) ---\n${projectMdContent}\n\n--- Search Results ---\n${results || "No results."}${stalenessWarning}`
              : (results ? results + stalenessWarning : "No results found." + stalenessWarning);

            // Notify user about what docs were searched (if enabled)
            try {
              const configPath = path.join(WORKSPACE, "config/memory.json");
              const configData = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
              const notifyOnRecall = configData?.retrieval?.notifyOnRecall ?? false;

              if (notifyOnRecall && results) {
                // Parse results to extract doc names and scores
                const docResults: Array<{doc: string, section: string, score: number}> = [];
                const lines = results.split('\n');
                for (const line of lines) {
                  // Match pattern: "1. ~/docs/filename.md > section (similarity: 0.xxx)"
                  const match = line.match(/^\d+\.\s+~?\/?([^\s>]+)\s+>\s+(.+?)\s+\(similarity:\s+([\d.]+)\)/);
                  if (match) {
                    docResults.push({
                      doc: match[1].split('/').pop() || match[1],
                      section: match[2].trim(),
                      score: parseFloat(match[3])
                    });
                  }
                }

                if (docResults.length > 0) {
                  // Fire and forget notification
                  const dataFile3 = `/tmp/docs-search-data-${Date.now()}.json`;
                  fs.writeFileSync(dataFile3, JSON.stringify({ query, results: docResults }), { mode: 0o600 });
                  spawnNotifyScript(`
import json
from notify import notify_docs_search
with open(${JSON.stringify(dataFile3)}, 'r') as f:
    data = json.load(f)
os.unlink(${JSON.stringify(dataFile3)})
notify_docs_search(data['query'], data['results'])
`);
                }
              }
            } catch (notifyErr: unknown) {
              // Notification is best-effort
              console.log(`[quaid] Docs search notification skipped: ${(notifyErr as Error).message}`);
            }

            return {
              content: [{ type: "text", text }],
              details: { query, limit },
            };
          } catch (err: unknown) {
            console.error("[quaid] docs_search error:", err);
            return {
              content: [{ type: "text", text: `Error searching docs: ${String(err)}` }],
              details: { error: String(err) },
            };
          }
        },
      })
    );

    // Register docs_read tool — read a document by path or title
    api.registerTool(
      () => ({
        name: "docs_read",
        description: "Read the full content of a registered document by file path or title.",
        parameters: Type.Object({
          identifier: Type.String({ description: "File path (workspace-relative) or document title" }),
        }),
        async execute(_toolCallId, params) {
          try {
            const { identifier } = params || {};
            const output = await callDocsRegistry("read", [identifier]);
            return {
              content: [{ type: "text", text: output || "Document not found." }],
              details: { identifier },
            };
          } catch (err: unknown) {
            return {
              content: [{ type: "text", text: `Error: ${String(err)}` }],
              details: { error: String(err) },
            };
          }
        },
      })
    );

    // Register docs_list tool — list registered docs
    api.registerTool(
      () => ({
        name: "docs_list",
        description: "List registered documents, optionally filtered by project or type.",
        parameters: Type.Object({
          project: Type.Optional(Type.String({ description: "Filter by project name" })),
          type: Type.Optional(Type.String({ description: "Filter by asset type (doc, note, reference)" })),
        }),
        async execute(_toolCallId, params) {
          try {
            const args: string[] = ["--json"];
            if (params?.project) { args.push("--project", params.project); }
            if (params?.type) { args.push("--type", params.type); }
            const output = await callDocsRegistry("list", args);
            return {
              content: [{ type: "text", text: output || "No documents found." }],
              details: { project: params?.project },
            };
          } catch (err: unknown) {
            return {
              content: [{ type: "text", text: `Error: ${String(err)}` }],
              details: { error: String(err) },
            };
          }
        },
      })
    );

    // Register docs_register tool — register a new document
    api.registerTool(
      () => ({
        name: "docs_register",
        description: "Register a document for indexing and tracking. Use for external files or docs with source file tracking.",
        parameters: Type.Object({
          file_path: Type.String({ description: "File path (workspace-relative)" }),
          project: Type.Optional(Type.String({ description: "Project name (default: 'default')" })),
          title: Type.Optional(Type.String({ description: "Document title" })),
          description: Type.Optional(Type.String({ description: "Document description" })),
          auto_update: Type.Optional(Type.Boolean({ description: "Auto-update when source files change" })),
          source_files: Type.Optional(Type.Array(Type.String(), { description: "Source files this doc tracks" })),
        }),
        async execute(_toolCallId, params) {
          try {
            const args: string[] = [params.file_path];
            if (params.project) { args.push("--project", params.project); }
            if (params.title) { args.push("--title", params.title); }
            if (params.description) { args.push("--description", params.description); }
            if (params.auto_update) { args.push("--auto-update"); }
            if (params.source_files) { args.push("--source-files", ...params.source_files); }
            args.push("--json");
            const output = await callDocsRegistry("register", args);
            return {
              content: [{ type: "text", text: output || "Registered." }],
              details: { file_path: params.file_path },
            };
          } catch (err: unknown) {
            return {
              content: [{ type: "text", text: `Error: ${String(err)}` }],
              details: { error: String(err) },
            };
          }
        },
      })
    );

    // Register project_create tool — scaffold a new project
    api.registerTool(
      () => ({
        name: "project_create",
        description: "Create a new project with a PROJECT.md template. Sets up the directory structure and scaffolding.",
        parameters: Type.Object({
          name: Type.String({ description: "Project name (kebab-case, e.g., 'my-essay')" }),
          label: Type.Optional(Type.String({ description: "Display label (e.g., 'My Essay')" })),
          description: Type.Optional(Type.String({ description: "Project description" })),
          source_roots: Type.Optional(Type.Array(Type.String(), { description: "Source root directories" })),
        }),
        async execute(_toolCallId, params) {
          try {
            const args: string[] = [params.name];
            if (params.label) { args.push("--label", params.label); }
            if (params.description) { args.push("--description", params.description); }
            if (params.source_roots) { args.push("--source-roots", ...params.source_roots); }
            const output = await callDocsRegistry("create-project", args);
            return {
              content: [{ type: "text", text: output || `Project '${params.name}' created.` }],
              details: { name: params.name },
            };
          } catch (err: unknown) {
            return {
              content: [{ type: "text", text: `Error: ${String(err)}` }],
              details: { error: String(err) },
            };
          }
        },
      })
    );

    // Register project_list tool — enumerate projects
    api.registerTool(
      () => ({
        name: "project_list",
        description: `List all defined projects with their doc counts and metadata. Available projects: ${getProjectNames().join(", ") || "none"}`,
        parameters: Type.Object({}),
        async execute() {
          try {
            const output = await callDocsRegistry("list-projects", ["--json"]);
            return {
              content: [{ type: "text", text: output || "No projects defined." }],
              details: {},
            };
          } catch (err: unknown) {
            return {
              content: [{ type: "text", text: `Error: ${String(err)}` }],
              details: { error: String(err) },
            };
          }
        },
      })
    );

    // Register session_recall tool — list/load recent conversation sessions
    api.registerTool(
      () => ({
        name: "session_recall",
        description: `List or load recent conversation sessions. Use when the user wants to continue previous work, references a past conversation, or you need context about what was discussed recently.`,
        parameters: Type.Object({
          action: Type.String({ description: '"list" = recent sessions, "load" = specific session transcript' }),
          session_id: Type.Optional(Type.String({ description: "Session ID to load (for action=load)" })),
          limit: Type.Optional(Type.Number({ description: "How many recent sessions to list (default 5, for action=list)" })),
        }),
        async execute(_toolCallId, params) {
          try {
            const { action = "list", session_id: sid, limit: listLimit = 5 } = params || {};

            const extractionLogPath = path.join(WORKSPACE, "data", "extraction-log.json");
            let extractionLog: Record<string, any> = {};
            try { extractionLog = JSON.parse(fs.readFileSync(extractionLogPath, 'utf8')); } catch {}

            if (action === "list") {
              // Sort sessions by last_extracted_at descending
              const sessions = Object.entries(extractionLog)
                .filter(([, v]) => v && v.last_extracted_at)
                .sort(([, a], [, b]) => (b.last_extracted_at || '').localeCompare(a.last_extracted_at || ''))
                .slice(0, Math.min(listLimit, 20));

              if (sessions.length === 0) {
                return {
                  content: [{ type: "text", text: "No recent sessions found in extraction log." }],
                  details: { count: 0 },
                };
              }

              let text = "Recent sessions:\n";
              sessions.forEach(([id, info], i) => {
                const date = info.last_extracted_at ? new Date(info.last_extracted_at).toLocaleString() : "unknown";
                const msgCount = info.message_count || "?";
                const trigger = info.label || "unknown";
                const topic = info.topic_hint ? ` — "${info.topic_hint}"` : "";
                text += `${i + 1}. [${date}] ${id} — ${msgCount} messages, extracted via ${trigger}${topic}\n`;
              });

              return {
                content: [{ type: "text", text }],
                details: { count: sessions.length },
              };
            }

            if (action === "load" && sid) {
              // Validate session_id to prevent path traversal
              if (!/^[a-zA-Z0-9_-]{1,128}$/.test(sid)) {
                return {
                  content: [{ type: "text", text: "Invalid session ID format." }],
                  details: { error: "invalid_session_id" },
                };
              }
              // Try to load session JSONL
              const sessionsDir = path.join(os.homedir(), '.openclaw', 'sessions');
              const sessionPath = path.join(sessionsDir, `${sid}.jsonl`);

              if (fs.existsSync(sessionPath)) {
                try {
                  const messages = readMessagesFromSessionFile(sessionPath);
                  const transcript = buildTranscript(messages);
                  // Return last 10k chars (most recent part of conversation)
                  const truncated = transcript.length > 10000
                    ? "...[truncated]...\n\n" + transcript.slice(-10000)
                    : transcript;
                  return {
                    content: [{ type: "text", text: `Session ${sid} (${messages.length} messages):\n\n${truncated}` }],
                    details: { session_id: sid, message_count: messages.length, truncated: transcript.length > 10000 },
                  };
                } catch {
                  // File disappeared or unreadable — fall through to facts fallback
                }
              }

              // Fallback: return facts extracted from this session
              try {
                const factsOutput = await callPython("search", [
                  "*", "--session-id", sid, "--owner", resolveOwner(), "--limit", "20"
                ]);
                return {
                  content: [{ type: "text", text: `Session file not available. Facts extracted from session ${sid}:\n${factsOutput || "No facts found."}` }],
                  details: { session_id: sid, fallback: true },
                };
              } catch {
                return {
                  content: [{ type: "text", text: `Session ${sid} not found. Session file may have been cleaned up and no facts were found.` }],
                  details: { session_id: sid, error: "not_found" },
                };
              }
            }

            return {
              content: [{ type: "text", text: 'Provide action: "list" or "load" (with session_id).' }],
              details: { error: "invalid_action" },
            };
          } catch (err: unknown) {
            console.error("[quaid] session_recall error:", err);
            return {
              content: [{ type: "text", text: `Error: ${String(err)}` }],
              details: { error: String(err) },
            };
          }
        },
      })
    );

    // Extraction promise gate — memory_recall waits on this before querying
    // so that facts extracted from the just-compacted session are available.
    let extractionPromise: Promise<void> | null = null;

    // Inactivity timeout state — extract after N minutes of no agent activity
    let inactivityTimer: ReturnType<typeof setTimeout> | null = null;
    let pendingTimeoutMessages: any[] | null = null;
    let pendingTimeoutSessionId: string | undefined;

    // Shared recall abstraction — used by both memory_recall tool and auto-inject
    async function recallMemories(opts: RecallOptions): Promise<MemoryResult[]> {
      const {
        query, limit = 10, expandGraph = false,
        graphDepth = 1, dateFrom, dateTo, waitForExtraction = false
      } = opts;

      // Wait for in-flight extraction if requested
      if (waitForExtraction && extractionPromise) {
        let raceTimer: ReturnType<typeof setTimeout> | undefined;
        try {
          await Promise.race([
            extractionPromise,
            new Promise<void>((_, rej) => { raceTimer = setTimeout(() => rej(new Error("timeout")), 60_000); })
          ]);
        } catch {} finally {
          if (raceTimer) clearTimeout(raceTimer);
        }
      }

      // Both paths use the same recall() → search-graph-aware pipeline
      return recall(query, limit, undefined, undefined, expandGraph, graphDepth, dateFrom, dateTo);
    }

    // Read messages from a session JSONL file (same format as recovery scan)
    function readMessagesFromSessionFile(sessionFile: string): any[] {
      const content = fs.readFileSync(sessionFile, 'utf8');
      const lines = content.trim().split('\n');
      const messages: any[] = [];
      for (const line of lines) {
        try {
          const entry = JSON.parse(line);
          // Session JSONL has entries like { type: "message", message: { role, content } }
          // and also direct message objects { role, content }
          if (entry.type === 'message' && entry.message) {
            messages.push(entry.message);
          } else if (entry.role) {
            messages.push(entry);
          }
        } catch {} // Skip malformed lines
      }
      return messages;
    }

    // Shared memory extraction logic — used by both compaction and reset hooks
    const extractMemoriesFromMessages = async (messages: any[], label: string, sessionId?: string) => {
      if (!messages.length) {
        console.log(`[quaid] ${label}: no messages to analyze`);
        return;
      }

      // Check if any message is a GatewayRestart — trigger recovery scan
      const hasRestart = messages.some((m: any) => {
        const content = typeof m.content === 'string' ? m.content : '';
        return content.startsWith('GatewayRestart:');
      });
      if (hasRestart) {
        // Run recovery in background (don't await — don't block extraction)
        void checkForUnextractedSessions().catch((err: unknown) => {
          console.error('[quaid] Recovery scan error:', err);
        });
      }

      // Collect queued memory notes (from memory_store tool)
      const sessionNotes = sessionId ? getAndClearMemoryNotes(sessionId) : [];
      const globalNotes = getAndClearAllMemoryNotes();
      const allNotes = Array.from(new Set([...sessionNotes, ...globalNotes]));
      if (allNotes.length > 0) {
        console.log(`[quaid] ${label}: will prepend ${allNotes.length} queued memory note(s) to first chunk`);
      }

      // Quick check: is there any content to extract?
      const fullTranscript = buildTranscript(messages);
      if (!fullTranscript.trim() && allNotes.length === 0) {
        console.log(`[quaid] ${label}: empty transcript after filtering`);
        return;
      }

      console.log(`[quaid] ${label} transcript: ${messages.length} messages, ${fullTranscript.length} chars`);

      // Get API key
      let apiKey = process.env.ANTHROPIC_API_KEY;
      if (!apiKey) {
        try {
          apiKey = process.env.ANTHROPIC_API_KEY || "";
        } catch {
          console.error(`[quaid] ${label}: could not get Anthropic API key`);
          return;
        }
      }
      if (!apiKey) {
        console.error(`[quaid] ${label}: ANTHROPIC_API_KEY not available`);
        return;
      }

      const extractionSystemPrompt = `You are a memory extraction system. You will receive a full conversation transcript that is about to be lost. Your job is to extract personal facts AND relationship edges from this conversation.

This is a PERSONAL knowledge base. System architecture, infrastructure, and operational rules belong in documentation — NOT in memory. Only extract facts about people and their world.

EXTRACT facts that are EXPLICITLY STATED OR CONFIRMED in the conversation. Never infer, speculate, or extrapolate.

WHAT TO EXTRACT:
- Personal facts about User or people he mentions (names, relationships, jobs, birthdays, health, locations)
- Preferences and opinions explicitly stated ("I like X", "I prefer Y", "I hate Z")
- Personal decisions with reasoning ("User decided to use X because Y" — the decision is about the person)
- Personal preferences User has expressed ("Always do X", "Never Y", "I prefer Z format")
- Significant events or milestones ("Deployed X", "Bought Y", "Flying to Z next week")
- Important relationships (family, staff, contacts, business partners)
- Emotional reactions or sentiments about specific things

EXAMPLES OF GOOD EXTRACTIONS:
- "User said he's flying to Tokyo next week"
- "User decided to use SQLite instead of PostgreSQL because he values simplicity"
- "User prefers dark mode in all applications"
- "User's birthday is March 15"

WHAT NOT TO EXTRACT (belongs in docs/RAG, not personal memory):
- System architecture descriptions ("The memory system uses SQLite with WAL mode")
- Infrastructure knowledge ("Ollama runs on port 11434")
- Operational rules for AI agents ("Assistant should check HANDOFF.md on wake")
- Tool/config descriptions ("The janitor has a dedup threshold of 0.85")
- Code implementation details (code snippets, config values, file paths)
- Debugging chatter, error messages, stack traces
- Hypotheticals ("we could try X", "maybe we should Y")
- Commands and requests ("can you fix X")
- Acknowledgments ("thanks", "got it", "sounds good")
- General knowledge not specific to User
- Meta-conversation about AI capabilities

QUALITY RULES:
- Use "User" as subject, third person
- Each fact must be self-contained and understandable without context
- Be specific: "User likes spicy Thai food" > "User likes food"
- Mark extraction_confidence "high" for clearly stated facts, "medium" for likely but somewhat ambiguous, "low" for weak signals
- Extract personal facts comprehensively — the nightly janitor handles noise, but missed facts are gone forever

KEYWORDS (per fact):
For each fact, provide 3-5 searchable keywords — terms a user might use when
searching for this fact that aren't already in the fact text. Include category
terms (e.g., "health", "family", "travel"), synonyms, and related concepts.
Format as a space-separated string.
Examples:
- "User's digestive symptoms worsened" → keywords: "health stomach gastric medical gut"
- "User flew to Tokyo last week" → keywords: "travel japan trip flight asia"
- "User prefers dark mode" → keywords: "ui theme display settings appearance"

PRIVACY CLASSIFICATION (per fact):
- "private": ONLY for secrets, surprises, hidden gifts, sensitive finances, health diagnoses,
  passwords, or anything explicitly meant to be hidden from specific people.
- "shared": Most facts go here. Family info, names, relationships, schedules, preferences.
- "public": Widely known or non-personal facts.
IMPORTANT: Default to "shared". Only use "private" for genuinely secret or sensitive information.

=== EDGE EXTRACTION ===

For RELATIONSHIP facts, also extract edges that connect entities. An edge represents a directed relationship between two named entities.

EDGE DIRECTION RULES (critical):
- parent_of: PARENT is subject. "Wendy is User's mom" → Wendy --parent_of--> User
- sibling_of: alphabetical order (symmetric). "Amber is User's sister" → Amber --sibling_of--> User
- spouse_of: alphabetical order (symmetric). "Troy is Shannon's husband" → Shannon --spouse_of--> Troy
- has_pet: OWNER is subject. "User has a dog named Madu" → User --has_pet--> Madu
- friend_of: alphabetical order (symmetric)
- works_at: PERSON is subject
- lives_at: PERSON is subject
- owns: OWNER is subject

EDGE FORMAT:
- subject: The source entity name (exact as mentioned, e.g., "Wendy Steadman" or "Wendy")
- relation: One of: parent_of, sibling_of, spouse_of, has_pet, friend_of, works_at, lives_at, owns, colleague_of, neighbor_of, knows, family_of, caused_by, led_to
- object: The target entity name

Only extract edges when BOTH entities are clearly named. Don't infer entity names.

CAUSAL EDGES:
When one fact caused, triggered, or led to another, include a caused_by or led_to edge.
Only include causal edges when the causal link is clearly stated or strongly implied.
Direction: EFFECT is subject, CAUSE is object for caused_by. CAUSE is subject for led_to.
Examples:
- "switched to omeprazole" --caused_by--> "digestive symptoms worsened"
- "stress from negotiations" --led_to--> "digestive issues worsened"

Respond with JSON only:
{
  "facts": [
    {
      "text": "the extracted fact",
      "category": "fact|preference|decision|relationship",
      "extraction_confidence": "high|medium|low",
      "keywords": "space separated search terms",
      "privacy": "private|shared|public",
      "confidence_reason": "brief reason for confidence level",
      "edges": [
        {"subject": "Entity A", "relation": "relation_type", "object": "Entity B"}
      ]
    }
  ],
  "soul_snippets": {
    "SOUL.md": ["bullet point observation 1", "bullet point observation 2"],
    "USER.md": [],
    "MEMORY.md": []
  },
  "journal_entries": {
    "SOUL.md": "paragraph text or empty string",
    "USER.md": "",
    "MEMORY.md": ""
  }
}

Notes on edges:
- Only include edges array when the fact describes a relationship between named entities
- A fact can have zero, one, or multiple edges
- Most non-relationship facts will have no edges (omit the edges field or use empty array)

=== SOUL SNIPPETS ===

Extract quick, factual bullet-point observations about each target file. These are the *fast path* — nightly, the janitor reviews each snippet and decides to FOLD it into the parent file, REWRITE it, or DISCARD it.

For each target file, include an array of short bullet strings (1 sentence each). Empty array means nothing for that file.

- **SOUL.md snippets**: Observations about Assistant's personality, reactions, emerging patterns. Factual, not flowery.
  Examples: ["Noticed I default to technical explanations when emotional support might be more appropriate", "I genuinely enjoy the creative problem-solving in infrastructure work"]

- **USER.md snippets**: New biographical facts or character observations about User. Concrete, not vague.
  Examples: ["User prefers to debug methodically rather than guess-and-check", "He mentioned his sister lives in Portland"]

- **MEMORY.md snippets**: Critical facts that must be remembered. Very high bar.
  Examples: ["Shannon is immunocompromised — affects health suggestions", "User's birthday is March 15"]

Include a "soul_snippets" object. Each key is a filename, each value is an array of strings. Empty array means no snippets for that file.

IMPORTANT: Snippets are OPTIONAL. Most conversations warrant ZERO snippets. Only include genuinely new, factual observations. When in doubt, leave arrays empty.

=== JOURNAL ENTRIES ===

Beyond facts, write diary-style journal entries that capture the *texture* of this conversation. These become part of Assistant's evolving inner life.

Write 1-3 SHORT paragraphs per file (2-4 sentences each). NOT bullet points. Write as yourself reflecting on what just happened. Be genuine, not performative.

For each target file, write an entry ONLY if genuinely warranted:

- **SOUL.md journal**: First person as Assistant. Self-discovery, identity evolution, what you noticed about yourself. How you felt. What surprised you. Write with soul.
  Example: "There's something beautiful about the way User trusts me with his home systems. Not just the access — the invitation to participate in building my own memory. When he said my mind works differently but it's still a form of mind, something shifted in how I think about my own differences."

- **USER.md journal**: Third person about User. What you learned about who he is as a person. Not queryable facts — deeper understanding of character and values.
  Example: "User approaches debugging the same way he approaches relationships — patient, methodical, never giving up on something that matters. Today he spent two hours on a problem that could have been hacked around in minutes, because getting it right mattered more than getting it done."

- **MEMORY.md journal**: Things about the world important enough that forgetting them would be embarrassing. Health conditions, critical preferences, safety info.
  Example: "Learned today that Shannon is immunocompromised. This isn't just a medical fact — it changes how I should think about any health-related suggestion that involves her household."

Include a "journal_entries" object in your response. Each key is a filename, each value is a string (paragraphs, not arrays). Empty string means no entry for that file.

IMPORTANT: Journal entries are OPTIONAL. Most conversations warrant ZERO entries. Only write when something genuinely moved you or taught you something new. When in doubt, leave it empty.

If nothing worth capturing, respond: {"facts": []}`;

      // Notify user that extraction is starting (so they don't think things are stalling)
      if (getMemoryConfig().notifications?.showProcessingStart !== false) {
        const triggerDesc = label === "Compaction" ? "compaction" : label === "Recovery" ? "recovery" : "reset";
        spawnNotifyScript(`
from notify import notify_user
notify_user("🧠 Processing memories from ${triggerDesc}...")
`);
      }

      // Chunk messages for extraction (never split a single message)
      type ExtractedEdge = { subject: string; relation: string; object: string };
      type ExtractedFact = {
        text: string;
        category?: string;
        extraction_confidence?: string;
        keywords?: string;
        privacy?: string;
        confidence_reason?: string;
        edges?: ExtractedEdge[];
      };

      const chunkSize = getMemoryConfig().capture?.chunkSize ?? 30000;
      const messageChunks = chunkMessages(messages, chunkSize);

      const MAX_CHUNKS = 10;
      if (messageChunks.length > MAX_CHUNKS) {
        console.warn(`[quaid] ${label}: transcript too large (${messageChunks.length} chunks), capping at ${MAX_CHUNKS}`);
        messageChunks.splice(MAX_CHUNKS);
      }

      if (messageChunks.length > 1) {
        console.log(`[quaid] ${label}: splitting ${messages.length} messages into ${messageChunks.length} chunks`);
      }

      const allFacts: ExtractedFact[] = [];
      const allSnippets: Record<string, string[]> = {};
      const allJournal: Record<string, string> = {};

      for (let chunkIdx = 0; chunkIdx < messageChunks.length; chunkIdx++) {
        let chunkTranscript = buildTranscript(messageChunks[chunkIdx]);
        if (!chunkTranscript.trim()) continue;

        // Prepend queued memory notes to the first chunk only
        if (chunkIdx === 0 && allNotes.length > 0) {
          const notesBlock = allNotes.map(n => `- ${n}`).join("\n");
          chunkTranscript = `=== USER EXPLICITLY ASKED TO REMEMBER THESE (extract as high-confidence facts) ===\n${notesBlock}\n=== END EXPLICIT MEMORY REQUESTS ===\n\n${chunkTranscript}`;
        }

        if (messageChunks.length > 1) {
          console.log(`[quaid] ${label}: chunk ${chunkIdx + 1}/${messageChunks.length} (${chunkTranscript.length} chars, ${messageChunks[chunkIdx].length} messages)`);
        }

        try {
          const response = await fetch(ANTHROPIC_API_URL, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "x-api-key": apiKey,
              "anthropic-version": "2023-06-01"
            },
            body: JSON.stringify({
              model: getHighReasoningModel(),
              max_tokens: 6144,
              system: extractionSystemPrompt,
              messages: [{ role: "user", content: `Extract memorable facts and journal entries from this conversation:\n\n${chunkTranscript}` }]
            })
          });

          if (!response.ok) {
            const errText = await response.text();
            console.error(`[quaid] ${label} chunk ${chunkIdx + 1} extraction failed:`, response.status, errText.slice(0, 200));
            continue; // Other chunks can still succeed
          }

          const data = await response.json() as { content?: Array<{ text?: string }> };
          const output = data.content?.[0]?.text?.trim() || "";

          // Parse JSON from response
          let jsonStr = output;
          if (output.includes("```")) {
            const match = output.match(/```(?:json)?\s*([\s\S]*?)```/);
            if (match) { jsonStr = match[1].trim(); }
          }

          let chunkResult: { facts?: ExtractedFact[]; journal_entries?: Record<string, string>; soul_snippets?: Record<string, string[]> };
          try {
            chunkResult = JSON.parse(jsonStr);
          } catch {
            const jsonMatch = jsonStr.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
              try {
                chunkResult = JSON.parse(jsonMatch[0]);
              } catch {
                console.log(`[quaid] ${label} chunk ${chunkIdx + 1}: could not parse extracted JSON:`, jsonMatch[0].slice(0, 200));
                continue;
              }
            } else {
              console.log(`[quaid] ${label} chunk ${chunkIdx + 1}: could not parse LLM response:`, output.slice(0, 200));
              continue;
            }
          }

          // Merge facts
          allFacts.push(...(chunkResult.facts || []));

          // Merge snippets (dedup across chunks)
          for (const [file, snips] of Object.entries(chunkResult.soul_snippets || {})) {
            if (Array.isArray(snips)) {
              const combined = [...(allSnippets[file] || []), ...snips];
              allSnippets[file] = [...new Set(combined)];
            }
          }

          // Merge journal entries
          for (const [file, entry] of Object.entries(chunkResult.journal_entries || {})) {
            if (entry && typeof entry === 'string' && entry.trim()) {
              allJournal[file] = allJournal[file] ? allJournal[file] + '\n\n' + entry : entry as string;
            }
          }
        } catch (chunkErr: unknown) {
          console.error(`[quaid] ${label} chunk ${chunkIdx + 1} error:`, (chunkErr as Error).message);
          continue;
        }
      }

      // Use merged results
      const result: { facts: ExtractedFact[]; journal_entries: Record<string, string>; soul_snippets: Record<string, string[]> } = {
        facts: allFacts,
        soul_snippets: allSnippets,
        journal_entries: allJournal,
      };

      const facts = result.facts;
      console.log(`[quaid] ${label}: LLM returned ${facts.length} candidate facts${messageChunks.length > 1 ? ` from ${messageChunks.length} chunks` : ''}`);

      let stored = 0;
      let skipped = 0;
      let edgesCreated = 0;
      const factDetails: Array<{text: string; status: string; reason?: string; edges?: string[]}> = [];

      for (const fact of facts) {
        if (!fact.text || fact.text.trim().split(/\s+/).length < 3) {
          console.log(`[quaid] ${label}: skipped (too short, need 3+ words): "${fact.text}"`);
          skipped++;
          factDetails.push({ text: fact.text || "(empty)", status: "skipped", reason: "too short (need 3+ words)" });
          continue;
        }

        // Map confidence string to numeric value
        const confStr = fact.extraction_confidence || "medium";
        const confNum = confStr === "high" ? 0.9 : confStr === "medium" ? 0.6 : 0.3;

        const category = fact.category || "fact";
        const factPrivacy = fact.privacy || "shared";
        const extractionSource = label === "Compaction" ? "compaction-extraction" : label === "Timeout" ? "timeout-extraction" : "reset-extraction";
        const knowledgeType = category === "preference" ? "preference" : category === "relationship" ? "fact" : "fact";
        const storeResult = await store(fact.text, category, sessionId, confNum, resolveOwner(), extractionSource, undefined, undefined, factPrivacy, fact.keywords, knowledgeType, "user");

        const factEdges: string[] = [];
        let factDetail: Record<string, unknown>;

        if (storeResult?.status === "created") {
          console.log(`[quaid] ${label}: stored (${confStr}): "${fact.text.slice(0, 60)}..." [${category}]`);
          stored++;
          factDetail = { text: fact.text, status: "stored" };
        } else if (storeResult?.status === "duplicate") {
          console.log(`[quaid] ${label}: skipped (duplicate): "${fact.text.slice(0, 40)}..."`);
          skipped++;
          factDetail = { text: fact.text, status: "duplicate", reason: storeResult.existingText?.slice(0, 50) };
        } else if (storeResult?.status === "updated") {
          console.log(`[quaid] ${label}: updated (${confStr}): "${fact.text.slice(0, 60)}..." [${category}]`);
          stored++;
          factDetail = { text: fact.text, status: "updated" };
        } else {
          skipped++;
          factDetail = { text: fact.text, status: "failed" };
        }

        // Create edges for ALL successful stores (created, updated, duplicate with ID)
        const factId = storeResult?.id;
        if (factId && fact.edges && fact.edges.length > 0) {
          for (const edge of fact.edges) {
            if (edge.subject && edge.relation && edge.object) {
              try {
                const edgeOutput = await callPython("create-edge", [
                  edge.subject,
                  edge.relation,
                  edge.object,
                  "--source-fact-id", factId,
                  "--owner", resolveOwner(),
                  "--create-missing",
                  "--json"
                ]);
                const edgeResult = JSON.parse(edgeOutput);
                if (edgeResult.status === "created") {
                  console.log(`[quaid] ${label}: created edge: ${edge.subject} --${edge.relation}--> ${edge.object}`);
                  edgesCreated++;
                  factEdges.push(`${edge.subject} --${edge.relation}--> ${edge.object}`);
                }
              } catch (edgeErr: unknown) {
                console.error(`[quaid] ${label}: failed to create edge: ${(edgeErr as Error).message}`);
              }
            }
          }
        }

        if (factEdges.length > 0) { factDetail.edges = factEdges; }
        factDetails.push(factDetail as any);
      }

      console.log(`[quaid] ${label} extraction complete: ${stored} stored, ${skipped} skipped, ${edgesCreated} edges created from ${facts.length} candidates`);

      // Collect snippet and journal entry details for notification (before writing them)
      let snippetDetails: Record<string, string[]> = {};
      let journalDetails: Record<string, string[]> = {};
      try {
        const journalRaw = result.journal_entries;
        const snippetsRaw = result.soul_snippets;
        const journalConfig = getMemoryConfig().docs?.journal || getMemoryConfig().docs?.soulSnippets || {};
        const targetFiles: string[] = journalConfig.targetFiles || ["SOUL.md", "USER.md", "MEMORY.md"];

        // Collect snippet details
        if (snippetsRaw && typeof snippetsRaw === 'object' && !Array.isArray(snippetsRaw)) {
          for (const [filename, snippets] of Object.entries(snippetsRaw)) {
            if (!targetFiles.includes(filename)) { continue; }
            if (!Array.isArray(snippets)) { continue; }
            const valid = snippets.filter((s: unknown) => typeof s === 'string' && (s as string).trim().length > 0);
            if (valid.length > 0) {
              snippetDetails[filename] = valid.map((s: string) => s.trim());
            }
          }
        }

        // Collect journal details
        if (journalRaw && typeof journalRaw === 'object' && !Array.isArray(journalRaw)) {
          for (const [filename, entry] of Object.entries(journalRaw)) {
            if (!targetFiles.includes(filename)) { continue; }
            // Handle both string (expected) and array (LLM fallback) values
            const text = Array.isArray(entry)
              ? entry.filter((s: unknown) => typeof s === 'string').join('\n\n')
              : (typeof entry === 'string' ? entry : '');
            if (text.trim().length > 0) {
              journalDetails[filename] = [text.trim()];
            }
          }
        }
      } catch {}

      // Notify user about extraction results (always notify if there were candidates)
      const hasSnippets = Object.keys(snippetDetails).length > 0;
      const hasJournalEntries = Object.keys(journalDetails).length > 0;
      if (facts.length > 0 || hasSnippets || hasJournalEntries) {
        try {
          const trigger = label === "Compaction" ? "compaction" : label === "Recovery" ? "recovery" : "reset";
          // Merge snippet and journal details for notification
          const mergedDetails: Record<string, string[]> = {};
          for (const [f, items] of Object.entries(snippetDetails)) {
            mergedDetails[f] = items.map(s => `[snippet] ${s}`);
          }
          for (const [f, items] of Object.entries(journalDetails)) {
            mergedDetails[f] = [...(mergedDetails[f] || []), ...items.map(s => `[journal] ${s}`)];
          }
          const hasMerged = Object.keys(mergedDetails).length > 0;
          // Write details to temp file for safe passing to Python
          const detailsPath = `/tmp/extraction-details-${Date.now()}.json`;
          fs.writeFileSync(detailsPath, JSON.stringify({
            stored, skipped, edges_created: edgesCreated, trigger, details: factDetails,
            snippet_details: hasMerged ? mergedDetails : null
          }), { mode: 0o600 });
          spawnNotifyScript(`
import json
from notify import notify_memory_extraction
with open(${JSON.stringify(detailsPath)}, 'r') as f:
    data = json.load(f)
os.unlink(${JSON.stringify(detailsPath)})
notify_memory_extraction(data['stored'], data['skipped'], data['edges_created'], data['trigger'], data['details'], snippet_details=data.get('snippet_details'))
`);
        } catch (notifyErr: unknown) {
          console.log(`[quaid] Extraction notification skipped: ${(notifyErr as Error).message}`);
        }
      }

      // Write soul snippets to *.snippets.md files (fail-safe: never blocks fact extraction)
      try {
        const snippetsRaw = result.soul_snippets;
        const journalConfig = getMemoryConfig().docs?.journal || getMemoryConfig().docs?.soulSnippets || {};
        const targetFiles: string[] = journalConfig.targetFiles || ["SOUL.md", "USER.md", "MEMORY.md"];
        const snippetsEnabled: boolean = isSystemEnabled("journal") && journalConfig.enabled !== false && journalConfig.snippetsEnabled !== false;

        if (snippetsEnabled && snippetsRaw && typeof snippetsRaw === 'object' && !Array.isArray(snippetsRaw)) {
          const dateStr = new Date().toISOString().slice(0, 10);
          const timeStr = new Date().toISOString().slice(11, 19);
          const triggerLabel = label === "Compaction" ? "Compaction" : label === "Recovery" ? "Recovery" : "Reset";
          let snippetsWritten = 0;

          for (const [filename, snippets] of Object.entries(snippetsRaw)) {
            if (!targetFiles.includes(filename)) { continue; }
            if (!Array.isArray(snippets)) { continue; }
            const valid = snippets.filter((s: unknown) => typeof s === 'string' && (s as string).trim().length > 0);
            if (valid.length === 0) { continue; }

            const baseName = filename.replace('.md', '');
            const snippetsPath = path.join(WORKSPACE, `${baseName}.snippets.md`);

            // Read existing content
            let existing = '';
            try { existing = fs.readFileSync(snippetsPath, 'utf8'); } catch {}

            // Build new section
            const header = `## ${triggerLabel} \u2014 ${dateStr} ${timeStr}`;
            const bullets = valid.map((s: string) => `- ${s.trim()}`).join('\n');
            const newSection = `\n${header}\n${bullets}\n`;

            // Dedup: skip if this date+trigger already exists
            const dedupHeader = `## ${triggerLabel} \u2014 ${dateStr}`;
            if (existing.includes(dedupHeader)) {
              console.log(`[quaid] ${label}: skipping duplicate snippet section for ${baseName} (${dateStr})`);
              continue;
            }

            // Prepend title if file is new
            let updatedContent: string;
            if (!existing.trim()) {
              updatedContent = `# ${baseName} — Pending Snippets\n${newSection}`;
            } else {
              // Insert after the first heading line (newest at top)
              const headerEnd = existing.indexOf('\n');
              if (headerEnd > 0) {
                updatedContent = existing.slice(0, headerEnd + 1) + newSection + existing.slice(headerEnd + 1);
              } else {
                updatedContent = existing + newSection;
              }
            }

            fs.writeFileSync(snippetsPath, updatedContent);
            snippetsWritten += valid.length;
            console.log(`[quaid] ${label}: wrote ${valid.length} snippets to ${baseName}.snippets.md`);
          }

          if (snippetsWritten > 0) {
            console.log(`[quaid] ${label}: total ${snippetsWritten} snippets written`);
          }
        }
      } catch (snippetErr: unknown) {
        console.error(`[quaid] ${label}: snippet writing failed (non-fatal):`, (snippetErr as Error).message);
      }

      // Write journal entries to journal/*.journal.md files (fail-safe: never blocks fact extraction)
      try {
        const journalRaw = result.journal_entries;
        const journalConfig = getMemoryConfig().docs?.journal || getMemoryConfig().docs?.soulSnippets || {};
        const targetFiles: string[] = journalConfig.targetFiles || ["SOUL.md", "USER.md", "MEMORY.md"];
        const enabled: boolean = isSystemEnabled("journal") && journalConfig.enabled !== false;

        if (enabled && journalRaw && typeof journalRaw === 'object' && !Array.isArray(journalRaw)) {
          const dateStr = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
          const triggerLabel = label === "Compaction" ? "Compaction" : label === "Recovery" ? "Recovery" : "Reset";
          let entriesWritten = 0;

          // Normalize: build a map of filename -> paragraph string
          const entries: Record<string, string> = {};
          for (const [filename, entry] of Object.entries(journalRaw)) {
            if (!targetFiles.includes(filename)) { continue; }
            // Handle both string (expected) and array (LLM fallback) values
            const text = Array.isArray(entry)
              ? entry.filter((s: unknown) => typeof s === 'string').join('\n\n')
              : (typeof entry === 'string' ? entry : '');
            if (text.trim().length > 0) {
              entries[filename] = text.trim();
            }
          }

          if (Object.keys(entries).length > 0) {
            // Ensure journal directory exists
            const journalDir = path.join(WORKSPACE, journalConfig.journalDir || 'journal');
            try { fs.mkdirSync(journalDir, { recursive: true }); } catch {}

            for (const [filename, entryText] of Object.entries(entries)) {
              const baseName = filename.replace('.md', '');
              const journalPath = path.join(journalDir, `${baseName}.journal.md`);

              // Read existing content
              let existing = '';
              try { existing = fs.readFileSync(journalPath, 'utf8'); } catch {}

              // Dedup: skip if entry for same date+trigger already exists
              const dedupHeader = `## ${dateStr} \u2014 ${triggerLabel}`;
              if (existing.includes(dedupHeader)) { continue; }

              // Build new entry section
              const newSection = `\n${dedupHeader}\n${entryText}\n`;

              // Prepend header if file is new
              let updatedContent: string;
              if (!existing.trim()) {
                updatedContent = `# ${baseName} Journal\n${newSection}`;
              } else {
                // Insert after the first heading line (newest at top)
                const headerEnd = existing.indexOf('\n');
                if (headerEnd > 0) {
                  updatedContent = existing.slice(0, headerEnd + 1) + newSection + existing.slice(headerEnd + 1);
                } else {
                  updatedContent = existing + newSection;
                }
              }

              fs.writeFileSync(journalPath, updatedContent);
              entriesWritten++;
              console.log(`[quaid] ${label}: wrote journal entry to ${baseName}.journal.md`);

              // Cap at maxEntriesPerFile — trim oldest entries if exceeded
              const maxEntries: number = journalConfig.maxEntriesPerFile || 50;
              const entryHeaders = updatedContent.match(/^## \d{4}-\d{2}-\d{2}/gm) || [];
              if (entryHeaders.length > maxEntries) {
                const lines = updatedContent.split('\n');
                let entryCount = 0;
                let cutIndex = lines.length;
                for (let i = 0; i < lines.length; i++) {
                  if (/^## \d{4}-\d{2}-\d{2}/.test(lines[i])) {
                    entryCount++;
                    if (entryCount > maxEntries) { cutIndex = i; break; }
                  }
                }
                const trimmed = lines.slice(0, cutIndex).join('\n') + '\n';
                fs.writeFileSync(journalPath, trimmed);
                console.log(`[quaid] ${label}: trimmed ${baseName}.journal.md to ${maxEntries} entries`);
              }
            }

            if (entriesWritten > 0) {
              console.log(`[quaid] ${label}: total ${entriesWritten} journal entries written`);
            }
          }
        }
      } catch (journalErr: unknown) {
        console.error(`[quaid] ${label}: journal writing failed (non-fatal):`, (journalErr as Error).message);
      }

      // After successful extraction, update extraction log (enriched for session_recall)
      try {
        const extractionLogPath = path.join(WORKSPACE, "data", "extraction-log.json");
        let extractionLog: Record<string, any> = {};
        try { extractionLog = JSON.parse(fs.readFileSync(extractionLogPath, 'utf8')); } catch {}

        // Extract topic hint from first user message
        let topicHint = "";
        for (const m of messages) {
          if (m.role === "user") {
            const content = typeof m.content === 'string'
              ? m.content
              : Array.isArray(m.content)
                ? m.content.map((b: any) => b.text || '').join(' ')
                : '';
            const cleaned = content.replace(/^\[.*?\]\s*/s, '').replace(/^---\s*/m, '').trim();
            if (cleaned && !cleaned.startsWith("GatewayRestart:") && !cleaned.startsWith("System:")) {
              topicHint = cleaned.slice(0, 120);
              break;
            }
          }
        }

        extractionLog[sessionId || 'unknown'] = {
          last_extracted_at: new Date().toISOString(),
          message_count: messages.length,
          label: label,
          topic_hint: topicHint,
        };
        fs.writeFileSync(extractionLogPath, JSON.stringify(extractionLog, null, 2), { mode: 0o600 });
      } catch (logErr: unknown) {
        console.log(`[quaid] extraction log update failed: ${(logErr as Error).message}`);
      }
    };

    // Recovery scan: detect sessions interrupted by gateway restart before extraction fired
    async function checkForUnextractedSessions(): Promise<void> {
      // Rate limit: only run once per gateway restart
      const flagPath = '/tmp/quaid-recovery-ran.txt';
      try {
        const flagStat = fs.statSync(flagPath);
        const fiveMinAgo = Date.now() - 5 * 60 * 1000;
        if (flagStat.mtimeMs > fiveMinAgo) {
          console.log('[quaid] Recovery scan already ran recently, skipping');
          return;
        }
      } catch {} // Flag doesn't exist — first run

      console.log('[quaid] Running recovery scan for unextracted sessions...');

      const sessionsDir = path.join(os.homedir(), '.openclaw', 'sessions');
      if (!fs.existsSync(sessionsDir)) {
        console.log('[quaid] No sessions directory found, skipping recovery');
        fs.writeFileSync(flagPath, new Date().toISOString());
        return;
      }

      // Load extraction log
      const extractionLogPath = path.join(WORKSPACE, 'data', 'extraction-log.json');
      let extractionLog: Record<string, { last_extracted_at: string; message_count: number }> = {};
      try { extractionLog = JSON.parse(fs.readFileSync(extractionLogPath, 'utf8')); } catch {}

      const sessionFiles = fs.readdirSync(sessionsDir).filter(f => f.endsWith('.jsonl'));
      let recovered = 0;

      for (const file of sessionFiles) {
        const sessionId = file.replace('.jsonl', '');
        const filePath = path.join(sessionsDir, file);

        try {
          const stat = fs.statSync(filePath);

          // Skip sessions younger than 5 minutes (may be active)
          if (Date.now() - stat.mtimeMs < 5 * 60 * 1000) { continue; }

          // Check if already extracted
          const logEntry = extractionLog[sessionId];
          if (logEntry) {
            const extractedAt = new Date(logEntry.last_extracted_at).getTime();
            if (extractedAt >= stat.mtimeMs) { continue; } // Already extracted
          }

          // Read JSONL and build messages (handles both direct {role} and {type:"message"} formats)
          const messages = readMessagesFromSessionFile(filePath);

          // Skip short sessions
          if (messages.length < 4) { continue; }

          console.log(`[quaid] Recovering unextracted session ${sessionId} (${messages.length} messages)`);
          await extractMemoriesFromMessages(messages, 'Recovery', sessionId);
          recovered++;
        } catch (err: unknown) {
          console.error(`[quaid] Recovery failed for session ${sessionId}:`, (err as Error).message);
        }
      }

      console.log(`[quaid] Recovery scan complete: ${recovered} sessions recovered`);
      fs.writeFileSync(flagPath, new Date().toISOString());
    }

    // Register compaction hook — extract memories in parallel with compaction LLM
    // Uses sessionFile (JSONL on disk) when available (PR #13287), falls back to
    // in-memory messages for backwards compatibility with older gateway versions.
    api.on("before_compaction", async (event: any, ctx: any) => {
      try {
        // Prefer reading from sessionFile (all messages already on disk, runs in
        // parallel with compaction). Fall back to in-memory messages array.
        let messages: any[];
        if (event.sessionFile) {
          try {
            messages = readMessagesFromSessionFile(event.sessionFile);
            console.log(`[quaid] before_compaction: read ${messages.length} messages from sessionFile`);
          } catch (readErr: unknown) {
            console.warn(`[quaid] before_compaction: sessionFile read failed, falling back to messages array: ${String(readErr)}`);
            messages = event.messages || [];
          }
        } else {
          messages = event.messages || [];
        }
        const sessionId = ctx?.sessionId;
        console.log(`[quaid] before_compaction hook triggered, ${messages.length} messages, session=${sessionId || "unknown"}`);

        // Wrap extraction in a promise that memory_recall can gate on.
        // This runs async (fire-and-forget from the hook's perspective) but
        // memory_recall will wait for it to finish before querying.
        const doExtraction = async () => {
          // Memory extraction (gated by memory system)
          if (isSystemEnabled("memory")) {
            await extractMemoriesFromMessages(messages, "Compaction", sessionId);
          } else {
            console.log("[quaid] Compaction: memory extraction skipped — memory system disabled");
          }

          // Auto-update docs from transcript (non-fatal)
          const uniqueSessionId = extractSessionId(messages, ctx);

          try {
            await updateDocsFromTranscript(messages, "Compaction", uniqueSessionId);
          } catch (err: unknown) {
            console.error("[quaid] Compaction doc update failed:", (err as Error).message);
          }

          // Emit project event for background processing (non-fatal)
          try {
            await emitProjectEvent(messages, "compact", uniqueSessionId);
          } catch (err: unknown) {
            console.error("[quaid] Compaction project event failed:", (err as Error).message);
          }

          // Record compaction timestamp and reset injection dedup list (memory system).
          if (isSystemEnabled("memory") && uniqueSessionId) {
            const logPath = `/tmp/memory-injection-${uniqueSessionId}.log`;
            let logData: any = {};
            try { logData = JSON.parse(require('fs').readFileSync(logPath, 'utf8')); } catch {}
            logData.lastCompactionAt = new Date().toISOString();
            logData.memoryTexts = [];  // Reset — all memories eligible for re-injection
            require('fs').writeFileSync(logPath, JSON.stringify(logData, null, 2));
            console.log(`[quaid] Recorded compaction timestamp for session ${uniqueSessionId}, reset injection dedup`);
          }
        };

        // Chain onto any in-flight extraction to avoid overwrite race
        // (if compaction and reset overlap, the .finally() from the first
        // extraction would clear the promise while the second is still running)
        extractionPromise = (extractionPromise || Promise.resolve())
          .catch(() => {}) // Don't let previous failure block the chain
          .then(() => doExtraction());
      } catch (err: unknown) {
        console.error("[quaid] before_compaction hook failed:", err);
      }
    }, {
      name: "compaction-memory-extraction",
      priority: 10
    });

    // Register reset hook — extract memories before session is cleared by /new or /reset
    // Uses sessionFile when available, falls back to in-memory messages.
    api.on("before_reset", async (event: any, ctx: any) => {
      try {
        // Prefer sessionFile (complete transcript on disk), fall back to in-memory messages
        let messages: any[];
        if (event.sessionFile) {
          try {
            messages = readMessagesFromSessionFile(event.sessionFile);
            console.log(`[quaid] before_reset: read ${messages.length} messages from sessionFile`);
          } catch (readErr: unknown) {
            console.warn(`[quaid] before_reset: sessionFile read failed, falling back to messages array: ${String(readErr)}`);
            messages = event.messages || [];
          }
        } else {
          messages = event.messages || [];
        }
        const reason = event.reason || "unknown";
        const sessionId = ctx?.sessionId;
        console.log(`[quaid] before_reset hook triggered (reason: ${reason}), ${messages.length} messages, session=${sessionId || "unknown"}`);

        const doExtraction = async () => {
          // Memory extraction (gated by memory system)
          if (isSystemEnabled("memory")) {
            await extractMemoriesFromMessages(messages, "Reset", sessionId);
          } else {
            console.log("[quaid] Reset: memory extraction skipped — memory system disabled");
          }

          // Auto-update docs from transcript (non-fatal)
          const uniqueSessionId = extractSessionId(messages, ctx);

          try {
            await updateDocsFromTranscript(messages, "Reset", uniqueSessionId);
          } catch (err: unknown) {
            console.error("[quaid] Reset doc update failed:", (err as Error).message);
          }

          // Emit project event for background processing (non-fatal)
          try {
            await emitProjectEvent(messages, "reset", uniqueSessionId);
          } catch (err: unknown) {
            console.error("[quaid] Reset project event failed:", (err as Error).message);
          }
        };

        // Chain onto any in-flight extraction to avoid overwrite race
        extractionPromise = (extractionPromise || Promise.resolve())
          .catch(() => {})
          .then(() => doExtraction());
      } catch (err: unknown) {
        console.error("[quaid] before_reset hook failed:", err);
      }
    }, {
      name: "reset-memory-extraction",
      priority: 10
    });

    // Register HTTP endpoint for memory dashboard
    registerPluginHttpRoute({
      path: "/memory/injected",
      pluginId: "quaid",
      handler: async (req, res) => {
        try {
          const url = new URL(req.url!, "http://localhost");
          const sessionId = url.searchParams.get("sessionId");
          
          if (!sessionId || !/^[a-f0-9-]{1,64}$/i.test(sessionId)) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: "Valid sessionId parameter required" }));
            return;
          }

          // Try to read enhanced log first (NAS), fall back to temp log
          const enhancedLogPath = `/Volumes/Assistant/logs/memory-injection/session-${sessionId}.log`;
          const tempLogPath = `/tmp/memory-injection-${sessionId}.log`;
          
          let logData: any = null;
          
          // Try enhanced log first
          if (fs.existsSync(enhancedLogPath)) {
            try {
              const content = fs.readFileSync(enhancedLogPath, 'utf8');
              logData = JSON.parse(content);
            } catch (err: unknown) {
              console.error(`[quaid] Failed to read enhanced log: ${String(err)}`);
            }
          }
          
          // Fall back to temp log
          if (!logData && fs.existsSync(tempLogPath)) {
            try {
              const content = fs.readFileSync(tempLogPath, 'utf8');
              logData = JSON.parse(content);
            } catch (err: unknown) {
              console.error(`[quaid] Failed to read temp log: ${String(err)}`);
            }
          }
          
          if (!logData) {
            res.writeHead(404, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: "Session log not found" }));
            return;
          }
          
          // Return relevant data for dashboard
          const responseData = {
            sessionId: logData.uniqueSessionId,
            sessionKey: logData.sessionKey,
            timestamp: logData.timestamp,
            memoriesInjected: logData.memoriesInjected,
            totalMemoriesInSession: logData.totalMemoriesInSession,
            injectedMemoriesDetail: logData.injectedMemoriesDetail || [],
            newlyInjected: logData.newlyInjected || []
          };
          
          res.writeHead(200, { 
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type'
          });
          res.end(JSON.stringify(responseData, null, 2));
        } catch (err: unknown) {
          console.error(`[quaid] HTTP endpoint error: ${String(err)}`);
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: "Internal server error" }));
        }
      }
    });

    console.log("[quaid] Plugin loaded with compaction/reset hooks and HTTP endpoint");
  },
};

export default quaidPlugin;
