import { spawn, spawnSync } from 'child_process';
import { existsSync } from 'fs';
import { mkdtemp, readFile, rm } from 'fs/promises';
import { tmpdir } from 'os';
import path, { join } from 'path';
import { buildContinuationPrompt, buildInitPrompt, buildObservationPrompt, buildSummaryPrompt } from '../../sdk/prompts.js';
import { SettingsDefaultsManager } from '../../shared/SettingsDefaultsManager.js';
import { USER_SETTINGS_PATH } from '../../shared/paths.js';
import { logger } from '../../utils/logger.js';
import { ModeManager } from '../domain/ModeManager.js';
import type { ModeConfig } from '../domain/types.js';
import type { ActiveSession, ConversationMessage } from '../worker-types.js';
import { DatabaseManager } from './DatabaseManager.js';
import { SessionManager } from './SessionManager.js';
import {
  isAbortError,
  processAgentResponse,
  type WorkerRef
} from './agents/index.js';

const DEFAULT_MODEL = 'gpt-5.3-codex-spark';
const DEFAULT_CODEX_PATH = 'codex';
const DEFAULT_MAX_CONTEXT_MESSAGES = 20;
const DEFAULT_MAX_ESTIMATED_TOKENS = 100000;
const DEFAULT_TIMEOUT_MS = 180000;
const CHARS_PER_TOKEN_ESTIMATE = 4;
const MAX_PROCESS_OUTPUT_CAPTURE = 20000;
const WINDOWS_CODEX_COMMAND_NAMES = new Set(['codex', 'codex.exe', 'codex.cmd']);

type CodexReasoningEffort = 'low' | 'medium' | 'high' | 'xhigh';

interface CodexConfig {
  commandPath: string;
  model: string;
  reasoningEffort: CodexReasoningEffort;
  timeoutMs: number;
}

interface CodexExecResult {
  content: string;
  tokensUsed?: number;
  sawAgentMessage?: boolean;
}

interface CodexJsonEvent {
  type?: string;
  item?: {
    type?: string;
    text?: string;
  };
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    total_tokens?: number;
  };
}

function getWindowsCodexInstallCandidates(): string[] {
  const localAppDataCandidates = [
    process.env.LOCALAPPDATA,
    process.env.USERPROFILE ? join(process.env.USERPROFILE, 'AppData', 'Local') : undefined
  ].filter((candidate): candidate is string => Boolean(candidate));

  const uniqueLocalAppData = [...new Set(localAppDataCandidates.map((candidate) => path.normalize(candidate)))];

  return uniqueLocalAppData.flatMap((localAppData) => [
    join(localAppData, 'OpenAI', 'Codex', 'bin', 'codex.exe'),
    join(localAppData, 'OpenAI', 'Codex', 'bin', 'codex.cmd')
  ]);
}

function isBareCodexCommand(commandPath: string): boolean {
  const trimmed = commandPath.trim();
  const hasPathSeparator = trimmed.includes('\\') || trimmed.includes('/');
  return !hasPathSeparator
    && !path.isAbsolute(trimmed)
    && WINDOWS_CODEX_COMMAND_NAMES.has(trimmed.toLowerCase());
}

function resolveCodexCommand(commandPath: string): string {
  const trimmed = commandPath.trim() || DEFAULT_CODEX_PATH;

  if (process.platform !== 'win32' || !isBareCodexCommand(trimmed)) {
    return trimmed;
  }

  return getWindowsCodexInstallCandidates().find((candidate) => existsSync(candidate)) || trimmed;
}

function commandExistsOnPath(commandPath: string): boolean {
  const checker = process.platform === 'win32' ? 'where.exe' : 'which';
  try {
    const result = spawnSync(checker, [commandPath], {
      stdio: 'ignore',
      windowsHide: true
    });
    return result.status === 0;
  } catch {
    return false;
  }
}

export class CodexProvider {
  private dbManager: DatabaseManager;
  private sessionManager: SessionManager;

  constructor(dbManager: DatabaseManager, sessionManager: SessionManager) {
    this.dbManager = dbManager;
    this.sessionManager = sessionManager;
  }

  async startSession(session: ActiveSession, worker?: WorkerRef): Promise<void> {
    const { commandPath, model, reasoningEffort, timeoutMs } = this.getCodexConfig();

    if (!session.memorySessionId) {
      const syntheticMemorySessionId = `codex-${session.contentSessionId}-${Date.now()}`;
      session.memorySessionId = syntheticMemorySessionId;
      this.dbManager.getSessionStore().updateMemorySessionId(session.sessionDbId, syntheticMemorySessionId);
      logger.info('SESSION', `MEMORY_ID_GENERATED | sessionDbId=${session.sessionDbId} | provider=Codex`);
    }

    const mode = ModeManager.getInstance().getActiveMode();

    const initPrompt = session.lastPromptNumber === 1
      ? buildInitPrompt(session.project, session.contentSessionId, session.userPrompt, mode)
      : buildContinuationPrompt(session.userPrompt, session.lastPromptNumber, session.contentSessionId, mode);

    session.conversationHistory.push({ role: 'user', content: initPrompt });

    try {
      const initResponse = await this.queryCodexMultiTurn(
        session.conversationHistory,
        commandPath,
        model,
        reasoningEffort,
        timeoutMs,
        session.abortController.signal
      );
      await this.handleInitResponse(initResponse, session, worker, model);
    } catch (error: unknown) {
      if (error instanceof Error) {
        logger.error('SDK', 'Codex init failed', { sessionId: session.sessionDbId, model }, error);
      } else {
        logger.error('SDK', 'Codex init failed with non-Error', { sessionId: session.sessionDbId, model }, new Error(String(error)));
      }
      await this.handleSessionError(error, session, worker);
      return;
    }

    let lastCwd: string | undefined;

    try {
      for await (const message of this.sessionManager.getMessageIterator(session.sessionDbId)) {
        lastCwd = await this.processOneMessage(
          session,
          message,
          lastCwd,
          commandPath,
          model,
          reasoningEffort,
          timeoutMs,
          worker,
          mode
        );
      }
    } catch (error: unknown) {
      if (error instanceof Error) {
        logger.error('SDK', 'Codex message processing failed', { sessionId: session.sessionDbId, model }, error);
      } else {
        logger.error('SDK', 'Codex message processing failed with non-Error', { sessionId: session.sessionDbId, model }, new Error(String(error)));
      }
      await this.handleSessionError(error, session, worker);
      return;
    }

    const sessionDuration = Date.now() - session.startTime;
    logger.success('SDK', 'Codex agent completed', {
      sessionId: session.sessionDbId,
      duration: `${(sessionDuration / 1000).toFixed(1)}s`,
      historyLength: session.conversationHistory.length,
      model
    });
  }

  private prepareMessageMetadata(session: ActiveSession, message: { agentId?: string | null; agentType?: string | null }): void {
    session.pendingAgentId = message.agentId ?? null;
    session.pendingAgentType = message.agentType ?? null;
  }

  private async handleInitResponse(
    initResponse: CodexExecResult,
    session: ActiveSession,
    worker: WorkerRef | undefined,
    model: string
  ): Promise<void> {
    if (initResponse.content) {
      session.conversationHistory.push({ role: 'assistant', content: initResponse.content });
      const tokensUsed = initResponse.tokensUsed || 0;
      session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
      session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);

      await processAgentResponse(
        initResponse.content, session, this.dbManager, this.sessionManager,
        worker, tokensUsed, null, 'Codex', undefined, model
      );
    } else {
      logger.debug('SDK', 'Empty Codex init response - continuing with observation queue', {
        sessionId: session.sessionDbId, model
      });
    }
  }

  private async processOneMessage(
    session: ActiveSession,
    message: { _persistentId: number; agentId?: string | null; agentType?: string | null; type: 'observation' | 'summarize'; cwd?: string; prompt_number?: number; tool_name?: string; tool_input?: unknown; tool_response?: unknown; last_assistant_message?: string },
    lastCwd: string | undefined,
    commandPath: string,
    model: string,
    reasoningEffort: CodexReasoningEffort,
    timeoutMs: number,
    worker: WorkerRef | undefined,
    mode: ModeConfig
  ): Promise<string | undefined> {
    this.prepareMessageMetadata(session, message);

    if (message.cwd) {
      lastCwd = message.cwd;
    }
    const originalTimestamp = session.earliestPendingTimestamp;

    if (message.type === 'observation') {
      await this.processObservationMessage(
        session, message, originalTimestamp, lastCwd,
        commandPath, model, reasoningEffort, timeoutMs, worker
      );
    } else if (message.type === 'summarize') {
      await this.processSummaryMessage(
        session, message, originalTimestamp, lastCwd,
        commandPath, model, reasoningEffort, timeoutMs, worker, mode
      );
    }

    return lastCwd;
  }

  private async processObservationMessage(
    session: ActiveSession,
    message: { prompt_number?: number; tool_name?: string; tool_input?: unknown; tool_response?: unknown; cwd?: string },
    originalTimestamp: number | null,
    lastCwd: string | undefined,
    commandPath: string,
    model: string,
    reasoningEffort: CodexReasoningEffort,
    timeoutMs: number,
    worker: WorkerRef | undefined
  ): Promise<void> {
    if (message.prompt_number !== undefined) {
      session.lastPromptNumber = message.prompt_number;
    }

    if (!session.memorySessionId) {
      throw new Error('Cannot process observations: memorySessionId not yet captured. This session may need to be reinitialized.');
    }

    const obsPrompt = buildObservationPrompt({
      id: 0,
      tool_name: message.tool_name!,
      tool_input: JSON.stringify(message.tool_input),
      tool_output: JSON.stringify(message.tool_response),
      created_at_epoch: originalTimestamp ?? Date.now(),
      cwd: message.cwd
    });

    session.conversationHistory.push({ role: 'user', content: obsPrompt });
    const obsResponse = await this.queryCodexMultiTurn(
      session.conversationHistory,
      commandPath,
      model,
      reasoningEffort,
      timeoutMs,
      session.abortController.signal
    );

    let tokensUsed = 0;
    if (obsResponse.content) {
      session.conversationHistory.push({ role: 'assistant', content: obsResponse.content });
      tokensUsed = obsResponse.tokensUsed || 0;
      session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
      session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);
    }

    await processAgentResponse(
      obsResponse.content || '', session, this.dbManager, this.sessionManager,
      worker, tokensUsed, originalTimestamp, 'Codex', lastCwd, model
    );
  }

  private async processSummaryMessage(
    session: ActiveSession,
    message: { last_assistant_message?: string },
    originalTimestamp: number | null,
    lastCwd: string | undefined,
    commandPath: string,
    model: string,
    reasoningEffort: CodexReasoningEffort,
    timeoutMs: number,
    worker: WorkerRef | undefined,
    mode: ModeConfig
  ): Promise<void> {
    if (!session.memorySessionId) {
      throw new Error('Cannot process summary: memorySessionId not yet captured. This session may need to be reinitialized.');
    }

    const summaryPrompt = buildSummaryPrompt({
      id: session.sessionDbId,
      memory_session_id: session.memorySessionId,
      project: session.project,
      user_prompt: session.userPrompt,
      last_assistant_message: message.last_assistant_message || ''
    }, mode);

    session.conversationHistory.push({ role: 'user', content: summaryPrompt });
    const summaryResponse = await this.queryCodexMultiTurn(
      session.conversationHistory,
      commandPath,
      model,
      reasoningEffort,
      timeoutMs,
      session.abortController.signal
    );

    let tokensUsed = 0;
    if (summaryResponse.content) {
      session.conversationHistory.push({ role: 'assistant', content: summaryResponse.content });
      tokensUsed = summaryResponse.tokensUsed || 0;
      session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
      session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);
    }

    await processAgentResponse(
      summaryResponse.content || '', session, this.dbManager, this.sessionManager,
      worker, tokensUsed, originalTimestamp, 'Codex', lastCwd, model
    );
  }

  private async handleSessionError(error: unknown, session: ActiveSession, _worker?: WorkerRef): Promise<never> {
    if (isAbortError(error)) {
      logger.warn('SDK', 'Codex agent aborted', { sessionId: session.sessionDbId });
      throw error;
    }

    logger.failure('SDK', 'Codex agent error', { sessionDbId: session.sessionDbId }, error instanceof Error ? error : new Error(String(error)));
    throw error;
  }

  private estimateTokens(text: string): number {
    return Math.ceil(text.length / CHARS_PER_TOKEN_ESTIMATE);
  }

  private truncateHistory(history: ConversationMessage[]): ConversationMessage[] {
    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);

    const maxContextMessages = parseInt(settings.CLAUDE_MEM_CODEX_MAX_CONTEXT_MESSAGES, 10) || DEFAULT_MAX_CONTEXT_MESSAGES;
    const maxEstimatedTokens = parseInt(settings.CLAUDE_MEM_CODEX_MAX_TOKENS, 10) || DEFAULT_MAX_ESTIMATED_TOKENS;

    if (history.length <= maxContextMessages) {
      const totalTokens = history.reduce((sum, m) => sum + this.estimateTokens(m.content), 0);
      if (totalTokens <= maxEstimatedTokens) {
        return history;
      }
    }

    const truncated: ConversationMessage[] = [];
    let tokenCount = 0;

    for (let i = history.length - 1; i >= 0; i--) {
      const msg = history[i];
      const msgTokens = this.estimateTokens(msg.content);

      if (truncated.length >= maxContextMessages || tokenCount + msgTokens > maxEstimatedTokens) {
        logger.warn('SDK', 'Codex context window truncated to prevent runaway CLI usage', {
          originalMessages: history.length,
          keptMessages: truncated.length,
          droppedMessages: i + 1,
          estimatedTokens: tokenCount,
          tokenLimit: maxEstimatedTokens
        });
        break;
      }

      truncated.unshift(msg);
      tokenCount += msgTokens;
    }

    return truncated;
  }

  private conversationToPrompt(history: ConversationMessage[]): string {
    const parts = [
      'You are the claude-mem memory compression worker.',
      'Use only the conversation text below. Do not inspect files, run commands, browse, or modify anything.',
      'Return only the assistant response requested by the latest user message. Do not add commentary outside the requested XML or text.',
      '',
      'Conversation history follows.'
    ];

    history.forEach((msg, index) => {
      parts.push(
        '',
        `--- BEGIN ${msg.role.toUpperCase()} MESSAGE ${index + 1} ---`,
        msg.content,
        `--- END ${msg.role.toUpperCase()} MESSAGE ${index + 1} ---`
      );
    });

    return parts.join('\n');
  }

  private async queryCodexMultiTurn(
    history: ConversationMessage[],
    commandPath: string,
    model: string,
    reasoningEffort: CodexReasoningEffort,
    timeoutMs: number,
    abortSignal?: AbortSignal
  ): Promise<CodexExecResult> {
    const truncatedHistory = this.truncateHistory(history);
    const prompt = this.conversationToPrompt(truncatedHistory);
    const totalChars = truncatedHistory.reduce((sum, m) => sum + m.content.length, 0);
    const estimatedInputTokens = this.estimateTokens(prompt);

    logger.debug('SDK', `Querying Codex CLI (${model})`, {
      turns: truncatedHistory.length,
      totalChars,
      estimatedTokens: estimatedInputTokens,
      reasoningEffort,
      timeoutMs
    });

    const execResult = await this.runCodexExec(commandPath, model, reasoningEffort, prompt, timeoutMs, abortSignal);
    const content = execResult.content;
    if (!content) {
      logger.debug('SDK', 'Empty response from Codex CLI');
      return { content: '' };
    }

    const outputTokens = this.estimateTokens(content);
    const tokensUsed = execResult.tokensUsed ?? (estimatedInputTokens + outputTokens);

    logger.info('SDK', execResult.tokensUsed ? 'Codex CLI usage' : 'Codex CLI usage estimate', {
      model,
      reasoningEffort,
      ...(execResult.tokensUsed ? {} : { inputTokens: estimatedInputTokens, outputTokens }),
      totalTokens: tokensUsed,
      messagesInContext: truncatedHistory.length
    });

    return { content, tokensUsed };
  }

  private async runCodexExec(
    commandPath: string,
    model: string,
    reasoningEffort: CodexReasoningEffort,
    prompt: string,
    timeoutMs: number,
    abortSignal?: AbortSignal
  ): Promise<CodexExecResult> {
    const tempDir = await mkdtemp(join(tmpdir(), 'claude-mem-codex-'));
    const outputPath = join(tempDir, 'last-message.txt');
    let timedOut = false;
    let aborted = false;
    let stdout = '';
    let stderr = '';
    let killTimer: ReturnType<typeof setTimeout> | null = null;

    try {
      const command = this.normalizeCodexCommand(commandPath);
      const child = spawn(command, [
        'exec',
        '--json',
        '--ephemeral',
        '--ignore-user-config',
        '--ignore-rules',
        '--skip-git-repo-check',
        '--sandbox',
        'read-only',
        '--model',
        model,
        '-c',
        `model_reasoning_effort="${reasoningEffort}"`,
        '--output-last-message',
        outputPath,
        '-'
      ], {
        env: process.env,
        shell: process.platform === 'win32' && /\.(cmd|bat)$/i.test(command),
        stdio: ['pipe', 'pipe', 'pipe'],
        windowsHide: true
      });

      const timeout = setTimeout(() => {
        timedOut = true;
        child.kill('SIGTERM');
        killTimer = setTimeout(() => child.kill('SIGKILL'), 5000);
      }, timeoutMs);

      const abortHandler = () => {
        aborted = true;
        child.kill('SIGTERM');
      };
      abortSignal?.addEventListener('abort', abortHandler, { once: true });

      child.stdout.on('data', (chunk: Buffer) => {
        stdout = this.appendProcessOutput(stdout, chunk);
      });
      child.stderr.on('data', (chunk: Buffer) => {
        stderr = this.appendProcessOutput(stderr, chunk);
      });

      const exit = await new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((resolve, reject) => {
        child.on('error', reject);
        child.on('close', (code, signal) => resolve({ code, signal }));
        child.stdin.end(prompt);
      });

      clearTimeout(timeout);
      if (killTimer) clearTimeout(killTimer);
      abortSignal?.removeEventListener('abort', abortHandler);

      if (aborted) {
        throw this.createAbortError();
      }
      if (timedOut) {
        throw new Error(`Codex CLI timed out after ${timeoutMs}ms`);
      }
      if (exit.code !== 0) {
        throw new Error(`Codex CLI exited with code ${exit.code ?? 'unknown'}${exit.signal ? ` (signal ${exit.signal})` : ''}. ${this.describeProcessOutput(stdout, stderr)}`);
      }

      const jsonResult = this.extractCodexJsonResult(stdout);
      const outputFileContent = await this.readOptionalFile(outputPath);
      const content = outputFileContent || jsonResult.content;
      if (!content && !jsonResult.sawAgentMessage) {
        throw new Error(`Codex CLI completed without writing a final message. ${this.describeProcessOutput(stdout, stderr)}`);
      }

      return {
        content,
        tokensUsed: jsonResult.tokensUsed,
        sawAgentMessage: jsonResult.sawAgentMessage || outputFileContent.length > 0
      };
    } finally {
      await rm(tempDir, { recursive: true, force: true });
    }
  }

  private appendProcessOutput(current: string, chunk: Buffer): string {
    const next = current + chunk.toString('utf-8');
    return next.length > MAX_PROCESS_OUTPUT_CAPTURE
      ? next.slice(-MAX_PROCESS_OUTPUT_CAPTURE)
      : next;
  }

  private async readOptionalFile(filePath: string): Promise<string> {
    try {
      return (await readFile(filePath, 'utf-8')).trim();
    } catch {
      return '';
    }
  }

  private extractCodexJsonResult(stdout: string): CodexExecResult {
    let content = '';
    let tokensUsed: number | undefined;
    let sawAgentMessage = false;

    for (const line of stdout.split(/\r?\n/)) {
      if (!line.trim()) continue;

      let event: CodexJsonEvent;
      try {
        event = JSON.parse(line) as CodexJsonEvent;
      } catch {
        continue;
      }

      if (event.type === 'item.completed'
          && event.item?.type === 'agent_message'
          && typeof event.item.text === 'string') {
        sawAgentMessage = true;
        content = event.item.text.trim();
      }

      if (event.type === 'turn.completed' && event.usage) {
        const inputTokens = event.usage.input_tokens || 0;
        const outputTokens = event.usage.output_tokens || 0;
        tokensUsed = event.usage.total_tokens || inputTokens + outputTokens || undefined;
      }
    }

    return { content, tokensUsed, sawAgentMessage };
  }

  private describeProcessOutput(stdout: string, stderr: string): string {
    const stdoutLength = stdout.trim().length;
    const stderrLength = stderr.trim().length;
    if (stdoutLength === 0 && stderrLength === 0) {
      return 'No process output was captured.';
    }
    return `Captured stdout=${stdoutLength} chars, stderr=${stderrLength} chars; check worker logs for Codex CLI details.`;
  }

  private createAbortError(): Error {
    const error = new Error('Codex CLI aborted');
    error.name = 'AbortError';
    return error;
  }

  private normalizeCodexCommand(commandPath: string): string {
    return resolveCodexCommand(commandPath);
  }

  private getCodexConfig(): CodexConfig {
    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);

    const configuredTimeoutMs = parseInt(settings.CLAUDE_MEM_CODEX_TIMEOUT_MS, 10);
    const timeoutMs = Number.isFinite(configuredTimeoutMs) && configuredTimeoutMs >= 10000
      ? configuredTimeoutMs
      : DEFAULT_TIMEOUT_MS;

    const configuredReasoningEffort = settings.CLAUDE_MEM_CODEX_REASONING_EFFORT;
    const validReasoningEfforts: CodexReasoningEffort[] = ['low', 'medium', 'high', 'xhigh'];
    const reasoningEffort = validReasoningEfforts.includes(configuredReasoningEffort as CodexReasoningEffort)
      ? configuredReasoningEffort as CodexReasoningEffort
      : 'low';

    return {
      commandPath: settings.CLAUDE_MEM_CODEX_PATH || DEFAULT_CODEX_PATH,
      model: settings.CLAUDE_MEM_CODEX_MODEL || DEFAULT_MODEL,
      reasoningEffort,
      timeoutMs
    };
  }
}

export function isCodexAvailable(): boolean {
  const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
  const commandPath = resolveCodexCommand(settings.CLAUDE_MEM_CODEX_PATH || DEFAULT_CODEX_PATH);
  const hasPathSeparator = commandPath.includes('\\') || commandPath.includes('/');

  if (!hasPathSeparator && !path.isAbsolute(commandPath)) {
    return commandExistsOnPath(commandPath);
  }

  if (existsSync(commandPath)) {
    return true;
  }

  if (process.platform === 'win32' && !path.extname(commandPath)) {
    return existsSync(`${commandPath}.cmd`) || existsSync(`${commandPath}.exe`);
  }

  return false;
}

export function isCodexSelected(): boolean {
  const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
  return settings.CLAUDE_MEM_PROVIDER === 'codex';
}
