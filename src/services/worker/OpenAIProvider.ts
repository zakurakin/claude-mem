import { buildContinuationPrompt, buildInitPrompt, buildObservationPrompt, buildSummaryPrompt } from '../../sdk/prompts.js';
import { getCredential } from '../../shared/EnvManager.js';
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

const OPENAI_DEFAULT_BASE_URL = 'https://api.openai.com/v1';
const DEFAULT_MAX_CONTEXT_MESSAGES = 20;
const DEFAULT_MAX_ESTIMATED_TOKENS = 100000;
const DEFAULT_MAX_OUTPUT_TOKENS = 4096;
const CHARS_PER_TOKEN_ESTIMATE = 4;

type OpenAIReasoningEffort = 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh';

interface OpenAIInputMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface OpenAIResponseTextPart {
  type?: string;
  text?: string | { value?: string };
}

interface OpenAIResponseOutputItem {
  type?: string;
  content?: OpenAIResponseTextPart[];
  text?: string;
}

interface OpenAIResponsesResponse {
  output_text?: string;
  output?: OpenAIResponseOutputItem[];
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    total_tokens?: number;
  };
  error?: {
    message?: string;
    type?: string;
    code?: string;
  };
}

export class OpenAIProvider {
  private dbManager: DatabaseManager;
  private sessionManager: SessionManager;

  constructor(dbManager: DatabaseManager, sessionManager: SessionManager) {
    this.dbManager = dbManager;
    this.sessionManager = sessionManager;
  }

  async startSession(session: ActiveSession, worker?: WorkerRef): Promise<void> {
    const { apiKey, model, baseUrl, maxOutputTokens, reasoningEffort } = this.getOpenAIConfig();

    if (!apiKey) {
      throw new Error('OpenAI API key not configured. Set CLAUDE_MEM_OPENAI_API_KEY in settings or OPENAI_API_KEY in ~/.claude-mem/.env.');
    }

    if (!session.memorySessionId) {
      const syntheticMemorySessionId = `openai-${session.contentSessionId}-${Date.now()}`;
      session.memorySessionId = syntheticMemorySessionId;
      this.dbManager.getSessionStore().updateMemorySessionId(session.sessionDbId, syntheticMemorySessionId);
      logger.info('SESSION', `MEMORY_ID_GENERATED | sessionDbId=${session.sessionDbId} | provider=OpenAI`);
    }

    const mode = ModeManager.getInstance().getActiveMode();

    const initPrompt = session.lastPromptNumber === 1
      ? buildInitPrompt(session.project, session.contentSessionId, session.userPrompt, mode)
      : buildContinuationPrompt(session.userPrompt, session.lastPromptNumber, session.contentSessionId, mode);

    session.conversationHistory.push({ role: 'user', content: initPrompt });

    try {
      const initResponse = await this.queryOpenAIMultiTurn(
        session.conversationHistory,
        apiKey,
        model,
        baseUrl,
        maxOutputTokens,
        reasoningEffort
      );
      await this.handleInitResponse(initResponse, session, worker, model);
    } catch (error: unknown) {
      if (error instanceof Error) {
        logger.error('SDK', 'OpenAI init failed', { sessionId: session.sessionDbId, model }, error);
      } else {
        logger.error('SDK', 'OpenAI init failed with non-Error', { sessionId: session.sessionDbId, model }, new Error(String(error)));
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
          apiKey,
          model,
          baseUrl,
          maxOutputTokens,
          reasoningEffort,
          worker,
          mode
        );
      }
    } catch (error: unknown) {
      if (error instanceof Error) {
        logger.error('SDK', 'OpenAI message processing failed', { sessionId: session.sessionDbId, model }, error);
      } else {
        logger.error('SDK', 'OpenAI message processing failed with non-Error', { sessionId: session.sessionDbId, model }, new Error(String(error)));
      }
      await this.handleSessionError(error, session, worker);
      return;
    }

    const sessionDuration = Date.now() - session.startTime;
    logger.success('SDK', 'OpenAI agent completed', {
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
    initResponse: { content: string; tokensUsed?: number },
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
        worker, tokensUsed, null, 'OpenAI', undefined, model
      );
    } else {
      logger.error('SDK', 'Empty OpenAI init response - session may lack context', {
        sessionId: session.sessionDbId, model
      });
    }
  }

  private async processOneMessage(
    session: ActiveSession,
    message: { _persistentId: number; agentId?: string | null; agentType?: string | null; type: 'observation' | 'summarize'; cwd?: string; prompt_number?: number; tool_name?: string; tool_input?: unknown; tool_response?: unknown; last_assistant_message?: string },
    lastCwd: string | undefined,
    apiKey: string,
    model: string,
    baseUrl: string,
    maxOutputTokens: number,
    reasoningEffort: OpenAIReasoningEffort | undefined,
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
        apiKey, model, baseUrl, maxOutputTokens, reasoningEffort, worker
      );
    } else if (message.type === 'summarize') {
      await this.processSummaryMessage(
        session, message, originalTimestamp, lastCwd,
        apiKey, model, baseUrl, maxOutputTokens, reasoningEffort, worker, mode
      );
    }

    return lastCwd;
  }

  private async processObservationMessage(
    session: ActiveSession,
    message: { prompt_number?: number; tool_name?: string; tool_input?: unknown; tool_response?: unknown; cwd?: string },
    originalTimestamp: number | null,
    lastCwd: string | undefined,
    apiKey: string,
    model: string,
    baseUrl: string,
    maxOutputTokens: number,
    reasoningEffort: OpenAIReasoningEffort | undefined,
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
    const obsResponse = await this.queryOpenAIMultiTurn(
      session.conversationHistory,
      apiKey,
      model,
      baseUrl,
      maxOutputTokens,
      reasoningEffort
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
      worker, tokensUsed, originalTimestamp, 'OpenAI', lastCwd, model
    );
  }

  private async processSummaryMessage(
    session: ActiveSession,
    message: { last_assistant_message?: string },
    originalTimestamp: number | null,
    lastCwd: string | undefined,
    apiKey: string,
    model: string,
    baseUrl: string,
    maxOutputTokens: number,
    reasoningEffort: OpenAIReasoningEffort | undefined,
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
    const summaryResponse = await this.queryOpenAIMultiTurn(
      session.conversationHistory,
      apiKey,
      model,
      baseUrl,
      maxOutputTokens,
      reasoningEffort
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
      worker, tokensUsed, originalTimestamp, 'OpenAI', lastCwd, model
    );
  }

  private async handleSessionError(error: unknown, session: ActiveSession, _worker?: WorkerRef): Promise<never> {
    if (isAbortError(error)) {
      logger.warn('SDK', 'OpenAI agent aborted', { sessionId: session.sessionDbId });
      throw error;
    }

    logger.failure('SDK', 'OpenAI agent error', { sessionDbId: session.sessionDbId }, error instanceof Error ? error : new Error(String(error)));
    throw error;
  }

  private estimateTokens(text: string): number {
    return Math.ceil(text.length / CHARS_PER_TOKEN_ESTIMATE);
  }

  private truncateHistory(history: ConversationMessage[]): ConversationMessage[] {
    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);

    const maxContextMessages = parseInt(settings.CLAUDE_MEM_OPENAI_MAX_CONTEXT_MESSAGES, 10) || DEFAULT_MAX_CONTEXT_MESSAGES;
    const maxEstimatedTokens = parseInt(settings.CLAUDE_MEM_OPENAI_MAX_TOKENS, 10) || DEFAULT_MAX_ESTIMATED_TOKENS;

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
        logger.warn('SDK', 'OpenAI context window truncated to prevent runaway costs', {
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

  private conversationToOpenAIInput(history: ConversationMessage[]): OpenAIInputMessage[] {
    return history.map(msg => ({
      role: msg.role === 'assistant' ? 'assistant' : 'user',
      content: msg.content
    }));
  }

  private extractOutputText(data: OpenAIResponsesResponse): string {
    if (typeof data.output_text === 'string' && data.output_text.length > 0) {
      return data.output_text;
    }

    const parts: string[] = [];
    for (const item of data.output ?? []) {
      if (typeof item.text === 'string') {
        parts.push(item.text);
      }

      for (const contentPart of item.content ?? []) {
        const text = contentPart.text;
        if (typeof text === 'string') {
          parts.push(text);
        } else if (text && typeof text.value === 'string') {
          parts.push(text.value);
        }
      }
    }

    return parts.join('\n').trim();
  }

  private async queryOpenAIMultiTurn(
    history: ConversationMessage[],
    apiKey: string,
    model: string,
    baseUrl: string,
    maxOutputTokens: number,
    reasoningEffort?: OpenAIReasoningEffort
  ): Promise<{ content: string; tokensUsed?: number }> {
    const truncatedHistory = this.truncateHistory(history);
    const input = this.conversationToOpenAIInput(truncatedHistory);
    const totalChars = truncatedHistory.reduce((sum, m) => sum + m.content.length, 0);
    const estimatedTokens = this.estimateTokens(truncatedHistory.map(m => m.content).join(''));

    logger.debug('SDK', `Querying OpenAI Responses API (${model})`, {
      turns: truncatedHistory.length,
      totalChars,
      estimatedTokens,
      reasoningEffort: reasoningEffort || 'omitted'
    });

    const requestBody: Record<string, unknown> = {
      model,
      input,
      max_output_tokens: maxOutputTokens
    };

    if (reasoningEffort) {
      requestBody.reasoning = { effort: reasoningEffort };
    }

    const normalizedBaseUrl = baseUrl.replace(/\/+$/, '') || OPENAI_DEFAULT_BASE_URL;
    const response = await fetch(`${normalizedBaseUrl}/responses`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`OpenAI API error: ${response.status} - ${errorText}`);
    }

    const data = await response.json() as OpenAIResponsesResponse;

    if (data.error) {
      const code = data.error.code || data.error.type || 'unknown';
      throw new Error(`OpenAI API error: ${code} - ${data.error.message || 'Unknown error'}`);
    }

    const content = this.extractOutputText(data);
    if (!content) {
      logger.error('SDK', 'Empty response from OpenAI');
      return { content: '' };
    }

    const tokensUsed =
      data.usage?.total_tokens
      ?? (((data.usage?.input_tokens || 0) + (data.usage?.output_tokens || 0)) || undefined);

    if (tokensUsed) {
      logger.info('SDK', 'OpenAI API usage', {
        model,
        inputTokens: data.usage?.input_tokens || 0,
        outputTokens: data.usage?.output_tokens || 0,
        totalTokens: tokensUsed,
        messagesInContext: truncatedHistory.length
      });
    }

    return { content, tokensUsed };
  }

  private getOpenAIConfig(): {
    apiKey: string;
    model: string;
    baseUrl: string;
    maxOutputTokens: number;
    reasoningEffort?: OpenAIReasoningEffort;
  } {
    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);

    const apiKey = settings.CLAUDE_MEM_OPENAI_API_KEY || getCredential('OPENAI_API_KEY') || '';
    const model = settings.CLAUDE_MEM_OPENAI_MODEL || 'gpt-5.2-codex';
    const baseUrl = settings.CLAUDE_MEM_OPENAI_BASE_URL || OPENAI_DEFAULT_BASE_URL;

    const configuredOutputTokens = parseInt(settings.CLAUDE_MEM_OPENAI_MAX_OUTPUT_TOKENS, 10);
    const maxOutputTokens = Number.isFinite(configuredOutputTokens) && configuredOutputTokens > 0
      ? configuredOutputTokens
      : DEFAULT_MAX_OUTPUT_TOKENS;

    const configuredReasoningEffort = settings.CLAUDE_MEM_OPENAI_REASONING_EFFORT;
    const validReasoningEfforts: OpenAIReasoningEffort[] = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh'];
    const reasoningEffort = validReasoningEfforts.includes(configuredReasoningEffort as OpenAIReasoningEffort)
      ? configuredReasoningEffort as OpenAIReasoningEffort
      : undefined;

    return { apiKey, model, baseUrl, maxOutputTokens, reasoningEffort };
  }
}

export function isOpenAIAvailable(): boolean {
  const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
  return !!(settings.CLAUDE_MEM_OPENAI_API_KEY || getCredential('OPENAI_API_KEY'));
}

export function isOpenAISelected(): boolean {
  const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
  return settings.CLAUDE_MEM_PROVIDER === 'openai';
}
