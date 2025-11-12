import { createClaudeCode } from 'ai-sdk-provider-claude-code';
import { generateText } from 'ai';
import { ToolSet } from '../../../mcp/types.js';
import { MCPManager } from '../../../mcp/manager.js';
import { UnifiedToolManager, CombinedToolSet } from '../../tools/unified-tool-manager.js';
import { ContextManager } from '../messages/manager.js';
import { ImageData } from '../messages/types.js';
import { ILLMService, LLMServiceConfig } from './types.js';
import { logger } from '../../../logger/index.js';
import { formatToolResult } from '../utils/tool-result-formatter.js';
import { EventManager } from '../../../events/event-manager.js';
import { SessionEvents } from '../../../events/event-types.js';
import { v4 as uuidv4 } from 'uuid';

export class ClaudeCodeService implements ILLMService {
	private provider: ReturnType<typeof createClaudeCode>;
	private model: string;
	private mcpManager: MCPManager;
	private unifiedToolManager: UnifiedToolManager | undefined;
	private contextManager: ContextManager;
	private maxIterations: number;
	private eventManager?: EventManager;

	constructor(
		provider: ReturnType<typeof createClaudeCode>,
		model: string,
		mcpManager: MCPManager,
		contextManager: ContextManager,
		maxIterations: number = 5,
		unifiedToolManager?: UnifiedToolManager
	) {
		this.provider = provider;
		this.model = model;
		this.mcpManager = mcpManager;
		this.contextManager = contextManager;
		this.maxIterations = maxIterations;
		this.unifiedToolManager = unifiedToolManager;
	}

	/**
	 * Set the event manager for emitting LLM response events
	 */
	setEventManager(eventManager: EventManager): void {
		this.eventManager = eventManager;
	}

	async generate(userInput: string, imageData?: ImageData, _stream?: boolean): Promise<string> {
		await this.contextManager.addUserMessage(userInput, imageData);

		const messageId = uuidv4();
		const startTime = Date.now();

		// Try to get sessionId from contextManager if available, otherwise undefined
		const sessionId = (this.contextManager as any)?.sessionId;

		// Emit LLM response started event
		if (this.eventManager && sessionId) {
			this.eventManager.emitSessionEvent(sessionId, SessionEvents.LLM_RESPONSE_STARTED, {
				sessionId,
				messageId,
				model: this.model,
				timestamp: startTime,
			});
		}

		// Use unified tool manager if available, otherwise fall back to MCP manager
		let formattedTools: any;
		if (this.unifiedToolManager) {
			formattedTools = await this.unifiedToolManager.getToolsForProvider('anthropic');
		} else {
			const rawTools = await this.mcpManager.getAllTools();
			formattedTools = this.formatToolsForAISDK(rawTools);
		}

		logger.silly(`Formatted tools for AI SDK: ${JSON.stringify(formattedTools, null, 2)}`);

		let iterationCount = 0;
		try {
			while (iterationCount < this.maxIterations) {
				iterationCount++;

				// Attempt to get a response, with retry logic
				const { response, toolCalls } = await this.getAIResponseWithRetries(formattedTools, userInput);

				// Extract text content
				const textContent = response;

				// If there are no tool calls, we're done
				if (!toolCalls || toolCalls.length === 0) {
					// Add assistant message to history
					await this.contextManager.addAssistantMessage(textContent);

					// Emit LLM response completed event
					if (this.eventManager && sessionId) {
						this.eventManager.emitSessionEvent(sessionId, SessionEvents.LLM_RESPONSE_COMPLETED, {
							sessionId,
							messageId,
							model: this.model,
							duration: Date.now() - startTime,
							timestamp: Date.now(),
							response: textContent,
						});
					}

					return textContent;
				}

				// Log thinking steps when assistant provides reasoning before tool calls
				if (textContent && textContent.trim()) {
					logger.info(`💭 ${textContent.trim()}`);

					// Emit thinking event
					if (this.eventManager && sessionId) {
						this.eventManager.emitSessionEvent(sessionId, SessionEvents.LLM_THINKING, {
							sessionId,
							messageId,
							timestamp: Date.now(),
						});
					}
				}

				// Transform tool calls into the format expected by ContextManager
				const formattedToolCalls = toolCalls.map((toolCall: any) => ({
					id: toolCall.toolCallId,
					type: 'function' as const,
					function: {
						name: toolCall.toolName,
						arguments: JSON.stringify(toolCall.args),
					},
				}));

				// Add assistant message with tool calls to history
				await this.contextManager.addAssistantMessage(textContent, formattedToolCalls);

				// Handle tool calls
				for (const toolCall of toolCalls) {
					logger.debug(`Tool call initiated: ${JSON.stringify(toolCall, null, 2)}`);
					logger.info(`🔧 Using tool: ${toolCall.toolName}`);
					const toolName = toolCall.toolName;
					const args = toolCall.args;
					const toolCallId = toolCall.toolCallId;

					// Execute tool
					try {
						let result: any;
						if (this.unifiedToolManager) {
							result = await this.unifiedToolManager.executeTool(toolName, args, sessionId);
						} else {
							result = await this.mcpManager.executeTool(toolName, args);
						}

						// Display formatted tool result
						const formattedResult = formatToolResult(toolName, result);
						logger.info(`📋 Tool Result:\n${formattedResult}`);

						// Add tool result to message manager
						await this.contextManager.addToolResult(toolCallId, toolName, result);
					} catch (error) {
						// Handle tool execution error
						const errorMessage = error instanceof Error ? error.message : String(error);
						logger.error(`Tool execution error for ${toolName}: ${errorMessage}`);

						// Add error as tool result
						await this.contextManager.addToolResult(toolCallId, toolName, {
							error: errorMessage,
						});
					}
				}
			}

			// If we reached max iterations, return a message
			logger.warn(`Reached maximum iterations (${this.maxIterations}) for task.`);
			const finalResponse = 'Task completed but reached maximum tool call iterations.';
			await this.contextManager.addAssistantMessage(finalResponse);
			return finalResponse;
		} catch (error) {
			// Handle API errors
			const errorMessage = error instanceof Error ? error.message : String(error);

			// Emit LLM response error event
			if (this.eventManager && sessionId) {
				this.eventManager.emitSessionEvent(sessionId, SessionEvents.LLM_RESPONSE_ERROR, {
					sessionId,
					messageId,
					model: this.model,
					error: errorMessage,
					timestamp: Date.now(),
				});
			}
			logger.error(`Error in Claude Code service API call: ${errorMessage}`, { error });
			await this.contextManager.addAssistantMessage(`Error processing request: ${errorMessage}`);
			return `Error processing request: ${errorMessage}`;
		}
	}

	/**
	 * Direct generate method that bypasses conversation context
	 * Used for internal tool operations that shouldn't pollute conversation history
	 * @param userInput - The input to generate a response for
	 * @param systemPrompt - Optional system prompt to use
	 * @returns Promise<string> - The generated response
	 */
	async directGenerate(userInput: string, systemPrompt?: string): Promise<string> {
		let attempts = 0;
		const MAX_ATTEMPTS = 3;

		logger.debug('ClaudeCodeService: Direct generate call (bypassing conversation context)', {
			inputLength: userInput.length,
			hasSystemPrompt: !!systemPrompt,
		});

		while (attempts < MAX_ATTEMPTS) {
			attempts++;
			try {
				// Create a minimal message array for direct API call
				const messages = [
					{
						role: 'user' as const,
						content: userInput,
					},
				];

				// Make direct API call without adding to conversation context
				const result = await generateText({
					model: this.provider(this.model),
					messages: messages,
					...(systemPrompt && { system: systemPrompt }),
					// No tools for direct calls - this is for simple text generation
				});

				const textContent = result.text;

				logger.debug('ClaudeCodeService: Direct generate completed', {
					responseLength: textContent.length,
				});

				return textContent;
			} catch (error) {
				const apiError = error as any;
				const errorMessage = apiError.message || 'Unknown error';

				logger.error(
					`Error in Claude Code direct generate call (Attempt ${attempts}/${MAX_ATTEMPTS}): ${errorMessage}`,
					{
						attempt: attempts,
						maxAttempts: MAX_ATTEMPTS,
						inputLength: userInput.length,
					}
				);

				// Check if this is a retryable error
				const isRetryable = this.isRetryableError(apiError);

				if (attempts >= MAX_ATTEMPTS || !isRetryable) {
					if (!isRetryable) {
						logger.error(`Non-retryable error in direct generate: ${errorMessage}`);
					} else {
						logger.error(`Failed direct generate after ${MAX_ATTEMPTS} attempts.`);
					}
					throw new Error(`Direct generate failed: ${errorMessage}`);
				}

				// Calculate delay with exponential backoff and jitter
				const delay = this.calculateRetryDelay(attempts);
				logger.info(
					`Retrying direct generate in ${delay}ms... (Attempt ${attempts + 1}/${MAX_ATTEMPTS})`
				);

				await new Promise(resolve => setTimeout(resolve, delay));
			}
		}

		throw new Error('Direct generate failed after maximum retry attempts');
	}

	async getAllTools(): Promise<ToolSet | CombinedToolSet> {
		if (this.unifiedToolManager) {
			return await this.unifiedToolManager.getAllTools();
		}
		return this.mcpManager.getAllTools();
	}

	getConfig(): LLMServiceConfig {
		return {
			provider: 'claude-code',
			model: this.model,
		};
	}

	// Helper methods
	private async getAIResponseWithRetries(
		tools: any,
		_userInput: string
	): Promise<{ response: string; toolCalls: any[] | undefined }> {
		let attempts = 0;
		const MAX_ATTEMPTS = 3;

		logger.debug(`Tools in response: ${Object.keys(tools).length}`);

		while (attempts < MAX_ATTEMPTS) {
			attempts++;
			try {
				// Get all formatted messages from conversation history
				const formattedMessages = await this.contextManager.getAllFormattedMessages();

				// For AI SDK, we need to separate system messages from the messages array
				const systemMessage = formattedMessages.find(msg => msg.role === 'system');
				const nonSystemMessages = formattedMessages.filter(msg => msg.role !== 'system');

				// Convert messages to AI SDK format
				const aiSdkMessages = nonSystemMessages.map(msg => {
					if (msg.role === 'user') {
						// Handle user messages with possible images
						if (Array.isArray(msg.content)) {
							return {
								role: 'user' as const,
								content: msg.content.map((part: any) => {
									if (part.type === 'text') {
										return { type: 'text' as const, text: part.text };
									} else if (part.type === 'image_url') {
										// Convert image_url to AI SDK format
										return {
											type: 'image' as const,
											image: part.image_url.url,
										};
									}
									return part;
								}),
							};
						}
						return { role: 'user' as const, content: msg.content };
					} else if (msg.role === 'assistant') {
						// Handle assistant messages with possible tool calls
						if (msg.tool_calls && msg.tool_calls.length > 0) {
							return {
								role: 'assistant' as const,
								content: msg.content || '',
								toolInvocations: msg.tool_calls.map((tc: any) => ({
									state: 'result' as const,
									toolCallId: tc.id,
									toolName: tc.function.name,
									args: JSON.parse(tc.function.arguments),
									result: undefined, // Will be filled by tool results
								})),
							};
						}
						return { role: 'assistant' as const, content: msg.content || '' };
					} else if (msg.role === 'tool') {
						// Tool results are handled differently in AI SDK
						// They should be part of the toolInvocations in assistant messages
						return null; // Skip tool messages, they're merged into assistant messages
					}
					return msg;
				}).filter(msg => msg !== null);

				// Merge tool results into assistant messages with toolInvocations
				for (let i = 0; i < aiSdkMessages.length; i++) {
					const msg = aiSdkMessages[i];
					if (msg.role === 'assistant' && msg.toolInvocations) {
						// Find corresponding tool results
						for (const invocation of msg.toolInvocations) {
							// Look for tool result in original messages
							const toolResultMsg = nonSystemMessages.find(
								m => m.role === 'tool' && m.tool_call_id === invocation.toolCallId
							);
							if (toolResultMsg) {
								invocation.result = toolResultMsg.content;
							}
						}
					}
				}

				// Call AI SDK generateText
				const result = await generateText({
					model: this.provider(this.model),
					messages: aiSdkMessages as any,
					...(systemMessage && { system: systemMessage.content as string }),
					tools: attempts === 1 ? tools : undefined, // Only offer tools on first attempt
					maxSteps: 1, // Only one step, we handle iterations ourselves
				});

				logger.silly('CLAUDE CODE MESSAGE RESPONSE: ', JSON.stringify(result, null, 2));

				if (!result || !result.text) {
					throw new Error('Received empty response from Claude Code');
				}

				// Extract tool calls if any
				const toolCalls = result.toolCalls || [];

				return { response: result.text, toolCalls };
			} catch (error) {
				const apiError = error as any;
				const errorMessage = apiError.message || 'Unknown error';

				logger.error(
					`Error in Claude Code API call (Attempt ${attempts}/${MAX_ATTEMPTS}): ${errorMessage}`,
					{
						attempt: attempts,
						maxAttempts: MAX_ATTEMPTS,
					}
				);

				// Check if this is a retryable error
				const isRetryable = this.isRetryableError(apiError);

				if (attempts >= MAX_ATTEMPTS || !isRetryable) {
					if (!isRetryable) {
						logger.error(`Non-retryable error encountered: ${errorMessage}`);
					} else {
						logger.error(`Failed to get response from Claude Code after ${MAX_ATTEMPTS} attempts.`);
					}
					throw error;
				}

				// Calculate delay with exponential backoff and jitter
				const delay = this.calculateRetryDelay(attempts);
				logger.info(`Retrying in ${delay}ms... (Attempt ${attempts + 1}/${MAX_ATTEMPTS})`);

				await new Promise(resolve => setTimeout(resolve, delay));
			}
		}

		throw new Error('Failed to get response after maximum retry attempts');
	}

	/**
	 * Determines if an error is retryable
	 */
	private isRetryableError(error: any): boolean {
		const errorMessage = error.message || '';
		const errorCode = error.code || '';

		// Non-retryable errors
		if (errorMessage.includes('authentication') || errorMessage.includes('unauthorized')) {
			return false;
		}

		if (errorMessage.includes('invalid request') || errorMessage.includes('bad request')) {
			return false;
		}

		// Retryable errors: network errors, rate limits, server errors, etc.
		if (
			errorMessage.includes('network') ||
			errorMessage.includes('timeout') ||
			errorMessage.includes('rate limit') ||
			errorMessage.includes('overloaded') ||
			errorMessage.includes('503') ||
			errorMessage.includes('502') ||
			errorMessage.includes('500') ||
			errorCode === 'ECONNRESET' ||
			errorCode === 'ETIMEDOUT'
		) {
			return true;
		}

		return true; // Default to retryable for unknown errors
	}

	/**
	 * Calculates retry delay with exponential backoff and jitter
	 */
	private calculateRetryDelay(attempt: number): number {
		const baseDelay = 1000; // Base delay of 1 second

		// Exponential backoff: 2^attempt * baseDelay
		const exponentialDelay = Math.pow(2, attempt - 1) * baseDelay;

		// Add jitter (random factor between 0.5 and 1.5)
		const jitter = 0.5 + Math.random();
		const finalDelay = Math.min(exponentialDelay * jitter, 30000); // Cap at 30 seconds

		return Math.round(finalDelay);
	}

	/**
	 * Format tools from MCP format to AI SDK format
	 */
	private formatToolsForAISDK(tools: ToolSet): Record<string, any> {
		const aiSdkTools: Record<string, any> = {};

		Object.entries(tools).forEach(([toolName, tool]) => {
			const input_schema: { type: string; properties: any; required: string[] } = {
				type: 'object',
				properties: {},
				required: [],
			};

			// Map tool parameters to JSON Schema format
			if (tool.parameters) {
				// The actual parameters structure appears to be a JSON Schema object
				const jsonSchemaParams = tool.parameters as any;

				if (jsonSchemaParams.type === 'object' && jsonSchemaParams.properties) {
					input_schema.properties = jsonSchemaParams.properties;
					if (Array.isArray(jsonSchemaParams.required)) {
						input_schema.required = jsonSchemaParams.required;
					}
				} else {
					logger.warn(`Unexpected parameters format for tool ${toolName}:`, jsonSchemaParams);
				}
			} else {
				// Handle case where tool might have no parameters
				logger.debug(`Tool ${toolName} has no defined parameters.`);
			}

			aiSdkTools[toolName] = {
				description: tool.description,
				parameters: input_schema,
			};
		});

		return aiSdkTools;
	}
}
