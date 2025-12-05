import { Anthropic } from "@anthropic-ai/sdk"
import OpenAI from "openai"

import { clearReasoningContentFromMessages } from "./clear-reasoning-content"

type ContentPartText = OpenAI.Chat.ChatCompletionContentPartText
type ContentPartImage = OpenAI.Chat.ChatCompletionContentPartImage
type UserMessage = OpenAI.Chat.ChatCompletionUserMessageParam
type AssistantMessage = OpenAI.Chat.ChatCompletionAssistantMessageParam
type Message = OpenAI.Chat.ChatCompletionMessageParam
type AnthropicMessage = Anthropic.Messages.MessageParam

/**
 * Converts Anthropic messages to OpenAI format while merging consecutive messages with the same role.
 * This is required for DeepSeek Reasoner which does not support successive messages with the same role.
 * 
 * According to DeepSeek API documentation, `reasoning_content` from previous turns should NOT be
 * included when sending messages for new turns. This function automatically clears reasoning_content
 * from assistant messages before conversion to ensure compliance with DeepSeek API requirements.
 *
 * @param messages Array of Anthropic messages (may contain reasoning_content from previous turns)
 * @returns Array of OpenAI messages where consecutive messages with the same role are combined,
 *          and reasoning_content has been cleared from assistant messages
 */
export function convertToR1Format(messages: AnthropicMessage[]): Message[] {
	// Clear reasoning_content from assistant messages before conversion
	// This ensures compliance with DeepSeek API requirements that reasoning_content
	// from previous turns should not be sent in new conversation turns
	const cleanedMessages = clearReasoningContentFromMessages(messages)

	return cleanedMessages.reduce<Message[]>((merged, message) => {
		const lastMessage = merged[merged.length - 1]
		let messageContent: string | (ContentPartText | ContentPartImage)[] = ""
		let hasImages = false

		// Convert content to appropriate format
		if (Array.isArray(message.content)) {
			const textParts: string[] = []
			const imageParts: ContentPartImage[] = []

			message.content.forEach((part) => {
				if (part.type === "text") {
					textParts.push(part.text)
				}
				if (part.type === "image") {
					hasImages = true
					imageParts.push({
						type: "image_url",
						image_url: { url: `data:${part.source.media_type};base64,${part.source.data}` },
					})
				}
			})

			if (hasImages) {
				const parts: (ContentPartText | ContentPartImage)[] = []
				if (textParts.length > 0) {
					parts.push({ type: "text", text: textParts.join("\n") })
				}
				parts.push(...imageParts)
				messageContent = parts
			} else {
				messageContent = textParts.join("\n")
			}
		} else {
			messageContent = message.content
		}

		// If last message has same role, merge the content
		if (lastMessage?.role === message.role) {
			if (typeof lastMessage.content === "string" && typeof messageContent === "string") {
				lastMessage.content += `\n${messageContent}`
			}
			// If either has image content, convert both to array format
			else {
				const lastContent = Array.isArray(lastMessage.content)
					? lastMessage.content
					: [{ type: "text" as const, text: lastMessage.content || "" }]

				const newContent = Array.isArray(messageContent)
					? messageContent
					: [{ type: "text" as const, text: messageContent }]

				if (message.role === "assistant") {
					const mergedContent = [...lastContent, ...newContent] as AssistantMessage["content"]
					lastMessage.content = mergedContent
				} else {
					const mergedContent = [...lastContent, ...newContent] as UserMessage["content"]
					lastMessage.content = mergedContent
				}
			}
		} else {
			// Add as new message with the correct type based on role
			if (message.role === "assistant") {
				const newMessage: AssistantMessage = {
					role: "assistant",
					content: messageContent as AssistantMessage["content"],
				}
				merged.push(newMessage)
			} else {
				const newMessage: UserMessage = {
					role: "user",
					content: messageContent as UserMessage["content"],
				}
				merged.push(newMessage)
			}
		}

		return merged
	}, [])
}
