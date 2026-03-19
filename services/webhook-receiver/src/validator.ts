/**
 * Input validation for email webhook payloads.
 */

export interface EmailWebhookPayload {
  id: string;
  sender: string;
  subject: string;
  body: string;
  recipients: string[];
  attachments?: string[];
  receivedAt?: string;
}

interface ValidationResult {
  valid: boolean;
  errors: string[];
}

export function validateEmailWebhook(payload: unknown): ValidationResult {
  const errors: string[] = [];

  if (!payload || typeof payload !== "object") {
    return { valid: false, errors: ["Payload must be a JSON object"] };
  }

  const p = payload as Record<string, unknown>;

  if (!p.id || typeof p.id !== "string") {
    errors.push("'id' is required and must be a string");
  }

  if (!p.sender || typeof p.sender !== "string") {
    errors.push("'sender' is required and must be a string");
  }

  if (!p.subject || typeof p.subject !== "string") {
    errors.push("'subject' is required and must be a string");
  }

  if (!p.body || typeof p.body !== "string") {
    errors.push("'body' is required and must be a string");
  }

  if (!Array.isArray(p.recipients)) {
    errors.push("'recipients' is required and must be an array");
  }

  return { valid: errors.length === 0, errors };
}
