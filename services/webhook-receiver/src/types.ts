/**
 * Shared type definitions for the webhook receiver.
 */

export interface QueuedEmail {
  tenantId: string;
  email: {
    id: string;
    sender: string;
    subject: string;
    body: string;
    recipients: string[];
    attachments?: string[];
  };
  receivedAt: string;
}
