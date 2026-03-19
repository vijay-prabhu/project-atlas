/**
 * Shared type definitions for the CDC processor.
 */

export interface EmailRecord {
  id: string;
  subject: string;
  body: string;
  sender: string;
  tenantId: string;
  category: string;
  projectId: string;
  receivedAt: string;
}

export interface IndexDocument {
  id: string;
  embedding: number[];
  metadata: {
    tenant_id: string;
    subject: string;
    sender: string;
    category: string;
    project_id: string;
    received_at: string;
  };
}
