/**
 * CDC Processor Lambda — handles DynamoDB Streams events.
 *
 * When an email is inserted, modified, or removed from the EmailsTable,
 * this Lambda processes the change and updates the search index.
 *
 * Event types:
 * - INSERT: New email → generate embeddings → index in Pinecone
 * - MODIFY: Email updated → re-embed changed fields → update index
 * - REMOVE: Email deleted → remove vectors from index
 */

import { DynamoDBStreamEvent, DynamoDBRecord } from "aws-lambda";
import { CDCEvent, processEvent } from "./indexer";
import { EmailRecord } from "./types";

export const handler = async (event: DynamoDBStreamEvent): Promise<void> => {
  console.log(
    `Processing ${event.Records.length} DynamoDB stream records`
  );

  const processedIds: string[] = [];
  const errors: Array<{ recordId: string; error: string }> = [];

  for (const record of event.Records) {
    try {
      const cdcEvent = parseDynamoDBRecord(record);
      if (cdcEvent) {
        await processEvent(cdcEvent);
        processedIds.push(cdcEvent.recordId);
      }
    } catch (error) {
      const recordId = record.dynamodb?.Keys?.pk?.S ?? "unknown";
      console.error(`Error processing record ${recordId}:`, error);
      errors.push({
        recordId,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  console.log(
    `Processed: ${processedIds.length}, Errors: ${errors.length}`
  );

  if (errors.length > 0) {
    console.error("Failed records:", JSON.stringify(errors));
  }
};

function parseDynamoDBRecord(record: DynamoDBRecord): CDCEvent | null {
  const eventName = record.eventName;
  if (!eventName) return null;

  const newImage = record.dynamodb?.NewImage;
  const oldImage = record.dynamodb?.OldImage;

  // Extract tenant and email IDs from the key
  const pk = record.dynamodb?.Keys?.pk?.S ?? "";
  const sk = record.dynamodb?.Keys?.sk?.S ?? "";

  // Key format: pk=TENANT#tenant_id, sk=EMAIL#email_id
  const tenantId = pk.replace("TENANT#", "");
  const emailId = sk.replace("EMAIL#", "");

  if (!tenantId || !emailId) {
    console.warn("Skipping record with invalid key format:", { pk, sk });
    return null;
  }

  return {
    eventType: eventName as "INSERT" | "MODIFY" | "REMOVE",
    recordId: emailId,
    tenantId,
    newRecord: newImage ? unmarshallEmail(newImage) : undefined,
    oldRecord: oldImage ? unmarshallEmail(oldImage) : undefined,
    timestamp: record.dynamodb?.ApproximateCreationDateTime ?? Date.now() / 1000,
  };
}

function unmarshallEmail(image: Record<string, any>): EmailRecord {
  return {
    id: image.email_id?.S ?? "",
    subject: image.subject?.S ?? "",
    body: image.body?.S ?? "",
    sender: image.sender?.S ?? "",
    tenantId: image.tenant_id?.S ?? "",
    category: image.category?.S ?? "",
    projectId: image.project_id?.S ?? "",
    receivedAt: image.received_at?.S ?? "",
  };
}
