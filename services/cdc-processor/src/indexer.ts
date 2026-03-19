/**
 * Search index operations for CDC events.
 *
 * Handles INSERT/MODIFY/REMOVE events by updating the
 * vector search index (Pinecone) accordingly.
 */

import { EmailRecord } from "./types";

export interface CDCEvent {
  eventType: "INSERT" | "MODIFY" | "REMOVE";
  recordId: string;
  tenantId: string;
  newRecord?: EmailRecord;
  oldRecord?: EmailRecord;
  timestamp: number;
}

// Track processed event IDs for idempotency
const processedEvents = new Set<string>();

export async function processEvent(event: CDCEvent): Promise<void> {
  // Idempotency check
  const eventKey = `${event.tenantId}:${event.recordId}:${event.timestamp}`;
  if (processedEvents.has(eventKey)) {
    console.log(`Skipping duplicate event: ${eventKey}`);
    return;
  }

  switch (event.eventType) {
    case "INSERT":
      await handleInsert(event);
      break;
    case "MODIFY":
      await handleModify(event);
      break;
    case "REMOVE":
      await handleRemove(event);
      break;
    default:
      console.warn(`Unknown event type: ${event.eventType}`);
  }

  processedEvents.add(eventKey);

  // Prevent memory leak — keep last 10K events
  if (processedEvents.size > 10000) {
    const iterator = processedEvents.values();
    for (let i = 0; i < 5000; i++) {
      processedEvents.delete(iterator.next().value!);
    }
  }
}

async function handleInsert(event: CDCEvent): Promise<void> {
  const record = event.newRecord;
  if (!record) {
    console.warn("INSERT event missing newRecord");
    return;
  }

  console.log(
    `Indexing new email: ${record.id} for tenant: ${event.tenantId}`
  );

  // In production: generate embedding and upsert to Pinecone
  // const embedding = await generateEmbedding(record.subject + " " + record.body);
  // await pinecone.upsert({
  //   vectors: [{
  //     id: record.id,
  //     values: embedding,
  //     metadata: {
  //       tenant_id: event.tenantId,
  //       subject: record.subject,
  //       sender: record.sender,
  //       category: record.category,
  //       project_id: record.projectId,
  //     },
  //   }],
  //   namespace: event.tenantId,
  // });

  console.log(`Indexed email ${record.id} to namespace ${event.tenantId}`);
}

async function handleModify(event: CDCEvent): Promise<void> {
  const newRecord = event.newRecord;
  const oldRecord = event.oldRecord;
  if (!newRecord) {
    console.warn("MODIFY event missing newRecord");
    return;
  }

  // Check if searchable fields changed
  const fieldsChanged =
    newRecord.subject !== oldRecord?.subject ||
    newRecord.body !== oldRecord?.body ||
    newRecord.category !== oldRecord?.category ||
    newRecord.projectId !== oldRecord?.projectId;

  if (!fieldsChanged) {
    console.log(`No searchable field changes for ${newRecord.id}, skipping reindex`);
    return;
  }

  console.log(`Reindexing modified email: ${newRecord.id}`);

  // In production: re-embed and update in Pinecone
  // Same as INSERT but with updated content

  console.log(`Reindexed email ${newRecord.id}`);
}

async function handleRemove(event: CDCEvent): Promise<void> {
  console.log(
    `Removing email ${event.recordId} from index (tenant: ${event.tenantId})`
  );

  // In production: delete from Pinecone
  // await pinecone.delete({
  //   ids: [event.recordId],
  //   namespace: event.tenantId,
  // });

  console.log(`Removed email ${event.recordId} from namespace ${event.tenantId}`);
}
