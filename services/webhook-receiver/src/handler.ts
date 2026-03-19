/**
 * Webhook Receiver Lambda — accepts incoming email notifications
 * and queues them for processing via SQS.
 */

import { APIGatewayProxyEvent, APIGatewayProxyResult } from "aws-lambda";
import { SQSClient, SendMessageCommand } from "@aws-sdk/client-sqs";
import { validateEmailWebhook, EmailWebhookPayload } from "./validator";

const sqsClient = new SQSClient({});
const QUEUE_URL = process.env.QUEUE_URL ?? "";

export const handler = async (
  event: APIGatewayProxyEvent
): Promise<APIGatewayProxyResult> => {
  try {
    if (!event.body) {
      return response(400, { error: "Missing request body" });
    }

    const payload = JSON.parse(event.body);
    const validation = validateEmailWebhook(payload);

    if (!validation.valid) {
      return response(400, { error: "Validation failed", details: validation.errors });
    }

    const emailPayload = payload as EmailWebhookPayload;

    // Extract tenant from header
    const tenantId = event.headers["X-Tenant-ID"] ?? event.headers["x-tenant-id"];
    if (!tenantId) {
      return response(401, { error: "Missing X-Tenant-ID header" });
    }

    // Send to SQS for async processing
    await sqsClient.send(
      new SendMessageCommand({
        QueueUrl: QUEUE_URL,
        MessageBody: JSON.stringify({
          tenantId,
          email: emailPayload,
          receivedAt: new Date().toISOString(),
        }),
        MessageAttributes: {
          TenantId: {
            DataType: "String",
            StringValue: tenantId,
          },
        },
      })
    );

    return response(202, {
      status: "queued",
      emailId: emailPayload.id,
      tenantId,
    });
  } catch (error) {
    console.error("Webhook processing error:", error);
    return response(500, {
      error: "Internal server error",
    });
  }
};

function response(statusCode: number, body: object): APIGatewayProxyResult {
  return {
    statusCode,
    headers: {
      "Content-Type": "application/json",
      "X-Content-Type-Options": "nosniff",
    },
    body: JSON.stringify(body),
  };
}
