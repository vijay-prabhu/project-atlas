#!/usr/bin/env bash
set -e

LOG_DIR="/tmp/project-atlas-logs"
mkdir -p "$LOG_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$LOG_DIR/pids"

log() { echo -e "${BLUE}[atlas]${NC} $1"; }
ok()  { echo -e "${GREEN}[✓]${NC} $1"; }
err() { echo -e "${RED}[✗]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }

# Clean up old PIDs
> "$PID_FILE"

# ─── 1. Docker Compose (DynamoDB Local, DynamoDB Admin) ──────────────
log "Starting Docker Compose services..."
cd "$PROJECT_ROOT"
docker compose up -d > "$LOG_DIR/docker-compose.log" 2>&1

# Wait for DynamoDB Local
for i in $(seq 1 15); do
  if curl -sf http://localhost:8001 > /dev/null 2>&1; then
    ok "DynamoDB Local ready"
    break
  fi
  sleep 1
done

if ! curl -sf http://localhost:8001 > /dev/null 2>&1; then
  err "DynamoDB Local failed to start. Check $LOG_DIR/docker-compose.log"
  exit 1
fi

ok "DynamoDB Admin UI ready → http://localhost:8002"

# ─── 2. Create DynamoDB tables ──────────────────────────────────────
log "Creating DynamoDB tables..."
cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

python3 -c "
import boto3, sys

endpoint = 'http://localhost:8001'
client = boto3.client('dynamodb', endpoint_url=endpoint, region_name='ca-central-1',
                      aws_access_key_id='local', aws_secret_access_key='local')

tables = [
    ('atlas-emails', 'pk', 'sk'),
    ('atlas-projects', 'pk', 'sk'),
    ('atlas-checkpoints', 'pk', 'sk'),
    ('atlas-feedback', 'pk', 'sk'),
]

existing = client.list_tables()['TableNames']
for table_name, pk, sk in tables:
    if table_name in existing:
        continue
    client.create_table(
        TableName=table_name,
        KeySchema=[
            {'AttributeName': pk, 'KeyType': 'HASH'},
            {'AttributeName': sk, 'KeyType': 'RANGE'},
        ],
        AttributeDefinitions=[
            {'AttributeName': pk, 'AttributeType': 'S'},
            {'AttributeName': sk, 'AttributeType': 'S'},
        ],
        BillingMode='PAY_PER_REQUEST',
    )
    print(f'  Created table: {table_name}')

print(f'  Tables ready: {len(tables)}')
" 2>&1 | while read -r line; do echo -e "  ${GREEN}$line${NC}"; done

# ─── 3. Seed sample data into DynamoDB ──────────────────────────────
log "Seeding sample data..."
python3 -c "
import boto3, json, os

endpoint = 'http://localhost:8001'
dynamodb = boto3.resource('dynamodb', endpoint_url=endpoint, region_name='ca-central-1',
                          aws_access_key_id='local', aws_secret_access_key='local')

# Seed projects
projects_table = dynamodb.Table('atlas-projects')
data_dir = os.path.join('$PROJECT_ROOT', 'data')
with open(os.path.join(data_dir, 'sample_projects.json')) as f:
    data = json.load(f)

with projects_table.batch_writer() as batch:
    for project in data['projects']:
        batch.put_item(Item={
            'pk': 'TENANT#demo',
            'sk': f'PROJECT#{project[\"id\"]}',
            **project,
        })
    for rfi in data['rfis']:
        batch.put_item(Item={
            'pk': 'TENANT#demo',
            'sk': f'RFI#{rfi[\"id\"]}',
            **rfi,
        })

# Seed emails
emails_table = dynamodb.Table('atlas-emails')
email_count = 0
for filename in os.listdir(os.path.join(data_dir, 'sample_emails')):
    if not filename.endswith('.json'):
        continue
    with open(os.path.join(data_dir, 'sample_emails', filename)) as f:
        emails = json.load(f)
    with emails_table.batch_writer() as batch:
        for email in emails:
            batch.put_item(Item={
                'pk': 'TENANT#demo',
                'sk': f'EMAIL#{email[\"id\"]}',
                **{k: v for k, v in email.items() if v is not None},
            })
            email_count += 1

print(f'  Seeded {len(data[\"projects\"])} projects, {len(data[\"rfis\"])} RFIs, {email_count} emails')
" 2>&1 | while read -r line; do echo -e "  ${GREEN}$line${NC}"; done

# ─── 4. FastAPI Application (port 8000) ─────────────────────────────
log "Starting FastAPI server..."
cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload > "$LOG_DIR/api.log" 2>&1 &
echo "$!" >> "$PID_FILE"

for i in $(seq 1 15); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    ok "API ready → http://localhost:8000"
    break
  fi
  sleep 1
done

if ! curl -sf http://localhost:8000/health > /dev/null 2>&1; then
  err "API failed to start. Check $LOG_DIR/api.log"
  exit 1
fi

# ─── Summary ─────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN} Project Atlas — All services running${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  API Server        → http://localhost:8000"
echo "  API Docs          → http://localhost:8000/docs"
echo "  DynamoDB Local    → http://localhost:8001"
echo "  DynamoDB Admin    → http://localhost:8002"
echo ""
echo "  Logs:  $LOG_DIR/"
echo "    api.log, docker-compose.log"
echo ""
echo "  Test email filing:"
echo "    curl -X POST http://localhost:8000/api/v1/emails/file \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -H 'X-Tenant-ID: demo' \\"
echo "      -d '{\"email\": {\"sender\": \"s.chen@pacificsteel.com\", \"subject\": \"RE: RFI-247 - Steel Connection\", \"body\": \"Following up on RFI-247 regarding the connection detail.\"}}'"
echo ""
echo -e "  Stop all:  ${YELLOW}./dev-stop.sh${NC}"
echo ""
