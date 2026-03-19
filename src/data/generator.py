"""Scale data generator for AECO emails.

Generates realistic construction industry emails at scale using:
1. Faker for metadata (names, dates, companies, project codes)
2. Custom AECO provider for industry-specific terms
3. Template-based email body generation

Usage:
    # Generate 1,000 emails to JSON
    python -m src.data.generator --count 1000 --output data/generated_emails.json

    # Generate 100,000 emails and load into DynamoDB Local
    python -m src.data.generator --count 100000 --dynamodb --endpoint http://localhost:8001

    # Generate for specific tenant
    python -m src.data.generator --count 5000 --tenant tenant_a --output data/tenant_a_emails.json
"""

import argparse
import json
import random
import sys
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional

from faker import Faker
from faker.providers import BaseProvider

fake = Faker()


# ─── Custom AECO Faker Provider ──────────────────────────────────────


class AECOProvider(BaseProvider):
    """Custom Faker provider for AECO industry terms."""

    COMPANIES = [
        ("Pacific Steel Erectors", "PSE"),
        ("Vertex Waterproofing", "VW"),
        ("Morrison Engineering Group", "MEG"),
        ("Stronghold Contractors", "SC"),
        ("Bridge Builders Corp", "BBC"),
        ("Summit Mechanical", "SM"),
        ("Greenfield Landscape", "GL"),
        ("Industrial Construction Co", "ICC"),
        ("Thornton Architects", "TA"),
        ("Apex Electrical Systems", "AES"),
        ("Ironclad Structural", "IS"),
        ("Precision Plumbing", "PP"),
        ("Coastal Civil Engineering", "CCE"),
        ("Metro Fire Protection", "MFP"),
        ("Skyline Concrete", "SKC"),
        ("Elite Drywall Systems", "EDS"),
        ("TerraForm Excavation", "TFE"),
        ("AllStar Roofing", "ASR"),
        ("ClearView Glazing", "CVG"),
        ("ProSpec Testing Labs", "PTL"),
    ]

    PROJECTS = [
        ("Waterfront Mixed-Use Tower", "P-2024-0847", "Pacific Development Corp"),
        ("City Hall Renovation Phase 2", "P-2024-1203", "City of Riverside"),
        ("Highway 401 Bridge Replacement", "P-2025-0156", "Ministry of Transportation"),
        ("Westfield Medical Center Expansion", "P-2023-0592", "Westfield Health System"),
        ("Oakridge Elementary School", "P-2025-0301", "Unified School District #47"),
        ("Industrial Distribution Center", "P-2024-0720", "Atlas Logistics Inc"),
        ("Harborview Condominiums", "P-2025-0412", "Harborview Properties LLC"),
        ("Central Library Modernization", "P-2024-0933", "City Public Library Board"),
        ("Airport Terminal B Expansion", "P-2023-0801", "Regional Airport Authority"),
        ("Riverside Office Park Phase 3", "P-2025-0567", "Riverside REIT"),
    ]

    DISCIPLINES = [
        "structural", "mechanical", "electrical", "architectural",
        "plumbing", "civil", "landscape", "fire_protection",
    ]

    SPEC_SECTIONS = [
        ("03 30 00", "Cast-in-Place Concrete"),
        ("05 12 00", "Structural Steel Framing"),
        ("07 21 00", "Thermal Insulation"),
        ("08 44 13", "Glazed Aluminum Curtain Walls"),
        ("09 29 00", "Gypsum Board"),
        ("13 34 19", "Metal Rolling Doors"),
        ("22 11 16", "Domestic Water Piping"),
        ("23 05 93", "Testing Adjusting and Balancing for HVAC"),
        ("23 37 00", "Air Outlets and Inlets"),
        ("26 24 16", "Panelboards"),
        ("31 23 16", "Excavation"),
        ("32 12 16", "Asphalt Paving"),
        ("32 31 13", "Chain Link Fences and Gates"),
        ("33 11 00", "Water Utility Distribution Piping"),
    ]

    RFI_SUBJECTS = [
        "Structural Steel Connection Detail at Grid Line {grid}",
        "Waterproofing Membrane Specification at {location}",
        "HVAC Diffuser Layout Conflict with Structural Beams",
        "Anchor Bolt Spacing at {location}",
        "Temporary Shoring Design for {purpose}",
        "Elevator Pit Drainage Detail",
        "Fire Rating of Partition Wall at {location}",
        "Concrete Mix Design for {element}",
        "Electrical Panel Location Conflict at {location}",
        "Window Sill Height Clarification at {location}",
        "Rebar Splice Length for {element}",
        "Roof Drain Location vs Structural Framing",
        "Door Hardware Schedule Discrepancy",
        "Ceiling Height Conflict at {location}",
        "Foundation Depth at {location}",
    ]

    SUBMITTAL_DESCRIPTIONS = [
        "HVAC Diffusers - Linear Slot Type",
        "Curtain Wall System - Unitized Panels",
        "Gypsum Board Assemblies - Fire-Rated",
        "Precast Concrete Bridge Girders",
        "Medical Equipment - Patient Lifts",
        "Chain Link Fencing and Gates",
        "Overhead Coiling Doors - Insulated",
        "Panelboards - 480/277V Distribution",
        "Structural Steel Shop Drawings",
        "Waterproofing Membrane System",
        "Fire Sprinkler System Layout",
        "Elevator Cab Finishes",
        "Acoustical Ceiling Tiles",
        "Landscape Planting Plan",
    ]

    CO_REASONS = [
        "Additional structural steel at roof level",
        "Unforeseen soil conditions during excavation",
        "Owner-requested finish upgrades at lobby",
        "Code-required fire suppression addition",
        "Hazardous material abatement - asbestos",
        "Revised mechanical equipment layout",
        "Additional electrical capacity for tenant fit-out",
        "Storm damage repair to temporary structures",
        "Accelerated schedule - overtime premium",
        "Revised foundation design per updated geotech",
    ]

    def aeco_company(self):
        return random.choice(self.COMPANIES)

    def aeco_project(self):
        return random.choice(self.PROJECTS)

    def aeco_discipline(self):
        return random.choice(self.DISCIPLINES)

    def aeco_spec_section(self):
        return random.choice(self.SPEC_SECTIONS)

    def aeco_grid_line(self):
        letter = random.choice("ABCDEFGHJKLM")
        number = random.randint(1, 20)
        return f"{letter}-{number}"

    def aeco_location(self):
        locations = [
            "Level 3 West Wing", "Parking Level P2", "Roof Mechanical Room",
            "Ground Floor Lobby", "Level 15 Typical Floor", "Basement",
            "East Stairwell", "Corridor 2B", "Room 405", "Penthouse Level",
            "South Facade", "Loading Dock Area", "Generator Room",
        ]
        return random.choice(locations)

    def aeco_element(self):
        elements = [
            "grade beams", "pile caps", "shear walls", "transfer beams",
            "elevated slab", "retaining wall", "pedestal", "mat foundation",
        ]
        return random.choice(elements)


fake.add_provider(AECOProvider)


# ─── Email Templates ─────────────────────────────────────────────────


def _generate_rfi_email(project, company, rfi_num):
    """Generate a realistic RFI email."""
    person = fake.name()
    grid = fake.aeco_grid_line()
    location = fake.aeco_location()
    element = fake.aeco_element()

    subject_template = random.choice(AECOProvider.RFI_SUBJECTS)
    subject = f"RFI-{rfi_num} - {subject_template.format(grid=grid, location=location, purpose='traffic maintenance', element=element)}"

    body = f"""{fake.first_name()},

We've identified an issue on the {project[0]} project that requires clarification.

{random.choice([
    f"The detail shown on drawing S-{random.randint(100, 499)} at grid line {grid} conflicts with the specification requirements.",
    f"During field work at {location}, we discovered conditions that differ from what's shown on the contract documents.",
    f"The current design at {location} doesn't accommodate the {random.choice(['equipment', 'ductwork', 'piping', 'conduit'])} routing as coordinated.",
    f"We need clarification on the {random.choice(['connection detail', 'material specification', 'dimension', 'finish requirement'])} at {location}.",
])}

{random.choice([
    "Please advise on the design intent so we can proceed with our work.",
    f"This is on our critical path - we need a response by {fake.date_between('+3d', '+14d').strftime('%B %d')} to maintain the schedule.",
    "We've attached marked-up drawings showing the conflict. Please review and respond.",
    f"Our crew is standing by at {location} waiting for direction on this issue.",
])}

{random.choice([
    "Attached: Field photos and marked-up drawings.",
    f"See attached conflict markup on drawing {random.choice(['S', 'A', 'M', 'E'])}-{random.randint(100, 499)}.",
    "",
])}

Regards,
{person}
{random.choice(['Project Manager', 'Project Engineer', 'Superintendent', 'Field Engineer', 'Senior Engineer', 'Construction Manager'])}
{company[0]}
({fake.phone_number()})"""

    return {
        "subject": subject,
        "body": body,
        "category": "rfi",
        "attachments": [f"RFI-{rfi_num}_markup.pdf"] if random.random() > 0.3 else [],
    }


def _generate_submittal_email(project, company, sub_num):
    """Generate a realistic submittal email."""
    person = fake.name()
    spec = fake.aeco_spec_section()
    desc = random.choice(AECOProvider.SUBMITTAL_DESCRIPTIONS)

    subject = f"Submittal SUB-{sub_num:03d} - {desc} - {project[0]}"

    body = f"""{fake.first_name()},

Please find attached the submittal package for {desc} per Spec Section {spec[0]} for the {project[0]} project ({project[1]}).

Submittal includes:
- Product data sheets{random.choice([' and performance data', ''])}
- {random.choice(['Shop drawings', 'Manufacturer cut sheets', 'Sample board photos', 'Installation details'])}
- {random.choice(['Color selections', 'Warranty information', 'Test reports', 'Compliance certificates'])}

{random.choice([
    "Requested review period: 10 business days.",
    "Please expedite review - this item is on the critical path.",
    "This is a resubmission per your previous review comments.",
    f"Note: Lead time for this material is {random.randint(6, 20)} weeks from order.",
])}

{person}
{company[0]}"""

    return {
        "subject": subject,
        "body": body,
        "category": "submittal",
        "attachments": [f"SUB-{sub_num:03d}_data.pdf", f"SUB-{sub_num:03d}_drawings.pdf"],
    }


def _generate_change_order_email(project, company, co_num):
    """Generate a realistic change order email."""
    person = fake.name()
    reason = random.choice(AECOProvider.CO_REASONS)
    amount = random.randint(5, 300) * 1000

    subject = f"CO #{co_num} - {reason} - {project[0]}"

    body = f"""{fake.first_name()},

We are submitting Change Order #{co_num} for the {project[0]} project.

Scope: {reason}

{random.choice([
    f"This additional work was identified during {random.choice(['field work', 'coordination meetings', 'exploratory demolition', 'site investigations'])}.",
    f"The owner has requested {random.choice(['an upgrade to', 'modifications to', 'additional scope for'])} this area.",
    "This work is required due to conditions that differ materially from the contract documents.",
])}

Proposed price: ${amount:,}
Schedule impact: {random.randint(0, 15)} working days

{random.choice([
    "Detailed pricing breakdown attached.",
    "Please review and advise. We need approval to proceed with material procurement.",
    "This change order includes labor, material, and equipment costs.",
])}

{person}
{company[0]}"""

    return {
        "subject": subject,
        "body": body,
        "category": "change_order",
        "attachments": [f"CO{co_num}_pricing.pdf"],
    }


def _generate_general_email(project, company):
    """Generate a general correspondence email."""
    person = fake.name()
    topics = [
        ("schedule update", f"Quick update on the {project[0]} schedule. {random.choice(['We are tracking 3 days ahead.', 'The critical path has shifted due to material delays.', 'Phase 2 is now scheduled to start next Monday.'])}"),
        ("coordination", f"Wanted to coordinate {random.choice(['crane usage', 'material deliveries', 'concrete pours', 'shutdown windows'])} for next week on the {project[0]} project."),
        ("safety", f"Reminder: {random.choice(['toolbox talk tomorrow at 7 AM', 'safety stand-down on Friday', 'updated site safety plan attached', 'PPE inspection this Thursday'])} for the {project[0]} site."),
        ("general", f"Following up on our conversation about the {project[0]} project. {random.choice(['Please let me know if you need anything else.', 'Looking forward to the site visit next week.', 'Can we schedule a coordination meeting?'])}"),
    ]
    topic, body_text = random.choice(topics)

    return {
        "subject": f"{topic.title()} - {project[0]}",
        "body": f"{fake.first_name()},\n\n{body_text}\n\n{person}\n{company[0]}",
        "category": "general",
        "attachments": [],
    }


def _generate_meeting_minutes_email(project, company):
    """Generate meeting minutes email."""
    meeting_num = random.randint(1, 50)
    date = fake.date_between("-30d", "today")
    person = fake.name()

    subject = f"Meeting Minutes - OAC Meeting #{meeting_num} - {date.strftime('%B %d, %Y')} - {project[0]}"
    body = f"""All,

Please find the minutes from OAC Meeting #{meeting_num} for the {project[0]} project held on {date.strftime('%B %d, %Y')}.

Key Discussion Items:
1. {random.choice(['Schedule update - overall project tracking on schedule', 'Budget review - contingency usage at 35%', 'Safety report - zero incidents this month'])}
2. {random.choice(['Structural work progressing per plan', 'Mechanical rough-in started on Level 3', 'Envelope work 60% complete'])}
3. {random.choice(['Pending RFIs reviewed - 5 outstanding', 'Submittal log reviewed - 12 pending', 'Change orders - 2 under review'])}

Action Items:
- {fake.name()}: {random.choice(['Respond to outstanding RFIs by next meeting', 'Submit revised schedule', 'Provide pricing for additional scope'])}
- {fake.name()}: {random.choice(['Schedule mock-up testing', 'Coordinate shutdowns with building operations', 'Submit closeout documents'])}

Next meeting: {fake.date_between('+5d', '+14d').strftime('%B %d, %Y')}

{person}
{company[0]}"""

    return {
        "subject": subject,
        "body": body,
        "category": "meeting_minutes",
        "attachments": [f"OAC_{meeting_num}_Minutes.pdf"],
    }


# ─── Main Generator ──────────────────────────────────────────────────


CATEGORY_WEIGHTS = {
    "rfi": 0.30,
    "submittal": 0.20,
    "change_order": 0.10,
    "general": 0.25,
    "meeting_minutes": 0.15,
}


def generate_email(tenant_id: str = "demo") -> dict:
    """Generate a single realistic AECO email."""
    company = fake.aeco_company()
    project = fake.aeco_project()

    # Weighted category selection
    category = random.choices(
        list(CATEGORY_WEIGHTS.keys()),
        weights=list(CATEGORY_WEIGHTS.values()),
    )[0]

    if category == "rfi":
        email_data = _generate_rfi_email(project, company, random.randint(1, 500))
    elif category == "submittal":
        email_data = _generate_submittal_email(project, company, random.randint(1, 200))
    elif category == "change_order":
        email_data = _generate_change_order_email(project, company, random.randint(1, 50))
    elif category == "meeting_minutes":
        email_data = _generate_meeting_minutes_email(project, company)
    else:
        email_data = _generate_general_email(project, company)

    sender_name = fake.name()
    sender_domain = company[0].lower().replace(" ", "").replace(",", "")[:12]

    return {
        "id": f"gen_{uuid.uuid4().hex[:12]}",
        "sender": f"{sender_name.split()[0].lower()}.{sender_name.split()[-1].lower()}@{sender_domain}.com",
        "sender_name": sender_name,
        "recipients": [f"{fake.first_name().lower()}@{fake.aeco_company()[0].lower().replace(' ', '')[:10]}.com"],
        "subject": email_data["subject"],
        "body": email_data["body"],
        "attachments": email_data["attachments"],
        "received_at": fake.date_time_between("-90d", "now").isoformat(),
        "thread_id": f"thread_{uuid.uuid4().hex[:8]}",
        "project_hint": project[0],
        "tenant_id": tenant_id,
        "expected_category": email_data["category"],
    }


def generate_batch(count: int, tenant_id: str = "demo") -> list[dict]:
    """Generate a batch of emails."""
    return [generate_email(tenant_id) for _ in range(count)]


def load_to_dynamodb(emails: list[dict], endpoint: str = "http://localhost:8001"):
    """Load generated emails into DynamoDB Local using batch_writer."""
    import boto3

    dynamodb = boto3.resource(
        "dynamodb",
        endpoint_url=endpoint,
        region_name="ca-central-1",
        aws_access_key_id="local",
        aws_secret_access_key="local",
    )
    table = dynamodb.Table("atlas-emails")

    loaded = 0
    start = time.perf_counter()

    with table.batch_writer() as batch:
        for email in emails:
            tenant_id = email.get("tenant_id", "demo")
            batch.put_item(Item={
                "pk": f"TENANT#{tenant_id}",
                "sk": f"EMAIL#{email['id']}",
                "email_id": email["id"],
                "sender": email["sender"],
                "subject": email["subject"],
                "body": email["body"][:5000],  # DynamoDB item size limit
                "category": email.get("expected_category", ""),
                "received_at": email["received_at"],
            })
            loaded += 1
            if loaded % 1000 == 0:
                elapsed = time.perf_counter() - start
                rate = loaded / elapsed
                print(f"  Loaded {loaded:,} emails ({rate:.0f}/sec)")

    elapsed = time.perf_counter() - start
    print(f"  Done: {loaded:,} emails in {elapsed:.1f}s ({loaded/elapsed:.0f}/sec)")


def main():
    parser = argparse.ArgumentParser(description="Generate AECO test emails at scale")
    parser.add_argument("--count", type=int, default=1000, help="Number of emails to generate")
    parser.add_argument("--tenant", type=str, default="demo", help="Tenant ID")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--dynamodb", action="store_true", help="Load into DynamoDB Local")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8001", help="DynamoDB endpoint")
    args = parser.parse_args()

    print(f"Generating {args.count:,} AECO emails for tenant '{args.tenant}'...")
    start = time.perf_counter()

    emails = generate_batch(args.count, args.tenant)

    gen_time = time.perf_counter() - start
    print(f"Generated {len(emails):,} emails in {gen_time:.1f}s ({len(emails)/gen_time:.0f}/sec)")

    # Category distribution
    from collections import Counter
    dist = Counter(e["expected_category"] for e in emails)
    print(f"Distribution: {dict(dist)}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(emails, f, indent=2, default=str)
        print(f"Saved to {args.output}")

    if args.dynamodb:
        print(f"Loading into DynamoDB at {args.endpoint}...")
        load_to_dynamodb(emails, args.endpoint)


if __name__ == "__main__":
    main()
