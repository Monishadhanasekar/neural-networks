import os
import json
import hashlib
import re
import numpy as np
from typing import List, Dict, Tuple
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

#Part 1: Setup and first API calls to free model

# setup openrouter

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

FREE_MODEL = "openrouter/free"
TOOL_MODEL = "openrouter/free"


DOCUMENTS = [
    {
        "title": "Company Q3 2024 Report",
        "source": "q3_report.pdf",
        "content": """
Q3 2024 Financial Results — ACME Corporation

Revenue and Growth
ACME Corporation reported total revenue of $4.2 billion for Q3 2024, representing a 12% increase year-over-year. The cloud services division was the primary growth driver, contributing $2.1 billion, up 28% from the same period last year. Hardware sales remained flat at $1.4 billion, while consulting services brought in $700 million, a 5% decline attributed to reduced enterprise spending.

Employee Growth
The company added 1,847 new employees during Q3, bringing total headcount to 34,500. Engineering teams grew by 1,200, with significant hiring in the AI and machine learning divisions. The company opened new offices in Austin, Texas and Bangalore, India to accommodate growth.

Product Launches
ACME launched three major products in Q3: CloudSync Pro (enterprise file synchronization), ACME Shield (zero-trust security platform), and DataFlow 2.0 (real-time analytics engine). CloudSync Pro achieved 50,000 enterprise customers within its first month. ACME Shield received FedRAMP authorization, opening government sector opportunities.

Challenges and Risks
Supply chain constraints continued to impact hardware margins, with gross margin declining to 34% from 38% in Q2. The company faces increasing competition in the cloud services market from major players. Additionally, regulatory scrutiny in the EU around data privacy may impact expansion plans.

Outlook
Management expects Q4 revenue of $4.5-4.7 billion, with continued strong growth in cloud services offsetting hardware softness. The company plans to invest $800 million in R&D during Q4, with a focus on AI capabilities.
"""
    },
    {
        "title": "Employee Handbook — Remote Work Policy",
        "source": "handbook_remote.pdf",
        "content": """
Remote Work Policy — ACME Corporation Employee Handbook, Section 4.3

Eligibility
All full-time employees who have completed their 90-day probationary period are eligible for remote work arrangements. Contractors and part-time employees must receive written approval from their department head. Employees in customer-facing roles requiring physical presence (retail, field service) are exempt from this policy.

Work Schedule
Remote employees must maintain core hours of 10:00 AM to 3:00 PM in their designated time zone. Outside of core hours, employees may structure their work schedule flexibly, provided they complete their standard 40-hour work week. All meetings during core hours are mandatory unless pre-approved absence is granted.

Equipment and Expenses
The company provides a one-time home office stipend of $1,500 for new remote workers. This covers desk, chair, and peripherals. IT will provide a company laptop and monitor. Internet expenses are reimbursed up to $75 per month with receipt submission through the expense portal.

Performance and Accountability
Remote employees are evaluated using the same performance metrics as in-office staff. Managers conduct weekly 1:1 check-ins and quarterly performance reviews. Employees who fail to meet performance expectations may be required to return to in-office work.

Security Requirements
All remote workers must use the company VPN for accessing internal systems. Two-factor authentication is mandatory. Work must be performed from a private, secure location — public wifi networks are prohibited for accessing company systems without VPN protection.
"""
    },
    {
        "title": "Engineering Wiki — Authentication System",
        "source": "wiki_auth.pdf",
        "content": """
Authentication System Architecture — Engineering Wiki

Overview
ACME's authentication system uses a microservices architecture with three core components: AuthGateway (the entry point), TokenService (JWT issuance and validation), and IdentityProvider (user credential management). The system handles approximately 2.3 million authentication requests per day.

AuthGateway
The AuthGateway is a Node.js service running on Kubernetes. It receives all authentication requests, performs initial rate limiting (100 requests per minute per IP), and routes to the appropriate authentication flow. Supported flows include: password-based login, OAuth 2.0 (Google, GitHub, Microsoft), SAML for enterprise SSO, and magic link email authentication.

TokenService
TokenService issues JWTs with a 15-minute access token lifetime and 7-day refresh token lifetime. Tokens are signed using RS256 with rotating keys (rotated every 30 days). The service maintains a token blacklist in Redis for immediate revocation. Token refresh requires presenting a valid refresh token and passes through fraud detection checks.

IdentityProvider
The IdentityProvider stores user credentials in PostgreSQL with bcrypt hashing (cost factor 12). It supports MFA via TOTP (Google Authenticator) and WebAuthn (hardware keys). Password requirements: minimum 12 characters, at least one uppercase, one number, one special character. Account lockout occurs after 5 failed attempts with a 30-minute cooldown.

Known Issues
- Token blacklist in Redis is not replicated across regions (tracked in JIRA AUTH-2847). Revoked tokens may remain valid for up to 60 seconds in non-primary regions.
- SAML integration with Okta has intermittent timeout issues during peak hours (AUTH-3021).
- The magic link flow does not enforce rate limiting separately from password-based login, allowing potential abuse (AUTH-3156).

Recent Changes
- 2024-09-15: Migrated from HS256 to RS256 token signing
- 2024-08-20: Added WebAuthn support for hardware key MFA
- 2024-07-01: Increased bcrypt cost factor from 10 to 12
"""
    },
    {
        "title": "Engineering Wiki — Deployment Process",
        "source": "wiki_deploy.pdf",
        "content": """
Deployment Process — Engineering Wiki

Overview
ACME uses a continuous deployment pipeline managed through GitHub Actions and ArgoCD. All services deploy to Kubernetes clusters across three regions: us-east-1, eu-west-1, and ap-southeast-1. Deployments follow a canary release pattern with automated rollback.

Pipeline Stages
1. PR Merge: Code merged to main triggers the pipeline
2. Build: Docker image built, tagged with git SHA
3. Test: Unit tests, integration tests, and security scanning (Snyk)
4. Stage: Deployed to staging environment for 30 minutes of automated end-to-end tests
5. Canary: 5% of production traffic routed to new version
6. Monitor: 15-minute observation window checking error rates, latency, and resource usage
7. Rollout: Gradual increase to 25%, 50%, 100% over 45 minutes
8. Verify: Post-deployment smoke tests

Rollback
Automatic rollback triggers if: error rate exceeds 1% (baseline + 0.5%), p99 latency exceeds 500ms, or memory usage exceeds 80% of pod limits. Manual rollback can be initiated by any on-call engineer through the ArgoCD dashboard or via Slack command: /deploy rollback [service] [version].

Hotfix Process
For critical production issues (P0/P1), engineers can bypass the staging environment. Hotfix PRs require approval from two senior engineers and the on-call lead. The canary phase is shortened to 5 minutes with heightened monitoring alerts.

Freeze Periods
Code freezes are enforced during: major product launches (48 hours before and 24 hours after), Black Friday / Cyber Monday week, and end-of-quarter financial reporting periods. Emergency hotfixes are exempt from freezes with VP-level approval.
"""
    },
    {
        "title": "Board Meeting Notes — AI Strategy",
        "source": "board_ai_strategy.pdf",
        "content": """
Board Meeting Minutes — AI Strategy Discussion, October 2024

Attendees: CEO, CTO, CFO, VP Engineering, VP Product, Board Members

AI Investment Overview
The CTO presented ACME's AI strategy for 2025, requesting a $500 million investment over 18 months. The proposal includes: $200M for AI infrastructure (GPU clusters, training pipelines), $150M for AI product development (embedding AI features across the product suite), $100M for AI talent acquisition (target: 500 ML engineers), and $50M for strategic AI partnerships and acquisitions.

Current AI Capabilities
ACME currently uses AI in three areas: CloudSync Pro's intelligent file categorization (using fine-tuned BERT models), customer support chatbot handling 40% of tier-1 tickets, and DataFlow 2.0's anomaly detection engine. The CTO noted these are 'point solutions' and the company needs a platform approach.

Competitive Analysis
The board discussed competitor moves: TechCorp launched an AI assistant integrated across their entire suite. GlobalSoft acquired an AI startup for $2B. CloudFirst announced AI-powered security features similar to ACME Shield's roadmap. The consensus was that ACME risks falling behind without aggressive investment.

Board Discussion
The CFO raised concerns about the investment timeline, suggesting a phased approach with $300M in year one and $200M in year two, contingent on year-one milestones. Board Member Dr. Sarah Chen emphasized the need for responsible AI development, recommending an AI ethics board and third-party audits. Board Member James Wright questioned whether building in-house was more cost-effective than partnering with foundation model providers like OpenAI or Anthropic.

Resolution
The board approved a phased $500M AI investment with quarterly milestone reviews. An AI Ethics Advisory Board will be established within 60 days. The CTO will present a detailed technical roadmap at the December meeting.
"""
    }
]

DISTRACTOR_DOCUMENTS = [
    {
        "title": "Engineering Wiki — Logging and Monitoring",
        "source": "wiki_logging.pdf",
        "content": """
Logging and Monitoring Standards — Engineering Wiki

Overview
All ACME services must emit structured logs in JSON format to the centralized logging platform (Datadog). Logs are retained for 90 days in hot storage and 1 year in cold storage. Each service must implement health check endpoints at /healthz and /readyz.

Log Levels
- ERROR: Unexpected failures requiring immediate attention
- WARN: Degraded functionality that doesn't block users
- INFO: Standard operational events (request received, job completed)
- DEBUG: Detailed diagnostic information, disabled in production by default

Metrics and Alerting
Services must expose Prometheus metrics on port 9090. Required metrics include: request_count, request_latency_seconds, error_rate, and active_connections. Alerts are configured in PagerDuty with escalation policies: P0 alerts page the on-call engineer immediately, P1 alerts page after 5 minutes if unacknowledged, P2 alerts create a Slack notification in #eng-alerts.

Dashboards
Each team maintains a service dashboard in Grafana showing the four golden signals: latency, traffic, errors, and saturation. Dashboards must be reviewed and updated quarterly. The SRE team provides dashboard templates for common service patterns.

Incident Response
When an alert fires, the on-call engineer has 15 minutes to acknowledge and 30 minutes to begin mitigation. All P0/P1 incidents require a post-mortem within 48 hours. Post-mortems are stored in Confluence under the Incident Reviews space. The blameless post-mortem format includes: timeline, impact assessment, root cause analysis, and action items with owners.

Audit Logging
All authentication events, permission changes, and data access events must be logged to the audit trail. Audit logs are immutable and retained for 7 years to comply with SOC 2 requirements. The audit logging service runs independently from application logging to ensure availability during outages.
"""
    },
    {
        "title": "Engineering Wiki — Database Standards",
        "source": "wiki_database.pdf",
        "content": """
Database Standards and Practices — Engineering Wiki

Supported Databases
ACME's approved database technologies are: PostgreSQL 15+ for relational data, Redis 7+ for caching and session storage, MongoDB 6+ for document storage, and Amazon DynamoDB for high-throughput key-value workloads. Any new database technology requires Architecture Review Board approval.

Schema Management
All schema changes must go through a migration process using Flyway. Migrations are versioned and applied automatically during deployment. Backward-incompatible changes (column drops, type changes) require a two-phase migration: first deploy code that handles both schemas, then apply the breaking change in a subsequent release. Schema changes must be reviewed by the database team before merging.

Connection Pooling
All services must use PgBouncer for PostgreSQL connection pooling. Maximum connections per service are allocated based on traffic tier: Tier 1 (>10K RPM): 50 connections, Tier 2 (1K-10K RPM): 20 connections, Tier 3 (<1K RPM): 10 connections. Connection pool exhaustion triggers a P1 alert.

Backup and Recovery
PostgreSQL databases are backed up continuously using WAL archiving to S3. Point-in-time recovery is available for the last 30 days. Full snapshots are taken daily at 03:00 UTC. Recovery testing is performed quarterly — the database team restores a production backup to staging and verifies data integrity. RTO target is 1 hour, RPO target is 5 minutes.

Data Classification
All database tables must be tagged with a data classification level: PUBLIC, INTERNAL, CONFIDENTIAL, or RESTRICTED. RESTRICTED data (PII, financial records) requires encryption at rest using AES-256 and column-level encryption for sensitive fields. Access to RESTRICTED databases requires approval from the Data Governance team.
"""
    },
    {
        "title": "Engineering Wiki — API Design Guidelines",
        "source": "wiki_api.pdf",
        "content": """
API Design Guidelines — Engineering Wiki

REST API Standards
All public APIs follow REST conventions with JSON request and response bodies. API versioning uses URL path versioning (e.g., /v1/users, /v2/users). Deprecation notices must be communicated 6 months before removal. All endpoints require authentication via Bearer tokens in the Authorization header.

Rate Limiting
Public APIs are rate limited to 1000 requests per minute per API key. Internal service-to-service calls use a separate rate limit of 10,000 RPM. Rate limit headers (X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset) must be included in all responses. Clients exceeding the rate limit receive a 429 Too Many Requests response with a Retry-After header.

Error Handling
Error responses follow the RFC 7807 Problem Details format. All errors include: type (URI reference), title (human-readable), status (HTTP status code), detail (specific explanation), and instance (URI of the specific occurrence). Internal errors must not leak implementation details — stack traces and internal service names are stripped from external responses.

Pagination
List endpoints use cursor-based pagination with the following query parameters: limit (default 20, max 100), cursor (opaque string), and order (asc/desc). Response includes next_cursor and has_more fields. Offset-based pagination is prohibited for datasets over 10,000 records due to performance degradation.

Documentation
All APIs must be documented using OpenAPI 3.0 specifications. Documentation is auto-generated and published to the API portal at api.acme.com/docs. Each endpoint must include: description, request/response schemas, example payloads, error codes, and rate limit information. API documentation is reviewed as part of the PR process.
"""
    },
    {
        "title": "Employee Handbook — Travel and Expenses Policy",
        "source": "handbook_travel.pdf",
        "content": """
Travel and Expenses Policy — ACME Corporation Employee Handbook, Section 5.1

Travel Authorization
All business travel must be pre-approved by the employee's direct manager via the TravelDesk portal. International travel requires additional approval from the department VP. Travel requests should be submitted at least 14 business days in advance. Emergency travel can be approved retroactively with VP sign-off within 48 hours of return.

Booking Standards
Flights must be booked through the corporate travel portal (managed by Corporate Travel Management). Economy class is standard for flights under 6 hours; premium economy is permitted for flights over 6 hours. Business class requires VP approval and is generally reserved for executive leadership. Hotel bookings should not exceed the GSA per diem rate for the destination city.

Expense Reimbursement
Employees must submit expense reports within 30 days of travel completion through the Concur expense system. Receipts are required for all expenses over $25. Meal expenses follow the company per diem: $75/day domestic, $100/day international. Alcohol is not reimbursable. Reimbursements are processed in the next payroll cycle following approval.

Company Credit Cards
Employees traveling more than 4 times per year are eligible for a corporate Amex card. Monthly statements must be reconciled within 10 business days. Personal charges on corporate cards must be flagged and reimbursed immediately. Misuse of corporate cards may result in card revocation and disciplinary action.

Mileage and Ground Transportation
Personal vehicle mileage is reimbursed at the current IRS rate ($0.67/mile for 2024). Ride-share services (Uber, Lyft) are preferred over rental cars for trips under 100 miles. Rental cars require manager approval and must include the company's insurance waiver. Parking and tolls are reimbursable with receipts.
"""
    },
    {
        "title": "Employee Handbook — Performance Review Process",
        "source": "handbook_performance.pdf",
        "content": """
Performance Review Process — ACME Corporation Employee Handbook, Section 6.2

Review Cycle
ACME conducts formal performance reviews twice per year: mid-year (June) and year-end (December). The mid-year review is a checkpoint focused on progress against goals and course correction. The year-end review is comprehensive and determines compensation adjustments, bonuses, and promotion eligibility.

Goal Setting
At the start of each fiscal year (January), employees work with their managers to set 3-5 measurable goals aligned with team and company objectives. Goals follow the SMART framework (Specific, Measurable, Achievable, Relevant, Time-bound). Goals are documented in Workday and can be adjusted at mid-year with manager approval.

Rating Scale
Employees are rated on a 5-point scale: Exceptional (5), Exceeds Expectations (4), Meets Expectations (3), Needs Improvement (2), and Unsatisfactory (1). Ratings are calibrated across teams during leadership calibration sessions to ensure consistency. A rating of 2 or below triggers a Performance Improvement Plan (PIP).

360 Feedback
Year-end reviews include 360 feedback from peers, direct reports (for managers), and cross-functional partners. Employees nominate 3-5 peers for feedback, and managers may add additional reviewers. Feedback is collected anonymously through Workday and shared with the employee during the review discussion.

Promotion Criteria
Promotions require: sustained high performance (rating of 4+ for two consecutive cycles), demonstrated capability at the next level, manager nomination, and skip-level approval. Promotion decisions are finalized during the February talent review. Off-cycle promotions require VP approval and are limited to exceptional circumstances.
"""
    },
    {
        "title": "Employee Handbook — Code of Conduct",
        "source": "handbook_conduct.pdf",
        "content": """
Code of Conduct — ACME Corporation Employee Handbook, Section 1.1

Core Values
ACME is committed to maintaining a workplace built on integrity, respect, and accountability. All employees, contractors, and partners are expected to uphold these standards in every interaction, whether with colleagues, customers, or the public.

Conflicts of Interest
Employees must disclose any personal, financial, or familial relationships that could influence business decisions. Outside employment is permitted with manager approval, provided it does not conflict with ACME responsibilities or involve competitors. Board positions at other companies require General Counsel approval. All conflicts must be reported through the Ethics Hotline or directly to the Compliance team.

Anti-Harassment Policy
ACME maintains a zero-tolerance policy for harassment, discrimination, and retaliation. This includes but is not limited to harassment based on race, gender, sexual orientation, religion, disability, age, or national origin. Employees who experience or witness harassment should report it to HR, their manager, or the anonymous Ethics Hotline (1-800-555-ACME). All reports are investigated promptly and confidentially.

Gifts and Entertainment
Employees may accept business gifts valued at $100 or less per occasion. Gifts exceeding $100 must be reported to the Compliance team. Government employees and officials may not be offered gifts of any value. Entertainment (meals, events) is permissible if it has a clear business purpose and does not create an obligation.

Confidentiality
All proprietary information, trade secrets, and non-public business data must be treated as confidential. Employees must not share confidential information outside the company without authorization. This obligation continues after employment ends. Breaches of confidentiality may result in termination and legal action.

Social Media
Employees may use personal social media but must not represent themselves as speaking for ACME unless authorized. Sharing confidential business information, internal discussions, or unreleased product details on social media is prohibited. The Communications team manages all official ACME social media accounts.
"""
    },
    {
        "title": "Q2 2024 Financial Report",
        "source": "q2_report.pdf",
        "content": """
Q2 2024 Financial Results — ACME Corporation

Revenue Summary
ACME Corporation reported total revenue of $3.9 billion for Q2 2024, representing a 9% increase year-over-year. Cloud services revenue reached $1.8 billion, up 22% year-over-year. Hardware revenue was $1.5 billion, flat compared to Q2 2023. Consulting services contributed $600 million, down 8% due to project completion cycles.

Operating Margins
Gross margin for Q2 was 38%, down from 40% in Q1 due to increased hardware component costs. Operating expenses totaled $2.8 billion, with R&D spending at $650 million (17% of revenue). Sales and marketing expenses increased 12% to support the CloudSync Pro launch preparation. G&A expenses remained flat at $280 million.

Customer Metrics
Total enterprise customers grew to 28,000, up 15% year-over-year. Net revenue retention rate was 118%, indicating strong expansion within existing accounts. Customer acquisition cost decreased 8% due to improved marketing efficiency. Churn rate remained below 3% for the sixth consecutive quarter.

Workforce Update
Total headcount at end of Q2 was 32,653. The company hired 1,200 employees during the quarter, with 60% in engineering roles. The voluntary attrition rate was 8.5%, below the industry average of 12%. The company announced plans to open new offices in Austin and Bangalore in Q3.

Capital Allocation
The company generated $800 million in free cash flow during Q2. Share repurchases totaled $200 million. Capital expenditure was $350 million, primarily for data center expansion. The company maintains $4.2 billion in cash and equivalents with zero long-term debt.
"""
    },
    {
        "title": "Engineering Wiki — CI/CD Pipeline Configuration",
        "source": "wiki_cicd.pdf",
        "content": """
CI/CD Pipeline Configuration Guide — Engineering Wiki

GitHub Actions Setup
All repositories must use the shared workflow templates in the .github/workflows directory of the platform-configs repository. Custom workflows require Platform Engineering team approval. Workflow runs are limited to 60 minutes; long-running jobs must be split into parallel stages. Self-hosted runners are available for builds requiring GPU access or specialized hardware.

Build Configuration
Docker images are built using multi-stage Dockerfiles to minimize image size. Base images must use the company-approved base (acme-base:latest) which includes security patches and monitoring agents. Images are scanned with Trivy for vulnerabilities — builds with CRITICAL or HIGH CVEs are blocked automatically. Images are pushed to the internal container registry at registry.acme.internal.

Test Requirements
All PRs must pass the following checks before merge: unit tests (minimum 80% coverage), integration tests, linting (ESLint for JS/TS, Black for Python), type checking (TypeScript strict mode, mypy for Python), and security scanning (Snyk for dependencies, Semgrep for code patterns). Flaky tests must be quarantined within 48 hours or the responsible team is paged.

Environment Management
ACME maintains four environments: development (auto-deployed on PR creation), staging (deployed on merge to main), canary (5% production traffic), and production. Environment variables and secrets are managed through HashiCorp Vault. Each environment has isolated database instances — production data never flows to non-production environments.

Artifact Management
Build artifacts (Docker images, npm packages, Python wheels) are stored in JFrog Artifactory with 90-day retention for development builds and indefinite retention for production releases. Artifact provenance is tracked using SLSA Level 3 attestations. Rollback artifacts are pre-cached in each deployment region for sub-minute rollback times.
"""
    },
    {
        "title": "Engineering Wiki — Service Mesh and Networking",
        "source": "wiki_networking.pdf",
        "content": """
Service Mesh and Networking — Engineering Wiki

Service Discovery
All internal services register with Consul for service discovery. Service-to-service communication uses gRPC by default, with HTTP/REST as a fallback for legacy services. Service endpoints are resolved via DNS (service-name.acme.internal) with round-robin load balancing.

Istio Service Mesh
ACME uses Istio as the service mesh layer. All inter-service traffic is encrypted with mutual TLS (mTLS). Traffic policies are defined using Istio VirtualService and DestinationRule resources. Circuit breakers are configured with a 50% error threshold — when a downstream service exceeds this, requests are short-circuited for 30 seconds.

Network Policies
Kubernetes NetworkPolicies enforce least-privilege network access. By default, pods cannot communicate with each other unless explicitly allowed. Each service defines its allowed ingress and egress targets in a network-policy.yaml file reviewed during PR. External egress is blocked by default — services requiring external API access must request a firewall rule through the Security team.

Load Balancing
External traffic enters through AWS ALB (Application Load Balancer) with WAF (Web Application Firewall) rules. Internal traffic uses Kubernetes service load balancing. For latency-sensitive services, topology-aware routing directs traffic to the nearest available pod. Health checks run every 10 seconds with a 3-failure threshold for removing unhealthy pods.

DNS and Certificates
Internal DNS is managed through Route 53 private hosted zones. External DNS uses Cloudflare with automatic DNSSEC. TLS certificates are provisioned automatically through cert-manager using Let's Encrypt for external services and the internal CA for inter-service communication. Certificate rotation happens automatically 30 days before expiration.
"""
    },
    {
        "title": "Employee Handbook — Benefits Overview",
        "source": "handbook_benefits.pdf",
        "content": """
Benefits Overview — ACME Corporation Employee Handbook, Section 3.1

Health Insurance
ACME offers three health insurance plans through UnitedHealthcare: Standard PPO ($150/month employee contribution), Premium PPO ($250/month), and High-Deductible Health Plan with HSA ($75/month). Dental and vision coverage is included in all plans at no additional cost. Coverage begins on the first day of employment. Dependents can be added during open enrollment (November) or within 30 days of a qualifying life event.

Retirement Benefits
The company offers a 401(k) plan with immediate eligibility and a 50% employer match up to 6% of salary (effectively 3% match). Vesting is immediate for employee contributions and follows a 3-year cliff vesting schedule for employer matching. An after-tax Roth 401(k) option is also available. Financial planning consultations are available quarterly through Fidelity.

Paid Time Off
Employees receive PTO based on tenure: 0-2 years: 15 days, 3-5 years: 20 days, 6+ years: 25 days. PTO accrues per pay period and can be carried over up to 5 days into the next calendar year. Unused PTO beyond the carryover limit is forfeited. Additionally, ACME provides 10 paid holidays, 5 sick days, and 3 personal days annually.

Parental Leave
Primary caregivers receive 16 weeks of fully paid parental leave. Secondary caregivers receive 8 weeks of fully paid leave. Parental leave can be taken within 12 months of birth or adoption. Employees may request a gradual return-to-work schedule for up to 4 weeks following parental leave.

Professional Development
ACME provides an annual learning budget of $3,000 per employee for courses, certifications, conferences, and books. Tuition reimbursement is available for degree programs up to $10,000 per year, subject to manager approval and a minimum grade of B. Employees may take up to 5 days of paid study leave per year for exam preparation.
"""
    },
    {
        "title": "Product Spec — ACME Shield Security Platform",
        "source": "spec_shield.pdf",
        "content": """
ACME Shield — Product Specification Document v2.1

Product Overview
ACME Shield is a zero-trust security platform designed for enterprise environments. The platform provides continuous identity verification, micro-segmentation, and real-time threat detection across hybrid cloud infrastructures. ACME Shield integrates with major identity providers (Okta, Azure AD, Google Workspace) and cloud platforms (AWS, Azure, GCP).

Core Features
Identity Verification: Continuous authentication using behavioral biometrics and device trust scoring. Users are re-verified based on risk signals rather than static session timeouts. Risk scores combine device health, location anomalies, and behavioral patterns.

Micro-Segmentation: Network traffic is segmented at the application layer using policy-as-code. Policies are defined in YAML and version-controlled. Changes are automatically tested against simulation environments before deployment. Lateral movement between segments requires explicit authorization.

Threat Detection: Real-time analysis of network flows, API calls, and user behavior using ML-based anomaly detection. The detection engine processes 50,000 events per second per node. Alert fatigue is reduced through automated correlation — related alerts are grouped into incidents with confidence scores.

Compliance and Certifications
ACME Shield has achieved: FedRAMP Moderate authorization, SOC 2 Type II compliance, ISO 27001 certification, and HIPAA readiness. The platform supports automated compliance reporting for PCI-DSS, GDPR, and CCPA requirements. Audit logs are tamper-proof and retained for 7 years.

Deployment Architecture
ACME Shield deploys as a Kubernetes operator in the customer's environment. The control plane runs in ACME's cloud; the data plane runs locally. No customer data leaves the customer's environment — only anonymized telemetry is sent to the control plane for ML model updates. Initial deployment takes approximately 4 hours with the guided setup wizard.
"""
    },
    {
        "title": "Engineering Wiki — Incident Management Runbook",
        "source": "wiki_incidents.pdf",
        "content": """
Incident Management Runbook — Engineering Wiki

Severity Classification
P0 (Critical): Complete service outage affecting all users. Revenue impact > $100K/hour. Target response: 5 minutes. Target resolution: 1 hour.
P1 (Major): Significant degradation affecting >25% of users. Revenue impact > $10K/hour. Target response: 15 minutes. Target resolution: 4 hours.
P2 (Minor): Limited impact affecting <25% of users or non-critical functionality. Target response: 1 hour. Target resolution: 24 hours.
P3 (Low): Cosmetic issues, minor bugs, or improvement requests. Target resolution: 1 sprint.

On-Call Rotation
Each team maintains a primary and secondary on-call rotation using PagerDuty. On-call shifts are 1 week, Monday to Monday. The primary on-call must acknowledge pages within 5 minutes. If unacknowledged, the page escalates to the secondary on-call, then to the Engineering Manager. On-call engineers receive a $500/week stipend and compensatory time off.

Incident Communication
P0/P1 incidents require a dedicated Slack channel named #inc-YYYYMMDD-description. The Incident Commander (IC) posts status updates every 15 minutes for P0 and every 30 minutes for P1. Customer-facing incidents require coordination with the Customer Support team who manage external communications through StatusPage. The VP of Engineering is notified automatically for all P0 incidents.

War Room Protocol
P0 incidents automatically create a Zoom bridge. The IC assigns roles: IC (coordinates response), Tech Lead (drives investigation), Communications Lead (manages stakeholder updates), and Scribe (documents timeline). Decisions are made by the IC with input from the Tech Lead. Disagreements are escalated to the VP of Engineering.

Post-Mortem Process
All P0/P1 incidents require a written post-mortem within 48 hours. The post-mortem template includes: incident summary, timeline of events, impact assessment (users affected, duration, revenue impact), root cause analysis (using the 5 Whys method), contributing factors, action items with owners and deadlines, and lessons learned. Post-mortems are presented in the weekly engineering all-hands meeting.
"""
    }
]

DOCUMENTS = DOCUMENTS + DISTRACTOR_DOCUMENTS

print(f"Loaded {len(DOCUMENTS)} documents")
for doc in DOCUMENTS:
    print(f"  • {doc['title']} ({len(doc['content'].split())} words)")

#Excercise 1 - chunking

# --- Strategy 1: Fixed-size (the naive way) ---

def chunk_fixed(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """Split text into fixed-size word chunks with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

# --- Strategy 2: Recursive/Semantic (split on natural boundaries) ---

def chunk_recursive(text: str, max_size: int = 200) -> List[str]:
    """Split text on natural boundaries: paragraphs, then sentences."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= max_size:
            chunks.append(para)
        else:
            sentences = para.replace(". ", ".\n").split("\n")
            current, current_len = [], 0
            for sent in sentences:
                sent_len = len(sent.split())
                if current_len + sent_len > max_size and current:
                    chunks.append(" ".join(current))
                    current, current_len = [sent], sent_len
                else:
                    current.append(sent)
                    current_len += sent_len
            if current:
                chunks.append(" ".join(current))

    return [c for c in chunks if len(c.split()) > 10]

# --- Compare them ---

sample = DOCUMENTS[0]["content"]

fixed = chunk_fixed(sample, chunk_size=100, overlap=20)
recursive = chunk_recursive(sample, max_size=100)

print("=" * 60)
print("FIXED-SIZE CHUNKING")
print("=" * 60)
for i, c in enumerate(fixed[:3]):
    print(f"\nChunk {i+1} ({len(c.split())} words):")
    print(c[:150], "...")

print("\n" + "=" * 60)
print("RECURSIVE CHUNKING")
print("=" * 60)
for i, c in enumerate(recursive[:3]):
    print(f"\nChunk {i+1} ({len(c.split())} words):")
    print(c[:150], "...")

print("\n💡 Fixed cuts mid-thought. Recursive respects paragraph boundaries.")

#Exercise 2: Naive RAG — Build it, then break it

import chromadb
from typing import List
from sentence_transformers import SentenceTransformer


# -------------------------------
# EMBEDDINGS (FREE - LOCAL)
# -------------------------------
print("🔄 Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(texts):
    return embedding_model.encode(texts).tolist()

def get_embedding(text):
    return embedding_model.encode([text])[0].tolist()

# -------------------------------
# CHUNK ALL DOCUMENTS
# -------------------------------
all_chunks = []
chunk_meta = []

for doc in DOCUMENTS:
    chunks = chunk_recursive(doc["content"], max_size=100)

    for chunk in chunks:
        all_chunks.append(chunk)
        chunk_meta.append({
            "title": doc["title"],
            "source": doc["source"]
        })

print(f"\n📄 Total chunks: {len(all_chunks)}")

# -------------------------------
# VECTOR DB (Chroma)
# -------------------------------
print("🔄 Creating Chroma DB...")

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="rag_demo")

# -------------------------------
# STORE EMBEDDINGS
# -------------------------------
print("🔄 Generating embeddings...")

embeddings = get_embeddings(all_chunks)

for i, (chunk, emb, meta) in enumerate(zip(all_chunks, embeddings, chunk_meta)):
    collection.add(
        ids=[str(i)],
        embeddings=[emb],
        documents=[chunk],
        metadatas=[meta]
    )

print("✅ Stored all chunks in vector DB")

# -------------------------------
# RETRIEVAL FUNCTION
# -------------------------------
def retrieve(query: str, top_k: int = 3):
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results["documents"][0], results["metadatas"][0]

# -------------------------------
# RAG ANSWER (Simple Version)
# -------------------------------
def rag_answer(query: str):
    docs, metas = retrieve(query)

    context = "\n\n".join(docs)

    answer = f"""
🔍 Query: {query}

📚 Retrieved Context:
{context}

💡 Answer (based on retrieved info):
{docs[0]}
"""
    return answer

# -------------------------------
# TEST QUERIES
# -------------------------------
if __name__ == "__main__":
    queries = [
        "What is the refund period?",
        "How many leave days do employees get?",
        "Are digital products refundable?"
    ]

    for q in queries:
        print("\n" + "=" * 60)
        print(rag_answer(q))

#Exercise 3: Hybrid Search — Semantic + BM25

import re
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi

# -------------------------------
# TOKENIZER (for BM25)
# -------------------------------
def tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase + words only"""
    return re.findall(r'\w+', text.lower())

# -------------------------------
# BUILD BM25 INDEX
# -------------------------------
bm25 = BM25Okapi([tokenize(c) for c in all_chunks])

print(f"✅ Built BM25 index over {len(all_chunks)} chunks")

# -------------------------------
# RECIPROCAL RANK FUSION (RRF)
# -------------------------------
def reciprocal_rank_fusion(
    semantic: List[Tuple[int, float]],
    keyword: List[Tuple[int, float]],
    k: int = 60
) -> List[Tuple[int, float]]:
    """
    Combine semantic + keyword rankings
    """
    scores = {}

    # Semantic scores
    for rank, (idx, _) in enumerate(semantic):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

    # Keyword scores
    for rank, (idx, _) in enumerate(keyword):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

    # Sort by combined score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# -------------------------------
# HYBRID SEARCH
# -------------------------------
def hybrid_search(question: str, k: int = 5) -> List[Dict]:
    """Combine semantic + BM25 search"""

    # -----------------------
    # 1. Semantic Search
    # -----------------------
    sem = collection.query(
        query_embeddings=[get_embedding(question)],
        n_results=20
    )

    # FIXED: IDs are "0", "1", "2" → no split needed
    sem_ranked = [
        (int(id), dist)
        for id, dist in zip(sem["ids"][0], sem["distances"][0])
    ]

    # -----------------------
    # 2. BM25 Keyword Search
    # -----------------------
    scores = bm25.get_scores(tokenize(question))

    bm25_ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )[:20]

    # -----------------------
    # 3. Fuse Results (RRF)
    # -----------------------
    fused = reciprocal_rank_fusion(sem_ranked, bm25_ranked)

    # -----------------------
    # 4. Return Top Results
    # -----------------------
    return [
        {
            "chunk": all_chunks[idx],
            "meta": chunk_meta[idx],
            "score": sc
        }
        for idx, sc in fused[:k]
    ]

# -------------------------------
# HYBRID RAG (NO LLM - FREE)
# -------------------------------
def hybrid_rag(question: str, k: int = 5, verbose: bool = True) -> str:
    """Hybrid RAG without LLM (returns best chunk)"""

    results = hybrid_search(question, k)

    if verbose:
        print(f"\n🔍 Query: '{question}'")
        print(f"\n📚 Retrieved {len(results)} chunks (Hybrid Search):")

        for i, r in enumerate(results):
            print(f"  [{i+1}] RRF Score={r['score']:.4f} | {r['meta']['title']}")
            print(f"      {r['chunk'][:100]}...")

    # -----------------------
    # SIMPLE ANSWER (FREE)
    # -----------------------
    best_chunk = results[0]["chunk"]

    answer = f"""
💡 Answer (from best match):
{best_chunk}
"""

    print(answer)
    return answer

# -------------------------------
# TEST QUERIES
# -------------------------------
if __name__ == "__main__":
    queries = [
        "What is the refund period?",
        "When will I get my money back?",
        "How many leave days do employees get?",
        "Are digital products refundable?"
    ]

    for q in queries:
        print("\n" + "=" * 60)
        hybrid_rag(q)

#Exercise 4: Contextual Retrieval — Anthropic's Technique

#Before embedding, use an LLM to prepend context to each chunk.

#BEFORE: "The company's revenue grew by 3%..."
#AFTER:  "[From ACME Corp Q2 2023 SEC filing] The company's revenue grew by 3%..."
#The embedding now captures what the chunk is about, not just what it says.

import re
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi

# -------------------------------
# SIMPLE TOKENIZER
# -------------------------------
def tokenize(text: str) -> List[str]:
    return re.findall(r'\w+', text.lower())

# -------------------------------
# FREE CONTEXTUALIZATION (NO LLM)
# -------------------------------
def contextualize_chunk_free(chunk: str, full_doc: str, title: str) -> str:
    """
    Add simple context WITHOUT LLM
    """

    # Take first 30 words of document as summary
    doc_summary = " ".join(full_doc.split()[:30])

    context = f"""
[Document: {title}]
[Summary: {doc_summary}]
[Section Content]:
"""

    return context.strip() + "\n\n" + chunk

print("🔄 Contextualizing chunks (FREE version)...")

ctx_chunks = []

for i, (chunk, meta) in enumerate(zip(all_chunks, chunk_meta)):
    doc = next(d for d in DOCUMENTS if d["title"] == meta["title"])

    ctx_chunk = contextualize_chunk_free(
        chunk,
        doc["content"],
        doc["title"]
    )

    ctx_chunks.append(ctx_chunk)

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(all_chunks)} done")

print(f"\n✅ Contextualized {len(ctx_chunks)} chunks (FREE)")

# -------------------------------
# BEFORE vs AFTER
# -------------------------------
print("\n--- BEFORE ---")
print(all_chunks[0][:200])

print("\n--- AFTER ---")
print(ctx_chunks[0][:300])

# -------------------------------
# STORE IN CHROMA
# -------------------------------
try:
    chroma_client.delete_collection("ctx_rag")
except:
    pass

ctx_collection = chroma_client.create_collection(name="ctx_rag")

print("🔄 Generating embeddings...")

ctx_embs = get_embeddings(ctx_chunks)

ctx_collection.add(
    ids=[f"ctx_{i}" for i in range(len(ctx_chunks))],
    embeddings=ctx_embs,
    documents=ctx_chunks,
    metadatas=chunk_meta
)

print(f"✅ Stored {len(ctx_chunks)} contextualized chunks")

# -------------------------------
# BM25 ON CONTEXTUAL CHUNKS
# -------------------------------
ctx_bm25 = BM25Okapi([tokenize(c) for c in ctx_chunks])

# -------------------------------
# RRF (same as before)
# -------------------------------
def reciprocal_rank_fusion(
    semantic: List[Tuple[int, float]],
    keyword: List[Tuple[int, float]],
    k: int = 60
) -> List[Tuple[int, float]]:

    scores = {}

    for rank, (idx, _) in enumerate(semantic):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

    for rank, (idx, _) in enumerate(keyword):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# -------------------------------
# CONTEXTUAL HYBRID SEARCH
# -------------------------------
def ctx_hybrid_search(question: str, k: int = 5) -> List[Dict]:

    # Semantic
    sem = ctx_collection.query(
        query_embeddings=[get_embedding(question)],
        n_results=20
    )

    sem_ranked = [
        (int(id.split("_")[1]), dist)
        for id, dist in zip(sem["ids"][0], sem["distances"][0])
    ]

    # BM25
    scores = ctx_bm25.get_scores(tokenize(question))
    bm25_ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:20]

    # Fuse
    fused = reciprocal_rank_fusion(sem_ranked, bm25_ranked)

    return [
        {
            "chunk": ctx_chunks[idx],
            "original": all_chunks[idx],
            "meta": chunk_meta[idx],
            "score": sc
        }
        for idx, sc in fused[:k]
    ]

# -------------------------------
# TEST
# -------------------------------
if __name__ == "__main__":
    query = "What is ACME's AI strategy and how does it connect to current products?"

    results = ctx_hybrid_search(query)

    print("\n🔍 Query:", query)
    print("\nTop Results:\n")

    for i, r in enumerate(results):
        print(f"[{i+1}] {r['meta']['title']} (Score: {r['score']:.4f})")
        print(f"Contextual Chunk:\n{r['chunk'][:200]}...\n")
