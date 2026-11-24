# Invite-only private workspace beta API

Enable the feature by setting APICALAI_WORKSPACE_DB to a writable SQLite path
(e.g. /app/data/workspaces.sqlite3 in Compose), APICALAI_WORKSPACE_INVITE to
a long random invite, and optionally APICALAI_WORKSPACE_QUOTA (default 100).
Use HTTPS if ever accessed remotely. Disabled by default.

All operations are POST /api/workspace/<operation> with JSON:
- register: username, password (12+ characters), invite
- login: username, password; returns an 8-hour session token
- logout: bearer session token
- documents: bearer token; returns only own document ids
- save: bearer token; id and text; up to 20 docs of 20,000 chars per account
- delete: bearer token; id
- ask: bearer token; question; retrieves own documents, with UTC daily query quota

Account ids are derived only from a hashed bearer session, never client input.
Passwords use salted PBKDF2-HMAC-SHA256; session tokens are hashed at rest.
Workspace documents are explicitly saved as plaintext in the private server DB;
operators can access them. Do not use sensitive data. General inference
endpoints still use separate shared API-key protection; no billing or universal
per-user inference quota exists. Password recovery, throttled authentication,
TLS, backups, audit logging, retention/deletion policy, abuse prevention and
independent security review are required before public beta.
