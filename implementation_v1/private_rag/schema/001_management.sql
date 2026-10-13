-- Only public management metadata and immutable protected-object references.
CREATE TABLE IF NOT EXISTS schema_version(version integer PRIMARY KEY CHECK(version=1));
INSERT INTO schema_version VALUES(1) ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS accounts(
 account_id uuid PRIMARY KEY, role text NOT NULL CHECK(role IN ('D','U')),
 certificate_sha256 text UNIQUE NOT NULL CHECK(length(certificate_sha256)=64), active boolean NOT NULL DEFAULT true,
 key_version bigint NOT NULL DEFAULT 0, authorization_version bigint NOT NULL DEFAULT 0,
 query_count bigint NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS recipient_keys(
 user_id uuid REFERENCES accounts, key_version bigint NOT NULL CHECK(key_version>0),
 public_key bytea NOT NULL CHECK(octet_length(public_key)=32), active boolean NOT NULL,
 PRIMARY KEY(user_id,key_version));
CREATE TABLE IF NOT EXISTS documents(
 document_id uuid PRIMARY KEY, owner_id uuid NOT NULL REFERENCES accounts, organization_id text NOT NULL,
 internal_title text NOT NULL, current_version bigint);
CREATE TABLE IF NOT EXISTS document_versions(
 document_id uuid REFERENCES documents, version bigint NOT NULL CHECK(version>0), block_count integer NOT NULL CHECK(block_count>0),
 model_numeric_version text NOT NULL CHECK(length(model_numeric_version)=64), state text NOT NULL CHECK(state IN ('staging','published')),
 PRIMARY KEY(document_id,version));
CREATE TABLE IF NOT EXISTS acl(
 document_id uuid REFERENCES documents, user_id uuid REFERENCES accounts, allowed boolean NOT NULL,
 generation bigint NOT NULL CHECK(generation>0), PRIMARY KEY(document_id,user_id));
CREATE TABLE IF NOT EXISTS user_slots(
 document_id uuid REFERENCES documents, user_id uuid REFERENCES accounts, block_index integer NOT NULL CHECK(block_index>=0),
 opaque_slot uuid UNIQUE NOT NULL, PRIMARY KEY(document_id,user_id,block_index));
CREATE TABLE IF NOT EXISTS sessions(
 session_id uuid PRIMARY KEY, account_id uuid REFERENCES accounts, kind text NOT NULL CHECK(kind IN ('prepare','query')),
 state text NOT NULL CHECK(state IN ('accepted','computing','releasing','completed','aborted')),
 snapshot jsonb NOT NULL, started_at timestamptz NOT NULL DEFAULT now(), ended_at timestamptz,
 public_fault text, total_bytes bigint CHECK(total_bytes>=0), release_acknowledged boolean NOT NULL DEFAULT false,
 worker_claimed boolean NOT NULL DEFAULT false, worker_owner uuid,
 cancel_requested boolean NOT NULL DEFAULT false);
-- Unknown measurements stay NULL; zero must mean an actual measured zero.
ALTER TABLE sessions ALTER COLUMN total_bytes DROP NOT NULL;
ALTER TABLE sessions ALTER COLUMN total_bytes DROP DEFAULT;
DO $$ BEGIN
 IF NOT EXISTS (SELECT 1 FROM information_schema.columns
   WHERE table_schema=current_schema() AND table_name='sessions' AND column_name='worker_claimed') THEN
  ALTER TABLE sessions ADD COLUMN worker_claimed boolean NOT NULL DEFAULT false;
  -- A pre-upgrade active row may already have an old live worker. Preserve
  -- the global slot until operator-confirmed cleanup; never infer no worker.
  UPDATE sessions SET worker_claimed=true WHERE state IN ('accepted','computing','releasing');
 END IF;
END $$;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS worker_owner uuid;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS cancel_requested boolean NOT NULL DEFAULT false;
CREATE TABLE IF NOT EXISTS session_cancel_intents(
 account_id uuid NOT NULL REFERENCES accounts, session_id uuid NOT NULL,
 requested_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(account_id,session_id));
CREATE UNIQUE INDEX IF NOT EXISTS one_global_secure_session ON sessions((true)) WHERE state IN ('accepted','computing','releasing');
CREATE TABLE IF NOT EXISTS preparations(
 preparation_id uuid PRIMARY KEY, document_id uuid, document_version bigint, block_index integer NOT NULL,
 session_id uuid NOT NULL REFERENCES sessions, server_object text NOT NULL CHECK(length(server_object)=64),
 FOREIGN KEY(document_id,document_version) REFERENCES document_versions);
ALTER TABLE preparations DROP CONSTRAINT IF EXISTS preparations_document_id_document_version_block_index_key;
CREATE TABLE IF NOT EXISTS deliveries(
 delivery_id uuid PRIMARY KEY, document_id uuid, document_version bigint, user_id uuid REFERENCES accounts,
 opaque_slot uuid REFERENCES user_slots(opaque_slot), block_index integer NOT NULL,
 key_version bigint NOT NULL, share_generation bigint NOT NULL, acl_generation bigint NOT NULL,
 binding jsonb NOT NULL, server_object text NOT NULL CHECK(length(server_object)=64),
 package_object text NOT NULL CHECK(length(package_object)=64),
 FOREIGN KEY(document_id,document_version) REFERENCES document_versions,
 FOREIGN KEY(user_id,key_version) REFERENCES recipient_keys,
 UNIQUE(user_id,opaque_slot,share_generation));
CREATE TABLE IF NOT EXISTS published_deliveries(
 document_id uuid REFERENCES documents, user_id uuid REFERENCES accounts, block_index integer NOT NULL,
 delivery_id uuid NOT NULL REFERENCES deliveries, PRIMARY KEY(document_id,user_id,block_index));
CREATE TABLE IF NOT EXISTS session_references(
 session_id uuid REFERENCES sessions, delivery_id uuid REFERENCES deliveries, PRIMARY KEY(session_id,delivery_id));
CREATE TABLE IF NOT EXISTS materials(
 material_id uuid PRIMARY KEY, session_id uuid NOT NULL REFERENCES sessions, operation_id bigint NOT NULL,
 kind text NOT NULL CHECK(kind IN ('triple','ot_prg','bfv_mask','enc0','session_randomness')),
 descriptor jsonb NOT NULL, state text NOT NULL CHECK(state IN ('reserved','consumed','burned')),
 reserved_at timestamptz NOT NULL DEFAULT now(), finished_at timestamptz,
 UNIQUE(session_id,operation_id,kind));
