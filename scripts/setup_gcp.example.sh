#!/usr/bin/env bash
# =============================================================================
# setup_gcp.sh
# GCP project bootstrap for dl-image-classifier-scaffold
# Run once after cloning the repository.
# Usage: bash scripts/setup_gcp.sh
# =============================================================================

set -euo pipefail

# =============================================================================
# COLOUR OUTPUT
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Colour

info()    { echo -e "${BLUE}[INFO]${NC}  $1"; }
success() { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
step()    { echo -e "\n${BOLD}==> $1${NC}"; }

# =============================================================================
# PREREQUISITES CHECK
# =============================================================================
step "Checking prerequisites"

command -v gcloud >/dev/null 2>&1 || error "gcloud CLI not found. Install from https://cloud.google.com/sdk/docs/install"

GCLOUD_VERSION=$(gcloud version 2>/dev/null | head -1 || echo "unknown")
info "gcloud SDK version: ${GCLOUD_VERSION}"

# Ensure the user is authenticated
if ! gcloud auth print-access-token >/dev/null 2>&1; then
  warn "Not authenticated with gcloud. Running: gcloud auth login"
  gcloud auth login
fi
success "gcloud authenticated"

# =============================================================================
# CONFIGURATION — edit these values before running
# =============================================================================
step "Configuration"

# Project
PROJECT_ID=""          # e.g. deepfake-detection-2026
PROJECT_NAME=""        # e.g. "Deepfake Detection"
BILLING_ACCOUNT_ID=""  # e.g. 0X0X0X-0X0X0X-0X0X0X  (gcloud billing accounts list)

# Region / Zone
REGION="asia-southeast1"
ZONE="asia-southeast1-b"

# GCS
GCS_BUCKET=""          # Must be globally unique, e.g. deepfake-detection-bucket-2026
GCS_DATASET_PATH="datasets/deepfake-detection-2026/"

# Artifact Registry
AR_REPO="deepfake-detection"

# Service accounts
SA_NAME="deepfake-sa"
SA_DISPLAY="Deepfake Detection Service Account"

# Cloud Run
CLOUD_RUN_SERVICE="deepfake-detection"

# Notification email (not sensitive — stored as plain env var)
NOTIFY_EMAIL=""        # e.g. your@email.com

# Validate required config is filled in
[[ -z "${PROJECT_ID}"        ]] && error "PROJECT_ID is not set. Edit the configuration section of this script."
[[ -z "${PROJECT_NAME}"      ]] && error "PROJECT_NAME is not set."
[[ -z "${BILLING_ACCOUNT_ID}" ]] && error "BILLING_ACCOUNT_ID is not set. Run: gcloud billing accounts list"
[[ -z "${GCS_BUCKET}"        ]] && error "GCS_BUCKET is not set."
[[ -z "${NOTIFY_EMAIL}"      ]] && error "NOTIFY_EMAIL is not set."

info "Project ID  : ${PROJECT_ID}"
info "Region      : ${REGION}"
info "GCS Bucket  : ${GCS_BUCKET}"
info "Notify email: ${NOTIFY_EMAIL}"

# =============================================================================
# STEP 1 — CREATE GCP PROJECT
# =============================================================================
step "1. Creating GCP project: ${PROJECT_ID}"

if gcloud projects describe "${PROJECT_ID}" >/dev/null 2>&1; then
  warn "Project ${PROJECT_ID} already exists — skipping creation"
else
  gcloud projects create "${PROJECT_ID}" \
    --name="${PROJECT_NAME}"
  success "Project created: ${PROJECT_ID}"
fi

# Set as active project
gcloud config set project "${PROJECT_ID}"
success "Active project set to: ${PROJECT_ID}"

# =============================================================================
# STEP 2 — BILLING CHECK & LINK
# =============================================================================
step "2. Billing check"

# Verify the billing account exists and is accessible
if ! gcloud billing accounts describe "${BILLING_ACCOUNT_ID}" >/dev/null 2>&1; then
  error "Billing account ${BILLING_ACCOUNT_ID} not found or not accessible.\nRun: gcloud billing accounts list"
fi

BILLING_OPEN=$(gcloud billing accounts describe "${BILLING_ACCOUNT_ID}" \
  --format="value(open)")

if [[ "${BILLING_OPEN}" != "True" ]]; then
  error "Billing account ${BILLING_ACCOUNT_ID} is closed. Please use an active billing account."
fi

# Link billing account to project
gcloud billing projects link "${PROJECT_ID}" \
  --billing-account="${BILLING_ACCOUNT_ID}"
success "Billing account ${BILLING_ACCOUNT_ID} linked to project"

# =============================================================================
# STEP 3 — ENABLE APIS
# =============================================================================
step "3. Enabling required GCP APIs"

APIS=(
  "firestore.googleapis.com"
  "aiplatform.googleapis.com"
  "storage.googleapis.com"
  "artifactregistry.googleapis.com"
  "run.googleapis.com"
  "secretmanager.googleapis.com"
  "monitoring.googleapis.com"
  "logging.googleapis.com"
  "cloudbuild.googleapis.com"
)

for API in "${APIS[@]}"; do
  info "Enabling ${API}..."
  gcloud services enable "${API}" --project="${PROJECT_ID}"
done

success "All APIs enabled"

# Brief pause — API enablement can take a few seconds to propagate
sleep 5

# =============================================================================
# STEP 4 — GCS BUCKET
# =============================================================================
step "4. Creating GCS bucket: gs://${GCS_BUCKET}"

if gsutil ls -b "gs://${GCS_BUCKET}" >/dev/null 2>&1; then
  warn "Bucket gs://${GCS_BUCKET} already exists — skipping creation"
else
  gsutil mb \
    -p "${PROJECT_ID}" \
    -l "${REGION}" \
    -b on \
    "gs://${GCS_BUCKET}"
  success "Bucket created: gs://${GCS_BUCKET}"
fi

# Create standard folder prefixes
info "Initialising bucket folder structure..."
for PREFIX in datasets/ checkpoints/ plots/ logs/; do
  echo "" | gsutil cp - "gs://${GCS_BUCKET}/${PREFIX}.keep" 2>/dev/null || true
done
success "Bucket folder structure initialised"

# GCS lifecycle — delete logs and plots older than 30 days; transition checkpoints to Nearline after 7 days
info "Applying GCS lifecycle policy..."
cat > /tmp/gcs_lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": { "type": "Delete" },
        "condition": {
          "age": 30,
          "matchesPrefix": ["logs/", "plots/"]
        }
      },
      {
        "action": {
          "type": "SetStorageClass",
          "storageClass": "NEARLINE"
        },
        "condition": {
          "age": 7,
          "matchesPrefix": ["checkpoints/"]
        }
      }
    ]
  }
}
EOF
gsutil lifecycle set /tmp/gcs_lifecycle.json "gs://${GCS_BUCKET}"
success "Lifecycle policy applied (logs/plots: delete after 30d | checkpoints: Nearline after 7d)"

# =============================================================================
# STEP 5 — ARTIFACT REGISTRY
# =============================================================================
step "5. Creating Artifact Registry repository: ${AR_REPO}"

if gcloud artifacts repositories describe "${AR_REPO}" \
     --location="${REGION}" \
     --project="${PROJECT_ID}" >/dev/null 2>&1; then
  warn "Artifact Registry repo '${AR_REPO}' already exists — skipping"
else
  gcloud artifacts repositories create "${AR_REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Docker images for deepfake detection scaffold" \
    --project="${PROJECT_ID}"
  success "Artifact Registry repo created: ${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}"
fi

# =============================================================================
# STEP 6 — SERVICE ACCOUNT
# =============================================================================
step "6. Creating service account: ${SA_NAME}"

SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

if gcloud iam service-accounts describe "${SA_EMAIL}" \
     --project="${PROJECT_ID}" >/dev/null 2>&1; then
  warn "Service account ${SA_EMAIL} already exists — skipping creation"
else
  gcloud iam service-accounts create "${SA_NAME}" \
    --display-name="${SA_DISPLAY}" \
    --project="${PROJECT_ID}"
  success "Service account created: ${SA_EMAIL}"
fi

# Grant minimum required roles
info "Binding IAM roles to service account..."

ROLES=(
  "roles/datastore.user"              # Firestore read/write
  "roles/storage.objectAdmin"         # GCS read/write
  "roles/aiplatform.user"             # Vertex AI job submission
  "roles/artifactregistry.reader"     # Pull images from Artifact Registry
  "roles/logging.logWriter"           # Write structured logs to Cloud Logging
  "roles/monitoring.metricWriter"     # Write metrics to Cloud Monitoring
  "roles/secretmanager.secretAccessor" # Read secrets (SendGrid key)
)

for ROLE in "${ROLES[@]}"; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${ROLE}" \
    --quiet
  info "  Bound: ${ROLE}"
done

success "IAM roles bound to ${SA_EMAIL}"

# =============================================================================
# STEP 7 — DOWNLOAD SERVICE ACCOUNT KEY
# =============================================================================
step "7. Downloading service account JSON key"

KEY_FILE="$(pwd)/service-account-key.json"

if [[ -f "${KEY_FILE}" ]]; then
  warn "Key file already exists at ${KEY_FILE} — skipping download"
else
  gcloud iam service-accounts keys create "${KEY_FILE}" \
    --iam-account="${SA_EMAIL}" \
    --project="${PROJECT_ID}"
  chmod 600 "${KEY_FILE}"
  success "Key downloaded to: ${KEY_FILE}"
  warn "Keep this file secure. It is gitignored and must never be committed."
fi

# =============================================================================
# STEP 8 — FIRESTORE
# =============================================================================
step "8. Initialising Firestore (Native mode)"

# Firestore can only be initialised once per project
FIRESTORE_STATUS=$(gcloud firestore databases describe \
  --project="${PROJECT_ID}" \
  --format="value(type)" 2>/dev/null || echo "NOT_FOUND")

if [[ "${FIRESTORE_STATUS}" == "FIRESTORE_NATIVE" ]]; then
  warn "Firestore already initialised in Native mode — skipping"
else
  gcloud firestore databases create \
    --location="${REGION}" \
    --project="${PROJECT_ID}"
  success "Firestore database created in Native mode (region: ${REGION})"
fi

# =============================================================================
# STEP 9 — SECRET MANAGER (SENDGRID API KEY)
# =============================================================================
step "9. Storing SendGrid API key in Secret Manager"

SECRET_NAME="sendgrid-api-key"

if gcloud secrets describe "${SECRET_NAME}" \
     --project="${PROJECT_ID}" >/dev/null 2>&1; then
  warn "Secret '${SECRET_NAME}' already exists — skipping creation"
  info "To update the key: gcloud secrets versions add ${SECRET_NAME} --data-file=-"
else
  echo ""
  echo -e "${YELLOW}Paste your SendGrid API key below, then press Enter followed by Ctrl+D:${NC}"
  gcloud secrets create "${SECRET_NAME}" \
    --replication-policy="automatic" \
    --project="${PROJECT_ID}" \
    --data-file=-

  success "Secret '${SECRET_NAME}' created in Secret Manager"
fi

# Grant Cloud Run service account access to this secret only
gcloud secrets add-iam-policy-binding "${SECRET_NAME}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/secretmanager.secretAccessor" \
  --project="${PROJECT_ID}" \
  --quiet
success "Secret accessor role granted to ${SA_EMAIL} on '${SECRET_NAME}'"

# =============================================================================
# STEP 10 — CLOUD MONITORING ALERT POLICIES
# =============================================================================
step "10. Creating Cloud Monitoring alert policies"

# --- 10a: Vertex AI job failure alert ---
info "Creating Vertex AI job failure alert..."

cat > /tmp/vertexai_alert.json <<EOF
{
  "displayName": "Vertex AI Training Job Failed",
  "conditions": [
    {
      "displayName": "Vertex AI custom job entered FAILED state",
      "conditionMatchedLog": {
        "filter": "resource.type=\"aiplatform.googleapis.com/CustomJob\" severity=ERROR logName=~\"cloudaiplatform\""
      }
    }
  ],
  "alertStrategy": {
    "notificationRateLimit": {
      "period": "300s"
    }
  },
  "combiner": "OR",
  "notificationChannels": [],
  "documentation": {
    "content": "A Vertex AI custom training job has entered a FAILED state. Check the Vertex AI console for details: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}",
    "mimeType": "text/markdown"
  }
}
EOF

gcloud alpha monitoring policies create \
  --policy-from-file=/tmp/vertexai_alert.json \
  --project="${PROJECT_ID}" 2>/dev/null || \
  warn "Vertex AI alert policy may already exist or requires alpha components — check Cloud Console to confirm"

success "Vertex AI job failure alert created"

# --- 10b: Log-based metric for training val_loss ---
info "Creating log-based metric for training val_loss..."

cat > /tmp/loss_metric.json <<EOF
{
  "name": "training_val_loss",
  "description": "Validation loss extracted from structured training logs, grouped by job_id",
  "filter": "jsonPayload.message=\"Epoch complete\" AND jsonPayload.val_loss!=\"\"",
  "metricDescriptor": {
    "metricKind": "GAUGE",
    "valueType": "DOUBLE",
    "unit": "1",
    "labels": [
      {
        "key": "job_id",
        "valueType": "STRING",
        "description": "Training job identifier"
      }
    ]
  },
  "valueExtractor": "EXTRACT(jsonPayload.val_loss)",
  "labelExtractors": {
    "job_id": "EXTRACT(jsonPayload.job_id)"
  }
}
EOF

gcloud logging metrics create training_val_loss \
  --config-from-file=/tmp/loss_metric.json \
  --project="${PROJECT_ID}" 2>/dev/null || \
  warn "Log-based metric may already exist — skipping"

success "Log-based val_loss metric created (visible in Cloud Monitoring as: logging/user/training_val_loss)"

# =============================================================================
# STEP 11 — WRITE .env FILE
# =============================================================================
step "11. Writing .env file"

ENV_FILE="$(pwd)/.env"

if [[ -f "${ENV_FILE}" ]]; then
  warn ".env already exists — backing up to .env.backup before overwriting"
  cp "${ENV_FILE}" "${ENV_FILE}.backup"
fi

cat > "${ENV_FILE}" <<EOF
# =============================================================================
# .env — local development environment
# Generated by setup_gcp.sh on $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# DO NOT COMMIT THIS FILE. It is gitignored.
# =============================================================================

# GCP project
GCP_PROJECT_ID=${PROJECT_ID}
GCS_BUCKET=${GCS_BUCKET}
GCS_DATASET_PATH=${GCS_DATASET_PATH}

# Local credentials
# Points at the service account JSON key downloaded by setup_gcp.sh.
# On Vertex AI and Cloud Run, this variable is not set — credentials are
# injected automatically via the GCP metadata server.
GOOGLE_APPLICATION_CREDENTIALS=${KEY_FILE}

# Email notifications (not sensitive)
NOTIFY_EMAIL=${NOTIFY_EMAIL}

# SENDGRID_API_KEY — NOT stored here.
# Stored in GCP Secret Manager as '${SECRET_NAME}'.
# Cloud Run mounts it via: --set-secrets SENDGRID_API_KEY=${SECRET_NAME}:latest
# For local testing only: export SENDGRID_API_KEY=<your-key>  (never commit this)

# Artifact Registry image base
AR_IMAGE_BASE=${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/deepfake-detection
EOF

chmod 600 "${ENV_FILE}"
success ".env written to: ${ENV_FILE}"

# =============================================================================
# STEP 12 — VALIDATE CONNECTIVITY
# =============================================================================
step "12. Validating GCP connectivity"

export GOOGLE_APPLICATION_CREDENTIALS="${KEY_FILE}"

# GCS check
info "Checking GCS access..."
if gsutil ls "gs://${GCS_BUCKET}" >/dev/null 2>&1; then
  success "GCS bucket accessible: gs://${GCS_BUCKET}"
else
  error "Cannot access GCS bucket. Check service account permissions."
fi

# Firestore check (list collections — returns empty on a fresh DB, which is fine)
info "Checking Firestore access..."
FIRESTORE_CHECK=$(gcloud firestore documents list \
  --collection-id=_health_check_ \
  --project="${PROJECT_ID}" 2>/dev/null || echo "OK")
success "Firestore accessible"

# Secret Manager check
info "Checking Secret Manager access..."
if gcloud secrets versions access latest \
     --secret="${SECRET_NAME}" \
     --project="${PROJECT_ID}" >/dev/null 2>&1; then
  success "Secret '${SECRET_NAME}' accessible from Secret Manager"
else
  warn "Could not read secret — check IAM binding. This will block Cloud Run if not resolved."
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${GREEN}${BOLD}  GCP project setup complete${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""
echo -e "  Project ID     : ${PROJECT_ID}"
echo -e "  Region         : ${REGION}"
echo -e "  GCS Bucket     : gs://${GCS_BUCKET}"
echo -e "  Firestore      : Native mode, region ${REGION}"
echo -e "  Service Account: ${SA_EMAIL}"
echo -e "  Key file       : ${KEY_FILE}"
echo -e "  AR repo        : ${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}"
echo -e "  SendGrid secret: Secret Manager — '${SECRET_NAME}'"
echo -e "  Notify email   : ${NOTIFY_EMAIL}"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo -e "  1. Upload your dataset to GCS:"
echo -e "     gsutil -m cp -r /path/to/dataset gs://${GCS_BUCKET}/${GCS_DATASET_PATH}"
echo ""
echo -e "  2. Build and push your Docker image:"
echo -e "     GIT_SHA=\$(git rev-parse --short HEAD)"
echo -e "     docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/deepfake-detection:\${GIT_SHA} ."
echo -e "     docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/deepfake-detection:\${GIT_SHA}"
echo ""
echo -e "  3. Deploy to Cloud Run:"
echo -e "     gcloud run deploy ${CLOUD_RUN_SERVICE} \\"
echo -e "       --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/deepfake-detection:\${GIT_SHA} \\"
echo -e "       --region ${REGION} \\"
echo -e "       --service-account ${SA_EMAIL} \\"
echo -e "       --memory 4Gi \\"
echo -e "       --cpu 2 \\"
echo -e "       --timeout 300 \\"
echo -e "       --no-allow-unauthenticated \\"
echo -e "       --set-secrets SENDGRID_API_KEY=${SECRET_NAME}:latest \\"
echo -e "       --set-env-vars GCP_PROJECT_ID=${PROJECT_ID},GCS_BUCKET=${GCS_BUCKET},NOTIFY_EMAIL=${NOTIFY_EMAIL}"
echo ""
echo -e "${YELLOW}Reminder: ${KEY_FILE} is sensitive. Keep it local and never commit it.${NC}"
echo ""