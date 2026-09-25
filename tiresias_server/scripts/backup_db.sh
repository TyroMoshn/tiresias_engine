#!/usr/bin/env bash
# ==============================================================================
# TIRESIAS Recommendation Backend - Automated SQLite Backup & Maintenance
# ==============================================================================
# Designed for automated execution via cron on Linux VPS.
# Performs:
#   1. Safe online backup using sqlite3 .backup API (lock-free WAL replication)
#   2. Gzip compression of the resulting backup snapshot
#   3. Backup rotation (retains snapshots for 7 days)
#   4. WAL checkpoint truncation and index optimization
# ==============================================================================

set -euo pipefail

# Directory of this script and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Target SQLite database path (configurable via TIRESIAS_DB_PATH)
DB_PATH="${TIRESIAS_DB_PATH:-${PROJECT_DIR}/db/tiresias_user.db}"
if [[ ! -f "$DB_PATH" && -f "${PROJECT_DIR}/../data/tiresias_user.db" ]]; then
    DB_PATH="${PROJECT_DIR}/../data/tiresias_user.db"
fi

# Target backup destination directory (configurable via TIRESIAS_BACKUP_DIR)
BACKUP_DIR="${TIRESIAS_BACKUP_DIR:-${PROJECT_DIR}/backups}"
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
BACKUP_FILE="${BACKUP_DIR}/tiresias_user_${TIMESTAMP}.db"
RETENTION_DAYS="${TIRESIAS_BACKUP_RETENTION_DAYS:-7}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting automated SQLite backup..."
echo "[INFO] Source database: $DB_PATH"
echo "[INFO] Destination dir: $BACKUP_DIR"

# Verify source database file exists
if [[ ! -f "$DB_PATH" ]]; then
    echo "[ERROR] Database file not found at: $DB_PATH" >&2
    exit 1
fi

# Ensure backup destination directory exists
mkdir -p "$BACKUP_DIR"

# Check required binaries
if ! command -v sqlite3 >/dev/null 2>&1; then
    echo "[ERROR] 'sqlite3' CLI utility is required but not installed." >&2
    exit 1
fi
if ! command -v gzip >/dev/null 2>&1; then
    echo "[ERROR] 'gzip' utility is required but not installed." >&2
    exit 1
fi

# 1. Safe online backup using sqlite3 .backup API
echo "[INFO] Creating online SQLite snapshot..."
sqlite3 "$DB_PATH" ".backup '$BACKUP_FILE'"

# 2. Compress backup with gzip
echo "[INFO] Compressing snapshot with gzip..."
gzip -f "$BACKUP_FILE"
COMPRESSED_FILE="${BACKUP_FILE}.gz"

if [[ -f "$COMPRESSED_FILE" ]]; then
    FILE_SIZE="$(du -h "$COMPRESSED_FILE" | cut -f1)"
    echo "[SUCCESS] Backup created successfully: $COMPRESSED_FILE ($FILE_SIZE)"
else
    echo "[ERROR] Failed to find compressed backup file: $COMPRESSED_FILE" >&2
    exit 1
fi

# 3. Rotate old backups (delete archives older than 7 days)
echo "[INFO] Pruning backup archives older than ${RETENTION_DAYS} days..."
find "$BACKUP_DIR" -name "tiresias_user_*.db.gz" -mtime +"$RETENTION_DAYS" -delete

# 4. Perform database maintenance: WAL checkpoint truncation & index optimization
echo "[INFO] Executing PRAGMA wal_checkpoint(TRUNCATE) and PRAGMA optimize..."
sqlite3 "$DB_PATH" "PRAGMA wal_checkpoint(TRUNCATE); PRAGMA optimize;"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] SQLite backup and maintenance completed successfully."
