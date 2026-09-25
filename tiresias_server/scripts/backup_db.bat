@echo off
setlocal enabledelayedexpansion

:: =============================================================================
:: TIRESIAS Recommendation Backend - Automated SQLite Backup & Maintenance (Windows)
:: =============================================================================
:: Performs:
::   1. Safe online backup using sqlite3 .backup API (lock-free WAL replication)
::   2. Gzip compression of the snapshot
::   3. Backup rotation (retains snapshots for 7 days)
::   4. WAL checkpoint truncation and index optimization
:: =============================================================================

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."

:: Resolve source database path
if "%TIRESIAS_DB_PATH%"=="" (
    set "DB_PATH=%PROJECT_DIR%\db\tiresias_user.db"
) else (
    set "DB_PATH=%TIRESIAS_DB_PATH%"
)

if not exist "%DB_PATH%" (
    if exist "%PROJECT_DIR%\..\data\tiresias_user.db" (
        set "DB_PATH=%PROJECT_DIR%\..\data\tiresias_user.db"
    )
)

:: Resolve backup directory
if "%TIRESIAS_BACKUP_DIR%"=="" (
    set "BACKUP_DIR=%PROJECT_DIR%\backups"
) else (
    set "BACKUP_DIR=%TIRESIAS_BACKUP_DIR%"
)

if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"

:: Generate timestamp YYYYMMDD_HHMMSS
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do set dt=%%I
if "%dt%"=="" (
    set TIMESTAMP=%date:~-4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%%time:~6,2%
    set TIMESTAMP=%TIMESTAMP: =0%
) else (
    set TIMESTAMP=%dt:~0,8%_%dt:~8,6%
)

set "BACKUP_FILE=%BACKUP_DIR%\tiresias_user_%TIMESTAMP%.db"

echo [%date% %time%] Starting SQLite online backup...
echo [INFO] Source database: %DB_PATH%
echo [INFO] Destination dir: %BACKUP_DIR%

if not exist "%DB_PATH%" (
    echo [ERROR] Database file not found at: %DB_PATH%
    exit /b 1
)

:: Online backup using sqlite3 CLI if available, otherwise Python sqlite3 API
where sqlite3 >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [INFO] Executing online backup via sqlite3 CLI...
    sqlite3 "%DB_PATH%" ".backup '%BACKUP_FILE%'"
    echo [INFO] Performing WAL checkpoint and database optimization...
    sqlite3 "%DB_PATH%" "PRAGMA wal_checkpoint(TRUNCATE); PRAGMA optimize;"
) else (
    echo [INFO] sqlite3 CLI not found in PATH; executing online backup via Python sqlite3 API...
    python -c "import sqlite3; s = sqlite3.connect(r'%DB_PATH%'); d = sqlite3.connect(r'%BACKUP_FILE%'); s.backup(d); d.close(); s.execute('PRAGMA wal_checkpoint(TRUNCATE)'); s.execute('PRAGMA optimize'); s.close()"
)

if not exist "%BACKUP_FILE%" (
    echo [ERROR] Backup snapshot creation failed!
    exit /b 1
)

echo [INFO] Compressing backup with gzip...
python -c "import gzip, shutil, os; f_in = open(r'%BACKUP_FILE%', 'rb'); f_out = gzip.open(r'%BACKUP_FILE%.gz', 'wb'); shutil.copyfileobj(f_in, f_out); f_in.close(); f_out.close(); os.remove(r'%BACKUP_FILE%')"

if exist "%BACKUP_FILE%.gz" (
    echo [SUCCESS] Backup created and compressed: %BACKUP_FILE%.gz
) else (
    echo [ERROR] Compression failed!
    exit /b 1
)

:: Retention rotation: delete backups older than 7 days
echo [INFO] Rotating backups older than 7 days...
forfiles /p "%BACKUP_DIR%" /m "tiresias_user_*.db.gz" /d -7 /c "cmd /c del @path" 2>nul

echo [%date% %time%] SQLite backup and maintenance completed successfully.
