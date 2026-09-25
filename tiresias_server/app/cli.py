from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path
from typing import List, Optional

from app.config import ServerConfig
from app.core.auth import UserRole
from app.db.database import Database


ROLE_MAP = {
    "guest": int(UserRole.GUEST),
    "user": int(UserRole.USER),
    "tester": int(UserRole.TESTER),
    "moderator": int(UserRole.MODERATOR),
    "admin": int(UserRole.ADMIN),
    "0": int(UserRole.GUEST),
    "10": int(UserRole.USER),
    "20": int(UserRole.TESTER),
    "50": int(UserRole.MODERATOR),
    "100": int(UserRole.ADMIN),
}


def parse_role(role_raw: str) -> int:
    clean = str(role_raw).strip().lower()
    if clean in ROLE_MAP:
        return ROLE_MAP[clean]
    try:
        val = int(clean)
        return val
    except ValueError:
        raise ValueError(
            f"Invalid role '{role_raw}'. "
            f"Allowed values: guest, user, tester, moderator, admin (or 0, 10, 20, 50, 100)."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="TIRESIAS Security & Administration CLI Tool",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to Tiresias SQLite database (default from TIRESIAS_DB_PATH or server config).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. create-admin
    p_admin = subparsers.add_parser("create-admin", help="Create or update an administrator account")
    p_admin.add_argument("--username", "-u", required=True, help="Administrator username")
    p_admin.add_argument("--password", "-p", required=True, help="Administrator password")
    p_admin.add_argument("--display-name", default=None, help="Display name (optional)")

    # 2. set-role
    p_role = subparsers.add_parser("set-role", help="Change a user's role")
    p_role.add_argument("--user-id", "-u", required=True, help="User ID (user_id) or username")
    p_role.add_argument(
        "--role",
        "-r",
        required=True,
        help="Role: guest (0), user (10), tester (20), moderator (50), admin (100)",
    )

    # 3. invite-add
    p_inv_add = subparsers.add_parser("invite-add", help="Add site user ID to the invite whitelist")
    p_inv_add.add_argument("--site-user-id", "-id", type=int, required=True, help="Numeric site user ID on e621")
    p_inv_add.add_argument("--username", default="", help="Username (optional)")
    p_inv_add.add_argument("--note", default="", help="Note/comment (optional)")

    # 4. invite-list
    subparsers.add_parser("invite-list", help="List users in the invite whitelist")

    # 5. invite-remove
    p_inv_rm = subparsers.add_parser("invite-remove", help="Remove user from the invite whitelist")
    p_inv_rm.add_argument("--site-user-id", "-id", type=int, required=True, help="Numeric site user ID on e621")

    # 6. reset-password
    p_pw = subparsers.add_parser("reset-password", help="Reset user password")
    p_pw.add_argument("--user-id", "-u", required=True, help="User ID (user_id)")
    p_pw.add_argument("--password", "-p", required=True, help="New password")

    return parser


def execute_command(args: argparse.Namespace) -> int:
    cfg = ServerConfig()
    db_path = Path(args.db_path).resolve() if args.db_path else cfg.db_path

    if not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)

    with Database(db_path) as db:
        if args.command == "create-admin":
            user, token = db.create_or_update_admin(
                username=args.username,
                password=args.password,
                display_name=args.display_name,
            )
            print(f"[SUCCESS] Administrator '{user['username']}' successfully created/updated.")
            print(f"  - User ID:      {user['user_id']}")
            print(f"  - Role:         {user['role']} (ADMIN)")
            print(f"  - Access token: {token[:8]}...{token[-6:]}")
            return 0

        elif args.command == "set-role":
            role_int = parse_role(args.role)
            ok = db.set_user_role(args.user_id, role_int)
            if ok:
                role_name = UserRole(role_int).name if role_int in UserRole._value2member_map_ else str(role_int)
                print(f"[SUCCESS] User '{args.user_id}' role updated to {role_name} ({role_int}).")
                return 0
            else:
                print(f"[ERROR] User '{args.user_id}' not found in database.", file=sys.stderr)
                return 1

        elif args.command == "invite-add":
            db.add_invited_user(
                site_user_id=args.site_user_id,
                username=args.username,
                note=args.note,
            )
            print(
                f"[SUCCESS] User site_user_id={args.site_user_id} added to whitelist "
                f"(username='{args.username}', note='{args.note}')."
            )
            return 0

        elif args.command == "invite-list":
            invites = db.list_invited_users()
            if not invites:
                print("[INFO] Whitelist is empty.")
                return 0

            print("=" * 75)
            print(f"{'SITE USER ID':<16} {'USERNAME':<20} {'ADDED AT':<20} {'NOTE'}")
            print("-" * 75)
            for inv in invites:
                dt_str = datetime.datetime.fromtimestamp(inv["created_at"]).strftime("%Y-%m-%d %H:%M:%S")
                print(f"{inv['site_user_id']:<16} {inv.get('username') or '-':<20} {dt_str:<20} {inv.get('note') or ''}")
            print("=" * 75)
            print(f"Total whitelist entries: {len(invites)}")
            return 0

        elif args.command == "invite-remove":
            ok = db.remove_invited_user(args.site_user_id)
            if ok:
                print(f"[SUCCESS] User site_user_id={args.site_user_id} removed from whitelist.")
                return 0
            else:
                print(f"[INFO] User site_user_id={args.site_user_id} not found in whitelist.")
                return 0

        elif args.command == "reset-password":
            ok = db.set_user_password(args.user_id, args.password)
            if ok:
                print(f"[SUCCESS] Password for user '{args.user_id}' updated.")
                return 0
            else:
                print(f"[ERROR] User '{args.user_id}' not found in database.", file=sys.stderr)
                return 1

        else:
            print(f"[ERROR] Unknown command '{args.command}'.", file=sys.stderr)
            return 1


def main(args: Optional[List[str]] = None) -> int:
    parser = build_parser()
    parsed_args = parser.parse_args(args)
    try:
        return execute_command(parsed_args)
    except Exception as e:
        print(f"[ERROR] Command failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
