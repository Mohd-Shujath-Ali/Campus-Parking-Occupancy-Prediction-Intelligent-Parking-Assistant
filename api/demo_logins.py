"""
One-click demo sign-in personas for presentations, evaluations, and role testing.

Preconfigured for zero-friction demonstration across all campus roles:
  - Student (Aarav Sharma · School of CSE, Block 26)
  - Faculty (Dr. Meera Kapoor · Administration & CSE, Block 1)
  - Hostel Warden (Rajinder Singh · Boys Hostel 2)
  - Security Guard (Harpreet Kaur · Main Gate Operations)
  - Administrator (Chief Parking Administrator · Master Console)
  - Vendor (Punjab Caterers · MB Food Square)

Always ready on login screen with instant 1-click credential filling & sign-in.
"""
import csv
import logging
import os
from typing import Any, Dict, List

from fastapi import Request

from api import auth
from api.db import get_db

logger = logging.getLogger(__name__)

DEFAULT_DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "CampusPass@2026")

ROLE_ORDER = ["student", "faculty", "warden", "guard", "admin", "vendor"]

PRESENTATION_PERSONAS = [
    {
        "username": "12301234",
        "alias": "student",
        "role": "student",
        "role_label": "Student",
        "name": "Aarav Sharma",
        "subtitle": "School of Computer Science & Engg",
        "home_block": "Block 26 (School of CSE)",
        "db_block": "Block_26_CSE",
        "department": "School of CSE",
        "password": DEFAULT_DEMO_PASSWORD,
        "badge_color": "blue",
        "icon": "fa-user-graduate",
        "target": "Driver Recommendations (Block 26)",
        "dest_url": "/",
    },
    {
        "username": "26541",
        "alias": "faculty",
        "role": "faculty",
        "role_label": "Faculty",
        "name": "Dr. Meera Kapoor",
        "subtitle": "Administration & Senior Faculty",
        "home_block": "Block 1 (Central Administration)",
        "db_block": "Block_1",
        "department": "Administration",
        "password": DEFAULT_DEMO_PASSWORD,
        "badge_color": "emerald",
        "icon": "fa-chalkboard-user",
        "target": "Faculty Priority View (Block 1)",
        "dest_url": "/",
    },
    {
        "username": "W-BH2-01",
        "alias": "warden",
        "role": "warden",
        "role_label": "Hostel Warden",
        "name": "Rajinder Singh",
        "subtitle": "Boys Hostel 2 Administration",
        "home_block": "BH2 Hostel Grounds",
        "db_block": "BH2_Hostel",
        "department": "Hostel Operations",
        "password": DEFAULT_DEMO_PASSWORD,
        "badge_color": "amber",
        "icon": "fa-hotel",
        "target": "Estate Console & Overrides",
        "dest_url": "/admin",
    },
    {
        "username": "G-GATE-03",
        "alias": "guard",
        "role": "guard",
        "role_label": "Security Guard",
        "name": "Harpreet Kaur",
        "subtitle": "Main Gate Safety & Enforcement",
        "home_block": "Main Gate Operations",
        "db_block": None,
        "department": "Campus Security",
        "password": DEFAULT_DEMO_PASSWORD,
        "badge_color": "purple",
        "icon": "fa-shield-halved",
        "target": "Violations & Gate Status",
        "dest_url": "/admin",
    },
    {
        "username": "admin",
        "alias": "",
        "role": "admin",
        "role_label": "Administrator",
        "name": "Chief Parking Admin",
        "subtitle": "Directorate of Transportation",
        "home_block": "Central Traffic Operations",
        "db_block": "Block_1",
        "department": "Campus Transportation Directorate",
        "password": DEFAULT_DEMO_PASSWORD,
        "badge_color": "rose",
        "icon": "fa-crown",
        "target": "Full Master Control Center",
        "dest_url": "/admin",
    },
    {
        "username": "V-CANTEEN-07",
        "alias": "vendor",
        "role": "vendor",
        "role_label": "Vendor",
        "name": "Punjab Caterers",
        "subtitle": "Campus Dining & Delivery Supply",
        "home_block": "MB Food Square",
        "db_block": "MB_Food_Square",
        "department": "Food Services",
        "password": DEFAULT_DEMO_PASSWORD,
        "badge_color": "slate",
        "icon": "fa-truck-fast",
        "target": "Delivery Bay Parking",
        "dest_url": "/",
    },
]


def credentials_file() -> str:
    path = get_db().path
    base = os.path.dirname(os.path.abspath(path)) if path != ":memory:" else os.getcwd()
    return os.path.join(base, "demo_credentials.csv")


def enabled(request: Request = None) -> bool:
    mode = os.environ.get("DEMO_LOGINS", "on").strip().lower()
    return mode != "off"


def ensure_demo_users() -> None:
    """
    Ensure all presentation demo accounts exist in the SQLite database with active state,
    known demo password, and must_change_password=0 so they can sign in with 1 click.
    """
    csv_rows = []
    for p in PRESENTATION_PERSONAS:
        pw = p["password"]
        user = auth.get_user_by_username(p["username"])
        if not user:
            try:
                auth.create_user(
                    p["username"],
                    p["role"],
                    full_name=p["name"],
                    home_block=p["db_block"],
                    department=p["department"],
                    password=pw,
                )
                logger.info("Created presentation account '%s' (%s).", p["username"], p["role"])
            except Exception as e:
                logger.warning("Could not create presentation account '%s': %s", p["username"], e)
        else:
            auth.set_password(user["id"], pw, must_change=False)
            auth.update_user(user["id"], is_active=1, full_name=p["name"], home_block=p["db_block"], department=p["department"])

        csv_rows.append({
            "username": p["username"],
            "role": p["role"],
            "full_name": p["name"],
            "home_block": p["home_block"],
            "password": pw,
            "must_change": "no",
        })

        alias = p.get("alias")
        if alias:
            alias_user = auth.get_user_by_username(alias)
            if not alias_user:
                try:
                    auth.create_user(
                        alias,
                        p["role"],
                        full_name=f"{p['name']} (Alias)",
                        home_block=p["db_block"],
                        department=p["department"],
                        password=pw,
                    )
                    logger.info("Created presentation alias '%s' (%s).", alias, p["role"])
                except Exception as e:
                    logger.warning("Could not create presentation alias '%s': %s", alias, e)
            else:
                auth.set_password(alias_user["id"], pw, must_change=False)
                auth.update_user(alias_user["id"], is_active=1)

    try:
        path = credentials_file()
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["username", "role", "full_name", "home_block", "password", "must_change"])
            w.writeheader()
            w.writerows(csv_rows)
    except Exception as e:
        logger.warning("Could not write demo_credentials.csv: %s", e)


def demo_accounts(request: Request = None) -> List[Dict[str, Any]]:
    """Return presentation demo personas for the login screen."""
    if not enabled(request):
        return []
    try:
        ensure_demo_users()
    except Exception as e:
        logger.error("Error ensuring presentation users: %s", e)
    return PRESENTATION_PERSONAS
