"""
Create demo accounts for every role so each interface can be tried locally.

    python -m scripts.seed_demo_users                 # one-time passwords (must change at first login)
    python -m scripts.seed_demo_users --known-password "Demo-Pass-2026"   # same password for all, no forced change

The one-time passwords are printed and written to data/live/demo_credentials.csv (git-ignored).
Existing user IDs are left untouched. The IDs and names below are made-up samples.
For real accounts, use Users -> Bulk import with a CSV from the registrar / HR.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import auth  # noqa: E402
from api.db import get_db  # noqa: E402
from api.auth_routes import normalize_expiry  # noqa: E402

DEMO_USERS = [
    # username,       role,      full name,              home block,           department
    ("12301234",     "student", "Aarav Sharma",         "Block_26_CSE",       "School of CSE"),
    ("12305678",     "student", "Simran Kaur",          "Block_39_Innovation", "School of CSE"),           # adjacent -> Block 26
    ("12309876",     "student", "Rohan Verma",          "Block_55_Mech",      "School of Mechanical Engg."),  # adjacent -> Block 57
    ("12311122",     "student", "Ananya Iyer",          "Management_Block",   "Mittal School of Business"),   # adjacent -> Law Block
    ("26541",        "faculty", "Dr. Meera Kapoor",     "Block_1",            "Administration"),
    ("27310",        "faculty", "Dr. Gurpreet Sandhu",  "Block_57_Agri",      "School of Agriculture"),
    ("28102",        "faculty", "Prof. Kavya Menon",    "Product_Industrial", "School of Design"),           # adjacent -> Architecture
    ("W-BH2-01",     "warden",  "Rajinder Singh",       "BH2_Hostel",         "Boys Hostel 2"),
    ("W-GH5-01",     "warden",  "Neelam Bhatia",        "GH5_Hostel",         "Girls Hostel 5"),
    ("V-CANTEEN-07", "vendor",  "Punjab Caterers",      "MB_Food_Square",     "Food services"),
    ("V-PRINT-02",   "vendor",  "Campus Print Hub",     "Block_26_CSE",       "Printing"),
    ("G-GATE-03",    "guard",   "Harpreet Kaur",        None,                 "Security"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--known-password", help="Give every demo account this password (no forced change). Local demos only.")
    ap.add_argument("--out", default=None, help="Where to write the credentials CSV")
    args = ap.parse_args()

    auth.ensure_bootstrap_admin()
    rows = []
    for username, role, name, block, dept in DEMO_USERS:
        if auth.get_user_by_username(username):
            print(f"  skip  {username:<14} (exists)")
            continue
        expires = normalize_expiry(role, None, default_for_vendor=True)
        user, temp = auth.create_user(username, role, full_name=name, home_block=block, department=dept,
                                      expires_at=expires, password=args.known_password)
        pw = temp or args.known_password
        rows.append({"username": username, "role": role, "full_name": name, "home_block": block or "",
                     "password": pw, "must_change": "yes" if temp else "no"})
        print(f"  added {username:<14} {role:<8} {name:<22} {block or '-':<20} {pw}")

    if rows:
        out = args.out or os.path.join(os.path.dirname(os.path.abspath(get_db().path)), "demo_credentials.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} credentials to {out}. Delete it after use.")
    else:
        print("\nNo new accounts created.")


if __name__ == "__main__":
    main()
