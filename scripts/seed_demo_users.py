"""
Create demo accounts for every role so each interface can be tried locally and during presentations.

    python -m scripts.seed_demo_users
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import auth  # noqa: E402
from api import demo_logins  # noqa: E402


def main():
    print("=" * 70)
    print("  SEEDING LPU CAMPUS PARKING DEMO PERSONAS FOR PRESENTATION")
    print("=" * 70)
    auth.ensure_bootstrap_admin()
    demo_logins.ensure_demo_users()

    print(f"\nPassword for all presentation demo accounts: {demo_logins.DEFAULT_DEMO_PASSWORD}\n")
    print(f"{'Role':<12} {'Primary ID':<12} {'Alias':<10} {'Name':<24} {'Home Block':<25}")
    print("-" * 85)
    for p in demo_logins.PRESENTATION_PERSONAS:
        alias = p.get("alias") or "-"
        print(f"{p['role']:<12} {p['username']:<12} {alias:<10} {p['name']:<24} {p['home_block']:<25}")

    print("\nAll demo accounts are active, unlocked, and do NOT require password change.")
    print("Ready for 1-click login on /login screen!")
    print("=" * 70)


if __name__ == "__main__":
    main()
