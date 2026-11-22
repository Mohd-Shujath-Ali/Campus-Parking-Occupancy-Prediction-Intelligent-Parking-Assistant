"""
Test-wide setup. Runs before any test module imports the app, so the app uses a throw-away
account database and a known admin password.
"""
import os
import tempfile

from tests.auth_helpers import ADMIN_PASSWORD

_tmp = tempfile.mkdtemp(prefix="lpu_auth_test_")
os.environ["AUTH_DB"] = os.path.join(_tmp, "app.db")
os.environ["PARKING_DB"] = os.path.join(_tmp, "observations.db")   # never write test counts into data/live/
os.environ["ADMIN_USERNAME"] = "admin"
os.environ["ADMIN_PASSWORD"] = ADMIN_PASSWORD
os.environ["LOGIN_RATE_LIMIT"] = "10000/minute"   # tests log in many times from one "IP"
