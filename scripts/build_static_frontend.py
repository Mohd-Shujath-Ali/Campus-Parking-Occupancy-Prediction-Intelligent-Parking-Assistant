"""
Static Frontend Exporter for Render Static Site Deployment
Compiles Jinja2 templates into flat, production-ready static HTML files and asset bundles.
"""

import os
import shutil
import argparse
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "frontend" / "templates"
STATIC_DIR = PROJECT_ROOT / "public" / "static"
OUTPUT_DIR = PROJECT_ROOT / "frontend_dist"

PAGES = [
    {"template": "driver.html", "active_page": "driver", "output_names": ["index.html", "driver.html"]},
    {"template": "admin.html", "active_page": "admin", "output_names": ["admin.html", "admin/index.html"]},
    {"template": "vision.html", "active_page": "vision", "output_names": ["vision.html", "vision/index.html"]},
    {"template": "analytics.html", "active_page": "analytics", "output_names": ["analytics.html", "analytics/index.html"]},
    {"template": "zones.html", "active_page": "zones", "output_names": ["zones.html", "zones/index.html"]},
]


def build_static_site(backend_url: str = ""):
    print(f"=== Compiling Static Frontend for Render ===")
    print(f"Source Templates: {TEMPLATES_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    if backend_url:
        print(f"Injected Backend URL: {backend_url}")

    # Ensure clean output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize Jinja2 Environment
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)

    # Mock request object for templates that reference request
    class MockRequest:
        def __init__(self):
            self.base_url = "/"
            self.url = "/"

    mock_request = MockRequest()

    for p in PAGES:
        tmpl = env.get_template(p["template"])
        rendered = tmpl.render(
            request=mock_request,
            active_page=p["active_page"]
        )

        # If a backend URL is supplied, inject window.__API_BASE__ into <head>
        if backend_url:
            injected_script = f'<script>window.__API_BASE__ = "{backend_url.rstrip("/")}";</script>\n</head>'
            rendered = rendered.replace("</head>", injected_script, 1)

        for out_rel in p["output_names"]:
            out_path = OUTPUT_DIR / out_rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(rendered)
            print(f"  ✓ Generated: {out_rel}")

    # Copy static assets (css, js, images)
    dist_static = OUTPUT_DIR / "static"
    shutil.copytree(STATIC_DIR, dist_static, dirs_exist_ok=True)
    print(f"  ✓ Copied static assets to {dist_static}")

    # Copy favicon to root
    favicon_src = STATIC_DIR / "favicon.ico"
    if favicon_src.exists():
        shutil.copy(favicon_src, OUTPUT_DIR / "favicon.ico")
        print(f"  ✓ Copied favicon.ico to root")

    # Generate _redirects for clean URL routing on Render / Netlify
    redirects_content = (
        "/admin        /admin.html        200\n"
        "/vision       /vision.html       200\n"
        "/analytics    /analytics.html    200\n"
        "/zones        /zones.html        200\n"
        "/*            /index.html        200\n"
    )
    with open(OUTPUT_DIR / "_redirects", "w", encoding="utf-8") as f:
        f.write(redirects_content)
    print(f"  ✓ Generated _redirects for Render Static Site routing")

    print("\n✅ Static site build complete! Ready for Render Static Site deployment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build static frontend for Render")
    parser.add_argument("--backend-url", type=str, default=os.environ.get("BACKEND_URL", ""),
                        help="Optional backend URL to inject as default")
    args = parser.parse_args()
    build_static_site(backend_url=args.backend_url)
