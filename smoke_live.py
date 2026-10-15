"""Exercise a deployed SmartFin instance with synthetic data and clean it up."""

import argparse
import re

import httpx


SAMPLE = {
    "age": "30", "income": "75000", "expenses": "45000", "goal": "1000000",
    "risk": "Medium", "dependents": "1", "has_insurance": "none",
    "deduction_80c": "50000", "deduction_80d": "12000", "deduction_nps": "10000",
}


def check(response, status, label):
    if response.status_code != status:
        raise RuntimeError(f"{label}: expected HTTP {status}, got {response.status_code}")


def main(base_url):
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=120, follow_redirects=False) as client:
        check(client.get("/"), 200, "home")
        page = client.get("/plan")
        check(page, 200, "planner")
        match = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
        if not match:
            raise RuntimeError("planner has no CSRF token")
        csrf = match.group(1)
        created = False
        try:
            response = client.post("/generate", data={**SAMPLE, "csrf_token": csrf})
            check(response, 302, "generate")
            created = True
            check(client.get("/result"), 200, "report")
            exported = client.get("/export")
            check(exported, 200, "export")
            if exported.json()["plan"]["profile"]["age"] != 30:
                raise RuntimeError("export did not contain the synthetic plan")
            print("PASS: home, anonymous planner, plan generation, report, export")
        finally:
            if created:
                check(client.post("/delete", data={"csrf_token": csrf}), 302, "delete")
                check(client.get("/export"), 404, "post-delete export")
                print("PASS: saved plan deleted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="deployment URL, for example https://smartfin-advisor.vercel.app")
    args = parser.parse_args()
    main(args.base_url)
