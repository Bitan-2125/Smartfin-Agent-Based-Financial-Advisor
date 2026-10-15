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


def main(base_url, ai=False):
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
            response = client.post("/generate", data={**SAMPLE, "csrf_token": csrf, **({"ai_consent": "1"} if ai else {})})
            check(response, 302, "generate")
            created = True
            report = client.get("/result")
            check(report, 200, "report")
            exported = client.get("/export")
            check(exported, 200, "export")
            if exported.json()["plan"]["profile"]["age"] != 30:
                raise RuntimeError("export did not contain the synthetic plan")
            print("PASS: home, anonymous planner, plan generation, report, export")
            if ai:
                plan = exported.json()["plan"]
                if not plan["ai_consent"] or "AI commentary is unavailable" in plan["response"] or len(plan["response"]) < 300:
                    raise RuntimeError("Groq did not generate a detailed plan")
                if 'id="queryInput"' not in report.text:
                    raise RuntimeError("chat input is missing from the report")
                answer = client.post(
                    "/chat", json={"query": "Explain how to balance the emergency reserve gap and goal funding."},
                    headers={"X-CSRF-Token": csrf},
                )
                check(answer, 200, "AI chat")
                if len(answer.json()["response"]) < 100 or "AI commentary is unavailable" in answer.json()["response"]:
                    raise RuntimeError("Groq chat did not return a substantive answer")
                if len(client.get("/export").json()["chat_history"]) != 2:
                    raise RuntimeError("AI conversation was not saved")
                print("PASS: Groq plan, visible chat, AI reply, saved conversation")
        finally:
            if created:
                check(client.post("/delete", data={"csrf_token": csrf}), 302, "delete")
                check(client.get("/export"), 404, "post-delete export")
                print("PASS: saved plan deleted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="deployment URL, for example https://smartfin-advisor.vercel.app")
    parser.add_argument("--ai", action="store_true", help="also verify the configured Groq plan and chat")
    args = parser.parse_args()
    main(args.base_url, ai=args.ai)
