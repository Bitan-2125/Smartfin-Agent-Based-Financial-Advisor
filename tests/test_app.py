import os
import tempfile
import unittest
import sqlite3
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
import app as smartfin_app
from planning import build_plan, debt_payoff, future_value, required_monthly
from storage import load_history, load_plan, database, allow_anonymous_usage


PROFILE = {
    "age": "30",
    "income": "75000",
    "expenses": "45000",
    "goal": "1000000",
    "risk": "Medium",
    "dependents": "1",
    "has_insurance": "none",
    "deduction_80c": "50000",
    "deduction_80d": "12000",
    "deduction_nps": "10000",
    "ai_consent": "1",
}


class SmartFinTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        smartfin_app.app.config.update(
            TESTING=False,
            DATABASE_PATH=os.path.join(self.tempdir.name, "plans.sqlite3"),
        )
        self.client = smartfin_app.app.test_client()
        self.client.get("/plan")
        with self.client.session_transaction() as web_session:
            self.csrf = web_session["csrf_token"]

    def tearDown(self):
        self.tempdir.cleanup()

    def post(self, path, **kwargs):
        if "data" in kwargs:
            kwargs["data"] = dict(kwargs["data"], csrf_token=self.csrf)
        else:
            kwargs.setdefault("headers", {})["X-CSRF-Token"] = self.csrf
        return self.client.post(path, **kwargs)

    def make_plan(self):
        with patch.object(smartfin_app, "llm") as fake_llm:
            fake_llm.invoke.return_value = AIMessage(content="A sample plan")
            response = self.post("/generate", data=PROFILE)
        self.assertEqual(response.status_code, 302)
        with self.client.session_transaction() as session:
            return session["token"]

    def test_tax_uses_current_slabs_and_only_entered_deductions(self):
        estimate = smartfin_app.tax_estimate(100_000, 30)
        self.assertEqual(estimate["new_regime_tax"], 0)
        self.assertEqual(estimate["deduction_80c"], 0)
        self.assertEqual(estimate["deduction_80d"], 0)
        self.assertEqual(estimate["deduction_nps"], 0)
        self.assertGreater(estimate["old_regime_tax_after"], 0)
        self.assertGreater(smartfin_app.tax_estimate(110_000, 30)["new_regime_tax"], 0)
        self.assertIsNone(smartfin_app.tax_estimate(500_000, 30))

    def test_high_income_result_explains_tax_limit(self):
        with patch.object(smartfin_app, "llm") as fake_llm:
            fake_llm.invoke.return_value = AIMessage(content="A sample plan")
            response = self.post(
                "/generate",
                data=dict(PROFILE, income="500000", expenses="100000"),
                follow_redirects=True,
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Tax comparison is unavailable above", response.data)

    def test_validation_rejects_nonfinite_and_invalid_choices(self):
        bad = dict(PROFILE, income="NaN", dependents="abc", risk="Extreme")
        response = self.post("/generate", data=bad)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Choose a valid number", response.data)
        self.assertIn(b"Choose a risk level", response.data)
        self.assertIn("income", smartfin_app.validate_profile(dict(PROFILE, income="inf")))
        self.assertIn("deduction_80d", smartfin_app.validate_profile(dict(PROFILE, deduction_80d="50000")))

    def test_plan_and_history_persist_across_requests(self):
        token = self.make_plan()
        self.assertEqual(load_plan(smartfin_app.app.config["DATABASE_PATH"], token)["response"], "A sample plan")
        self.assertEqual(self.client.get("/result").status_code, 200)

        with patch.object(smartfin_app, "llm") as fake_llm:
            fake_llm.invoke.return_value = AIMessage(content="A sample reply")
            self.assertEqual(self.post("/chat", json={"query": "First question"}).status_code, 200)
            self.assertEqual(self.post("/chat", json={"query": "Second question"}).status_code, 200)
            messages = fake_llm.invoke.call_args.args[0]
            self.assertTrue(any(isinstance(m, HumanMessage) and m.content == "First question" for m in messages))
            self.assertTrue(any(isinstance(m, AIMessage) and m.content == "A sample reply" for m in messages))

        history = load_history(smartfin_app.app.config["DATABASE_PATH"], token)
        self.assertEqual([m["role"] for m in history], ["user", "assistant", "user", "assistant"])
        self.assertIn(b"First question", self.client.get("/result").data)

    def test_provider_failure_does_not_crash_routes(self):
        token = self.make_plan()
        with patch.object(smartfin_app, "search_tool") as fake_search, patch.object(smartfin_app, "llm") as fake_llm, patch.object(smartfin_app.logger, "exception"):
            fake_search.run.return_value = {"error": "ConnectionError"}
            fake_llm.invoke.return_value = AIMessage(content="Search is unavailable")
            response = self.post("/chat", json={"query": "How is the stock market?"})
            self.assertEqual(response.status_code, 200)
            self.assertIn("Current market search is unavailable", fake_llm.invoke.call_args.args[0][-1].content)
        self.assertEqual(len(load_history(smartfin_app.app.config["DATABASE_PATH"], token)), 2)

        with patch.object(smartfin_app, "llm") as fake_llm, patch.object(smartfin_app.logger, "exception"):
            fake_llm.invoke.side_effect = RuntimeError("provider offline")
            self.assertIn("planning snapshot", self.post("/chat", json={"query": "Next question"}).json["response"])
            self.assertEqual(self.post("/generate", data=PROFILE).status_code, 302)

    def test_chat_requires_valid_question_and_plan(self):
        self.assertEqual(self.post("/chat", json={}).status_code, 400)
        self.assertEqual(self.post("/chat", json={"query": "hello"}).status_code, 401)
        self.make_plan()
        self.assertEqual(self.post("/chat", json={"query": " "}).status_code, 400)
        self.assertEqual(self.post("/chat", json={"query": "x" * 2001}).status_code, 400)

    def test_planning_calculations_and_multiple_goals(self):
        extended = dict(PROFILE, monthly_take_home="65000", monthly_emi="5000",
                        debt_balance="100000", debt_rate="12", liquid_savings="30000",
                        investments="50000", goal_years="5", goal_monthly="3000",
                        return_low="0", return_base="0", return_high="0", inflation="0")
        profile, goals, assumptions, errors = smartfin_app.parse_plan_form(extended)
        self.assertFalse(errors)
        goals.append({"name": "Education", "amount": 120000, "years": 1, "saved": 0, "monthly": 0})
        result = build_plan(profile, goals, assumptions)
        self.assertEqual(result["monthly_surplus"], 15000)
        self.assertEqual(result["net_worth"], -20000)
        self.assertEqual(result["emergency_target"], 300000)
        self.assertEqual(result["goals"][1]["scenarios"]["base"]["required_monthly"], 10000)
        self.assertEqual(result["required_goal_monthly"], 26667)
        self.assertFalse(result["goals_affordable"])
        self.assertEqual(required_monthly(120000, 0, 0, 12), 10000)
        self.assertEqual(debt_payoff(1000, 0, 12)["status"], "payment_too_low")
        self.assertAlmostEqual(smartfin_app.sip_projection(3000, 60, .08)[-1], float(future_value(0, 3000, 8, 60)), delta=1)

    def test_data_export_delete_and_csrf(self):
        self.assertEqual(self.client.post("/generate", data=PROFILE).status_code, 403)
        token = self.make_plan()
        export = self.client.get("/export")
        self.assertEqual(export.status_code, 200)
        self.assertEqual(export.json["plan"]["profile"]["age"], 30)
        self.assertIn("attachment", export.headers["Content-Disposition"])
        self.assertEqual(self.client.post("/delete").status_code, 403)
        self.assertEqual(self.post("/delete").status_code, 302)
        self.assertIsNone(load_plan(smartfin_app.app.config["DATABASE_PATH"], token))
        self.assertEqual(self.client.get("/export").status_code, 404)

    def test_edit_prefills_and_replaces_prior_plan(self):
        original = self.make_plan()
        edit = self.client.get("/edit")
        self.assertEqual(edit.status_code, 200)
        self.assertIn(b'name="goal_years"', edit.data)
        self.assertIn(b'value="75000.0"', edit.data)
        with patch.object(smartfin_app, "llm", None):
            response = self.post("/generate", data=dict(PROFILE, goal="1200000"))
        self.assertEqual(response.status_code, 302)
        self.assertIsNone(load_plan(smartfin_app.app.config["DATABASE_PATH"], original))
        with self.client.session_transaction() as web_session:
            updated = load_plan(smartfin_app.app.config["DATABASE_PATH"], web_session["token"])
        self.assertEqual(updated["profile"]["goal_amount"], 1200000)

    def test_extra_goal_form_validation_and_cash_income(self):
        form = dict(PROFILE, monthly_take_home="60000", extra_goal_name=["Car"],
                    extra_goal_amount=["500000"], extra_goal_years=["3"],
                    extra_goal_saved=["100000"], extra_goal_monthly=["5000"])
        from werkzeug.datastructures import MultiDict
        profile, goals, _, errors = smartfin_app.parse_plan_form(MultiDict(form))
        self.assertFalse(errors)
        self.assertEqual(profile["cash_income"], 60000)
        self.assertEqual(goals[1]["name"], "Car")
        form["monthly_take_home"] = "80000"
        self.assertIn("monthly_take_home", smartfin_app.parse_plan_form(MultiDict(form))[3])

    def test_extra_goal_report_and_malformed_form(self):
        from werkzeug.datastructures import MultiDict
        form = MultiDict(PROFILE)
        for key, value in (("extra_goal_name", "Car"), ("extra_goal_amount", "500000"),
                           ("extra_goal_years", "3"), ("extra_goal_saved", "100000"),
                           ("extra_goal_monthly", "5000")):
            form.add(key, value)
        with patch.object(smartfin_app, "llm", None):
            response = self.post("/generate", data=form)
        self.assertEqual(response.status_code, 302)
        result = self.client.get("/result")
        self.assertIn(b"Goals and scenarios", result.data)
        self.assertIn(b"Car", result.data)

        malformed = MultiDict(PROFILE)
        malformed.add("extra_goal_name", "Incomplete")
        bad = self.post("/generate", data=malformed)
        self.assertEqual(bad.status_code, 400)
        self.assertIn(b"extra goals", bad.data)

    def test_anonymous_entry_and_plan_isolation(self):
        self.assertEqual(self.client.get("/plan").status_code, 200)
        self.assertEqual(self.client.get("/login").status_code, 302)
        self.assertEqual(self.client.get("/register").status_code, 404)
        token = self.make_plan()
        other = smartfin_app.app.test_client()
        other.get("/plan")
        with other.session_transaction() as web_session:
            csrf = web_session["csrf_token"]
        self.assertEqual(other.get("/result").status_code, 302)
        self.assertEqual(other.get("/export").status_code, 404)
        self.assertEqual(other.post("/chat", json={"query": "hello"}, headers={"X-CSRF-Token": csrf}).status_code, 401)
        self.assertEqual(other.post("/delete", data={"csrf_token": csrf}).status_code, 302)
        self.assertIsNotNone(load_plan(smartfin_app.app.config["DATABASE_PATH"], token))
        self.assertEqual(self.client.get("/result").status_code, 200)

    def test_ai_opt_in(self):
        no_ai = dict(PROFILE)
        no_ai.pop("ai_consent")
        with patch.object(smartfin_app, "llm") as fake_llm:
            self.assertEqual(self.post("/generate", data=no_ai).status_code, 302)
            fake_llm.invoke.assert_not_called()
            self.assertEqual(self.post("/chat", json={"query": "hello"}).status_code, 403)

    def test_legacy_database_migration_and_usage_limit(self):
        path = os.path.join(self.tempdir.name, "legacy.sqlite3")
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE plans (token TEXT PRIMARY KEY, data TEXT NOT NULL, expires_at INTEGER NOT NULL)")
        connection.execute("INSERT INTO plans VALUES ('old-token', '{}', 9999999999)")
        connection.commit()
        connection.close()
        with database(path) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(plans)")}
        self.assertIn("user_id", columns)
        self.assertIsNotNone(load_plan(path, "old-token"))
        with database(path) as connection:
            connection.execute("INSERT INTO plans (token, user_id, data, expires_at) VALUES (?, ?, ?, ?)",
                               ("expired", None, "{}", 1))
        self.assertIsNone(load_plan(path, "expired"))
        with database(path) as connection:
            self.assertIsNone(connection.execute("SELECT 1 FROM plans WHERE token = 'expired'").fetchone())
        self.assertTrue(allow_anonymous_usage(smartfin_app.app.config["DATABASE_PATH"], "127.0.0.1", "test", 2))
        self.assertTrue(allow_anonymous_usage(smartfin_app.app.config["DATABASE_PATH"], "127.0.0.1", "test", 2))
        self.assertFalse(allow_anonymous_usage(smartfin_app.app.config["DATABASE_PATH"], "127.0.0.1", "test", 2))


if __name__ == "__main__":
    unittest.main()
