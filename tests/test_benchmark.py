"""
tests/test_benchmark.py — Unit tests for evaluation benchmark runner and persona simulation.

Run with:
    pytest tests/test_benchmark.py -v
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.benchmark import (
    TurnResult,
    ScenarioResult,
    BenchmarkReport,
    _simulate_turn_mock,
    run_scenario,
    run_benchmark,
)
from agent.agent import AgentState, Intent


class TestScenariosFile:
    def test_scenarios_json_exists_and_valid(self):
        scenarios_file = _PROJECT_ROOT / "eval" / "scenarios.json"
        assert scenarios_file.exists(), "eval/scenarios.json should exist"

        with open(scenarios_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "scenarios" in data
        assert len(data["scenarios"]) >= 4

        for scenario in data["scenarios"]:
            assert "id" in scenario
            assert "name" in scenario
            assert "expected_final_intent" in scenario
            assert "dialog" in scenario
            assert len(scenario["dialog"]) > 0
            for turn in scenario["dialog"]:
                assert "message" in turn
                assert "expected_intent" in turn


class TestBenchmarkDataModels:
    def test_benchmark_report_metrics(self):
        report = BenchmarkReport(
            total_scenarios=4,
            passed_scenarios=3,
            total_turns=10,
            correct_intents=8,
            avg_latency_ms=12.5,
            p95_latency_ms=25.0,
        )
        assert report.intent_accuracy == 80.0
        assert report.scenario_pass_rate == 75.0

    def test_benchmark_report_empty_turns(self):
        report = BenchmarkReport()
        assert report.intent_accuracy == 0.0
        assert report.scenario_pass_rate == 0.0

    def test_benchmark_to_markdown_formatting(self):
        scenario = ScenarioResult(
            scenario_id="s1",
            name="Test Scenario",
            expected_final_intent="high_intent",
            actual_final_intent="high_intent",
            expected_lead_completion=True,
            actual_lead_completion=True,
            passed=True,
            turns=[
                TurnResult(
                    turn_index=0,
                    user_message="I want to sign up",
                    expected_intent="high_intent",
                    actual_intent="high_intent",
                    intent_match=True,
                    agent_reply="Great! What is your name?",
                    latency_ms=5.0,
                )
            ],
        )
        report = BenchmarkReport(
            total_scenarios=1,
            passed_scenarios=1,
            total_turns=1,
            correct_intents=1,
            avg_latency_ms=5.0,
            p95_latency_ms=5.0,
            scenario_results=[scenario],
        )

        md = report.to_markdown()
        assert "# AutoStream Agent Benchmark Report" in md
        assert "Test Scenario" in md
        assert "high_intent" in md
        assert "100.0%" in md


class TestMockSimulator:
    def test_simulate_greeting(self):
        state = AgentState()
        updated = _simulate_turn_mock("Hey there!", state)
        assert updated.intent == Intent.greeting
        assert len(updated.messages) == 2

    def test_simulate_product_inquiry(self):
        state = AgentState()
        updated = _simulate_turn_mock("How much is the Pro plan?", state)
        assert updated.intent == Intent.product_inquiry
        assert len(updated.messages) == 2

    def test_simulate_high_intent_lead_capture(self):
        state = AgentState()
        updated = _simulate_turn_mock("Sign me up! My name is Jordan, email jordan@test.com", state)
        assert updated.intent == Intent.high_intent
        assert updated.lead.name == "Jordan"
        assert updated.lead.email == "jordan@test.com"
        assert updated.lead_stage == "done"


class TestBenchmarkExecution:
    def test_run_scenario_mock(self):
        scenario = {
            "id": "direct_lead",
            "name": "Direct Lead",
            "expected_final_intent": "high_intent",
            "expected_lead_completion": True,
            "dialog": [
                {
                    "message": "Sign me up! I'm Elena, elena@streamers.net",
                    "expected_intent": "high_intent",
                }
            ],
        }
        res = asyncio.run(run_scenario(scenario, use_mock=True))
        assert res.passed is True
        assert res.actual_final_intent == "high_intent"
        assert res.actual_lead_completion is True
        assert len(res.turns) == 1
        assert res.turns[0].intent_match is True

    def test_run_benchmark_mock_suite(self):
        scenarios_file = str(_PROJECT_ROOT / "eval" / "scenarios.json")
        report = asyncio.run(run_benchmark(scenarios_file, use_mock=True))
        assert report.total_scenarios == 4
        assert report.passed_scenarios == 4
        assert report.intent_accuracy == 100.0
        assert report.scenario_pass_rate == 100.0

    def test_run_benchmark_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            asyncio.run(run_benchmark("non_existent_scenarios.json", use_mock=True))


