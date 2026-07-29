from scripts.audit import m3_c_a_goal_selection_check as check


def test_m3_c_a_design_is_fully_resolved_and_authority_bounded():
    result = check.validate()

    assert result["pass"] is True, result["errors"]
    assert result["drive_count"] == 8
    assert result["affect_axis_count"] == 63
    assert result["affect_status_counts"] == {"MAPPED": 59, "PROPOSED-DROP": 4}
    assert result["unresolved_count"] == 0
    assert result["authorization_digest"] == check.AUTHORIZATION_DIGEST
    assert result["active_store_role"] == check.ACTIVE_STORE_ROLE


def test_drive_state_only_counterfactual_flips_selected_goal():
    result = check.validate()
    rows = {row["condition"]: row for row in result["counterfactual"]}

    strained = rows["strain_mapped_affect"]
    recovered = rows["recovered_exploration"]

    assert strained["winner"] == "recover_operating_margin"
    assert recovered["winner"] == "explore_information_gap"
    assert strained["winner"] != recovered["winner"]
    assert strained["winner_score"] >= check.EXPECTED_POLICY["selection minimum score"]
    assert recovered["winner_score"] >= check.EXPECTED_POLICY["selection minimum score"]
    assert strained["margin"] >= check.EXPECTED_POLICY["initial winner margin"]
    assert recovered["margin"] >= check.EXPECTED_POLICY["initial winner margin"]


def test_checker_cli_fail_on_unresolved_succeeds_for_v1(capsys):
    assert check.main(["--summary-only", "--fail-on-unresolved"]) == 0
    output = capsys.readouterr().out
    assert '"pass": true' in output
    assert '"unresolved_count": 0' in output
