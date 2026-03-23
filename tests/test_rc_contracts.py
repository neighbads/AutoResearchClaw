import re

import pytest

from researchclaw.pipeline.contracts import CONTRACTS, StageContract
from researchclaw.pipeline.stages import GATE_STAGES, STAGE_SEQUENCE, Stage


def test_contracts_dict_has_exactly_24_entries():
    assert len(CONTRACTS) == 24


def test_every_stage_has_matching_contract_entry():
    assert set(CONTRACTS.keys()) == set(Stage)


@pytest.mark.parametrize("stage", STAGE_SEQUENCE)
def test_each_stage_member_resolves_to_stage_contract(stage: Stage):
    assert isinstance(CONTRACTS[stage], StageContract)


@pytest.mark.parametrize("stage,contract", tuple(CONTRACTS.items()))
def test_contract_stage_field_matches_dict_key(stage: Stage, contract: StageContract):
    assert contract.stage is stage


@pytest.mark.parametrize("contract", tuple(CONTRACTS.values()))
def test_output_files_is_non_empty_for_all_contracts(contract: StageContract):
    assert contract.output_files


@pytest.mark.parametrize("stage,contract", tuple(CONTRACTS.items()))
def test_error_code_starts_with_e_and_contains_stage_number(
    stage: Stage, contract: StageContract
):
    assert contract.error_code.startswith("E")
    assert f"{int(stage):02d}" in contract.error_code
    assert re.match(r"^E\d{2}_[A-Z0-9_]+$", contract.error_code)


@pytest.mark.parametrize("contract", tuple(CONTRACTS.values()))
def test_max_retries_is_non_negative_for_all_contracts(contract: StageContract):
    assert contract.max_retries >= 0


def test_gate_stages_have_expected_max_retries():
    assert CONTRACTS[Stage.LITERATURE_SCREEN].max_retries == 0
    assert CONTRACTS[Stage.EXPERIMENT_DESIGN].max_retries == 0
    assert CONTRACTS[Stage.QUALITY_GATE].max_retries == 0


@pytest.mark.parametrize("stage", tuple(GATE_STAGES))
def test_gate_stage_contracts_are_never_retried(stage: Stage):
    assert CONTRACTS[stage].max_retries == 0


def test_topic_init_contract_has_expected_input_output_files():
    contract = CONTRACTS[Stage.TOPIC_INIT]

    assert contract.input_files == ()
    assert contract.output_files == ("goal.md", "hardware_profile.json")


def test_export_publish_contract_has_expected_outputs():
    contract = CONTRACTS[Stage.EXPORT_PUBLISH]

    assert contract.output_files == ("paper_final.md", "code/")


@pytest.mark.parametrize("contract", tuple(CONTRACTS.values()))
def test_dod_is_non_empty_string_for_all_contracts(contract: StageContract):
    assert isinstance(contract.dod, str)
    assert contract.dod.strip()


@pytest.mark.parametrize("contract", tuple(CONTRACTS.values()))
def test_input_files_is_tuple_of_strings(contract: StageContract):
    assert isinstance(contract.input_files, tuple)
    assert all(isinstance(path, str) and path for path in contract.input_files)


@pytest.mark.parametrize("contract", tuple(CONTRACTS.values()))
def test_output_files_is_tuple_of_strings(contract: StageContract):
    assert isinstance(contract.output_files, tuple)
    assert all(isinstance(path, str) and path for path in contract.output_files)


def test_error_codes_are_unique_across_contracts():
    all_codes = [contract.error_code for contract in CONTRACTS.values()]
    assert len(all_codes) == len(set(all_codes))


def test_contracts_follow_stage_sequence_order():
    assert tuple(CONTRACTS.keys()) == STAGE_SEQUENCE


@pytest.mark.parametrize("stage", STAGE_SEQUENCE)
def test_contract_stage_int_matches_stage_enum_value(stage: Stage):
    assert int(CONTRACTS[stage].stage) == int(stage)


def test_seed_spec_ingest_contract_outputs():
    contract = CONTRACTS[Stage.SEED_SPEC_INGEST]

    assert contract.input_files == ()
    assert contract.output_files == (
        "seed_manifest.json",
        "seed_spec_outline.md",
        "seed_claims.json",
        "seed_open_questions.md",
        "seed_repo_inventory.json",
        "seed_repo_keyfiles.md",
        "seed_api_map.json",
        "seed_spec_code_alignment.md",
    )
    assert contract.error_code == "E00_SEED_INGEST_FAIL"


def test_topic_init_consumes_seed_artifacts():
    contract = CONTRACTS[Stage.TOPIC_INIT]

    assert contract.input_files == ()


def test_problem_decompose_consumes_seed_artifacts():
    contract = CONTRACTS[Stage.PROBLEM_DECOMPOSE]

    assert contract.input_files == ("goal.md",)


def test_experiment_design_consumes_seed_artifacts():
    contract = CONTRACTS[Stage.EXPERIMENT_DESIGN]

    assert contract.input_files == (
        "hypotheses.md",
        "seed_spec_code_alignment.md",
        "seed_repo_inventory.json",
        "seed_api_map.json",
    )
