from scripts.make_sample_data import make
from src.data_prep import clean


def _clean():
    return clean.run(make(6000).astype(str))


def test_target_is_binary_and_named_no_show():
    out = _clean()
    assert set(out["no_show"].unique()) <= {0, 1}


def test_invalid_age_removed():
    out = _clean()
    assert out["age"].between(0, 110).all()


def test_duplicate_appointment_ids_removed():
    out = _clean()
    assert out["appointment_id"].is_unique


def test_scheduled_never_after_appointment():
    out = _clean()
    assert (out["scheduled_date"] <= out["appointment_date"]).all()


def test_quality_report_records_each_rule():
    _clean()
    for rule in ["duplicate_appointment_id", "age_out_of_range_(<0_or_>110)", "scheduled_after_appointment"]:
        assert clean.quality_report[rule]["rows_affected"] >= 1
