import pandas as pd
import pytest
from campuspulse.academic import pipeline as P

def test_academic_runs():
    try:
        P.load_raw()
    except FileNotFoundError:
        pytest.skip("UCI data not present")
    r = P.run_pipeline({k: v[0] for k, v in P.DECISIONS.items()})
    assert set(r) == set(P.FINDINGS)


def test_stratified_debt_effect_reports_dropout_rates_per_approval_band():
    rows = []
    for approval_rate, debt_dropout_count, non_debt_dropout_count in (
        (0.50, 5, 2),
        (0.99, 8, 3),
        (1.00, 4, 1),
    ):
        for debtor, dropout_count in ((1, debt_dropout_count), (0, non_debt_dropout_count)):
            rows.extend(
                {"approval_rate_sem1": approval_rate, "Debtor": debtor, "dropout": int(i < dropout_count)}
                for i in range(10)
            )
    _, bands = P.stratified_debt_effect(pd.DataFrame(rows))

    assert list(bands["band"].astype(str)) == ["low (<=50%)", "mid", "all passed"]
    assert bands["n_debtors"].tolist() == [10, 10, 10]
    assert bands["n_non_debtors"].tolist() == [10, 10, 10]
    assert bands["debtor_dropout_rate"].tolist() == pytest.approx([0.5, 0.8, 0.4])
    assert bands["non_debtor_dropout_rate"].tolist() == pytest.approx([0.2, 0.3, 0.1])
    assert bands["gap"].tolist() == pytest.approx([0.3, 0.5, 0.3])
