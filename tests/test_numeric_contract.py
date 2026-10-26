"""Independent decimal ceiling and count-writing boundaries."""

from fractions import Fraction

import pytest

from netlab_topogen.numeric import covers, decimal, optic_count, written_count


def test_decimal_entry_has_no_binary_intermediate():
    assert decimal(0.1) == Fraction(1, 10)
    assert decimal(1.3) * 5000 == 6500
    assert decimal(9007199254740993) == 9007199254740993


@pytest.mark.parametrize(
    "w,c,k",
    [
        (60.0005, 1.2245, 49),
        (0.3333333333333333, 0.1111111111111111, 3),
        (258.3333333333333, 36.9047619047619, 7),
        (1e16, 1e16, 1),
        (1223510546430935.5, 21465.09354, 57000010001),
        (9e-323, 4.4e-323, 3),
    ],
)
def test_optic_count_and_coverage_have_one_exact_rule(w, c, k):
    assert optic_count(w, c) == k
    assert covers(k, c, w)
    assert not covers(k - 1, c, w)
    assert covers(k + 1, c, w)


def test_count_writability_is_not_a_size_limit():
    assert written_count(2**54, "edge source optic O") == float(2**54)
    assert written_count(10**304, "edge source optic O") == 1e304
    for k in (2**53 + 1, 10**314):
        with pytest.raises(
            ValueError,
            match="edge source optic O: optic count .* cannot be written exactly",
        ):
            written_count(k, "edge source optic O")
