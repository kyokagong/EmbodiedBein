from eb.stats import significant


def test_overlapping_ci_not_significant():
    a = [1.0, 1.1, 0.9, 1.05, 0.95]
    b = [1.02, 0.98, 1.04, 0.97, 1.01]
    assert significant(a, b) is False


def test_separated_means_significant():
    a = [0.0, 0.1, -0.1, 0.05, -0.05]
    b = [10.0, 10.1, 9.9, 10.05, 9.95]
    assert significant(a, b) is True
