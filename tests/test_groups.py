import pandas as pd
import pytest

from glaurlex.core.groups import Group, apply_group, select_informant_rows


@pytest.fixture
def informants():
    return pd.DataFrame(
        {
            "informant_id": [1, 2, 3, 4, 5],
            "ORIGIN": ["LOCAL", "HERITAGE", "LOCAL", "FOREIGN", "LOCAL"],
            "AGE": [19, 25, 31, 44, 28],
            "LEVEL": [1, 2, 2, 3, 3],
        }
    )


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({}, [1, 2, 3, 4, 5]),
        ({"ORIGIN": "LOCAL"}, [1, 3, 5]),
        ({"ORIGIN": ["LOCAL", "HERITAGE"]}, [1, 2, 3, 5]),
        ({"AGE": {">=": 20, "<=": 30}}, [2, 5]),
        ({"LEVEL": {"!=": 2}, "ORIGIN": ["LOCAL"]}, [1, 5]),
        ({"LEVEL": {"==": 3}, "AGE": {">=": 30}}, [4]),
    ],
)
def test_apply_group_combines_rules_with_and(informants, filters, expected):
    assert apply_group(informants, Group("g", filters))["informant_id"].tolist() == expected


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ({"op": "between", "min": 20, "max": 30}, [2, 5]),
        ({"op": "between", "min": 25, "max": 28}, [2, 5]),
        ({"op": ">", "value": 30}, [3, 4]),
        ({"op": ">", "value": 31}, [4]),
        ({"op": "<", "value": 25}, [1]),
        ({"op": ">=", "value": 31}, [3, 4]),
        ({"op": "<=", "value": 25}, [1, 2]),
        ({"op": "==", "value": 28}, [5]),
        ({"op": "!=", "value": 28}, [1, 2, 3, 4]),
        ({">": 25, "<": 44}, [3, 5]),
    ],
)
def test_apply_group_supports_operator_rules(informants, rule, expected):
    assert apply_group(informants, Group("g", {"AGE": rule}))["informant_id"].tolist() == expected


@pytest.mark.parametrize(
    ("rule", "match"),
    [
        ({"op": "~", "value": 1}, "unknown operator '~'"),
        ({"<>": 1}, "unknown operator '<>'"),
        ({"between": [20, 30]}, "unknown operator 'between'"),
        ({"op": ">", "value": 1, "min": 0}, "must have exactly the keys"),
        ({"op": "between", "min": 20}, "must have exactly the keys"),
        ({"op": "between", "max": 30}, "must have exactly the keys"),
        ({"op": ">"}, "must have exactly the keys"),
    ],
)
def test_apply_group_rejects_malformed_rules(informants, rule, match):
    with pytest.raises(ValueError, match=match):
        apply_group(informants, Group("g", {"AGE": rule}))


def test_select_informant_rows(informants):
    topic = pd.DataFrame(
        {"informant_id": [1, 1, 2, 4, 9], "pos": [0, 1, 0, 0, 0], "type": list("abcde")}
    )
    selected = select_informant_rows(
        topic, apply_group(informants, Group("local", {"ORIGIN": "LOCAL"}))
    )

    assert selected["type"].tolist() == ["a", "b"]
    with pytest.raises(ValueError, match="informants must have"):
        select_informant_rows(topic, informants.drop(columns="informant_id"))
