import pytest

from prepare import prepare_data


@pytest.fixture(scope="module")
def prepared_data():
    return prepare_data(seed=42)
