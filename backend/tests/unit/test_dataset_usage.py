from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from eb.api.routes_data import DatasetUsageIn, set_dataset_usage
from eb.store.models import Dataset


@pytest.mark.asyncio
async def test_dataset_usage_label_preserves_existing_metadata():
    row = Dataset(id='ds_test', name='test', meta={'source_revision':'fixed'})
    session = SimpleNamespace(get=AsyncMock(return_value=row), commit=AsyncMock())
    result = await set_dataset_usage('ds_test', DatasetUsageIn(usage='eval'),
                                     session, SimpleNamespace(id='user'))
    assert result['usage'] == 'eval'
    assert row.meta == {'source_revision':'fixed', 'usage':'eval'}
    session.commit.assert_awaited_once()
    with pytest.raises(ValidationError):
        DatasetUsageIn(usage='both')
