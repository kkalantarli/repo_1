import pytest as py
import pandas as pd
from src.model import drop_id_column


def test_drop_id_column():

   df_test = pd.DataFrame({"Id": 1, "name": "Jack", "age": 30}, index=[0])
   expected = len(df_test.columns) - 1
   actual = len(drop_id_column(df_test).columns)
   assert actual == expected
