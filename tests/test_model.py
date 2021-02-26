
import pandas as pd
import numpy as np
from numpy import NaN
from src.model import drop_id_column
from src.model import count_missing_values
from src.model import fill_missing_values_in_cat_columns


def test_drop_id_column():

   df_test = pd.DataFrame({"Id": 1, "name": "Jack", "age": 30}, index=[0])
   expected = len(df_test.columns) - 1
   actual = len(drop_id_column(df_test).columns)
   assert actual == expected


def test_count_missing_values():

   df_test_2 = pd.DataFrame({"Id": 1, "name": "Jack", "age": np.nan}, index=[0])
   expected = int(1)
   actual = count_missing_values(df_test_2).sum()
   assert actual == expected


def test_fill_missing_values_in_cat_columns():
   df_test_3 = pd.DataFrame([{"Id": 1, "name": np.nan, "age": 30}, {"Id": 2, "name": "Jack", "age": 30}], index=[0, 1])
   expected = pd.DataFrame([{"Id": 1, "name": "None", "age": 30}, {"Id": 2, "name": "Jack", "age": 30}], index=[0, 1])
   actual = fill_missing_values_in_cat_columns(df_test_3)
   assert actual.equals(expected) is True

