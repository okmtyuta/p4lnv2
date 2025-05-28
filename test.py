import polars as pl

df = pl.read_csv("data/ishihama/data.csv")

df = df.filter(pl.col('ccs').is_null())
print(df)
# df = df.rename(
#     {
#         "Seq": "seq",
#         "CalcMass": "mass",
#         "CalcMz": "mz",
#         "Charge": "charge",
#         "CCS": "ccs",
#         "retention_time": "rt",
#         "Type": "type",
#     }
# )
# df = df.with_row_index()
# df = df.drop("CCSpred (linear regression)")
# df = df.select("index", "seq", "rt", "ccs", "length", "mass", "charge", "mz", "ion_mobility", "type")

# df.write_csv("data/ishihama/data.csv")
