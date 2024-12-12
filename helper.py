import pandas as pd

def outliers(data, col):
    Q1 = data[col].quantile(0.1)
    Q3 = data[col].quantile(0.9)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    indices = data.index[
        (data[col] < lower_bound) |
        (data[col] > upper_bound)
    ]

    return indices

def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:
    df_numeric = data.drop(columns=['loan_status', 'transaction_date', 'customer_id'])\
                        .select_dtypes(['int64', 'float64'])

    indices = []
    for col in df_numeric.columns:
        indices.extend(outliers(df_numeric, col))
    
    indices = sorted(set(indices))
    print(f"Dropping {len(indices)} outliers")
    return data.drop(indices)