import duckdb
import os

def load_big_file_to_parquet(csv_path: str, parquet_path: str, overwrite: bool = False):
    """
    Converts massive CSV to parquet using DuckDB

    :param csv_path:
    :param parquet_path:
    :param overwrite:
    :return:
    """

    if os.path.exists(parquet_path) and not overwrite:
        print('Parquet already exists: {parquet_path}')
        return

    print(f'Loading and converting {csv_path} to Parquet')

    duckdb.sql(f"""
        COPY (
            SELECT * FROM read_csv_auto('{csv_path}', ignore_errors=true)
        ) TO '{parquet_path}' (FORMAT PARQUET)
    """)

    print("Conversion complete")

def query_parquet(parquet_path: str, limit: int = 1000):
    """
    Loads a portion of the Parquet file for testing/querying

    :param parquet_path:
    :param limit:
    :return:
    """
    print(f"Querying Parquet: {parquet_path}")
    df = duckdb.sql(f"SELECT * FROM '{parquet_path}' LIMIT {limit}").df()
    print(f"✅ Loaded {len(df)} rows.")
    return df

