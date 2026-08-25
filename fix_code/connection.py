"""Databricks SQL execution via the Statement Execution API."""
import os

import pandas as pd
import streamlit as st
from databricks.sdk import WorkspaceClient


def get_user_token():
    """Retrieve the current user's access token from app headers."""
    token = st.context.headers.get("x-forwarded-access-token")
    if not token:
        st.error("User access token not available. Ensure the 'sql' scope is configured under User Authorization in the app settings.")
        st.stop()
    return token


def get_workspace_client():
    """Create a WorkspaceClient authenticated as the current app user."""
    token = get_user_token()
    # Remove platform-injected service principal OAuth creds to avoid
    # "more than one authorization method configured" conflict with PAT auth
    os.environ.pop("DATABRICKS_CLIENT_ID", None)
    os.environ.pop("DATABRICKS_CLIENT_SECRET", None)
    return WorkspaceClient(
        host=os.environ.get("DATABRICKS_HOST"),
        token=token
    )


@st.cache_data(ttl=600, show_spinner=False)
def run_query(query):
    """Execute SQL via Databricks Statement Execution API with type casting."""
    import time
    import requests as _requests
    from databricks.sdk.service.sql import Disposition, Format, StatementState

    w = get_workspace_client()
    warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")
    if not warehouse_id:
        raise ValueError("DATABRICKS_WAREHOUSE_ID environment variable is not set.")

    # Use EXTERNAL_LINKS to avoid the 26MB inline byte limit
    result = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=query,
        wait_timeout="30s",
        disposition=Disposition.EXTERNAL_LINKS,
        format=Format.JSON_ARRAY,
    )

    # Poll if the query hasn't finished within the wait timeout
    while result.status and result.status.state in (
        StatementState.PENDING,
        StatementState.RUNNING,
    ):
        time.sleep(2)
        result = w.statement_execution.get_statement(result.statement_id)

    if result.status and result.status.error:
        raise RuntimeError(f"Query failed: {result.status.error.message}")

    columns = [col.name for col in result.manifest.schema.columns]
    col_types = [col.type_name for col in result.manifest.schema.columns]

    # Fetch data from external links (EXTERNAL_LINKS never uses data_array)
    data = []
    for chunk_info in result.manifest.chunks:
        chunk_result = w.statement_execution.get_statement_result_chunk_n(
            statement_id=result.statement_id, chunk_index=chunk_info.chunk_index
        )
        if chunk_result.external_links:
            for link in chunk_result.external_links:
                resp = _requests.get(link.external_link)
                resp.raise_for_status()
                data.extend(resp.json())
        elif chunk_result.data_array:
            data.extend(chunk_result.data_array)

    df = pd.DataFrame(data, columns=columns)
    INT_TYPES = ("INT", "BIGINT", "SMALLINT", "TINYINT", "LONG", "SHORT")
    FLOAT_TYPES = ("DOUBLE", "FLOAT", "DECIMAL")
    for col_name, col_type in zip(columns, col_types):
        ct = str(getattr(col_type, 'value', col_type)).upper()
        if ct in INT_TYPES:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype("Int64")
        elif ct in FLOAT_TYPES or ct.startswith("DECIMAL"):
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    return df
