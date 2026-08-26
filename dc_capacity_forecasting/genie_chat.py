"""AI Chat interface powered by Databricks Genie Agent.

Provides a floating chat button and modal dialog that persists across tabs.
Uses the Genie Conversation API via the Databricks SDK.
"""
import time
from datetime import timedelta
import os
import pandas as pd
import streamlit as st
from connection import get_workspace_client

GENIE_SPACE_ID = os.environ.get("GENIE_SPACE_ID")
_POLL_INTERVAL = 2  # seconds between status polls
_POLL_TIMEOUT = 120  # max seconds to wait for a response


# ---------------------------------------------------------------------------
# Genie API helpers
# ---------------------------------------------------------------------------

def _poll_message(w, space_id: str, conversation_id: str, message_id: str):
    """Poll until the Genie message reaches a terminal state."""
    from databricks.sdk.service.dashboards import MessageStatus

    deadline = time.time() + _POLL_TIMEOUT
    while time.time() < deadline:
        msg = w.genie.get_message(
            space_id=space_id,
            conversation_id=conversation_id,
            message_id=message_id,
        )
        if msg.status in (
            MessageStatus.COMPLETED,
            MessageStatus.FAILED,
            MessageStatus.CANCELLED,
        ):
            return msg
        time.sleep(_POLL_INTERVAL)
    return msg  # return whatever we have on timeout


def _send_message(question: str):
    """Send a question to the Genie Agent and return the completed message."""
    w = get_workspace_client()
    conv_id = st.session_state.get("genie_conversation_id")

    if conv_id:
        # Continue existing conversation
        resp = w.genie.create_message(
            space_id=GENIE_SPACE_ID,
            conversation_id=conv_id,
            content=question,
        )
    else:
        # Start a new conversation
        resp = w.genie.start_conversation(
            space_id=GENIE_SPACE_ID,
            content=question,
        )

    # SDK may return a Wait object (long-running op) or a direct response.
    if hasattr(resp, "result"):
        try:
            msg = resp.result(timeout=timedelta(seconds=_POLL_TIMEOUT))
        except Exception:
            # result() raises on non-COMPLETED states; fall back to polling
            conv_id_new = getattr(resp, "conversation_id", None)
            msg_id = getattr(resp, "message_id", None)
            if conv_id_new and msg_id:
                msg = _poll_message(w, GENIE_SPACE_ID, conv_id_new, msg_id)
                st.session_state["genie_conversation_id"] = conv_id_new
            else:
                raise
    else:
        # Direct response — extract IDs and poll
        if hasattr(resp, "conversation_id"):
            conv_id_new = resp.conversation_id
        elif hasattr(resp, "conversation") and resp.conversation:
            conv_id_new = resp.conversation.id
        else:
            conv_id_new = conv_id

        msg_id = resp.message.id if hasattr(resp, "message") else resp.id
        msg = _poll_message(w, GENIE_SPACE_ID, conv_id_new or conv_id, msg_id)
        if conv_id_new:
            st.session_state["genie_conversation_id"] = conv_id_new

    # Persist conversation ID for follow-ups
    if hasattr(msg, "conversation_id") and msg.conversation_id:
        st.session_state["genie_conversation_id"] = msg.conversation_id

    return w, msg


def _extract_response_text(msg) -> str:
    """Extract human-readable text from a completed Genie message."""
    from databricks.sdk.service.dashboards import MessageStatus

    # Handle failed / cancelled messages
    if getattr(msg, "status", None) in (MessageStatus.FAILED, MessageStatus.CANCELLED):
        error = getattr(msg, "error", None)
        reason = getattr(error, "message", None) or str(error) if error else None
        if reason:
            return f"\u26A0\uFE0F The assistant couldn't answer: {reason}"
        return "\u26A0\uFE0F The assistant was unable to process your question. Please try rephrasing."

    parts = []
    if msg.attachments:
        for att in msg.attachments:
            if hasattr(att, "text") and att.text and hasattr(att.text, "content"):
                parts.append(att.text.content)
            if hasattr(att, "query") and att.query:
                if getattr(att.query, "description", None):
                    parts.append(att.query.description)
                if getattr(att.query, "query", None):
                    parts.append(f"```sql\n{att.query.query}\n```")
    # Fallback to top-level content if no attachment text
    if not parts and getattr(msg, "content", None):
        parts.append(msg.content)
    if not parts:
        parts.append("I wasn't able to generate a response. Please try rephrasing your question.")
    return "\n\n".join(parts)


def _fetch_query_result(w, msg) -> pd.DataFrame | None:
    """Attempt to retrieve tabular query results from the message."""
    try:
        if not msg.attachments:
            return None
        for att in msg.attachments:
            if not (hasattr(att, "query") and att.query):
                continue
            att_id = getattr(att, "id", None)
            if not att_id:
                continue
            result = w.genie.get_message_query_result(
                space_id=GENIE_SPACE_ID,
                conversation_id=msg.conversation_id,
                message_id=msg.id,
                attachment_id=att_id,
            )
            if (
                result.statement_response
                and result.statement_response.manifest
                and result.statement_response.result
            ):
                columns = [
                    c.name for c in result.statement_response.manifest.schema.columns
                ]
                data = result.statement_response.result.data_array or []
                if data:
                    return pd.DataFrame(data, columns=columns)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

@st.dialog("\U0001F4AC AI Assistant", width="large")
def render_chat_dialog():
    """Render the Genie-powered chat modal dialog."""
    # Initialize chat history
    if "genie_messages" not in st.session_state:
        st.session_state["genie_messages"] = []

    # Header with reset button
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.caption("Ask questions about your supply chain data")
    with col2:
        if st.button("\U0001F504 New Chat", key="genie_reset"):
            st.session_state["genie_messages"] = []
            st.session_state.pop("genie_conversation_id", None)
            st.rerun(scope="fragment")

    # Display chat history
    chat_container = st.container(height=420)
    with chat_container:
        if not st.session_state["genie_messages"]:
            st.markdown(
                "_Hi! I can help you explore your distribution center capacity "
                "and inventory data. Ask me anything._"
            )
        for msg in st.session_state["genie_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("dataframe") is not None:
                    st.dataframe(msg["dataframe"], use_container_width=True)

    # Chat input
    if prompt := st.chat_input(
        "Ask a question about your data...", key="genie_chat_input"
    ):
        # Append user message
        st.session_state["genie_messages"].append(
            {"role": "user", "content": prompt}
        )

        # Get Genie response
        with st.spinner("Thinking..."):
            try:
                w, result_msg = _send_message(prompt)
                response_text = _extract_response_text(result_msg)
                df_result = _fetch_query_result(w, result_msg)
            except Exception as e:
                response_text = f"\u26A0\uFE0F Sorry, I encountered an error: {e}"
                df_result = None

        # Append assistant message
        st.session_state["genie_messages"].append(
            {
                "role": "assistant",
                "content": response_text,
                "dataframe": df_result,
            }
        )
        st.rerun(scope="fragment")

