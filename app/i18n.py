"""Lightweight i18n helper.

The language lives in st.session_state (shared by every page). Each page owns its
own STRINGS dict and looks text up with t():

    STRINGS = {"en": {"title": "Data"}, "fr": {"title": "Données"}}
    st.title(t(STRINGS, "title"))
"""
import streamlit as st

DEFAULT_LANG = "en"


def get_lang() -> str:
    return st.session_state.get("lang", DEFAULT_LANG)


def t(strings: dict, key: str, **fmt) -> str:
    """Text for the current language (falls back to English, then to the key), optionally formatted."""
    en = strings.get("en", {})
    text = strings.get(get_lang(), en).get(key, en.get(key, key))
    return text.format(**fmt) if fmt else text


def render_lang_toggle() -> None:
    """EN/FR toggle in the sidebar. Called once from Home.py, before st.navigation."""
    st.session_state.setdefault("lang", DEFAULT_LANG)
    with st.sidebar:
        choice = st.radio("Language", ["EN", "FR"], index=0 if get_lang() == "en" else 1,
                          horizontal=True, key="lang_toggle", label_visibility="collapsed")
    st.session_state["lang"] = choice.lower()
