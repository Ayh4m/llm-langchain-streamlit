import time
import json

import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback

from dotenv import load_dotenv

from prompts import (PROMPTS, OUTPUT_PARSERS, HEADINGS,
                     MULTIPLES_TBL_PROMPT, RISK_TBL_PROMPT, BARRIERS_TO_ENTRY_PROMPT)
from chains import get_chains_results, get_chain_result
from gpt_params import MODELS, MAX_TOKEN
from enums import FN
import formatters


# Environment Variables
load_dotenv()


# === Helpers ===
def to_formatted_string(input_str, formatter):
    return formatter(json.loads(input_str))


def get_chain_execution_result(chain, input_vars, multi=False):
    with st.spinner("Wait for it ..."):
        start_time = time.perf_counter()
        with get_openai_callback() as cb:
            result = get_chain_result(chain, input_vars) if not multi else get_chains_results(chain, input_vars)
        elapsed_time = time.perf_counter() - start_time
    return result, elapsed_time, cb


def show_chain_execution_info(elapsed_time, cb):
    st.success(f"Executed in {elapsed_time:.2f} seconds.", icon="✅")
    st.info(
        f"""
        Total Tokens: {cb.total_tokens}\n
        Prompt Tokens: {cb.prompt_tokens}\n
        Completion Tokens: {cb.completion_tokens}\n
        Total Cost: ${cb.total_cost:.4f}
        """,
        icon="ℹ️"
    )


# === APP ===
st.set_page_config(page_title="Industry Info", page_icon=":robot_face:")

# ** Configurations Sidebar **
with st.sidebar:
    llm_model = MODELS[st.radio("Model", list(MODELS.keys()))]
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, step=0.1, value=1.0)
    max_tokens = st.slider("Max Tokens", min_value=1, max_value=MAX_TOKEN[llm_model], step=1, value=2000)

# Large Language Model (LLM)
llm = ChatOpenAI(model=llm_model, temperature=temperature, max_tokens=max_tokens)

# Chains
CHAINS = [LLMChain(llm=llm, prompt=prompt, verbose=False) for prompt in PROMPTS]
MULTIPLE_TBL_CHAIN = LLMChain(llm=llm, prompt=MULTIPLES_TBL_PROMPT, verbose=True)
RISK_TBL_CHAIN = LLMChain(llm=llm, prompt=RISK_TBL_PROMPT, verbose=True)
BARRIERS_TO_ENTRY_CHAIN = LLMChain(llm=llm, prompt=BARRIERS_TO_ENTRY_PROMPT, verbose=True)

# ** Main **
st.header("Powered by :robot_face:")
st.write("")
selected_fn = st.selectbox(label="Select a function", options=[fn.value for fn in FN])
st.divider()

# == Function 1 ==
if selected_fn == FN.FN1.value:

    st.markdown("Function 1: The AI will give you some information about the entered industry")

    input_text = st.text_input(
        label="Industry Title",
        placeholder="Enter an industry title e.g. HR Consulting",
        max_chars=25,
        label_visibility="collapsed"
    )

    if input_text:

        results, exec_time, tkn_cb = get_chain_execution_result(CHAINS, [{"industry_title": input_text} for i in
                                                                         range(len(CHAINS))], multi=True)

        results = [OUTPUT_PARSERS[r].parse(result) if OUTPUT_PARSERS[r] is not None else result
                   for r, result in enumerate(results)]

        st.markdown(f"#### Here are your results")

        show_chain_execution_info(exec_time, tkn_cb)

        for r, result in enumerate(results):
            with st.expander(f"##### {HEADINGS[r]}"):
                if isinstance(result, str):
                    st.markdown(result)
                elif isinstance(result, list):
                    for item in result:
                        st.markdown(f"- {item.capitalize()}")

# == Function 2 ==
if selected_fn == FN.FN2.value:

    st.markdown("Function 2: The AI will explain the entered multiples table of an industry")

    input_text = st.text_area(
        label="Industry Multiples Table",
        placeholder="Enter a multiples table object",
        label_visibility="collapsed"
    )

    if input_text:

        result, exec_time, tkn_cb = get_chain_execution_result(MULTIPLE_TBL_CHAIN, {
            "multiples_table": to_formatted_string(input_text, formatters.multiples_tbl_formatter)})

        st.markdown(f"#### Here are your results")

        show_chain_execution_info(exec_time, tkn_cb)

        with st.expander("##### Explanation"):
            st.markdown(result)

# == Function 3 ==
if selected_fn == FN.FN3.value:

    st.markdown("Function 3: The AI will explain the entered risk table of an industry")

    input_text = st.text_area(
        label="Risk Table",
        placeholder="Enter a risk table object",
        label_visibility="collapsed"
    )

    if input_text:

        result, exec_time, tkn_cb = get_chain_execution_result(RISK_TBL_CHAIN, {
            "risk_table": to_formatted_string(input_text, formatters.risk_tbl_formatter)})

        st.markdown(f"#### Here are your results")

        show_chain_execution_info(exec_time, tkn_cb)

        with st.expander("##### Explanation"):
            st.markdown(result)

# == Function 4 ==
if selected_fn == FN.FN4.value:

    st.markdown("Function 4: The AI will explain the entered barriers to entry checklist of an industry")

    input_text = st.text_area(
        label="Barriers To Entry Checklist",
        placeholder="Enter a barriers to entry checklist object",
        label_visibility="collapsed"
    )

    if input_text:

        result, exec_time, tkn_cb = get_chain_execution_result(BARRIERS_TO_ENTRY_CHAIN, {
            "barriers_to_entry": to_formatted_string(input_text, formatters.barriers_to_entry_formatter)})

        st.markdown(f"#### Here are your results")

        show_chain_execution_info(exec_time, tkn_cb)

        with st.expander("##### Explanation"):
            st.markdown(result)
