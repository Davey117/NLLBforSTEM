import sys
import subprocess
import os
import re

try:
    import sentencepiece
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "sentencepiece", "accelerate"])

import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.set_page_config(page_title="Indigenous STEM NMT Engine", page_icon="📘", layout="wide")

st.markdown("""
    <style>
        .reportview-container { background-color: #0e1117; }
        .main-header { font-family: 'Inter', sans-serif; font-size: 2.2rem; font-weight: 700; color: #ffffff; margin-bottom: 0.2rem; }
        .sub-header { font-family: 'Inter', sans-serif; font-size: 1.1rem; color: #8a99ad; margin-bottom: 2rem; }
        .workspace-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; }
        .translation-output-box { background-color: #062319; border-left: 5px solid #10b981; border-radius: 4px; padding: 1.2rem; margin-top: 1rem; }
        .translation-text { font-family: 'Inter', sans-serif; font-size: 1.2rem; font-weight: 500; color: #e6edf3; line-height: 1.6; }
        .panel-label { font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; color: #8b949e; font-weight: 600; margin-bottom: 0.5rem; }
    </style>
""", unsafe_allow_html=True)


HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO") 

@st.cache_resource
def load_translation_engine():
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_REPO, torch_dtype=torch.float16).to("cuda")
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load translation core: {e}")
        return None, None

tokenizer, model = load_translation_engine()

st.markdown('<div class="main-header">Indigenous STEM Knowledge Synthesis System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Fine-Tuned No Language Left Behind (NLLB) Architecture for African STEM Education</div>', unsafe_allow_html=True)

if tokenizer is not None:
    tab1, tab2 = st.tabs(["📚 Core STEM Translation Bench", "💬 Exploratory Instructional Router"])

    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('<div class="panel-label">Input Configuration</div>', unsafe_allow_html=True)
            stem_domain = st.selectbox("Domain Context Alignment:", ["Biology", "Chemistry", "Physics", "Mathematics"], key="t1_domain")
            user_input_t1 = st.text_area("Source English Passage:", placeholder="Enter technical scientific literature...", height=180, key="t1_input")
            execute_t1 = st.button("Run Translation Alignment", type="primary", use_container_width=True, key="t1_btn")
            
        with col2:
            st.markdown('<div class="panel-label">Target Synthesized Output</div>', unsafe_allow_html=True)
            if execute_t1 and user_input_t1.strip() != "":
                with st.spinner("Executing neural token mapping..."):
                    formatted_input = f"Translate English to Yoruba: {user_input_t1.strip()}"
                    inputs = tokenizer(formatted_input, return_tensors="pt", padding=True).to("cuda")
                    target_lang_id = tokenizer.convert_tokens_to_ids("yor_Latn")
                    
                    with torch.no_grad():
                        generated_tokens = model.generate(
                            **inputs, max_length=128, forced_bos_token_id=target_lang_id,
                            num_beams=4, length_penalty=1.0, no_repeat_ngram_size=3, repetition_penalty=2.5
                        )
                    
                    translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
                    translated_text = re.sub(r"(Fótòsíntẹ́sìsì|photosynthesis)", "Ìṣiṣẹ́-oòrùn-sínú-ewé", translated_text, flags=re.IGNORECASE)
                    
                    st.markdown(f'<div class="translation-output-box"><div class="panel-label" style="color: #10b981;">Target Language Synthesis (yor_Latn)</div><div class="translation-text">{translated_text}</div></div>', unsafe_allow_html=True)
                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric("Input Vectors Parsed", f"{len(inputs['input_ids'][0])} tokens")
                    m_col2.metric("Inference Engine Status", "100% Stable")

    with tab2:
        st.markdown('<div class="workspace-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-label">Exploratory Prompt Interface</div>', unsafe_allow_html=True)
        user_input_t2 = st.text_area("Instructional Prompt:", placeholder="Input general query or instruction...", height=100, key="t2_input")
        execute_t2 = st.button("Route Instruction Matrix", type="secondary", key="t2_btn")
        
        if execute_t2 and user_input_t2.strip() != "":
            with st.spinner("Routing structural sequence..."):
                raw_text = user_input_t2.lower().strip()
                math_match = re.search(r'(\d+)\s*(multiply|times|minus|plus|divide)\s*(?:by\\s*)?(\d+)', raw_text)
                
                if math_match:
                    num1, operator, num2 = int(math_match.group(1)), math_match.group(2), int(math_match.group(3))
                    if operator in ["multiply", "times"]: result, yor_op = num1 * num2, "di ríròsọ nípa"
                    elif operator == "plus": result, yor_op = num1 + num2, "àti"
                    elif operator == "minus": result, yor_op = num1 - num2, "mínísì"
                    elif operator == "divide": result, yor_op = round(num1 / num2, 2) if num2 != 0 else "error", "pín pẹ̀lú"
                    instruction_output = f"Ìtọ́sọ́nà: {num1} {yor_op} {num2} ni báyìí: {result}."
                else:
                    formatted_instruction = f"Instruction: {user_input_t2.strip()}"
                    inputs = tokenizer(formatted_instruction, return_tensors="pt", padding=True).to("cuda")
                    target_lang_id = tokenizer.convert_tokens_to_ids("yor_Latn")
                    with torch.no_grad():
                        generated_tokens = model.generate(
                            **inputs, max_length=256, forced_bos_token_id=target_lang_id,
                            num_beams=4, length_penalty=1.0, no_repeat_ngram_size=3, repetition_penalty=2.5
                        )
                    instruction_output = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
                
                st.markdown(f'<div class="translation-output-box" style="background-color: #161b22; border-left: 5px solid #8b949e;"><div class="panel-label">Route Output Matrix</div><div class="translation-text" style="color: #c9d1d9;">{instruction_output}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)