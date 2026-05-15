import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import torch
import os

st.set_option('client.showErrorDetails', True)

st.set_page_config(page_title="Kazakh MT Calque Eliminator")
st.title("Eliminating Stylistic Calques in Kazakh MT")
st.write("Сравнение базового NLLB и дообученных моделей (LoRA и Full Fine-tuning)")

# --- ПУТИ ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_V1 = os.path.join(BASE_DIR, "models", "checkpoints", "checkpoint-565")
PATH_FINAL_ADAPTER = os.path.join(BASE_DIR, "models", "final_adapter_1000")
BASE_MODEL_NAME = "facebook/nllb-200-distilled-600M"

option = st.selectbox("Choose Model Version", 
                      ("Base NLLB", "LoRA 500 lines"))

input_text = st.text_area("Русский текст", height=150)

@st.cache_resource
def load_base():
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.float16, use_safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    return model, tokenizer
base_model, base_tokenizer = load_base()

if st.button("Перевести"):
    if not input_text.strip():
        st.warning("Введите текст для перевода")
    else:
        with st.spinner(f"Загрузка {option} и генерация..."):
            try:
                # 1. Очистка кеша GPU (важно для Windows/CUDA)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 2. ЛОГИКА ЗАГРУЗКИ (Индивидуально для каждого выбора)
                if option == "LoRA 500 lines":
                    # Загружаем CPU-модель как полноценную
                    current_model = AutoModelForSeq2SeqLM.from_pretrained(PATH_V1)
                    current_tokenizer = AutoTokenizer.from_pretrained(PATH_V1)
                
                else:
                    # Чистая базовая модель
                    current_model = base_model
                    current_tokenizer = base_tokenizer

                # 3. ПОДГОТОВКА И ГЕНЕРАЦИЯ
                device = "cuda" if torch.cuda.is_available() else "cpu"
                current_model.to(device)
                current_model.eval()

                inputs = current_tokenizer(input_text, return_tensors="pt").to(device)

                with torch.no_grad():
                    generated_tokens = current_model.generate(
                        **inputs,
                        forced_bos_token_id=current_tokenizer.convert_tokens_to_ids("kaz_Cyrl"),
                        max_length=128,
                        num_beams=5,
                    )

                result = current_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                
                st.success(f"Результат ({option}):")
                st.info(result)

            except Exception as e:
                st.error(f"Произошла ошибка: {e}")