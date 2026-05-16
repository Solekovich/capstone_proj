# Eliminating Stylistic Calques and Adapting Machine Translation into Kazakh Based on Literary Standards

This repository contains the Data Science capstone project focused on improving the quality of machine translation into the Kazakh language. The core objective is to eliminate unnatural syntax and structural "calques" (literal translations) often produced by mainstream translation systems, adapting the output to traditional linguistic standards.

## 🚀 Project Overview
Machine translation systems frequently translate texts from Russian to Kazakh literally, preserving Russian syntactic structures. This project implements a lightweight fine-tuning approach to realign translations with natural Kazakh phrasing, guided by the benchmark linguistic works of **Rabiga Syzdyk** and **Gerold Belger**.

### Key Features:
* **Base Model:** Meta's `NLLB-200` (No Language Left Behind) - `distilled-600M`.
* **Fine-Tuning Technique:** **LoRA (Low-Rank Adaptation)** via Hugging Face `PEFT` to adjust translation style without full parameter retraining.
* **Optimization:** 8-bit/4-bit quantization using `bitsandbytes` to ensure efficiency and accessibility on standard hardware.
* **Interactive UI:** Built with **Streamlit** to showcase a side-by-side comparison between the base model's literal translations and the fine-tuned model's natural outputs.

---

## 📊 Dataset & EDA Insights
The model was fine-tuned using subsets extracted from the parallel corpus **KazParC**. 
* **Key EDA Finding:** Exploratory Data Analysis revealed that natural, culturally adaptive Kazakh text is on average **20% more concise** in word count and possesses an entirely different conjunction/clause structure compared to literal machine-translated calques.
* Experiments were conducted across multiple training checkpoints (e.g., 500 and 1000 steps) to evaluate semantic preservation versus stylistic naturalness.

---

## 🛠️ Repository Structure

```text
├── demo_app/
│   └── main.py          # Interactive Streamlit application
├── data/                # Dataset processing and EDA notebooks
├── models/              # (Local only) Trained LoRA adapter weights (safetensors)
├── .gitignore           # Ignores heavy model weights (>2GB) for clean version control
├── requirements.txt     # Universal dependency file optimized for reproduction
└── README.md            # Project documentation
