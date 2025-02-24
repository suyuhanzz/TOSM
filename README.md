# TOSM


## 🌟 Introduction to TOSM
**TOSM** (Topic-Oriented Suggestion Mode) is a novel framework designed for **multi-round interview questioning skill prediction tasks**, which aims to bridge the gap between job requirements (from job postings), candidate skills (from resumes) and interview questioning (from interview reports). TOSM operates through two distinct stages: the Graph-based Topic Learning Module and the Topic-specific Keyword Suggestion Module.
In the first stage, we employ a topic-oriented approach to identify the underlying connections between application materials and interview questions. This is achieved by using a graph-based technique to map the co-occurrence relationships among skill-related terms. Integrated with a Neural Topic Model (NTM), this module elucidates potential semantic links between the textual representations of application skills and the graph representations of interview skills, effectively bridging their substantial expression gap of vocabulary mismatches.
In the second stage, we enhance attention-based multi-label learning by transforming the original label-specific identification into topic-specific identification. We introduce the ``Topic Query Attention'' mechanism, enabling cross-text topic queries to capture key information in the application context more effectively than single-interview label queries.
This module facilitates a multi-round focus, ensuring a comprehensive assessment across different interview stages.

---

## 📁 Data Description
### Data Privacy & Sample Data
Due to data privacy constraints, we cannot share the original dataset. However, we provide **anonymized sample data** in the `data_sample/` folder:
- **Text Preprocessing**: Non-skill keywords have been filtered to focus purely on skill-oriented content.
- **Input Flexibility**: While our experiments used filtered text, the model natively supports **raw full-text input**:
  - `jd_skill`: Input
  - `resume_skill`: Input
  - `assessment_skill`: Ground-truth labels


---

## 🧩 Model Implementations
We provide three key implementations:
1. **TOSM** (Main Model):  
   Code: `TOSM.py`  
   *the Graph-based Topic Learning Module and the Topic-specific Keyword Suggestion Module*

2. **Ablation Models**:  
   - **GSM-G** (GSM and Graph Structure):  
     Code: `GSM_G.py`  
     *Validates the importance of graph-based skill interdependencies*  
   - **GSM-AT** (attention mechanism):  
     Code: `GSM_AT.py`  
     *Evaluates the impact of topic attention mechanisms*

---

## 🚀 Usage
### Quick Start
1. **Install Dependencies**:
   ```bash
   pip install torch==1.12.0 transformers==4.25.1 numpy==1.22.3
