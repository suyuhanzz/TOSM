# TOSM


## 🌟 Introduction to TOSM
**TOSM** (Task-Oriented Skill Matching) is a novel framework designed for **interview skill prediction tasks**, which aims to bridge the gap between job requirements (from job postings) and candidate skills (from resumes). By leveraging textual data from both sources and labels derived from interview questioning reports, TOSM employs a hybrid approach combining graph-based representations and adaptive training mechanisms to predict skill alignment. Its key innovation lies in dynamically modeling skill interdependencies while filtering noise from non-skill-related text.

---

## 📁 Data Description
### Data Privacy & Sample Data
Due to data privacy constraints, we cannot share the original dataset. However, we provide **anonymized sample data** in the `data_sample/` folder:
- **Text Preprocessing**: Non-skill keywords (e.g., company names, personal identifiers) have been filtered to focus purely on skill-oriented content.
- **Input Flexibility**: While our experiments used filtered text, the model natively supports **raw full-text input** for both:
  - `job_posting.txt`: Job description text
  - `resume.txt`: Candidate resume text
  - `label.txt`: Ground-truth labels from interview reports

### Task Input Format
To run the model, simply concatenate the job posting and resume texts into a single input sequence. Labels should align with the interview reports.

---

## 🧩 Model Implementations
We provide three key implementations:
1. **TOSM** (Main Model):  
   Code: `TOSM.py`  
   *Hybrid graph-text architecture with adaptive skill alignment*

2. **Ablation Models**:  
   - **GSM-G** (Graph Structure Removed):  
     Code: `ablation/GSM_G.py`  
     *Validates the importance of graph-based skill interdependencies*  
   - **GSM-AT** (Adaptive Training Removed):  
     Code: `ablation/GSM_AT.py`  
     *Evaluates the impact of adaptive training mechanisms*

---

## 🚀 Usage
### Quick Start
1. **Install Dependencies**:
   ```bash
   pip install torch==1.12.0 transformers==4.25.1 numpy==1.22.3
