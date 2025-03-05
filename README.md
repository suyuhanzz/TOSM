# TOSM


## 🌟 Introduction to TOSM
**TOSM** (Topic-Oriented Suggestion Mode) is a novel multi-label classification task model designed based on the neural topic model, incorporating graph structure in the neural topic model, and designing a topic attention mechanism to combine the topic model and multi-label task.
In the context of our paper, TOSM is mainly used to do skill keyword prediction for multiple rounds of interview questioning. This model is also applicable to general multi-label classification tasks.
TOSM operates through two distinct stages: the Graph-based Topic Learning Module and the Topic-specific Keyword Suggestion Module.


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
   pip install -r requirements.txt
2. **Run**:
   ```bash
   python TOSM.py --device {device} --top_k {top_k} --nwindow {nwindow} --num_topic {num_topic} --seed {seed}
