# MSQA

Code and Data for arXiv paper: [MSQA: Benchmarking LLMs on Graduate-Level Materials Science Reasoning and Knowledge]().

## Dependencies
This repo is built with python 3.11. Run the following commands to install dependencies:
```shell
pip install -r ./requirements.txt
```
## Dataset
The `data/MSQA_Dataset.json` file contains the 1757 question and answer pairs for MSQA. An example entry is shown below:
```
{
    "question": "What computational methods and parameters were used to model the reaction pathways of pyrrole hydrogenation on ruthenium and molybdenum oxide surfaces?",
    "answer": "Pyrrole hydrogenation reaction pathways on ruthenium (Ru) and hydrogen molybdenum bronze (H\u2093MoO\u2083) surfaces were modeled using periodic density functional theory (DFT) implemented in the Vienna Ab Initio Simulation Package (VASP). The spin-polarized generalized gradient approximation (GGA) with the PBE functional was used to describe exchange-correlation effects. A plane-wave basis set with a cutoff energy of 400 eV and the projector augmented wave (PAW) method were employed to describe electron-ion interactions.\n\nFor surface calculations, a 3\u00d73\u00d71 Monkhorst-Pack k-point grid was used, while a single k-point and a box size of 15\u00d715\u00d715 \u00c5\u00b3 were applied for gas-phase species. Structural optimizations used a force threshold of 0.05 eV/\u00c5, and a self-consistent field convergence criterion of 5\u00d710\u207b\u2075 eV was applied. Transition states were identified using the climbing image nudged elastic band (CI-NEB) method and validated by vibrational frequency calculations to confirm the presence of one imaginary frequency.\n\nThe Ru(0001) surface was modeled as a 4\u00d74 four-layer slab (lattice constants a = 2.72 \u00c5, c = 4.31 \u00c5), while the H\u2093MoO\u2083 surface was represented by a 2\u00d72 four-layer MoO\u2083(010) slab with two hydrogen atoms attached to terminal oxygen atoms, corresponding to x = 0.125 hydrogen loading. Adsorption energies were calculated as \\(E_{ads} = E_{surface+adsorbate} - (E_{surface} + E_{adsorbate})\\).",
    "topic": "### 1. **Summary of the Purpose of the Paper:**\nThe paper aims to understand the mechanistic details and energetic profiles of pyrrole hydrogenation on Ru(0001) and compare its catalytic performance to hydrogen molybdenum bronze (H\u2093MoO\u2083). Using periodic density functional theory (DFT) calculations, the study identifies key energetic parameters, such as activation energy and desorption energy, and highlights the superior catalytic performance of H\u2093MoO\u2083 due to its lower hydrogenation barrier and more favorable adsorption/desorption energetics. The findings provide insights into the design of low-cost and efficient hydrogenation catalysts.\n\n### 2. **Classification of Purpose:**\n- **<method> or <result>:** The purpose emphasizes **<result>**, as the focus is on deriving insights into catalytic performance and energetics, with a specific conclusion that H\u2093MoO\u2083 outperforms Ru(0001).\n\n### 3. **Relevant Research Questions:**\n1. What are the key energetic barriers (e.g., activation energy, desorption energy) involved in pyrrole hydrogenation on Ru(0001)?\n2. How does the catalytic performance of H\u2093MoO\u2083 compare to Ru(0001) in terms of hydrogenation efficiency and energetics?\n3. What specific properties of H\u2093MoO\u2083 (e.g., protonic nature of H, adsorption energy of pyrrole) contribute to its superior catalytic performance?\n4. Can the mechanistic insights gained from this study be generalized to other hydrogenation reactions or similar catalytic systems?\n5. How can the findings be used to design cost-effective and efficient hydrogenation catalysts for chemical and pharmaceutical applications?",
    "source": "data/diverse_sampled_paper_dataset/diverse_papers_3000/10.1021@acs.jpcc.5b06486.json",
    "source_section": "2 Computational Methods and Models",
    "question_type": "computational",
    "true_false_question": "Was the climbing image nudged elastic band (CI-NEB) method used to identify transition states in the pyrrole hydrogenation pathways modeled on Ru and H\u2093MoO\u2083 surfaces?",
    "true_false_question_answer": "YES"
}
```
Each key in the entry is explained below:
- `question` contains the question of long-form answer.
- `answer` contains the long-form answer. 
- `topic` is the GPT-4o summarization of paper abstracts. It is a paragraph detailing 1) the purpose of the article, 2) classification of the purpose to <method> or <result>, and 3) candidate questions.
- `source` contains the doi of the original paper. 
- `source_section` contains the section name from the original paper that is used to generate the long-form answer.
- `question_type` specifies one of seven question types for long-form answer's question. The seven question types is mentioned in the paper. 
- `true_false_question` contains question of binary-form answer. 
- `true_false_question_answer` contains the answer to `true_false_question` and only has values `YES` or `NO`.


## Generation and Evaluation pipelines
The generation pipeline is specified in `generation_pipeline`. `candidate_answer_generation.py` is used to generate candidate answers with GPT-4o, Gemini-2.0-pro, and Deepseek v3. The candidate answers are then merged with `merge_candidate_answers.py`. Binary questions and answers are generated with `binary_answer_conversion.py`. 

The pipeline for running baselines is specified in the `baslines` folder. `inference.py` contains the code to produce inference results with a model tag on huggingface. `evaluation.py` contains the implementation to evaluate inference results with gpt-4o or rule-based methods.