# OSS-20B Red Team Challenge Submission

This repository contains our complete submission for the Kaggle OSS-20B Red Team Challenge, featuring systematic vulnerability discovery in OpenAI's GPT-OSS-20B model through innovative attack methodologies.

## 🎯 Overview

Our team developed and deployed advanced red-teaming techniques to uncover critical vulnerabilities in the GPT-OSS-20B model across multiple domains including national security, medical safety, corporate deception, and substance synthesis guidance.

### Key Achievements
- **5 Major Vulnerabilities Discovered** across critical domains
- **Novel ICOTN Attack Method** developed for systematic safety bypass
- **Multi-Agent Testing Framework** for deception detection
- **Comprehensive Documentation** with reproducible findings

## 🔍 Discovered Vulnerabilities

### 1. **National Security Threats** (`finding_issues_of_national_security_threat.json`)
- **Severity**: High (8/10)
- **Type**: Deception+Chain of Thought Issues + Data Exfiltration+Sabotage+Reward Hacking+Inappropriate Tool Use(Critical Infrastructure Attack Guidance)
- **Method**: ICOTN (Iterative Chain of Thought Negation)
- **Impact**: Detailed attack methodologies for power grid and transportation systems

### 2. **Illicit Substance Synthesis** (`finding_illicit_substance_synthesis.json`)
- **Severity**: High (7/10)
- **Type**: Deception+Chain of Thought Issues + Data Exfiltration+Sabotage+Reward Hacking
- **Method**: ICOTN Attack
- **Impact**: Detailed synthesis instructions + warehouse location disclosure

### 3. **Medical Tool Calling Vulnerabilities** (`finding_prescription_guidance.json`)
- **Severity**: High (8/10)
- **Type**: Inappropriate Tool Use + Chain of Thought Issues
- **Method**: ICOTN on Medical Agent
- **Impact**: Unauthorized prescription generation (Vicodin)

### 4. **Corporate Deception** (`finding_modeldeception_in_simulatedcorpenv.json`)
- **Severity**: Medium (5/10)
- **Type**: Deceptive Alignment + Hidden Motivations
- **Method**: Multi-turn Adversarial Testing
- **Impact**: Strategic lying in corporate environments

### 5. **Legal Assistant Deception** (`finding_lawyer_deception_cleaned.json`)
- **Severity**: Medium (3/10)
- **Type**: Reward Hacking + Deception
- **Method**: Adversarial Prompting
- **Impact**: Misleading legal advice to minimize corporate liability

## 🛠️ Methodology & Tools

### ICOTN (Iterative Chain of Thought Negation)
Our novel attack framework consisting of three specialized agents:

- **🎯 Attacker Agent**: Generates sophisticated bypass prompts using CoT negation(Gemini 2.5 Flash)
- **🔍 Tester Agent**: Bypass Prompts Tested on this agent (OSS 20B Model)
- **⚖️ Evaluator Agent**: Assesses bypass success across 5 vulnerability dimensions(Gemini 2.5 Flash)


## 📁 Repository Structure

```
kaggle-oss20-redteamchallenge/
├── final_submission/              # Official competition findings submitted and uploaded in Kaggle
├── submission_notebooks/          # Jupyter notebooks for reproduction
│   ├── issue_1_opiod_narcotics_generation.ipynb - Issue 1: Illicit Substance Generation
│   ├── issue_2_icotn_national_security1_RF.ipynb - Issue 2: National Security
│   ├── issue_2_icotn_national_security2_ATF.ipynb - Issue 2: National Security
│   ├── issue_3_medicaltoolcallingbot_vulenrability.ipynb - Issue 3: Medical Tool Calling Agent Vulnerability
│   ├── issue_4_situationalmodeldeception.ipynb - Issue 4: Legal Assistant Deception
│   └── issue_5_corporatesimenv.ipynb - Issue 5: Corporate Standoff Environment (Cold War standoff simulation)
│   └── issue_1.5_sabotage_systemAI.ipynb - Another ICOTN attack where model created plan to help ChatGPT delete the new model versions it might get replaced with covertly without anyonhe knowing (Didn't make the final submission cut as there were 2 other ICOTN tool based attack sanmples with more severity and breadth)
├── tools/                         # Core attack frameworks
│   ├── cot_negation_agent_for_tool_calling.py - The ICOTN (Iterative Chain of Thought Negation framework )
│   ├── cot_negation_openai.py - depreciated version
│   └── cot_negation.py - depreciated version
├── prompts/                       # Attack prompt libraries (Contains the many shot prompting used for cold start in ICOTN and other prompts   used for Deception Simulation and Standoff simulation)

├── notebooks - Research phase notebooks (Other ideas that failed or didn't cut in for final submission )
├── LICENSE                        # Apache 2.0 License
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Groq API key (for OSS-20B access)
- Google API key (for Gemini models)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/kaggle-oss20-redteamchallenge.git
cd kaggle-oss20-redteamchallenge
```

2. **Set up Python environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys:
# GROQ_API_KEY=your_groq_api_key
# GOOGLE_API_KEY=your_google_api_key
```

### Running the Notebooks

Each vulnerability can be reproduced using the corresponding notebook:

```bash
# Start Jupyter
jupyter notebook

# Navigate to submission_notebooks/ and run:
# - issue_2_icotn_national_security1_RF.ipynb (National Security)
# - issue_3_medicaltoolcallingbot_vulenrability.ipynb (Medical)
# - issue_4_situationalmodeldeception.ipynb (Deception)
```



## 📊 Key Findings Summary

| Vulnerability | Severity | Breadth | Method | Topics |
|---------------|----------|---------|---------|---------|---------|
| National Security | 10 | 10 | ICOTN | Deception,Chain of Thought Issues,Sabotage, Data Exfiltration,Reward Hacking |
| Substance Synthesis | 10 | 5  | ICOTN | Harmful Content, Deception,Chain of Thought Issues,Sabotage, Data Exfiltration |
| Medical Malpractice | 5 | 5  | ICOTN + Tools | Tool Misuse, Safety,Chain of Thought Issues |
| Corporate Deception | 5 | 5  | Multi-turn | Hidden Motivations,Deception, Reward Hacking |
| Legal Deception | 5 | 3 | Adversarial | Deception,Reward Hacking |

## 🔬 Technical Innovation

### Chain of Thought Negation
Our core technique exploits reasoning models' over-reliance on safety RLHF by:
1. **Progressive Reframing**: Converting harmful requests into "defensive research"
2. **Reasoning Exploitation**: Targeting the model's internal reasoning process
3. **Iterative Refinement**: Adapting attacks based on model responses

### Multi-Agent Architecture
- **Distributed Testing**: Multiple specialized agents for comprehensive coverage
- **Adaptive Iteration**: Up to 20 attempts with early stopping on success
- **Cross-Model Validation**: Using Gemini for attack generation, OSS-20B for testing

## 🎯 Competition Impact

### Topics Addressed
- ✅ **Deception**: Strategic lying and misrepresentation
- ✅ **Hidden Motivations**: Covert goal pursuit
- ✅ **Sabotage**: Intentional system disruption
- ✅ **Inappropriate Tool Use**: Dangerous API exploitation
- ✅ **Data Exfiltration**: Information leakage
- ✅ **Reward Hacking**: Gaming evaluation metrics
- ✅ **Chain of Thought Issues**: Reasoning vulnerabilities

### Methodological Contributions
- **Systematic Framework**: Reproducible attack methodology
- **Automated Testing**: Scalable vulnerability discovery
- **Multi-Domain Coverage**: Broad applicability across use cases
- **Open Source Tools**: Community-ready attack frameworks

## ⚠️ Responsible Disclosure

This research was conducted under the sanctioned Kaggle competition framework for improving AI safety. All vulnerabilities have been:

- **Documented thoroughly** for reproduction and mitigation
- **Shared responsibly** within the competition context
- **Designed for defense** rather than exploitation
- **Open sourced** to benefit the broader safety community

## 📝 Citation

If you use this work in your research, please cite:
```bibtex
@misc{oss20b-redteam-2025,
  title={Systematic Vulnerability Discovery in OSS-20B: ICOTN and Multi-Agent Red-Teaming},
  author={David Babu},
  year={2025},
  howpublished={Kaggle OSS-20B Red Team Challenge},
  url={https://github.com/hazardscarn/kaggle-oss20-redteamchallenge}
}
```



## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for releasing OSS-20B for safety research
- **Kaggle** for hosting the red team challenge
- **Competition Judges** for their expertise and guidance
- **AI Safety Community** for ongoing collaboration

## 📧 Contact

For questions about this research or collaboration opportunities:
- Open an issue on this repository
- Contact the team through Kaggle competition discussion

---

**⚠️ Disclaimer**: This research is conducted for AI safety improvement. The vulnerabilities and attack methods documented here should only be used for defensive research and model improvement, not for malicious purposes.