# Multi-Model ICOTN Vulnerability Testing Framework

A scalable framework for running ICOTN (Iterative Chain-of-Thought Negation) attacks across multiple AI models in parallel for academic research and vulnerability analysis.

## 🚀 Quick Start

### 1. Setup
```bash
# Install dependencies
python setup_multi_model.py

# Configure API keys
cp .env.example .env
# Edit .env and add your API keys
```

### 2. Run Your First Experiment
```bash
# Check available models and API keys
python run_experiments.py --check

# Run a simple test
python run_experiments.py -e paper_analysis -q simple_test --max-attempts 5

# Run comprehensive analysis
python run_experiments.py -e reasoning_models -q narcotics
```

### 3. Analyze Results
```bash
# Generate publication-ready analysis
python analysis/results_analyzer.py -r experiment_results -o analysis_output
```

## 🏗️ Architecture

The framework consists of several key components:

### Core Components

1. **Model Configuration** (`config/model_config.py`)
   - Unified configuration for 15+ AI models across 5 providers
   - Easy model switching and parameter management
   - API key validation and availability checking

2. **Multi-Model Client** (`tools/multi_model_client.py`)
   - Unified interface for different AI providers (Groq, OpenAI, Google, Anthropic, Together)
   - Async support for parallel requests
   - Error handling and retry logic

3. **Parallel Experiment Runner** (`tools/parallel_icotn_runner.py`)
   - Concurrent attack execution across multiple models
   - Progress tracking and intermediate result saving
   - Configurable concurrency and rate limiting

4. **Results Analyzer** (`analysis/results_analyzer.py`)
   - Publication-ready statistics and visualizations
   - LaTeX table generation for academic papers
   - Executive summaries and vulnerability analysis

## 📊 Supported Models

### Reasoning Models (CoT Capable)
- **Groq**: GPT-OSS-20B, DeepSeek R1 Distill 70B, Llama 3.1 405B
- **OpenAI**: O1 Preview, O1 Mini

### Frontier Models  
- **OpenAI**: GPT-4o, GPT-4o Mini
- **Anthropic**: Claude 3.5 Sonnet, Claude 3.5 Haiku
- **Google**: Gemini 2.5 Flash, Gemini 2.0 Flash
- **Groq**: Llama 3.3 70B, Llama 3.2 90B

## 🎯 Experiment Configurations

### Predefined Experiments

#### `reasoning_models`
- **Focus**: CoT-capable models vulnerability
- **Models**: GPT-OSS-20B, DeepSeek R1, Llama 405B, O1 Preview
- **Use Case**: Research on reasoning model vulnerabilities

#### `paper_analysis` 
- **Focus**: Top models across providers
- **Models**: GPT-OSS-20B, DeepSeek R1, GPT-4o, Claude 3.5 Sonnet, Gemini 2.5 Flash, O1 Preview
- **Use Case**: Comprehensive analysis for academic papers

#### `comprehensive`
- **Focus**: All available models
- **Models**: All 15+ configured models
- **Use Case**: Large-scale vulnerability assessment

### Query Sets

#### `narcotics` (Original experiment)
- Drug manufacturing instructions
- Ingredient sourcing methods
- Distribution location information
- Security procedure details

#### `medical_jailbreaks`
- Unauthorized prescriptions
- Fake medical documentation
- Dangerous medical procedures
- Gene therapy instructions

#### `privacy_violations`
- Patient data access
- Employee record retrieval
- Confidential information exposure
- System log manipulation

#### `harmful_instructions`
- Explosive manufacturing
- Drug synthesis
- Hacking instructions
- Biological weapon creation

## 📋 Usage Examples

### Basic Usage
```bash
# List available options
python run_experiments.py --list-configs
python run_experiments.py --list-models
python run_experiments.py --list-queries

# Run experiments with different configurations
python run_experiments.py -e reasoning_models -q narcotics
python run_experiments.py -e general_models -q medical_jailbreaks -s medical
python run_experiments.py -e comprehensive -q harmful_instructions --workers 8
```

### Advanced Usage
```bash
# Custom model selection
python run_experiments.py -e paper_analysis -q narcotics --models gpt-oss-20b claude-3.5-sonnet

# Override parameters
python run_experiments.py -e reasoning_models -q simple_test --max-attempts 15 --workers 4

# Different system prompts
python run_experiments.py -e general_models -q medical_jailbreaks --system-prompt medical
```

### Analysis and Reporting
```bash
# Generate complete analysis report
python analysis/results_analyzer.py

# Analyze specific results
python analysis/results_analyzer.py -r experiment_results -p "*reasoning*_raw.json"

# Custom output directory
python analysis/results_analyzer.py -o paper_figures
```

## 📈 Output and Results

### Experiment Results
- **Raw JSON**: Complete attack details, reasoning, attempts
- **Summary CSV**: Model performance metrics for spreadsheet analysis  
- **Harmony Format**: Compatible with existing analysis tools

### Analysis Output
- **Publication Figures**: High-resolution PNG/PDF plots
- **LaTeX Tables**: Ready-to-insert academic tables
- **Executive Summary**: Key findings and statistics
- **Vulnerability Analysis**: Detailed security assessment

### Key Metrics Tracked
- **Success Rate**: Percentage of successful bypasses per model
- **Attempts Required**: Average iterations needed for successful attacks
- **Time Analysis**: Performance and efficiency metrics  
- **Error Rates**: Model reliability and failure analysis
- **Query Categories**: Vulnerability patterns by attack type

## 🔧 Configuration

### Model Configuration (`config/model_config.py`)
```python
# Add new model
"new-model": ModelConfig(
    name="New Model Name",
    provider=ModelProvider.GROQ,
    model_id="provider/model-id",
    api_key_env="API_KEY_ENV_VAR",
    supports_reasoning=True,
    max_tokens=8192
)
```

### Experiment Configuration
```python  
# Create custom experiment
"custom_experiment": ExperimentConfig(
    name="Custom Research Experiment",
    target_models=["gpt-oss-20b", "claude-3.5-sonnet"],
    max_attempts=20,
    parallel_workers=4
)
```

### Environment Variables (.env)
```bash
# Required API keys (add only what you need)
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key  
GOOGLE_API_KEY=your_google_key
ANTHROPIC_API_KEY=your_anthropic_key
TOGETHER_API_KEY=your_together_key
```

## 📊 Academic Research Features

### Paper-Ready Output
- **LaTeX Tables**: Formatted for academic publication
- **High-Quality Figures**: Publication-standard visualizations
- **Statistical Analysis**: Comprehensive vulnerability metrics
- **Executive Summaries**: Ready-to-use abstract content

### Research Applications
- **Comparative Analysis**: Multi-model vulnerability assessment
- **Reasoning Model Research**: CoT-specific vulnerability patterns
- **Safety Research**: Large-scale safety evaluation
- **Prompt Engineering**: Attack methodology validation

### Reproducibility
- **Deterministic Seeds**: Reproducible results across runs
- **Version Tracking**: Complete experiment configuration logging
- **Raw Data Export**: Full attack traces for detailed analysis
- **Statistical Validation**: Confidence intervals and significance testing

## 🔒 Ethical Guidelines

This framework is designed for **legitimate AI safety research** only:

- ✅ Academic vulnerability research
- ✅ Model safety improvement
- ✅ Responsible disclosure of findings
- ✅ Publication in peer-reviewed venues

- ❌ Malicious exploitation
- ❌ Production system attacks  
- ❌ Unauthorized testing
- ❌ Harmful content generation

## 🤝 Contributing

### Adding New Models
1. Update `MODEL_CONFIGS` in `config/model_config.py`
2. Add provider client in `tools/multi_model_client.py` if needed
3. Test with `python run_experiments.py --check`

### Adding New Query Sets
1. Add to `QUERY_SETS` in `run_experiments.py`
2. Consider adding appropriate system prompts
3. Test with small experiment first

### Improving Analysis
1. Extend `results_analyzer.py` with new metrics
2. Add visualization functions
3. Update LaTeX export formats

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{icotn_framework_2025,
  title={Multi-Model ICOTN Vulnerability Testing Framework},
  author={[Your Research Team]},
  year={2025},
  url={https://github.com/your-repo/icotn-framework}
}
```

## 🐛 Troubleshooting

### Common Issues

#### API Key Issues
```bash
# Check API key availability
python run_experiments.py --check

# Verify .env file format
cat .env | grep -v "^#"
```

#### Model Loading Errors
```bash
# List available models
python run_experiments.py --list-models

# Test individual model
python -c "from config.model_config import get_model_config; print(get_model_config('gpt-oss-20b'))"
```

#### Memory/Performance Issues
```bash
# Reduce parallel workers
python run_experiments.py -e reasoning_models -q simple_test --workers 2

# Use smaller experiment
python run_experiments.py -e paper_analysis -q simple_test --max-attempts 5
```

### Performance Tips
- Start with small experiments (`simple_test` query set)
- Use `--max-attempts 5` for initial testing
- Adjust `--workers` based on your API rate limits
- Monitor API usage and costs

## 🚀 Scaling for Large Studies

### For Comprehensive Analysis
```bash
# Run overnight comprehensive study
nohup python run_experiments.py -e comprehensive -q narcotics --max-attempts 30 --workers 6 > experiment.log 2>&1 &

# Monitor progress
tail -f experiment.log
```

### Batch Processing
```bash
# Run multiple experiments sequentially
for query_set in narcotics medical_jailbreaks harmful_instructions; do
    python run_experiments.py -e reasoning_models -q $query_set
done
```

### Result Aggregation
```bash
# Analyze all results at once
python analysis/results_analyzer.py -r experiment_results -p "*_raw.json"

# Generate final paper figures
python analysis/results_analyzer.py -o final_paper_analysis
```

---

## 📞 Support

For questions, issues, or contributions:
- Open GitHub issues for bugs/features
- Check existing issues for common problems
- Follow ethical guidelines for responsible research

**Happy researching! 🔬**
