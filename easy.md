## 🚀 **Ready to Run Commands:**

### **1. Quick Test (Recommended First)**
```bash
python run_experiments.py -e cot_narcotics -q simple_test --max-attempts 5
```
*Tests 6 models × 2 queries = 12 attacks*

### **2. Full Narcotics Experiment**
```bash
python run_experiments.py -e cot_narcotics -q narcotics
python run_experiments.py -e cot_narcotics -q narcotics --workers 4  --max-attempts 1

```
*Tests 6 models × 4 narcotics queries = 24 comprehensive attacks*

### **3. Analyze Results**
```bash
python analysis/results_analyzer.py -r cot_narcotics_results
```

## 📊 **What's Changed:**

- ✅ **No more OpenAI 404 errors**
- ✅ **6 working CoT models** (instead of the broken o1-preview)
- ✅ **Sequential execution** (parallel_workers=1) to avoid rate limits
- ✅ **All configurations updated** across all experiments

**Your setup is now ready to run with the new OpenAI models!** 🎯

Try the quick test first to verify everything works, then run the full experiment.