#!/bin/bash
# Cleanup script to remove unnecessary files

echo "Cleaning up ShopSmart RAG project..."

# Remove old/unnecessary Python files
rm -f main.py demo.py download_dataset.py check_system.py quick_test.py check_ollama.py setup.py

# Remove old LLM handler (replaced by ollama_handler)
rm -f src/llm_handler.py

# Remove old README (use README_OLLAMA.md instead)
# mv README.md README_OLD.md 2>/dev/null

echo "✓ Cleanup complete!"
echo ""
echo "Remaining core files:"
ls -1 *.py 2>/dev/null
echo ""
echo "To run evaluation:"
echo "  python run_complete_evaluation.py"
