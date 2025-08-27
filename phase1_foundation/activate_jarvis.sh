#!/bin/bash
# JARVIS Environment Activation Script

echo "🤖 Activating JARVIS environment..."

# Activate virtual environment
source ~/JARVIS_PROJECT/jarvis_env/bin/activate

# Set environment variables
export JARVIS_PROJECT_ROOT="$HOME/JARVIS_PROJECT"
export JARVIS_MODELS_PATH="$JARVIS_PROJECT_ROOT/models"
export JARVIS_CACHE_PATH="$JARVIS_PROJECT_ROOT/cache"

# Add project to Python path
export PYTHONPATH="$JARVIS_PROJECT_ROOT/phase1_foundation:$PYTHONPATH"

echo "✅ JARVIS environment activated!"
echo "📁 Project root: $JARVIS_PROJECT_ROOT"
echo "🧠 Models path: $JARVIS_MODELS_PATH"
echo ""
echo "🚀 Ready to run JARVIS commands!"
        