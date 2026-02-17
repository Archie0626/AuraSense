#!/bin/bash

# HAR Activity Recognition System - Run Script

echo "🏃 Human Activity Recognition System"
echo "===================================="

# Function to check if Python is installed
check_python() {
    if command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        echo "❌ Python is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    echo "✓ Using $($PYTHON_CMD --version)"
}

# Function to check if pip is installed
check_pip() {
    if ! command -v pip3 &>/dev/null && ! command -v pip &>/dev/null; then
        echo "❌ pip is not installed. Please install pip."
        exit 1
    fi
    echo "✓ pip is installed"
}

# Function to create virtual environment
create_venv() {
    if [ ! -d "venv" ]; then
        echo "📦 Creating virtual environment..."
        $PYTHON_CMD -m venv venv
        echo "✓ Virtual environment created"
    else
        echo "✓ Virtual environment already exists"
    fi
}

# Function to activate virtual environment
activate_venv() {
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows
        source venv/Scripts/activate
    else
        # Unix/MacOS
        source venv/bin/activate
    fi
    echo "✓ Virtual environment activated"
}

# Function to install requirements
install_requirements() {
    echo "📚 Installing requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        echo "✓ Requirements installed successfully"
    else
        echo "❌ Failed to install requirements"
        exit 1
    fi
}

# Function to train models
train_models() {
    echo ""
    echo "🤖 Training models..."
    echo "======================"
    $PYTHON_CMD models/save_models.py
}

# Function to run the app
run_app() {
    echo ""
    echo "🚀 Starting Streamlit app..."
    echo "============================"
    streamlit run app.py
}

# Function to check for updates
check_updates() {
    echo ""
    echo "🔄 Checking for updates..."
    if [ -f ".git/config" ]; then
        git fetch
        UPDATES=$(git log HEAD..origin/main --oneline)
        if [ -n "$UPDATES" ]; then
            echo "📢 Updates available! Run 'git pull' to update."
        else
            echo "✓ System is up to date"
        fi
    fi
}

# Main execution
main() {
    echo ""
    check_python
    check_pip
    echo ""
    
    # Check if --skip-train flag is passed
    SKIP_TRAIN=false
    for arg in "$@"
    do
        if [ "$arg" == "--skip-train" ]; then
            SKIP_TRAIN=true
        fi
    done
    
    # Check if models exist
    if [ "$SKIP_TRAIN" = false ] && [ ! -f "models/random_forest_model.pkl" ]; then
        echo "⚠️  Models not found. Training required."
        TRAIN_REQUIRED=true
    else
        echo "✓ Models found"
        TRAIN_REQUIRED=false
    fi
    
    # Create virtual environment if it doesn't exist
    create_venv
    
    # Activate virtual environment
    activate_venv
    
    # Install requirements
    install_requirements
    
    # Train models if required
    if [ "$TRAIN_REQUIRED" = true ]; then
        train_models
    fi
    
    # Run the app
    run_app
    
    # Check for updates
    check_updates
}

# Parse command line arguments
case "$1" in
    --help|-h)
        echo "Usage: ./run.sh [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --skip-train   Skip model training (use existing models)"
        echo "  --train-only   Only train models, don't run the app"
        echo "  --install-only Only install dependencies"
        echo ""
        echo "Examples:"
        echo "  ./run.sh                 # Run with model training if needed"
        echo "  ./run.sh --skip-train    # Run without training"
        echo "  ./run.sh --train-only    # Only train models"
        exit 0
        ;;
    --train-only)
        check_python
        check_pip
        create_venv
        activate_venv
        install_requirements
        train_models
        exit 0
        ;;
    --install-only)
        check_python
        check_pip
        create_venv
        activate_venv
        install_requirements
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac