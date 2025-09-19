#!/bin/bash

# AI Co-Creator Setup Script
# Automated setup for development and production deployment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_header() {
    echo
    echo -e "${BLUE}🚀 $1${NC}"
    echo "=================================="
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        print_error "Python is not installed or not in PATH"
        exit 1
    fi

    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    REQUIRED_VERSION="3.9"

    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
        print_status "Python $PYTHON_VERSION detected (meets requirement: >= $REQUIRED_VERSION)"
    else
        print_error "Python $PYTHON_VERSION detected, but >= $REQUIRED_VERSION is required"
        exit 1
    fi
}

# Function to setup virtual environment
setup_venv() {
    print_header "Setting up Python Virtual Environment"

    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists"
        print_info "To recreate, delete 'venv' directory and run setup again"
    else
        print_info "Creating virtual environment..."
        $PYTHON_CMD -m venv venv
        print_status "Virtual environment created"
    fi

    # Activate virtual environment
    print_info "Activating virtual environment..."
    source venv/bin/activate || {
        print_error "Failed to activate virtual environment"
        exit 1
    }

    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip

    print_status "Virtual environment ready"
}

# Function to install dependencies
install_dependencies() {
    print_header "Installing Dependencies"

    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found!"
        exit 1
    fi

    print_info "Installing Python packages..."
    print_info "This may take several minutes (PyTorch with CUDA is large)..."

    # Install with timeout and retry logic
    pip install -r requirements.txt --timeout 300 || {
        print_warning "Installation failed, retrying with cache refresh..."
        pip install --no-cache-dir -r requirements.txt --timeout 300 || {
            print_error "Failed to install dependencies"
            print_info "Try manually: pip install -r requirements.txt"
            exit 1
        }
    }

    print_status "Dependencies installed successfully"
}

# Function to setup environment configuration
setup_environment() {
    print_header "Configuring Environment"

    # Copy .env.example to .env if it doesn't exist
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            print_info "Creating .env from template..."
            cp .env.example .env
            print_status ".env file created"
        else
            print_error ".env.example not found!"
            exit 1
        fi
    else
        print_info ".env file already exists"
    fi

    # Generate secure SECRET_KEY if using default
    print_info "Checking SECRET_KEY configuration..."
    if grep -q "your-secret-key-change-this-in-production" .env 2>/dev/null; then
        print_info "Generating secure SECRET_KEY..."
        SECRET_KEY=$($PYTHON_CMD -c "import secrets; print(secrets.token_urlsafe(32))")

        # Update .env file
        if command_exists sed; then
            sed -i.bak "s/SECRET_KEY=your-secret-key-change-this-in-production/SECRET_KEY=$SECRET_KEY/" .env
            rm -f .env.bak
            print_status "Secure SECRET_KEY generated"
        else
            print_warning "Please manually update SECRET_KEY in .env file"
        fi
    else
        print_status "SECRET_KEY already configured"
    fi

    print_status "Environment configuration complete"
}

# Function to initialize database
init_database() {
    print_header "Initializing Database"

    if [ -f "init_db.py" ]; then
        print_info "Running database initialization..."
        $PYTHON_CMD init_db.py || {
            print_error "Database initialization failed"
            print_info "Try manually: python init_db.py"
            exit 1
        }
        print_status "Database initialized successfully"
    else
        print_warning "init_db.py not found, skipping database initialization"
    fi
}

# Function to verify installation
verify_installation() {
    print_header "Verifying Installation"

    # Check if test file exists
    if [ -f "test_api.py" ]; then
        print_info "Running verification tests..."

        # Start server in background
        print_info "Starting test server..."
        $PYTHON_CMD -m uvicorn app.main:app --host 127.0.0.1 --port 8000 &
        SERVER_PID=$!

        # Wait for server to start
        sleep 5

        # Run tests
        if $PYTHON_CMD test_api.py; then
            print_status "All tests passed! Installation verified."
            TESTS_PASSED=true
        else
            print_error "Some tests failed. Check the output above."
            TESTS_PASSED=false
        fi

        # Stop server
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true

        if [ "$TESTS_PASSED" = true ]; then
            return 0
        else
            return 1
        fi
    else
        print_warning "test_api.py not found, skipping verification"
        return 0
    fi
}

# Function to show next steps
show_next_steps() {
    print_header "Setup Complete!"

    echo
    print_status "AI Co-Creator has been set up successfully!"
    echo
    print_info "Next steps:"
    echo "  1. Activate virtual environment:"
    echo "     source venv/bin/activate"
    echo
    echo "  2. Start development server:"
    echo "     uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
    echo
    echo "  3. Access the application:"
    echo "     • API Documentation: http://localhost:8000/docs"
    echo "     • Health Check: http://localhost:8000/health"
    echo "     • Main API: http://localhost:8000/api/v1/"
    echo
    echo "  4. Run tests anytime:"
    echo "     python test_api.py"
    echo
    print_info "For production deployment, see README.md"
    echo
}

# Function to handle errors
handle_error() {
    print_error "Setup failed at step: $1"
    print_info "Check the error messages above for details"
    print_info "You can also run steps manually following README.md"
    exit 1
}

# Main setup function
main() {
    echo
    echo "🤖 AI Co-Creator Automated Setup"
    echo "=================================="
    echo
    print_info "This script will set up AI Co-Creator for local development"
    print_warning "Make sure you have at least 5GB free space for dependencies"
    echo

    # Confirm proceed
    read -p "Continue with setup? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Setup cancelled"
        exit 0
    fi

    # Check prerequisites
    print_header "Checking Prerequisites"

    check_python_version || handle_error "Python version check"

    if command_exists git; then
        print_status "Git found"
    else
        print_warning "Git not found (recommended for development)"
    fi

    # Setup steps
    setup_venv || handle_error "Virtual environment setup"

    # Activate virtual environment for remaining steps
    source venv/bin/activate

    install_dependencies || handle_error "Dependency installation"
    setup_environment || handle_error "Environment configuration"
    init_database || handle_error "Database initialization"

    # Verify installation
    if verify_installation; then
        show_next_steps
    else
        print_warning "Setup completed but verification failed"
        print_info "You may need to manually check the configuration"
        show_next_steps
    fi
}

# Run main function
main "$@"