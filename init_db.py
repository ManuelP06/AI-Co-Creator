#!/usr/bin/env python3
"""
Simple database initialization script for AI Co-Creator
"""

import os
import sys
from pathlib import Path

def init_database():
    """Initialize the database with proper setup."""

    print("Initializing AI Co-Creator Database...")
    print("=" * 50)

    try:
        # Import after ensuring the app path is available
        from app.database import engine
        from app.models import Base

        # Create all tables
        print("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully")

        # Create necessary directories
        print("Creating required directories...")

        directories = [
            "uploads/videos",
            "uploads/temp",
            "outputs/videos",
            "outputs/thumbnails",
            "logs"
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"   {directory}")

        print("\nDatabase initialization completed successfully!")
        print("\nNext steps:")
        print("   1. Start the server: uvicorn app.main:app --reload")
        print("   2. Visit: http://localhost:8001/docs")
        print("   3. Test UI: http://localhost:3000")

        return True

    except Exception as e:
        print(f"Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main initialization function."""

    print("AI Co-Creator Database Setup")
    print("=" * 50)

    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Initialize database
    if init_database():
        print("\nSetup completed successfully!")
        sys.exit(0)
    else:
        print("\nSetup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()