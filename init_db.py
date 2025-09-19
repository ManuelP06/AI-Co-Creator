#!/usr/bin/env python3
"""
Database initialization script for AI Co-Creator
"""

import os
import sys
from pathlib import Path

def init_database():
    """Initialize the database with proper setup."""

    print("🔧 Initializing AI Co-Creator Database...")
    print("=" * 50)

    try:
        # Import after ensuring the app path is available
        from app.database import engine, SessionLocal
        from app.models import Base
        from app.core.auth import get_password_hash
        from app.models import User
        from app.config import settings

        # Create all tables
        print("📁 Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")

        # Check if admin user exists
        db = SessionLocal()
        try:
            admin_user = db.query(User).filter(User.username == "admin").first()

            if not admin_user:
                print("👤 Creating default admin user...")

                # Create admin user
                hashed_password = get_password_hash("admin123")
                admin_user = User(
                    username="admin",
                    email="admin@ai-creator.local",
                    hashed_password=hashed_password,
                    is_active=True,
                    is_superuser=True
                )

                db.add(admin_user)
                db.commit()
                db.refresh(admin_user)

                print("✅ Admin user created successfully")
                print("   Username: admin")
                print("   Password: admin123")
                print("   🚨 CHANGE PASSWORD IN PRODUCTION!")
            else:
                print("ℹ️  Admin user already exists")

            # Create test user for development
            test_user = db.query(User).filter(User.username == "testuser").first()
            if not test_user:
                print("🧪 Creating test user for development...")

                hashed_password = get_password_hash("testpass123")
                test_user = User(
                    username="testuser",
                    email="test@ai-creator.local",
                    hashed_password=hashed_password,
                    is_active=True,
                    is_superuser=False
                )

                db.add(test_user)
                db.commit()
                db.refresh(test_user)

                print("✅ Test user created successfully")
                print("   Username: testuser")
                print("   Password: testpass123")
            else:
                print("ℹ️  Test user already exists")

        finally:
            db.close()

        # Create necessary directories
        print("📂 Creating required directories...")

        directories = [
            "uploads/videos",
            "uploads/temp",
            "outputs/videos",
            "outputs/thumbnails",
            "logs"
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {directory}")

        print("\n🎉 Database initialization completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Start the server: uvicorn app.main:app --reload")
        print("   2. Visit: http://localhost:8000/docs")
        print("   3. Test with: python test_local.py")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you've installed dependencies: pip install -r requirements.txt")
        return False

    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_environment():
    """Check if environment is properly configured."""

    print("🔍 Checking environment configuration...")

    # Check .env file
    if not os.path.exists('.env'):
        print("⚠️  .env file not found")

        if os.path.exists('.env.example'):
            print("📋 Copying .env.example to .env...")
            import shutil
            shutil.copy('.env.example', '.env')
            print("✅ .env file created from template")
            print("🔧 Edit .env file to configure your settings")
        else:
            print("❌ .env.example not found!")
            return False
    else:
        print("✅ .env file found")

    # Check if SECRET_KEY is set
    from app.config import settings
    if settings.secret_key == "your-secret-key-change-this-in-production":
        print("🔐 Generating secure SECRET_KEY...")
        import secrets
        secret_key = secrets.token_urlsafe(32)

        # Update .env file
        with open('.env', 'r') as f:
            content = f.read()

        if 'SECRET_KEY=' in content:
            # Replace existing SECRET_KEY
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('SECRET_KEY='):
                    lines[i] = f'SECRET_KEY={secret_key}'
                    break
            content = '\n'.join(lines)
        else:
            # Add SECRET_KEY
            content += f'\nSECRET_KEY={secret_key}\n'

        with open('.env', 'w') as f:
            f.write(content)

        print("✅ Secure SECRET_KEY generated and saved")
    else:
        print("✅ SECRET_KEY is configured")

    return True

def main():
    """Main initialization function."""

    print("🚀 AI Co-Creator Database Setup")
    print("=" * 50)

    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Check environment first
    if not check_environment():
        print("❌ Environment check failed")
        sys.exit(1)

    # Initialize database
    if init_database():
        print("\n🎉 Setup completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()