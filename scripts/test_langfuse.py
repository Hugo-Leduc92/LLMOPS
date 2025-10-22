"""Test script for Langfuse integration."""

import os
from langfuse import Langfuse

# Initialize Langfuse
langfuse = Langfuse()

# Test basic functionality
print("✅ Langfuse initialized successfully!")

# Test environment variables
required_env_vars = [
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY", 
    "LANGFUSE_HOST"
]

missing_vars = []
for var in required_env_vars:
    if not os.getenv(var):
        missing_vars.append(var)

if missing_vars:
    print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
    print("Add these to your .env file:")
    for var in missing_vars:
        print(f"  {var}=your_value_here")
else:
    print("✅ All required Langfuse environment variables are set!")

print("\n📊 Langfuse will track:")
print("  - User messages and model responses")
print("  - Generation parameters (temperature, top_p, etc.)")
print("  - Model metadata and performance metrics")
print("  - Request/response timing")
