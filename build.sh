#!/bin/bash

# Build optimization script
echo "Starting optimized build process..."

# Clean up temporary files
echo "Cleaning temporary files..."
rm -rf /tmp/* /var/tmp/* 2>/dev/null || true

# Check space usage
echo "Checking space usage..."
du -sh * | sort -rh | head -20
df -h

# Install Python dependencies with optimizations
echo "Installing Python dependencies..."
pip install --no-cache-dir --disable-pip-version-check -r requirements.txt

# Clean up after installation
echo "Cleaning up after installation..."
pip cache purge 2>/dev/null || true
rm -rf /tmp/* /var/tmp/* 2>/dev/null || true

echo "Build optimization complete!"
