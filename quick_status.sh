#!/bin/bash

echo "========================================="
echo "EAR Challenge - Quick Status Check"
echo "========================================="

# Check if training is running
if screen -list | grep -q "three_stream_training"; then
    echo "✓ Training is RUNNING"
else
    echo "✗ Training is NOT running"
fi

# Check latest checkpoint
if [ -d "checkpoints" ]; then
    LATEST=$(ls -t checkpoints/three_stream_*.pth 2>/dev/null | head -1)
    if [ ! -z "$LATEST" ]; then
        echo "✓ Latest checkpoint: $(basename $LATEST)"
        echo "  Modified: $(stat -c %y "$LATEST" | cut -d'.' -f1)"
    fi
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Status:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk '{printf "  GPU Util: %s%%, Memory: %s/%s MB\n", $1, $2, $3}'
fi

# Check disk space
echo ""
echo "Disk Usage:"
du -sh data/ checkpoints/ 2>/dev/null | awk '{printf "  %s: %s\n", $2, $1}'

echo "========================================="
