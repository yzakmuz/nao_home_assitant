#!/bin/bash
LOG="/home/nao/server_log.txt"
SERVER="/home/nao/nao_body/server.py"
MAX_RETRIES=30

echo "$(date) -- start_server.sh launched" >> "$LOG"

# Wait for NAOqi
echo "$(date) -- Waiting for NAOqi ..." >> "$LOG"
retries=0
while [ $retries -lt $MAX_RETRIES ]; do
    if python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1',9559)); s.close()" 2>/dev/null; then
        echo "$(date) -- NAOqi is up" >> "$LOG"
        break
    fi
    retries=$((retries + 1))
    sleep 2
done

if [ $retries -eq $MAX_RETRIES ]; then
    echo "$(date) -- ERROR: NAOqi never came up. Aborting." >> "$LOG"
    exit 1
fi

# Wait for network
echo "$(date) -- Waiting for network ..." >> "$LOG"
retries=0
while [ $retries -lt $MAX_RETRIES ]; do
    if ip addr show eth0 2>/dev/null | grep -q "inet "; then
        echo "$(date) -- eth0 is up" >> "$LOG"
        break
    fi
    retries=$((retries + 1))
    sleep 2
done

if [ $retries -eq $MAX_RETRIES ]; then
    echo "$(date) -- WARNING: eth0 not up, starting anyway." >> "$LOG"
fi

sleep 5

# Restart loop
while true; do
    echo "$(date) -- Starting server.py" >> "$LOG"
    python "$SERVER" >> "$LOG" 2>&1
    EXIT_CODE=$?
    echo "$(date) -- server.py exited with code $EXIT_CODE" >> "$LOG"
    if [ $EXIT_CODE -eq 0 ]; then
        echo "$(date) -- Clean exit, not restarting." >> "$LOG"
        break
    fi
    echo "$(date) -- Restarting in 5 seconds ..." >> "$LOG"
    sleep 5
done