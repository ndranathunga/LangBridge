#!/bin/bash

# Name of the tmux session + time stamp
SESSION_NAME="train_lb_$(date +%s)"

# Path to the script to be run inside tmux
SCRIPT_PATH="scripts/train_lb/test.sh"

# Check if the session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null

# If the session does not exist, create it and run the script
if [ $? != 0 ]; then
  tmux new-session -d -s $SESSION_NAME
  tmux send-keys -t $SESSION_NAME "bash $SCRIPT_PATH" C-m
fi

# Attach to the session
tmux attach-session -t $SESSION_NAME