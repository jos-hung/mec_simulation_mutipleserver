#!/bin/bash

if [ "$(docker ps -q)" ]; then
    echo "Stopping all Docker containers..."
    docker stop $(docker ps -q)
fi

if [ "$(docker ps -a -q)" ]; then
    echo "Removing all Docker containers..."
    docker rm $(docker ps -a -q)
fi

echo "All Docker containers stopped and removed."

echo "Closing all other terminals except the current one..."

current_pid=$$
current_shell=$(basename "$SHELL")

# for pid in $(pgrep -x "$current_shell"); do
#     if [ "$pid" -ne "$current_pid" ]; then
#         kill -9 "$pid"
#     fi
# done

# if [[ "$OSTYPE" == darwin* ]]; then
#     current_app_pid=$(ps -p $$ -o ppid= | tr -d ' ')
#     for app in Terminal iTerm2; do
#         for pid in $(pgrep -x "$app"); do
#             if [ "$pid" -ne "$current_app_pid" ]; then
#                 kill -9 "$pid"
#             fi
#         done
#     done
# else
#     # for pid in $(pgrep -f "gnome-terminal|konsole|xterm"); do
#     #     if [ "$pid" -ne "$current_pid" ]; then
#     #         kill -9 "$pid"
#     #     fi
#     # done
# fi


# osascript <<EOD
# tell application "Terminal"
#     repeat with w in windows
#         if id of w is not $current_window_id then
#             close w
#         end if
#     end repeat
# end tell
# EOD

# echo "All other Terminal windows closed."

echo "All other terminal sessions have been closed."
