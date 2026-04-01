#!/usr/bin/env bash
set -euo pipefail

APP_PORT="${APP_PORT:-8000}"
API_URL="${API_URL:-http://localhost:${APP_PORT}/webhooks/twilio}"
FROM_NUMBER="${FROM_NUMBER:-+15555550111}"
TO_NUMBER="${TO_NUMBER:-+15555550222}"
ROUNDS="${ROUNDS:-60}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0.15}"

MESSAGES=(
  "yo whatup"
  "i need to finish the cad for the enclosure by tomorrow night"
  "prof just dropped another assignment"
  "in class rn"
  "later i need to send that email"
  "this weekend i need to finish my portfolio draft"
  "what do i need to get done tonight"
  "what do i have due this week"
  "i keep getting distracted because i need to fix the website first"
  "just finished the first draft"
  "my bad i underestimated this"
  "before studio i need to print the board"
  "are these canned responses or live ai generated?"
)

echo "sending ${ROUNDS} simulated inbound messages to ${API_URL}"
for i in $(seq 1 "${ROUNDS}"); do
  idx=$(( (i - 1) % ${#MESSAGES[@]} ))
  body="${MESSAGES[$idx]}"
  sid="SMSTRESS$(date +%s)${i}"

  curl -sS -X POST "$API_URL" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    --data-urlencode "From=$FROM_NUMBER" \
    --data-urlencode "To=$TO_NUMBER" \
    --data-urlencode "Body=$body" \
    --data-urlencode "MessageSid=$sid" \
    --data-urlencode "NumMedia=0" >/dev/null

  if (( i % 10 == 0 )); then
    echo "sent ${i}/${ROUNDS}"
  fi
  sleep "$SLEEP_SECONDS"
done

echo "done"
