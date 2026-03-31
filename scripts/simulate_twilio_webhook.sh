#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000/webhooks/twilio}"
FROM_NUMBER="${FROM_NUMBER:-+15555550111}"
TO_NUMBER="${TO_NUMBER:-+15555550222}"
MESSAGE_SID="${MESSAGE_SID:-SMLOCAL$(date +%s)}"
BODY="${1:-yo i need to finish the CAD for the enclosure by tomorrow night}"

curl -X POST "$API_URL" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "From=$FROM_NUMBER" \
  --data-urlencode "To=$TO_NUMBER" \
  --data-urlencode "Body=$BODY" \
  --data-urlencode "MessageSid=$MESSAGE_SID" \
  --data-urlencode "NumMedia=0"

echo

