#!/bin/bash
# watch-metrics.sh - Watch backend metrics

while true; do
  clear
  echo "=== Backend Metrics $(date) ==="
  echo ""
  
  # Total requests across all pods
  TOTAL=$(curl -s -G 'http://localhost:9090/api/v1/query' \
    --data-urlencode 'query=sum(flask_http_request_total{job="backend-service"})' | \
    jq -r '.data.result[0].value[1]')
  echo "Total Requests: $TOTAL"
  
  # Request rate
  RATE=$(curl -s -G 'http://localhost:9090/api/v1/query' \
    --data-urlencode 'query=sum(rate(flask_http_request_total{job="backend-service"}[5m]))' | \
    jq -r '.data.result[0].value[1]')
  echo "Request Rate: $RATE req/s"
  echo ""
  
  # Per-pod breakdown
  echo "Per Pod:"
  curl -s -G 'http://localhost:9090/api/v1/query' \
    --data-urlencode 'query=flask_http_request_total{job="backend-service"}' | \
    jq -r '.data.result[] | "  \(.metric.pod): \(.value[1]) requests"'
  
  echo ""
  echo "Press Ctrl+C to stop"
  sleep 5
done