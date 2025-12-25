#!/bin/bash
# quick-health-check.sh - Quick verification of Phase 1.1

echo "🏥 Quick Health Check - Phase 1.1"
echo "=================================="
echo ""

# Check monitoring stack
echo "📊 Monitoring Stack:"
./check_monitoring.sh | grep -E "✅|❌" | head -10
echo ""

# Check backend scraping
echo "🎯 Backend Scraping:"
TARGET_COUNT=$(curl -s 'http://localhost:9090/api/v1/targets' | \
  jq '[.data.activeTargets[] | select(.scrapePool | contains("backend"))] | length' 2>/dev/null)

if [ "$TARGET_COUNT" -eq "2" ]; then
    echo "✅ 2 backend targets being scraped"
else
    echo "⚠️  Found $TARGET_COUNT backend targets (expected 2)"
fi

# Check metrics exist
echo ""
echo "📈 Metrics:"
METRIC_COUNT=$(curl -s -G 'http://localhost:9090/api/v1/query' \
  --data-urlencode 'query=flask_http_request_total{job="backend-service"}' | \
  jq '.data.result | length' 2>/dev/null)

if [ "$METRIC_COUNT" -gt "0" ]; then
    echo "✅ $METRIC_COUNT metric series found"
else
    echo "⚠️  No metrics found"
fi

echo ""
echo "✅ Health check complete!"