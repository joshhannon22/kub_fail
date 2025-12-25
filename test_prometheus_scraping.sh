#!/bin/bash
# test-prometheus-scraping.sh - Focused test of Prometheus scraping

set +e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔍 Testing Prometheus Scraping Setup...${NC}"
echo ""

# 1. Check ServiceMonitor exists
echo -e "${BLUE}📋 ServiceMonitor:${NC}"
if kubectl get servicemonitor -n aiops-demo backend &>/dev/null; then
    echo -e "${GREEN}✅ ServiceMonitor exists${NC}"
else
    echo -e "${RED}❌ ServiceMonitor not found${NC}"
    exit 1
fi

# 2. Check Service
echo -e "${BLUE}🔌 Backend Service:${NC}"
if kubectl get svc -n aiops-demo backend-service &>/dev/null; then
    echo -e "${GREEN}✅ Service exists${NC}"
else
    echo -e "${RED}❌ Service not found${NC}"
    exit 1
fi

# 3. Check Prometheus targets (focused check)
echo -e "${BLUE}🎯 Prometheus Targets:${NC}"
TARGETS=$(curl -s 'http://localhost:9090/api/v1/targets' | \
  jq -r '.data.activeTargets[] | select(.scrapePool | contains("backend")) | "\(.scrapeUrl) - \(.health)"' 2>/dev/null)

if [ -z "$TARGETS" ]; then
    echo -e "${RED}❌ No backend targets found${NC}"
    echo "   Waiting 30 seconds and checking again..."
    sleep 30
    TARGETS=$(curl -s 'http://localhost:9090/api/v1/targets' | \
      jq -r '.data.activeTargets[] | select(.scrapePool | contains("backend")) | "\(.scrapeUrl) - \(.health)"' 2>/dev/null)
fi

if [ -z "$TARGETS" ]; then
    echo -e "${RED}❌ Still no backend targets found${NC}"
    exit 1
else
    echo "$TARGETS" | while read line; do
        if echo "$line" | grep -q "up"; then
            echo -e "${GREEN}✅ $line${NC}"
        else
            echo -e "${RED}❌ $line${NC}"
        fi
    done
fi

# 4. Check if metrics are being collected
echo -e "${BLUE}📊 Metrics Collection:${NC}"
METRICS=$(curl -s -G 'http://localhost:9090/api/v1/query' \
  --data-urlencode 'query=flask_http_request_total{job="backend-service"}' | \
  jq -r '.data.result | length' 2>/dev/null)

if [ "$METRICS" -gt "0" ]; then
    echo -e "${GREEN}✅ Found $METRICS metric series${NC}"
    
    # Show sample
    curl -s -G 'http://localhost:9090/api/v1/query' \
      --data-urlencode 'query=flask_http_request_total{job="backend-service"}' | \
      jq -r '.data.result[0] | "   Sample: \(.metric.pod) = \(.value[1])"' 2>/dev/null
else
    echo -e "${YELLOW}⚠️  No metrics found yet (may need to generate traffic)${NC}"
fi

echo ""
echo -e "${GREEN}✅ Test complete!${NC}"