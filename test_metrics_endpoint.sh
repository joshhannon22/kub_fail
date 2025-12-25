#!/bin/bash
# test-metrics-endpoint.sh - Test backend metrics endpoint

set +e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔍 Testing Backend Metrics Endpoint...${NC}"
echo ""

# Get a backend pod
POD=$(kubectl get pod -n aiops-demo -l app=backend -o jsonpath='{.items[0].metadata.name}')

if [ -z "$POD" ]; then
    echo -e "${RED}❌ No backend pods found${NC}"
    exit 1
fi

echo -e "${BLUE}📦 Testing pod: $POD${NC}"
echo ""

# Port-forward in background with better error handling
kubectl port-forward -n aiops-demo $POD 5000:5000 > /tmp/pf.log 2>&1 &
PF_PID=$!
sleep 3

# Check if port-forward is working
if ! kill -0 $PF_PID 2>/dev/null; then
    echo -e "${RED}❌ Port-forward failed${NC}"
    cat /tmp/pf.log
    exit 1
fi

# Test metrics endpoint
echo -e "${BLUE}📊 Checking /metrics endpoint:${NC}"
METRICS_RESPONSE=$(curl -s -w "%{http_code}" http://localhost:5000/metrics -o /tmp/metrics.txt)
HTTP_CODE="${METRICS_RESPONSE: -3}"

if [ "$HTTP_CODE" = "200" ]; then
    if grep -q "flask_http_request_total" /tmp/metrics.txt; then
        echo -e "${GREEN}✅ Metrics endpoint working${NC}"
        echo ""
        echo "Sample metrics:"
        grep "flask_http_request_total" /tmp/metrics.txt | head -3
    else
        echo -e "${YELLOW}⚠️  Metrics endpoint accessible but no Flask metrics found${NC}"
    fi
else
    echo -e "${RED}❌ Metrics endpoint returned HTTP $HTTP_CODE${NC}"
fi

# Cleanup
kill $PF_PID 2>/dev/null
wait $PF_PID 2>/dev/null
rm -f /tmp/pf.log /tmp/metrics.txt

echo ""
echo -e "${GREEN}✅ Test complete!${NC}"