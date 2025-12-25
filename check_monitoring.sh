#!/bin/bash
# check-monitoring.sh - Quick health check for monitoring stack

set +e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 Checking Prerequisites...${NC}"
echo ""

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}❌ kubectl not found. Please install kubectl.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ kubectl installed${NC}"

# Check helm
if ! command -v helm &> /dev/null; then
    echo -e "${RED}❌ helm not found. Please install helm.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ helm installed${NC}"

# Check cluster connection
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}❌ Cannot connect to cluster. Is your Kind cluster running?${NC}"
    echo "   Try: kind get clusters"
    exit 1
fi
echo -e "${GREEN}✅ Cluster connection OK${NC}"

# Function to setup Helm repos
setup_helm_repos() {
    echo -e "${YELLOW}📦 Setting up Helm repositories...${NC}"
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null
    helm repo add grafana https://grafana.github.io/helm-charts 2>/dev/null
    helm repo update > /dev/null 2>&1
    echo -e "${GREEN}✅ Helm repositories configured${NC}"
}

# Function to install Prometheus + Grafana
install_prometheus_grafana() {
    echo -e "${YELLOW}📊 Installing Prometheus + Grafana stack...${NC}"
    helm install prometheus prometheus-community/kube-prometheus-stack \
      --namespace monitoring \
      --create-namespace \
      --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
      --set prometheus.service.type=NodePort \
      --set prometheus.service.nodePort=30090 \
      --set grafana.service.type=NodePort \
      --set grafana.service.nodePort=30000 \
      > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Prometheus + Grafana installed${NC}"
        echo -e "${YELLOW}   Waiting for pods to be ready (this may take 1-2 minutes)...${NC}"
        kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=prometheus -n monitoring --timeout=120s > /dev/null 2>&1
        kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=grafana -n monitoring --timeout=120s > /dev/null 2>&1
    else
        echo -e "${YELLOW}⚠️  Prometheus + Grafana may already be installed or installation failed${NC}"
    fi
}

# Function to install Loki
install_loki() {
    echo -e "${YELLOW}📝 Installing Loki...${NC}"
    helm install loki grafana/loki \
      --namespace monitoring \
      --version 5.47.2 \
      --set loki.auth_enabled=false \
      --set loki.commonConfig.replication_factor=1 \
      --set loki.storage.type=filesystem \
      --set singleBinary.replicas=1 \
      > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Loki installed${NC}"
        echo -e "${YELLOW}   Waiting for Loki to be ready...${NC}"
        kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=loki -n monitoring --timeout=120s > /dev/null 2>&1
    else
        echo -e "${YELLOW}⚠️  Loki may already be installed or installation failed${NC}"
    fi
}

# Function to install Promtail
install_promtail() {
    echo -e "${YELLOW}📤 Installing Promtail...${NC}"
    helm install promtail grafana/promtail \
      --namespace monitoring \
      --set config.lokiAddress=http://loki:3100/loki/api/v1/push \
      > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Promtail installed${NC}"
        echo -e "${YELLOW}   Waiting for Promtail to be ready...${NC}"
        kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=promtail -n monitoring --timeout=120s > /dev/null 2>&1
    else
        echo -e "${YELLOW}⚠️  Promtail may already be installed or installation failed${NC}"
    fi
}

# Setup Helm repos (always check/update)
setup_helm_repos

# Check if monitoring namespace exists
if ! kubectl get namespace monitoring &> /dev/null; then
    echo -e "${YELLOW}📁 Creating monitoring namespace...${NC}"
    kubectl create namespace monitoring
    echo -e "${GREEN}✅ Monitoring namespace created${NC}"
else
    echo -e "${GREEN}✅ Monitoring namespace exists${NC}"
fi

echo ""
echo -e "${BLUE}📊 Checking Monitoring Stack Status...${NC}"
echo ""

# Check Prometheus pods
PROM_PODS=$(kubectl get pods -n monitoring -l app.kubernetes.io/name=prometheus --no-headers 2>/dev/null | wc -l | tr -d ' ')
if [ "$PROM_PODS" -eq "0" ]; then
    echo -e "${RED}❌ Prometheus pods not found${NC}"
    install_prometheus_grafana
else
    echo -e "${GREEN}✅ Prometheus: $PROM_PODS pod(s)${NC}"
fi

# Check Grafana pods
GRAFANA_PODS=$(kubectl get pods -n monitoring -l app.kubernetes.io/name=grafana --no-headers 2>/dev/null | wc -l | tr -d ' ')
if [ "$GRAFANA_PODS" -eq "0" ]; then
    echo -e "${RED}❌ Grafana pods not found${NC}"
    # Prometheus and Grafana are installed together, so if Grafana is missing but Prometheus exists, reinstall
    if [ "$PROM_PODS" -gt "0" ]; then
        echo -e "${YELLOW}⚠️  Grafana missing but Prometheus exists. You may need to reinstall the stack.${NC}"
    else
        install_prometheus_grafana
    fi
else
    echo -e "${GREEN}✅ Grafana: $GRAFANA_PODS pod(s)${NC}"
fi

# Check Loki pods
LOKI_PODS=$(kubectl get pods -n monitoring -l app.kubernetes.io/name=loki --no-headers 2>/dev/null | wc -l | tr -d ' ')
if [ "$LOKI_PODS" -eq "0" ]; then
    echo -e "${RED}❌ Loki pods not found${NC}"
    install_loki
else
    echo -e "${GREEN}✅ Loki: $LOKI_PODS pod(s)${NC}"
fi

# Check Promtail pods
PROMTAIL_PODS=$(kubectl get pods -n monitoring -l app.kubernetes.io/name=promtail --no-headers 2>/dev/null | wc -l | tr -d ' ')
if [ "$PROMTAIL_PODS" -eq "0" ]; then
    echo -e "${RED}❌ Promtail pods not found${NC}"
    install_promtail
else
    echo -e "${GREEN}✅ Promtail: $PROMTAIL_PODS pod(s)${NC}"
fi

echo ""
echo -e "${BLUE}🔗 Service Access:${NC}"
echo "   Grafana:   http://localhost:3000"
echo "   Prometheus: http://localhost:9090"
echo ""

# Get Grafana password if secret exists
if kubectl get secret -n monitoring prometheus-grafana &> /dev/null; then
    GRAFANA_PASSWORD=$(kubectl get secret -n monitoring prometheus-grafana -o jsonpath="{.data.admin-password}" 2>/dev/null | base64 --decode 2>/dev/null)
    if [ ! -z "$GRAFANA_PASSWORD" ]; then
        echo -e "${BLUE}🔑 Grafana credentials:${NC}"
        echo "   Username: admin"
        echo "   Password: $GRAFANA_PASSWORD"
    fi
fi

echo ""
echo -e "${GREEN}✅ Health check complete!${NC}"