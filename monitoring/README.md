# Monitoring Stack Setup

This directory contains instructions and configurations for deploying the monitoring stack.

## Deployment Instructions

### 1. Add Helm Repositories

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update
```

### 2. Install Prometheus + Grafana Stack

```bash
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.service.type=NodePort \
  --set prometheus.service.nodePort=30090 \
  --set grafana.service.type=NodePort \
  --set grafana.service.nodePort=30000
```

### 3. Install Loki (Log Aggregation)

```bash
helm install loki grafana/loki \
  --namespace monitoring \
  --version 5.47.2 \
  --set loki.auth_enabled=false \
  --set loki.commonConfig.replication_factor=1 \
  --set loki.storage.type=filesystem \
  --set singleBinary.replicas=1
```

### 4. Install Promtail (Log Collector)

```bash
helm install promtail grafana/promtail \
  --namespace monitoring \
  --set config.lokiAddress=http://loki:3100/loki/api/v1/push
```

## Access Services

### Grafana
- URL: http://localhost:3000
- Get password: `kubectl get secret -n monitoring prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo`
- Username: `admin`

### Prometheus
- URL: http://localhost:9090

## Verify Setup

1. Check pods are running:
   ```bash
   kubectl get pods -n monitoring
   ```

2. Verify Prometheus is scraping metrics:
   - Go to http://localhost:9090
   - Query: `flask_http_request_total`

3. Verify Grafana can access Prometheus:
   - Login to Grafana
   - Go to Connections → Data Sources
   - Prometheus should be pre-configured

4. Add Loki as data source in Grafana:
   - Go to Connections → Data Sources → Add data source
   - Select "Loki"
   - Set URL to: `http://loki:3100`
   - Click "Save & Test"

