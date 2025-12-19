# Project: Intelligent Kubernetes Auto-Scaler with Predictive Failure Detection

## Core Components

### 1. Observability Layer

- Deploy a multi-tier application on Kubernetes (e.g., web frontend, API backend, database)
- Set up Prometheus for metrics collection and Grafana for visualization
- Implement the ELK stack (Elasticsearch, Logstash, Kibana) or Loki for log aggregation
- Collect metrics like CPU, memory, request latency, error rates, pod restarts

### 2. Data Pipeline

- Build a Python service that continuously ingests metrics and logs from Prometheus and your logging system
- Store time-series data in a format suitable for ML (like InfluxDB or just structured files)
- Create feature engineering pipeline: calculate rolling averages, rate of change, anomaly scores

### 3. ML Intelligence Layer

- **Predictive Scaling Model:** Train an LSTM or time-series forecasting model (Prophet, ARIMA) to predict traffic patterns 5-15 minutes ahead, triggering proactive scaling before load spikes
- **Anomaly Detection:** Implement an autoencoder or isolation forest to detect unusual behavior patterns that indicate potential failures
- **Resource Optimization:** Build a reinforcement learning agent or simpler heuristic model that learns optimal resource allocation based on cost vs performance

### 4. Action Layer

- Create a controller service that interfaces with Kubernetes API
- Implement automated responses:
  - Scale deployments based on predictions
  - Restart unhealthy pods before they fail
  - Adjust resource requests/limits dynamically
  - Generate incident tickets for human review on critical anomalies

### 5. Feedback Loop

- Track the accuracy of predictions vs actual events
- Measure cost savings and performance improvements
- Continuously retrain models with new data


## STEP 1: Observability Layer

### 1.1 Deploy Simple Multi-Tier App

**Quick Demo App** - Use this simple microservices setup:

- **Frontend:** Nginx serving static HTML with a button to hit the API
- **Backend API:** Flask app with endpoints like /api/health, /api/data, /api/heavy (simulates load)
- **Database:** Redis for simple key-value storage

**Kubernetes Setup:**

```bash
# Create namespace
kubectl create namespace aiops-demo

# You'll need these manifests:
# - frontend-deployment.yaml (nginx)
# - backend-deployment.yaml (flask)
# - redis-deployment.yaml
# - services for each
# - ingress (optional)
```

**Resources:**

- Use a local cluster (minikube, kind, or k3d) or a cloud cluster (EKS, GKE, AKS)
- 3 deployments with 2 replicas each initially
- I can provide the complete YAML manifests if needed

### 1.2 Setup Prometheus + Grafana

Install using Helm (easiest way):

```bash
# Add helm repos
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace

# This includes Prometheus, Grafana, and Alertmanager
```

**What you get:**

- Prometheus automatically scrapes Kubernetes metrics
- Grafana pre-configured with dashboards
- Access Grafana: `kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80`

### 1.3 Setup Logging (Loki - simpler than ELK)

```bash
# Install Loki stack
helm install loki grafana/loki-stack \
  --namespace monitoring \
  --set promtail.enabled=true \
  --set grafana.enabled=false
```

**Instrument your Flask app:**

```python
# Add to your Flask app
import logging
from pythonjsonlogger import jsonlogger

logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)
```


## STEP 2: Data Pipeline

### 2.1 Setup Python Ingestion Service

**Architecture:**

- Python service runs as a Kubernetes CronJob or Deployment
- Queries Prometheus API every 30-60 seconds
- Queries Loki API for logs
- Stores in time-series format

**Project Structure:**

```
data-pipeline/
├── Dockerfile
├── requirements.txt
├── src/
│   ├── collectors/
│   │   ├── prometheus_collector.py
│   │   └── loki_collector.py
│   ├── storage/
│   │   └── timeseries_writer.py
│   ├── features/
│   │   └── feature_engineering.py
│   └── main.py
└── config/
    └── config.yaml
```

**prometheus_collector.py:**

```python
import requests
from datetime import datetime, timedelta

class PrometheusCollector:
    def __init__(self, prometheus_url):
        self.url = prometheus_url
    
    def collect_metrics(self):
        """Collect key metrics from Prometheus"""
        metrics = {
            'cpu_usage': 'sum(rate(container_cpu_usage_seconds_total[5m])) by (pod)',
            'memory_usage': 'sum(container_memory_usage_bytes) by (pod)',
            'request_rate': 'sum(rate(http_requests_total[5m])) by (pod)',
            'error_rate': 'sum(rate(http_requests_total{status=~"5.."}[5m])) by (pod)',
            'pod_restarts': 'kube_pod_container_status_restarts_total'
        }
        
        results = {}
        for metric_name, query in metrics.items():
            response = requests.get(
                f"{self.url}/api/v1/query",
                params={'query': query}
            )
            results[metric_name] = response.json()['data']['result']
        
        return results
```

**timeseries_writer.py:**

```python
import pandas as pd
from influxdb_client import InfluxDBClient, Point
from datetime import datetime

class TimeSeriesWriter:
    def __init__(self, influx_url, token, org, bucket):
        self.client = InfluxDBClient(url=influx_url, token=token, org=org)
        self.write_api = self.client.write_api()
        self.bucket = bucket
        self.org = org
    
    def write_metrics(self, metrics_data):
        """Write metrics to InfluxDB"""
        for metric_name, data in metrics_data.items():
            for result in data:
                point = Point(metric_name) \
                    .tag("pod", result['metric'].get('pod', 'unknown')) \
                    .field("value", float(result['value'][1])) \
                    .time(datetime.utcnow())
                
                self.write_api.write(bucket=self.bucket, org=self.org, record=point)
```

**main.py (the main loop):**

```python
import time
import yaml
from collectors.prometheus_collector import PrometheusCollector
from storage.timeseries_writer import TimeSeriesWriter
from features.feature_engineering import FeatureEngineer

def main():
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    prom_collector = PrometheusCollector(config['prometheus_url'])
    writer = TimeSeriesWriter(
        config['influx_url'],
        config['influx_token'],
        config['influx_org'],
        config['influx_bucket']
    )
    feature_engineer = FeatureEngineer()
    
    # Main collection loop
    while True:
        try:
            # Collect raw metrics
            metrics = prom_collector.collect_metrics()
            
            # Write raw metrics
            writer.write_metrics(metrics)
            
            # Calculate features
            features = feature_engineer.calculate_features(metrics)
            writer.write_metrics(features)
            
            print(f"[{datetime.now()}] Collected and stored metrics")
            
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(60)  # Collect every minute

if __name__ == "__main__":
    main()
```

### 2.2 Deploy InfluxDB for Storage

```bash
# Install InfluxDB
helm install influxdb influxdata/influxdb2 \
  --namespace monitoring \
  --set adminUser.user=admin \
  --set adminUser.password=adminpassword
```

### 2.3 Feature Engineering

**feature_engineering.py:**

```python
import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, window_sizes=[5, 15, 30]):
        self.window_sizes = window_sizes
    
    def calculate_features(self, metrics):
        """Calculate rolling averages, rates of change, etc."""
        features = {}
        
        for metric_name, data in metrics.items():
            df = self._to_dataframe(data)
            
            # Rolling averages
            for window in self.window_sizes:
                features[f"{metric_name}_rolling_{window}min"] = \
                    df['value'].rolling(window=window).mean()
            
            # Rate of change
            features[f"{metric_name}_rate_of_change"] = \
                df['value'].pct_change()
            
            # Standard deviation (volatility)
            features[f"{metric_name}_std"] = \
                df['value'].rolling(window=15).std()
        
        return features
    
    def _to_dataframe(self, prometheus_result):
        """Convert Prometheus result to pandas DataFrame"""
        # Implementation details...
        pass
```


## STEP 3: ML Intelligence Layer

### 3.1 Setup ML Training Environment

**Project Structure:**

```
ml-models/
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── train_predictor.ipynb
│   └── train_anomaly_detector.ipynb
├── models/
│   ├── predictor.py
│   ├── anomaly_detector.py
│   └── optimizer.py
├── training/
│   ├── train_scaling_predictor.py
│   └── train_anomaly_model.py
├── inference/
│   └── inference_service.py
└── requirements.txt
```

### 3.2 Predictive Scaling Model

**train_scaling_predictor.py:**

```python
import pandas as pd
from influxdb_client import InfluxDBClient
from prophet import Prophet
import joblib

class TrafficPredictor:
    def __init__(self):
        self.model = Prophet(
            seasonality_mode='multiplicative',
            daily_seasonality=True,
            weekly_seasonality=True
        )
    
    def prepare_data(self, influx_client):
        """Query historical data from InfluxDB"""
        query = '''
        from(bucket: "aiops")
          |> range(start: -30d)
          |> filter(fn: (r) => r["_measurement"] == "request_rate")
          |> aggregateWindow(every: 1m, fn: mean)
        '''
        
        result = influx_client.query_api().query_data_frame(query)
        
        # Format for Prophet (needs 'ds' and 'y' columns)
        df = pd.DataFrame({
            'ds': result['_time'],
            'y': result['_value']
        })
        
        return df
    
    def train(self, data):
        """Train the forecasting model"""
        self.model.fit(data)
        return self
    
    def predict(self, periods=15):
        """Predict next 15 minutes of traffic"""
        future = self.model.make_future_dataframe(periods=periods, freq='1min')
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def save_model(self, path):
        joblib.dump(self.model, path)
    
    @staticmethod
    def load_model(path):
        model = TrafficPredictor()
        model.model = joblib.load(path)
        return model

# Training script
if __name__ == "__main__":
    client = InfluxDBClient(url="http://influxdb:8086", token="your-token", org="aiops")
    
    predictor = TrafficPredictor()
    data = predictor.prepare_data(client)
    predictor.train(data)
    predictor.save_model('models/traffic_predictor.pkl')
    
    print("Model trained and saved!")
```

### 3.3 Anomaly Detection

**train_anomaly_model.py:**

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

class AnomalyDetector:
    def __init__(self, contamination=0.1):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
    
    def prepare_features(self, influx_client):
        """Get multiple metrics as features"""
        # Query multiple metrics
        metrics = ['cpu_usage', 'memory_usage', 'error_rate', 'request_latency']
        
        # Combine into feature matrix
        features = []
        # ... query and combine metrics
        
        return np.array(features)
    
    def train(self, X):
        """Train anomaly detection model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        return self
    
    def predict(self, X):
        """Predict anomalies (-1 = anomaly, 1 = normal)"""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        scores = self.model.score_samples(X_scaled)
        return predictions, scores
    
    def save_model(self, path):
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
```

### 3.4 Inference Service (Real-time Predictions)

**inference_service.py:**

```python
from flask import Flask, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load models at startup
traffic_predictor = joblib.load('models/traffic_predictor.pkl')
anomaly_detector = joblib.load('models/anomaly_detector.pkl')

@app.route('/predict/traffic', methods=['GET'])
def predict_traffic():
    """Predict next 15 minutes of traffic"""
    forecast = traffic_predictor.predict(periods=15)
    
    # Determine if we need to scale
    current_replicas = get_current_replicas()  # Query K8s
    predicted_load = forecast['yhat'].iloc[-1]
    
    should_scale = predicted_load > threshold
    
    return jsonify({
        'forecast': forecast.to_dict(),
        'should_scale': should_scale,
        'recommended_replicas': calculate_replicas(predicted_load)
    })

@app.route('/detect/anomaly', methods=['POST'])
def detect_anomaly():
    """Check current metrics for anomalies"""
    current_metrics = get_current_metrics()  # Query Prometheus
    
    prediction, score = anomaly_detector.predict([current_metrics])
    
    return jsonify({
        'is_anomaly': prediction[0] == -1,
        'anomaly_score': float(score[0]),
        'metrics': current_metrics
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## STEP 4: Action Layer

### 4.1 Kubernetes Controller Service

**controller_service.py:**

```python
from kubernetes import client, config
import requests
import time

class K8sController:
    def __init__(self, namespace='aiops-demo'):
        config.load_incluster_config()  # When running in cluster
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
        self.namespace = namespace
        self.inference_url = "http://ml-inference-service:5000"
    
    def scale_deployment(self, deployment_name, replicas):
        """Scale a deployment to specified replicas"""
        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            # Update replicas
            deployment.spec.replicas = replicas
            
            # Apply update
            self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
                body=deployment
            )
            
            print(f"Scaled {deployment_name} to {replicas} replicas")
            return True
            
        except Exception as e:
            print(f"Error scaling deployment: {e}")
            return False
    
    def restart_unhealthy_pod(self, pod_name):
        """Delete unhealthy pod (K8s will recreate it)"""
        try:
            self.core_v1.delete_namespaced_pod(
                name=pod_name,
                namespace=self.namespace
            )
            print(f"Restarted pod: {pod_name}")
            return True
        except Exception as e:
            print(f"Error restarting pod: {e}")
            return False
    
    def adjust_resources(self, deployment_name, cpu_request, memory_request):
        """Adjust resource requests for a deployment"""
        deployment = self.apps_v1.read_namespaced_deployment(
            name=deployment_name,
            namespace=self.namespace
        )
        
        # Update container resources
        for container in deployment.spec.template.spec.containers:
            container.resources.requests = {
                'cpu': cpu_request,
                'memory': memory_request
            }
        
        self.apps_v1.patch_namespaced_deployment(
            name=deployment_name,
            namespace=self.namespace,
            body=deployment
        )
    
    def control_loop(self):
        """Main control loop - runs continuously"""
        while True:
            try:
                # Get predictions from ML service
                response = requests.get(f"{self.inference_url}/predict/traffic")
                prediction = response.json()
                
                if prediction['should_scale']:
                    self.scale_deployment(
                        'backend-api',
                        prediction['recommended_replicas']
                    )
                
                # Check for anomalies
                anomaly_response = requests.post(f"{self.inference_url}/detect/anomaly")
                anomaly = anomaly_response.json()
                
                if anomaly['is_anomaly']:
                    print(f"⚠️  Anomaly detected! Score: {anomaly['anomaly_score']}")
                    self.handle_anomaly(anomaly)
                
            except Exception as e:
                print(f"Control loop error: {e}")
            
            time.sleep(60)  # Run every minute
    
    def handle_anomaly(self, anomaly_data):
        """Handle detected anomalies"""
        # Create incident ticket (integrate with PagerDuty, Slack, etc.)
        # Potentially take automated action based on anomaly type
        pass

if __name__ == "__main__":
    controller = K8sController()
    controller.control_loop()
```

### 4.2 Deploy Controller as Kubernetes Deployment

**controller-deployment.yaml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aiops-controller
  namespace: aiops-demo
spec:
  replicas: 1
  selector:
    matchLabels:
      app: aiops-controller
  template:
    metadata:
      labels:
        app: aiops-controller
    spec:
      serviceAccountName: aiops-controller
      containers:
      - name: controller
        image: your-registry/aiops-controller:latest
        env:
        - name: NAMESPACE
          value: "aiops-demo"
        - name: INFERENCE_SERVICE_URL
          value: "http://ml-inference-service:5000"
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: aiops-controller
  namespace: aiops-demo
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: aiops-controller
  namespace: aiops-demo
rules:
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "patch", "update"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: aiops-controller
  namespace: aiops-demo
subjects:
- kind: ServiceAccount
  name: aiops-controller
roleRef:
  kind: Role
  name: aiops-controller
  apiGroup: rbac.authorization.k8s.io
```


## STEP 5: Feedback Loop

### 5.1 Metrics Tracking

**metrics_tracker.py:**

```python
from prometheus_client import Counter, Histogram, Gauge
import time

class MetricsTracker:
    def __init__(self):
        # Prediction accuracy metrics
        self.prediction_accuracy = Gauge(
            'aiops_prediction_accuracy',
            'Accuracy of traffic predictions'
        )
        
        self.scaling_decisions = Counter(
            'aiops_scaling_decisions_total',
            'Total scaling decisions made',
            ['action']  # scale_up, scale_down, no_action
        )
        
        self.anomaly_detections = Counter(
            'aiops_anomaly_detections_total',
            'Total anomalies detected',
            ['severity']
        )
        
        self.cost_savings = Gauge(
            'aiops_estimated_cost_savings_dollars',
            'Estimated cost savings in dollars'
        )
        
        self.model_inference_time = Histogram(
            'aiops_model_inference_seconds',
            'Time to run model inference',
            ['model_type']
        )
    
    def track_prediction(self, predicted_value, actual_value):
        """Track prediction vs actual"""
        error = abs(predicted_value - actual_value) / actual_value
        accuracy = 1 - error
        self.prediction_accuracy.set(accuracy)
    
    def track_scaling_decision(self, action):
        """Track scaling decisions"""
        self.scaling_decisions.labels(action=action).inc()
    
    def calculate_cost_savings(self):
        """Calculate cost savings from intelligent scaling"""
        # Compare actual resource usage vs what would have been used
        # with traditional autoscaling
        pass
```

### 5.2 Model Retraining Pipeline

**retrain_pipeline.py:**

```python
import schedule
import time
from training.train_scaling_predictor import TrafficPredictor
from training.train_anomaly_model import AnomalyDetector

class RetrainingPipeline:
    def __init__(self):
        self.predictor = TrafficPredictor()
        self.anomaly_detector = AnomalyDetector()
    
    def retrain_traffic_model(self):
        """Retrain traffic prediction model"""
        print("Starting traffic model retraining...")
        
        # Get fresh data from last 30 days
        data = self.predictor.prepare_data(influx_client)
        
        # Train new model
        self.predictor.train(data)
        
        # Evaluate on holdout set
        accuracy = self.evaluate_model(self.predictor)
        
        # Save if improved
        if accuracy > self.get_current_model_accuracy():
            self.predictor.save_model('models/traffic_predictor.pkl')
            print(f"✅ Model updated! New accuracy: {accuracy}")
        else:
            print(f"⏭️  Model not improved. Keeping current model.")
    
    def retrain_anomaly_model(self):
        """Retrain anomaly detection model"""
        print("Starting anomaly model retraining...")
        
        # Get recent normal behavior data
        features = self.anomaly_detector.prepare_features(influx_client)
        
        # Retrain
        self.anomaly_detector.train(features)
        self.anomaly_detector.save_model('models/anomaly_detector.pkl')
        
        print("✅ Anomaly model updated!")
    
    def schedule_retraining(self):
        """Schedule periodic retraining"""
        # Retrain every week
        schedule.every().sunday.at("02:00").do(self.retrain_traffic_model)
        schedule.every().sunday.at("03:00").do(self.retrain_anomaly_model)
        
        while True:
            schedule.run_pending()
            time.sleep(60)

if __name__ == "__main__":
    pipeline = RetrainingPipeline()
    pipeline.schedule_retraining()
```

### 5.3 Dashboard for Monitoring

Create a Grafana dashboard with panels for:

- Prediction accuracy over time
- Scaling decisions made (count by type)
- Anomalies detected
- Cost savings estimate
- Model inference time

## Quick Start Sequence

- **Week 1:** Set up K8s cluster, deploy demo app, install monitoring stack
- **Week 2:** Build and deploy data pipeline, verify data collection
- **Week 3:** Train initial ML models with synthetic/generated traffic
- **Week 4:** Build inference service and controller
- **Week 5:** Deploy controller, test automated actions
- **Week 6:** Add feedback loop and retraining