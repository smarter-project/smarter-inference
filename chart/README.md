# smarter-fluent-bit

## TL;DR

Fluent-bit needs to communicate with fluentd so obtain your fluentd password by running the following command against your **cloud** k3s instance:
```bash
kubectl get secrets/fluentd-credentials --template={{.data.password}} | base64 -d
```
Register your cloud fluentd credentials in your **edge** cluster by running:
```bash
kubectl create secret generic {{ .Values.fluentDConfiguration.credentials }}  --from-literal=password=<YOUR FLUENTD PASSWORD> --namespace={{ .Values.application.namespace }}
```
and then install this helm chart

```console
helm install smarter-fluent-bit charts/fluent-bit
```

