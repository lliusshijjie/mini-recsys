# Kubernetes Deployment

This directory contains single-replica MVP manifests for Mini-RecSys.

Apply order:

```bash
kubectl apply -f configmap.yaml
kubectl apply -f pvc.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

The container expects model files mounted at `/models`:

- `/models/all-MiniLM-L6-v2.onnx`
- `/models/tokenizer.json`

`mini-recsys-data` stores Sled, HNSW, Tantivy, and behavior feedback state.
Multiple writer replicas are intentionally not supported for this MVP.
