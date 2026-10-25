# MNIST Digit Classifier: deploy on Render

A Streamlit web app. It runs the trained network with plain NumPy, so it does not need TensorFlow. The deploy stays small and fits Render's free tier.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web UI |
| `predictor.py` | Image preprocessing + NumPy forward pass |
| `weights.npz` | **You create this** from your trained notebook (step 1) |
| `requirements.txt` | streamlit, numpy, pillow |
| `render.yaml` | Render configuration |

## Step 1: Export your trained weights

Add this as a new cell at the end of your notebook (after `model.fit`) and run it:

```python
W = model.get_weights()
np.savez("weights.npz", W1=W[0], b1=W[1], W2=W[2], b2=W[3], W3=W[4], b3=W[5])
```

Copy the resulting `weights.npz` into this folder (next to `app.py`).

This works for your model: Flatten, Dense(50), Dense(50), Dense(10). If you change the layer sizes, update `predict()` in `predictor.py`.

## Step 2: Test locally (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Step 3: Push to GitHub

```bash
git init
git add .
git commit -m "MNIST classifier"
git branch -M main
git remote add origin https://github.com/<your-username>/mnist-render.git
git push -u origin main
```

Make sure `weights.npz` is committed.

## Step 4: Deploy on Render

1. Go to https://dashboard.render.com and choose **New +, then Web Service**.
2. Connect your GitHub repo.
3. Fill in:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - **Instance type:** Free
4. Click **Create Web Service**. The first build takes a few minutes.

Alternatively, choose **New +, then Blueprint**, and Render will read `render.yaml` for you.

## Notes

- The free tier sleeps after about 15 minutes of inactivity, so the first visit afterwards takes about 30-60 seconds.
- If you retrain, re-export `weights.npz` and push again.
