# RNN-LSTM Visualizer

An interactive Streamlit app for visualizing how hidden states evolve inside Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) models.

## Features
- Toggle between RNN and LSTM implementations
- Heatmap showing hidden activations across time
- Inspect any time step's hidden vector
- View final hidden/cell states
- Export the entire hidden-state table to CSV

## Tech Stack
- Python 3.10+
- PyTorch
- Streamlit
- NumPy / Pandas / Matplotlib

## Local Setup

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows
source .venv/bin/activate # macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

## Docker Deployment

Build and run the container locally (or on any container platform):

```bash
docker build -t rnn-lstm-visualizer .
docker run -p 8501:8501 rnn-lstm-visualizer
```

Visit [http://localhost:8501](http://localhost:8501) to use the app.

### Hosting on Render/Fly/Other Platforms

1. Push this repo (with the `Dockerfile`) to GitHub.
2. Create a new **Web Service** that builds from the Dockerfile.
3. Keep the default CMD so the container executes `streamlit run app.py`.
4. Expose port `8501` and enable HTTP/HTTPS routing.
5. Redeploy by pushing new commits; the platform rebuilds the image automatically.
