import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="RNN & LSTM Hidden State Visualizer", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body, .main, .block-container, .sidebar .sidebar-content {
    color: #ffffff;
    background-color: #0b1120;
}
h1, h2, h3, .sidebar .sidebar-content h1, .sidebar .sidebar-content h2, .sidebar .sidebar-content h3 {
    color: #ffffff;
}
.block-container {
    padding-top: 2rem;
}
.stMarkdown, .stText, label, .stRadio, .stSelectbox, .stSlider, .stDataFrame, .stDownloadButton button {
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("RNN and LSTM Hidden State Visualization")
st.markdown(
    "This interface visualizes the internal hidden states of Recurrent Neural Networks (RNN) "
    "and Long Short-Term Memory (LSTM) networks to demonstrate how memory evolves over time."
)
st.divider()

# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.header("Model Configuration")

model_type = st.sidebar.selectbox("Select Neural Network", ["RNN", "LSTM"])
hidden_size = st.sidebar.slider("Number of Hidden Units", 4, 64, 16)
sequence_input = st.sidebar.text_input("Input Sequence (comma-separated)", "1,2,3,4,5")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed for academic visualization of sequence models.")

# ---------------- INPUT PROCESSING ----------------
try:
    sequence = np.array(sequence_input.split(","), dtype=np.float32)
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).view(1, -1, 1)

    # ---------------- MODEL INITIALIZATION ----------------
    if model_type == "RNN":
        model = torch.nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        outputs, h_n = model(sequence_tensor)
        memory = outputs.squeeze(0).detach().numpy()
        final_memory = h_n.detach().numpy()

    else:
        model = torch.nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        outputs, (h_n, c_n) = model(sequence_tensor)
        memory = outputs.squeeze(0).detach().numpy()
        final_memory = h_n.detach().numpy()
        cell_memory = c_n.detach().numpy()

    # ---------------- LAYOUT STRUCTURE ----------------
    col1, col2 = st.columns([1.2, 1])

    # ---------------- HEATMAP ----------------
    with col1:
        st.subheader("Hidden State Heatmap Across Time Steps")
        fig, ax = plt.subplots(figsize=(6, 4))
        heatmap = ax.imshow(memory, aspect="auto")
        ax.set_xlabel("Hidden Neurons")
        ax.set_ylabel("Time Steps")
        ax.set_title(f"{model_type} Hidden State Evolution")
        plt.colorbar(heatmap)
        st.pyplot(fig)

    # ---------------- STEP VIEW ----------------
    with col2:
        st.subheader("Hidden State at a Selected Time Step")
        timestep = st.slider(
            "Select Time Step",
            1,
            memory.shape[0],
            1
        )
        st.write(f"Hidden state vector at time step {timestep}:")
        st.dataframe(
            pd.DataFrame(memory[timestep - 1]).T,
            use_container_width=True
        )

    st.divider()

    # ---------------- FINAL STATES ----------------
    st.subheader("Final Memory Representation")
    col3, col4 = st.columns(2)

    with col3:
        st.write("Final Hidden State")
        st.dataframe(
            pd.DataFrame(final_memory.squeeze(1)),
            use_container_width=True
        )

    if model_type == "LSTM":
        with col4:
            st.write("Final Cell State (Long-Term Memory)")
            st.dataframe(
                pd.DataFrame(cell_memory.squeeze(1)),
                use_container_width=True
            )

    st.divider()

    # ---------------- FULL MEMORY TABLE ----------------
    st.subheader("Complete Hidden State Table Across All Time Steps")
    df = pd.DataFrame(memory)
    st.dataframe(df, use_container_width=True)

    # ---------------- DOWNLOAD OPTION ----------------
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Hidden States as CSV",
        data=csv_data,
        file_name="hidden_states.csv",
        mime="text/csv"
    )

    st.divider()

    # ---------------- EXPLANATION SECTION ----------------
    with st.expander("Conceptual Explanation of Memory Handling"):
        if model_type == "RNN":
            st.markdown("""
            **Recurrent Neural Network (RNN):**

            - Maintains only a single hidden state.
            - Memory is overwritten at each time step.
            - Struggles with long-term dependencies.
            - Suffering from vanishing gradient problem.
            """)
        else:
            st.markdown("""
            **Long Short-Term Memory (LSTM):**

            - Maintains both hidden state and cell state.
            - Uses input, forget, and output gates.
            - Effectively preserves long-term memory.
            - Performs better for long sequential data.
            """)

except Exception as e:
    st.error("Invalid input. Please enter only numeric values separated by commas.")
    st.error(f"System Error Details: {e}")
