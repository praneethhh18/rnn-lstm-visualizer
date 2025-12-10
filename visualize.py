import matplotlib.pyplot as plt
import numpy as np

def plot_hidden_states(hidden_states, title):
    hidden_states = hidden_states.detach().numpy()
    
    plt.figure()
    plt.imshow(hidden_states, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Hidden Units")
    plt.ylabel("Time Steps")
    return plt
