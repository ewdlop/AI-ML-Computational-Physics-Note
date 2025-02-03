from brian2 import *

# Simulation Parameters
duration = 100*ms  # Total simulation time

# Neuron Parameters
tau = 10*ms  # Membrane time constant
V_rest = -65*mV  # Resting potential
V_reset = -70*mV  # Reset potential after a spike
V_threshold = -50*mV  # Spiking threshold
R = 10*Mohm  # Membrane resistance
I_ext = 1.5*nA  # External input current

# Define the Leaky Integrate-and-Fire(LIF) Neuron Model
eqs = '''
dV/dt = (V_rest - V + R*I_ext) / tau : volt
'''

# Create a Population of 10 Neurons
num_neurons = 10
neurons = NeuronGroup(num_neurons, eqs, threshold='V>V_threshold', reset='V = V_reset', method='exact')

# Synaptic Connections
synapses = Synapses(neurons, neurons, on_pre='V += 2*mV')  # When pre-neuron spikes, add 2mV to post-neuron
synapses.connect(p=0.2)  # 20% connection probability

# Record Data
M = StateMonitor(neurons, 'V', record=True)  # Monitor membrane potentials
spikemon = SpikeMonitor(neurons)  # Monitor spikes

# Run Simulation
run(duration)

# Plot Results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
for i in range(num_neurons):
    plt.plot(M.t/ms, M.V[i]/mV, label=f'Neuron {i}')
plt.axhline(V_threshold/mV, ls="--", color="red", label="Threshold")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend(loc="upper right", fontsize=7)
plt.title("Multi-Neuron Spiking Activity")
plt.show()

# Plot Raster Plot
plt.figure(figsize=(10, 4))
plt.plot(spikemon.t/ms, spikemon.i, 'k.', markersize=5)
plt.xlabel("Time (ms)")
plt.ylabel("Neuron Index")
plt.title("Raster Plot of Spiking Activity")
plt.show()

# Print Spike Times
print("Spike Times (ms):", spikemon.t/ms)
