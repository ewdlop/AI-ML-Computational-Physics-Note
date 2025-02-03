from brian2 import *

# Simulation Parameters
duration = 100*ms  # Simulation time

# Neuron Parameters
tau = 10*ms  # Membrane time constant
V_rest = -65*mV  # Resting potential
V_reset = -70*mV  # Reset potential
V_threshold = -50*mV  # Spiking threshold
R = 10*Mohm  # Membrane resistance
I_ext = 1.5*nA  # External input current

# Define the LIF Neuron Model
eqs = '''
dV/dt = (V_rest - V + R*I_ext) / tau : volt
'''

# Create a Neuron Group
neurons = NeuronGroup(1, eqs, threshold='V>V_threshold', reset='V = V_reset', method='exact')

# Record Data
M = StateMonitor(neurons, 'V', record=True)  # Monitor membrane potential
spikemon = SpikeMonitor(neurons)  # Monitor spikes

# Run Simulation
run(duration)

# Plot Results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(M.t/ms, M.V[0]/mV, label="Membrane Potential")
plt.axhline(V_threshold/mV, ls="--", color="red", label="Threshold")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.title("LIF Neuron Spiking Behavior")
plt.show()

# Print Spike Times
print("Spike Times (ms):", spikemon.t/ms)
