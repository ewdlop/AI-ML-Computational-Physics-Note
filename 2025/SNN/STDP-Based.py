from brian2 import *

# Simulation Parameters
duration = 200*ms  # Total simulation time

# Neuron Model Parameters
tau = 10*ms  # Membrane time constant
V_rest = -65*mV  # Resting potential
V_reset = -70*mV  # Reset potential after a spike
V_threshold = -50*mV  # Spiking threshold
R = 10*Mohm  # Membrane resistance
I_ext = 1.5*nA  # External input current

# Define the LIF Neuron Model
eqs = '''
dV/dt = (V_rest - V + R*I_ext) / tau : volt
'''

# Create Neuron Groups (Pre- and Post-Synaptic)
pre_neurons = NeuronGroup(10, eqs, threshold='V>V_threshold', reset='V = V_reset', method='exact')
post_neurons = NeuronGroup(10, eqs, threshold='V>V_threshold', reset='V = V_reset', method='exact')

# Synaptic Connections with STDP
synapses = Synapses(pre_neurons, post_neurons,
                    model='''
                    w : 1  # Synaptic weight
                    dApre/dt = -Apre / (20*ms) : 1 (event-driven)
                    dApost/dt = -Apost / (20*ms) : 1 (event-driven)
                    ''',
                    on_pre='''
                    V_post += w*mV
                    Apre += 0.01
                    w = clip(w + Apost, 0, 1)  # Weight increase
                    ''',
                    on_post='''
                    Apost += 0.01
                    w = clip(w + Apre, 0, 1)  # Weight decrease
                    ''')
synapses.connect(p=0.3)  # 30% connection probability
synapses.w = 0.5  # Initial synaptic weight

# Record Data
M = StateMonitor(pre_neurons, 'V', record=True)  # Monitor membrane potential
spikemon_pre = SpikeMonitor(pre_neurons)  # Pre-synaptic spikes
spikemon_post = SpikeMonitor(post_neurons)  # Post-synaptic spikes
weight_monitor = StateMonitor(synapses, 'w', record=True)  # Monitor weight changes

# Run Simulation
run(duration)

# Plot Spiking Activity
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(spikemon_pre.t/ms, spikemon_pre.i, 'bo', markersize=4, label="Pre-Synaptic Spikes")
plt.plot(spikemon_post.t/ms, spikemon_post.i, 'ro', markersize=4, label="Post-Synaptic Spikes")
plt.xlabel("Time (ms)")
plt.ylabel("Neuron Index")
plt.title("STDP Spiking Activity")
plt.legend()

# Plot Weight Changes
plt.subplot(2, 1, 2)
plt.plot(weight_monitor.t/ms, weight_monitor.w.T)
plt.xlabel("Time (ms)")
plt.ylabel("Synaptic Weight")
plt.title("STDP Weight Changes Over Time")

plt.tight_layout()
plt.show()

# Print Synaptic Weights
print("Final Synaptic Weights:", weight_monitor.w[:,-1])
