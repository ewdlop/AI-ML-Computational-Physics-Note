from qiskit.ignis.mitigation.measurement import CompleteMeasFitter, complete_meas_cal

# Create calibration circuits for measurement error
cal_circuits, cal_results = complete_meas_cal(qc=quantum_error_correction(), qr=QuantumCircuit(4, 1))

# Fit mitigation filter
meas_fitter = CompleteMeasFitter(cal_results, cal_circuits)
mitigation_filter = meas_fitter.filter

# Apply mitigation
mitigated_result = mitigation_filter.apply(result)
mitigated_counts = mitigated_result.get_counts()

# Plot corrected results
plot_histogram(mitigated_counts, title="Mitigated Readout Errors")
plt.show()