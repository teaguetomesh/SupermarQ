from typing import Union

import cirq
import networkx as nx
import numpy as np
import qiskit


def cirq_to_qiskit(circuit: cirq.Circuit) -> qiskit.circuit.QuantumCircuit:
    qasm = cirq.circuits.QasmOutput(circuit, sorted(circuit.all_qubits()))
    return qiskit.circuit.QuantumCircuit().from_qasm_str(str(qasm))


def compute_connectivity(circuit: Union[cirq.Circuit, qiskit.circuit.QuantumCircuit]) -> float:
    if isinstance(circuit, cirq.Circuit):
        circ = cirq_to_qiskit(circuit)
    else:
        circ = circuit

    N = circ.num_qubits
    dag = qiskit.converters.circuit_to_dag(circ)
    dag.remove_all_ops_named("barrier")

    G = nx.Graph()
    for op in dag.two_qubit_ops():
        q1, q2 = op.qargs
        G.add_edge(q1.index, q2.index)

    degree_sum = sum([G.degree(n) for n in G.nodes])

    return degree_sum / (N * (N - 1))


def compute_liveness(circuit: Union[cirq.Circuit, qiskit.circuit.QuantumCircuit]) -> float:
    if isinstance(circuit, cirq.Circuit):
        circ = cirq_to_qiskit(circuit)
    else:
        circ = circuit

    N = circ.num_qubits
    dag = qiskit.converters.circuit_to_dag(circ)
    dag.remove_all_ops_named("barrier")

    activity_matrix = np.zeros((N, dag.depth()))

    for i, layer in enumerate(dag.layers()):
        for op in layer["partition"]:
            for qubit in op:
                activity_matrix[qubit.index, i] = 1

    return np.sum(activity_matrix) / (N * dag.depth())


def compute_parallelism(circuit: Union[cirq.Circuit, qiskit.circuit.QuantumCircuit]) -> float:
    if isinstance(circuit, cirq.Circuit):
        circ = cirq_to_qiskit(circuit)
    else:
        circ = circuit

    dag = qiskit.converters.circuit_to_dag(circ)
    dag.remove_all_ops_named("barrier")
    return max(1 - (circ.depth() / len(dag.gate_nodes())), 0)


def compute_measurement(circuit: Union[cirq.Circuit, qiskit.circuit.QuantumCircuit]) -> float:
    """
    measurement feature = # of layers of mid-circuit measurement / circuit depth

    Input
    -----
    circ : QuantumCircuit
    """
    if isinstance(circuit, cirq.Circuit):
        circ = cirq_to_qiskit(circuit)
    else:
        circ = circuit

    dag = qiskit.converters.circuit_to_dag(circ)
    dag.remove_all_ops_named("barrier")

    reset_moments = 0
    gate_depth = dag.depth() - 1

    for layer in dag.layers():
        reset_present = False
        for op in layer["graph"].op_nodes():
            if op.name == "reset":
                reset_present += True
        if reset_present:
            reset_moments += 1

    return reset_moments / gate_depth


def compute_entanglement(circuit: Union[cirq.Circuit, qiskit.circuit.QuantumCircuit]) -> float:
    if isinstance(circuit, cirq.Circuit):
        circ = cirq_to_qiskit(circuit)
    else:
        circ = circuit

    dag = qiskit.converters.circuit_to_dag(circ)
    dag.remove_all_ops_named("barrier")

    return len(dag.two_qubit_ops()) / len(dag.gate_nodes())


def compute_depth(circuit: Union[cirq.Circuit, qiskit.circuit.QuantumCircuit]) -> float:
    if isinstance(circuit, cirq.Circuit):
        circ = cirq_to_qiskit(circuit)
    else:
        circ = circuit

    dag = qiskit.converters.circuit_to_dag(circ)
    dag.remove_all_ops_named("barrier")
    n_ed = 0
    two_q_gates = set([op.name for op in dag.two_qubit_ops()])
    for name in two_q_gates:
        try:
            n_ed += dag.count_ops_longest_path()[name]
        except KeyError:
            continue
    n_e = len(dag.two_qubit_ops())

    if n_ed == 0:
        return 0

    return n_ed / n_e
