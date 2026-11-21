# Quantum_Temporal_CNN
# CNN - Convolutional Neural Network 



from __future__ import annotations

import csv
import hashlib
import itertools
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate, StatePreparation
from qiskit.quantum_info import Statevector


random.seed(39)
np.random.seed(39)



FILES = (
    Path("/data/loto7_4724_k93.csv"),
    Path("/data/loto7_4724_k93_loto_2984.csv"),
    Path("/data/loto7_4724_k93_loto_plus_1740.csv"),
)



N = 39
K = 7
SCALES = (1, 4, 16)

REGULARIZATIONS = (1.0, 0.1, 0.01)
TRAIN_FRACTION = 0.70
VALIDATION_END = 0.85

GRADIENT_TOLERANCE = 3e-6
BFGS_ITERATIONS = 650
CG_ITERATIONS = 850

MAX_CONSECUTIVE = 4
DIAGNOSTICS = False

COUNTS = np.array(
    [int(i).bit_count() for i in range(8)],
    dtype=np.int64,
)

# (vrsta, ciljni qubit, maska kontrolnih qubita)
BLOCK_GATES = (
    ("z", 0, 0),
    ("z", 1, 0),
    ("z", 2, 0),
    ("y", 1, 1),
    ("y", 2, 2),
    ("y", 0, 4),
    ("y", 2, 3),
    ("y", 0, 6),
    ("y", 1, 5),
)

BLOCK_BASE = np.array(
    [0.17, -0.23, 0.31, 0.10, -0.08, 0.12, 0.07, -0.06, 0.09]
)


# ------------------------------------------------------------
# KONSTRUKCIJA
#
# Dva sloja deljenih vremenskih kvantnih filtera.
#
# Prvi sloj:
#   dva ista naucena filtera primenjuju se na vremenske
#   trojke kroz celu istoriju.
#
# Drugi sloj:
#   spaja njihove odzive sa tri vremenska polozaja.
#
# Ista konstrukcija primenjuje se na skalama 1, 4 i 16.
# Najudaljeniji ulaz je 128 kola unazad.
#
# Nelinearni odzivi prvog sloja jesu ocekivane vrednosti
# kvantnih opservabli, izracunate iz lokalnih amplituda.
# Njihovo prosledjivanje sledecem sloju je klasicno.
#
# Izlaz:
#   lokalne amplitude brojeva ulaze u trokubITne blokove
#   sa kontrolisanim RY i dvostruko kontrolisanim RY kapijama.
#
# Grupe od tri broja uce se iz treninga.
# Nisu numericke zone i ne odredjuju koliko brojeva mora
# biti izabrano iz pojedine grupe.
#
# P(S | istorija) =
#   proizvod Bornovih verovatnoca blokova / Z,
#   uz |S| = 7.
#
# Z se racuna tacno polinomskim dinamickim programiranjem.
# NEXT se dobija maksimalnim proizvodom i, kada je potrebno,
# ogranicenom pretragom sa tacnom gornjom granicom.
#
# Sedmorka ima kanonsko kodiranje u sedam registara po
# sest qubita. Veliki izlazni statevector se ne alocira.
# Racun koristi ekvivalentnu reprezentaciju prisustva brojeva.
#
# Ovo je nasa hibridna amplitudna adaptacija vremenske
# konvolucije. Nije reprodukcija odredjenog objavljenog QCNN-a.
#
# Nema projekcije Statevector -> 7x39, klasifikatora,
# slucajnih kombinacija ili naknadnog menjanja NEXT rezultata.
# ------------------------------------------------------------


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_draws(path):
    if not path.is_file():
        raise FileNotFoundError(f"CSV nije pronadjen: {path}")

    rows = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)

        try:
            dialect = csv.Sniffer().sniff(
                sample, delimiters=",;\t "
            )
        except csv.Error as exc:
            raise ValueError(
                "Nije prepoznat razdelnik CSV-a."
            ) from exc

        for line_number, row in enumerate(
            csv.reader(handle, dialect), start=1
        ):
            fields = [field.strip() for field in row if field.strip()]

            if not fields:
                continue

            if len(fields) != K:
                raise ValueError(
                    f"Red {line_number}: ocekujem {K} brojeva."
                )

            try:
                values = [int(field) for field in fields]
            except ValueError as exc:
                raise ValueError(
                    f"Red {line_number}: neispravan ceo broj."
                ) from exc

            if (
                len(set(values)) != K
                or min(values) < 1
                or max(values) > N
            ):
                raise ValueError(
                    f"Red {line_number}: neispravna kombinacija."
                )

            rows.append(sorted(value - 1 for value in values))

    draws = np.asarray(rows, dtype=np.int64)

    if len(draws) < 100:
        raise ValueError("Potrebno je najmanje 100 redova.")

    return draws


def binary_draws(draws, n):
    result = np.zeros((len(draws), n), dtype=np.float64)
    result[np.arange(len(draws))[:, None], draws] = 1.0
    return result


def learn_groups(targets):
    """Deterministicka struktura blokova, samo iz treninga."""
    n = targets.shape[1]
    require(n % 3 == 0, "Broj kanala mora biti deljiv sa tri.")

    information = np.zeros((n, n))

    for a in range(n):
        for b in range(a):
            codes = (
                2 * targets[:, a].astype(np.int64)
                + targets[:, b].astype(np.int64)
            )

            table = np.bincount(
                codes, minlength=4
            ).reshape(2, 2).astype(np.float64)

            table /= len(targets)
            marginal_a = table.sum(axis=1)
            marginal_b = table.sum(axis=0)

            value = 0.0

            for x in range(2):
                for y in range(2):
                    p = table[x, y]

                    if p > 0:
                        value += p * math.log(
                            p / (marginal_a[x] * marginal_b[y])
                        )

            information[a, b] = information[b, a] = max(value, 0.0)

    remaining = list(range(n))
    groups = []

    while remaining:
        best_pair = None
        best_value = -math.inf

        for i, a in enumerate(remaining):
            for b in remaining[i + 1:]:
                value = information[a, b]

                if value > best_value:
                    best_pair = (a, b)
                    best_value = value

        a, b = best_pair
        candidates = [c for c in remaining if c != a and c != b]

        c = max(
            candidates,
            key=lambda j: (
                information[a, j] + information[b, j],
                -j,
            ),
        )

        group = tuple(sorted((a, b, c)))
        groups.append(group)
        remaining = [j for j in remaining if j not in group]

    return np.asarray(groups, dtype=np.int64)


def kernel_base(inputs, variant):
    parameters = np.empty(1 + 3 * inputs)
    parameters[0] = 1.1 + 0.2 * variant

    for j in range(inputs):
        parameters[1 + 3 * j] = 0.65 + 0.07 * ((j + variant) % 3)
        parameters[2 + 3 * j] = 0.17 + 0.06 * variant
        parameters[3 + 3 * j] = 0.28 * (
            -1.0 if (j + variant) % 2 else 1.0
        )

    return parameters


class Layout:
    def __init__(self, groups, n, k, scales=SCALES):
        self.groups = np.asarray(groups, dtype=np.int64)
        self.n = n
        self.k = k
        self.scales = tuple(scales)

        require(
            self.groups.shape == (n // 3, 3)
            and np.array_equal(
                np.sort(self.groups.ravel()), np.arange(n)
            ),
            "Blokovi ne predstavljaju particiju svih brojeva.",
        )

        self.first_base = (
            kernel_base(3, 0),
            kernel_base(3, 1),
        )
        self.second_base = kernel_base(6, 2)

        position = 0
        self.first = []

        for base in self.first_base:
            self.first.append(slice(position, position + len(base)))
            position += len(base)

        self.second = slice(position, position + len(self.second_base))
        position += len(self.second_base)

        self.combine = slice(position, position + 3 * len(scales))
        position += 3 * len(scales)

        self.bias = slice(position, position + n)
        position += n

        self.blocks = slice(position, position + 9 * len(self.groups))
        position += 9 * len(self.groups)

        self.size = position
        self.bias_base = 2.0 * math.asin(math.sqrt(k / n))

        self.combine_base = np.tile(
            np.array([0.35, 0.35, 0.0]),
            (len(scales), 1),
        )


class Dataset:
    def __init__(self, history, queries, layout, observed=True):
        self.queries = np.asarray(queries, dtype=np.int64)
        self.size = len(self.queries)

        require(
            self.size > 0
            and self.queries.min() >= 0
            and self.queries.max() < len(history),
            "Neispravni indeksi istorije.",
        )

        codes = np.zeros(
            (self.size, layout.n, len(layout.scales)),
            dtype=np.int64,
        )

        for scale_index, dilation in enumerate(layout.scales):
            for tap in range(9):
                source = self.queries - dilation * tap
                available = source >= 0

                digits = np.zeros(
                    (self.size, layout.n), dtype=np.int64
                )
                digits[available] = (
                    1 + history[source[available]].astype(np.int64)
                )

                codes[:, :, scale_index] += digits * (3 ** tap)

        unique, inverse = np.unique(codes, return_inverse=True)
        self.ids = np.ascontiguousarray(
            inverse.reshape(codes.shape), dtype=np.int64
        )

        # 0: istorija nije dostupna.
        # 1: broj nije izasao.
        # 2: broj je izasao.
        mapping = np.array([0.0, -1.0, 1.0])
        self.patterns = np.empty((len(unique), 9))

        for tap in range(9):
            self.patterns[:, tap] = mapping[
                (unique // (3 ** tap)) % 3
            ]

        self.observed = None

        if observed:
            require(
                self.queries.max() + 1 < len(history),
                "Nedostaje ciljni ishod.",
            )

            targets = history[self.queries + 1]
            self.observed = np.sum(
                targets[:, layout.groups]
                * np.array([1, 2, 4])[None, None, :],
                axis=2,
            ).astype(np.int64)


# ------------------------------------------------------------
# LOKALNE KAP IJE I NJIHOVI ANALITICKI GRADIJENTI
# ------------------------------------------------------------


def ry(state, angle):
    c = np.cos(0.5 * angle)
    s = np.sin(0.5 * angle)

    return np.stack(
        (
            c * state[..., 0] - s * state[..., 1],
            s * state[..., 0] + c * state[..., 1],
        ),
        axis=-1,
    )


def rz(state, angle):
    phase = np.exp(-0.5j * angle)

    return np.stack(
        (
            phase * state[..., 0],
            np.conjugate(phase) * state[..., 1],
        ),
        axis=-1,
    )


def ry_backward(before, adjoint, angle):
    c = np.cos(0.5 * angle)
    s = np.sin(0.5 * angle)

    d0 = -0.5 * s * before[..., 0] - 0.5 * c * before[..., 1]
    d1 = 0.5 * c * before[..., 0] - 0.5 * s * before[..., 1]

    angle_gradient = 2.0 * np.real(
        np.conjugate(adjoint[..., 0]) * d0
        + np.conjugate(adjoint[..., 1]) * d1
    )

    previous = np.stack(
        (
            c * adjoint[..., 0] + s * adjoint[..., 1],
            -s * adjoint[..., 0] + c * adjoint[..., 1],
        ),
        axis=-1,
    )

    return previous, angle_gradient


def rz_backward(before, adjoint, angle):
    phase = np.exp(-0.5j * angle)
    after0 = phase * before[..., 0]
    after1 = np.conjugate(phase) * before[..., 1]

    angle_gradient = np.real(
        -1j * np.conjugate(adjoint[..., 0]) * after0
        + 1j * np.conjugate(adjoint[..., 1]) * after1
    )

    previous = np.stack(
        (
            np.conjugate(phase) * adjoint[..., 0],
            phase * adjoint[..., 1],
        ),
        axis=-1,
    )

    return previous, angle_gradient


def kernel_forward(inputs, parameters):
    state = np.empty((len(inputs), 2), dtype=np.complex128)
    state[:, 0] = math.cos(0.5 * parameters[0])
    state[:, 1] = math.sin(0.5 * parameters[0])

    cache = []

    for j in range(inputs.shape[1]):
        a, phase, rotation = parameters[1 + 3 * j:4 + 3 * j]
        angle = a * inputs[:, j] + phase

        before = state
        after_phase = rz(before, angle)
        state = ry(after_phase, rotation)

        cache.append((before, after_phase, angle))

    return state, cache


def kernel_backward(inputs, parameters, cache, adjoint):
    derivative = np.zeros_like(parameters)
    input_gradient = np.zeros_like(inputs)

    for j in range(inputs.shape[1] - 1, -1, -1):
        a, _, rotation = parameters[1 + 3 * j:4 + 3 * j]
        before, after_phase, angle = cache[j]

        adjoint, g_rotation = ry_backward(
            after_phase, adjoint, rotation
        )
        adjoint, g_phase = rz_backward(
            before, adjoint, angle
        )

        derivative[1 + 3 * j] = np.sum(g_phase * inputs[:, j])
        derivative[2 + 3 * j] = g_phase.sum()
        derivative[3 + 3 * j] = g_rotation.sum()

        input_gradient[:, j] = a * g_phase

    c = math.cos(0.5 * parameters[0])
    s = math.sin(0.5 * parameters[0])

    derivative[0] = 2.0 * np.real(
        np.sum(
            np.conjugate(adjoint[:, 0]) * (-0.5 * s)
            + np.conjugate(adjoint[:, 1]) * (0.5 * c)
        )
    )

    return derivative, input_gradient


def cnn_forward(parameters, data, layout):
    first_inputs = data.patterns.reshape(-1, 3)

    first_states = []
    first_caches = []
    first_parameters = []

    for channel in range(2):
        actual = (
            layout.first_base[channel]
            + parameters[layout.first[channel]]
        )

        state, cache = kernel_forward(first_inputs, actual)
        first_states.append(state)
        first_caches.append(cache)
        first_parameters.append(actual)

    features = np.stack(
        [
            np.abs(state[:, 0]) ** 2 - np.abs(state[:, 1]) ** 2
            for state in first_states
        ],
        axis=1,
    )

    second_inputs = features.reshape(len(data.patterns), 6)
    second_parameters = layout.second_base + parameters[layout.second]

    second_state, second_cache = kernel_forward(
        second_inputs, second_parameters
    )

    lookup = np.stack(
        (
            2.0 * np.real(
                np.conjugate(second_state[:, 0]) * second_state[:, 1]
            ),
            np.abs(second_state[:, 0]) ** 2
            - np.abs(second_state[:, 1]) ** 2,
        ),
        axis=1,
    )

    cache = (
        first_inputs,
        first_states,
        first_caches,
        first_parameters,
        second_inputs,
        second_state,
        second_cache,
        second_parameters,
    )

    return lookup, cache


def cnn_backward(lookup_gradient, cache, layout):
    (
        first_inputs,
        first_states,
        first_caches,
        first_parameters,
        second_inputs,
        second_state,
        second_cache,
        second_parameters,
    ) = cache

    gx = lookup_gradient[:, 0]
    gz = lookup_gradient[:, 1]

    adjoint = np.stack(
        (
            gx * second_state[:, 1] + gz * second_state[:, 0],
            gx * second_state[:, 0] - gz * second_state[:, 1],
        ),
        axis=1,
    )

    second_gradient, feature_gradient = kernel_backward(
        second_inputs, second_parameters, second_cache, adjoint
    )

    first_gradient = []
    feature_gradient = feature_gradient.reshape(-1, 2)

    for channel in range(2):
        state = first_states[channel]
        g = feature_gradient[:, channel]

        adjoint = np.stack(
            (g * state[:, 0], -g * state[:, 1]),
            axis=1,
        )

        derivative, _ = kernel_backward(
            first_inputs,
            first_parameters[channel],
            first_caches[channel],
            adjoint,
        )
        first_gradient.append(derivative)

    return first_gradient, second_gradient


def combine_forward(parameters, lookup, data, layout):
    bias = layout.bias_base + parameters[layout.bias]

    state = np.empty(
        (data.size, layout.n, 2), dtype=np.complex128
    )
    state[..., 0] = np.cos(0.5 * bias)[None, :]
    state[..., 1] = np.sin(0.5 * bias)[None, :]

    combine = (
        layout.combine_base
        + parameters[layout.combine].reshape(-1, 3)
    )

    cache = []

    for scale_index in range(len(layout.scales)):
        features = lookup[data.ids[:, :, scale_index]]
        fx = features[..., 0]
        fz = features[..., 1]
        a, b, offset = combine[scale_index]

        before = state
        after_phase = rz(before, a * fx)
        state = ry(after_phase, b * fz + offset)

        cache.append((before, after_phase, fx, fz))

    return state, (bias, combine, cache)


def combine_backward(adjoint, data, layout, cache):
    bias, combine, states = cache

    gradient = np.zeros_like(combine)
    lookup_gradient = np.zeros((len(data.patterns), 2))

    for scale_index in range(len(layout.scales) - 1, -1, -1):
        before, after_phase, fx, fz = states[scale_index]
        a, b, offset = combine[scale_index]

        adjoint, gy = ry_backward(
            after_phase, adjoint, b * fz + offset
        )
        adjoint, gz = rz_backward(
            before, adjoint, a * fx
        )

        gradient[scale_index, 0] = np.sum(gz * fx)
        gradient[scale_index, 1] = np.sum(gy * fz)
        gradient[scale_index, 2] = gy.sum()

        indices = data.ids[:, :, scale_index].ravel()

        lookup_gradient[:, 0] += np.bincount(
            indices,
            weights=(a * gz).ravel(),
            minlength=len(data.patterns),
        )
        lookup_gradient[:, 1] += np.bincount(
            indices,
            weights=(b * gy).ravel(),
            minlength=len(data.patterns),
        )

    bias_gradient = 2.0 * np.real(
        np.sum(
            np.conjugate(adjoint[..., 0])
            * (-0.5 * np.sin(0.5 * bias))[None, :]
            + np.conjugate(adjoint[..., 1])
            * (0.5 * np.cos(0.5 * bias))[None, :],
            axis=0,
        )
    )

    return gradient.ravel(), bias_gradient, lookup_gradient


# ------------------------------------------------------------
# TROKUBITNI ZAJEDNICKI IZLAZ
# ------------------------------------------------------------


def gate_pairs(target, controls):
    lower = np.array(
        [
            state for state in range(8)
            if not (state & (1 << target))
            and (state & controls) == controls
        ],
        dtype=np.int64,
    )
    return lower, lower | (1 << target)


def block_step(state, angle, specification):
    kind, target, controls = specification

    if kind == "z":
        sign = np.array(
            [1.0 if (state_index >> target) & 1 else -1.0
             for state_index in range(8)]
        )
        phase = np.exp(0.5j * angle[:, None] * sign[None, :])
        return state * phase[None, :, :]

    lower, upper = gate_pairs(target, controls)
    c = np.cos(0.5 * angle)[None, :, None]
    s = np.sin(0.5 * angle)[None, :, None]

    result = state.copy()
    result[:, :, lower] = (
        c * state[:, :, lower] - s * state[:, :, upper]
    )
    result[:, :, upper] = (
        s * state[:, :, lower] + c * state[:, :, upper]
    )
    return result


def block_step_backward(before, adjoint, angle, specification):
    kind, target, controls = specification

    if kind == "z":
        sign = np.array(
            [1.0 if (state_index >> target) & 1 else -1.0
             for state_index in range(8)]
        )
        phase = np.exp(0.5j * angle[:, None] * sign[None, :])
        after = before * phase[None, :, :]

        derivative = 2.0 * np.real(
            np.sum(
                np.conjugate(adjoint)
                * (0.5j * sign[None, None, :] * after),
                axis=(0, 2),
            )
        )

        return adjoint * np.conjugate(phase)[None, :, :], derivative

    lower, upper = gate_pairs(target, controls)
    c = np.cos(0.5 * angle)[None, :, None]
    s = np.sin(0.5 * angle)[None, :, None]

    d_lower = (
        -0.5 * s * before[:, :, lower]
        - 0.5 * c * before[:, :, upper]
    )
    d_upper = (
        0.5 * c * before[:, :, lower]
        - 0.5 * s * before[:, :, upper]
    )

    derivative = 2.0 * np.real(
        np.sum(
            np.conjugate(adjoint[:, :, lower]) * d_lower
            + np.conjugate(adjoint[:, :, upper]) * d_upper,
            axis=(0, 2),
        )
    )

    previous = adjoint.copy()
    previous[:, :, lower] = (
        c * adjoint[:, :, lower] + s * adjoint[:, :, upper]
    )
    previous[:, :, upper] = (
        -s * adjoint[:, :, lower] + c * adjoint[:, :, upper]
    )

    return previous, derivative


def blocks_forward(parameters, local_states, layout):
    local = local_states[:, layout.groups, :]
    state = np.ones(
        (len(local_states), len(layout.groups), 8),
        dtype=np.complex128,
    )

    for mask in range(8):
        for bit in range(3):
            state[:, :, mask] *= local[:, :, bit, (mask >> bit) & 1]

    angles = (
        BLOCK_BASE[None, :]
        + parameters[layout.blocks].reshape(-1, 9)
    )

    cache = []

    for gate_index, specification in enumerate(BLOCK_GATES):
        cache.append(state)
        state = block_step(
            state, angles[:, gate_index], specification
        )

    return state, (local, angles, cache)


def blocks_backward(adjoint, layout, cache):
    local, angles, states = cache
    derivative = np.zeros_like(angles)

    for gate_index in range(len(BLOCK_GATES) - 1, -1, -1):
        adjoint, derivative[:, gate_index] = block_step_backward(
            states[gate_index],
            adjoint,
            angles[:, gate_index],
            BLOCK_GATES[gate_index],
        )

    local_gradient = np.zeros_like(local)

    for mask in range(8):
        for bit in range(3):
            others = np.ones(
                local.shape[:2], dtype=np.complex128
            )

            for other in range(3):
                if other != bit:
                    others *= local[:, :, other, (mask >> other) & 1]

            local_gradient[:, :, bit, (mask >> bit) & 1] += (
                adjoint[:, :, mask] * np.conjugate(others)
            )

    result = np.zeros(
        (len(local), layout.n, 2), dtype=np.complex128
    )
    result[:, layout.groups.ravel(), :] = local_gradient.reshape(
        len(local), layout.n, 2
    )

    return result, derivative.ravel()


def count_prefix(probabilities, k):
    rows, groups, _ = probabilities.shape
    coefficients = np.zeros((rows, groups, 4))

    for count in range(4):
        coefficients[:, :, count] = probabilities[
            :, :, COUNTS == count
        ].sum(axis=2)

    prefix = np.zeros((rows, groups + 1, k + 1))
    prefix[:, 0, 0] = 1.0

    for group in range(groups):
        for count in range(min(3, k) + 1):
            prefix[:, group + 1, count:] += (
                coefficients[:, group, count, None]
                * prefix[:, group, :k + 1 - count]
            )

    return coefficients, prefix


def distribution_loss(states, observed, k, gradient=True):
    probabilities = np.abs(states) ** 2

    require(
        np.max(np.abs(probabilities.sum(axis=2) - 1.0)) < 1e-9,
        "Trokubitni blok nije normalizovan.",
    )

    rows, groups, _ = probabilities.shape
    coefficients, prefix = count_prefix(probabilities, k)
    normalization = prefix[:, -1, k]

    selected = np.take_along_axis(
        probabilities, observed[:, :, None], axis=2
    )[:, :, 0]

    require(
        np.all(normalization > 0)
        and np.all(selected > 0)
        and np.all(np.isfinite(normalization)),
        "Nulta ili nekonacna verovatnoca ciljnog ishoda.",
    )

    losses = np.log(normalization) - np.log(selected).sum(axis=1)

    if not gradient:
        return losses, None

    adjoint = np.zeros((rows, k + 1))
    adjoint[:, k] = 1.0 / normalization
    probability_gradient = np.zeros_like(probabilities)

    for group in range(groups - 1, -1, -1):
        coefficient_gradient = np.zeros((rows, 4))

        for count in range(min(3, k) + 1):
            coefficient_gradient[:, count] = np.sum(
                adjoint[:, count:]
                * prefix[:, group, :k + 1 - count],
                axis=1,
            )

        probability_gradient[:, group, :] = (
            coefficient_gradient[:, COUNTS]
        )

        previous = np.zeros_like(adjoint)

        for count in range(min(3, k) + 1):
            previous[:, :k + 1 - count] += (
                coefficients[:, group, count, None]
                * adjoint[:, count:]
            )

        adjoint = previous

    probability_gradient[
        np.arange(rows)[:, None],
        np.arange(groups)[None, :],
        observed,
    ] -= 1.0 / selected

    state_gradient = probability_gradient * states / rows
    return losses, state_gradient


def model_forward(parameters, data, layout):
    lookup, cnn_cache = cnn_forward(parameters, data, layout)

    local_states, combine_cache = combine_forward(
        parameters, lookup, data, layout
    )
    block_states, block_cache = blocks_forward(
        parameters, local_states, layout
    )

    return block_states, (
        cnn_cache,
        combine_cache,
        block_cache,
        lookup,
        local_states,
    )


def objective_parts(parameters, data, layout, gradient=True):
    require(data.observed is not None, "Nedostaju ciljevi.")

    states, cache = model_forward(parameters, data, layout)

    losses, state_gradient = distribution_loss(
        states, data.observed, layout.k, gradient
    )

    require(np.all(np.isfinite(losses)), "Nekonacan NLL.")

    if not gradient:
        return losses, None

    cnn_cache, combine_cache, block_cache, _, _ = cache

    local_gradient, block_gradient = blocks_backward(
        state_gradient, layout, block_cache
    )

    combine_gradient, bias_gradient, lookup_gradient = combine_backward(
        local_gradient, data, layout, combine_cache
    )

    first_gradient, second_gradient = cnn_backward(
        lookup_gradient, cnn_cache, layout
    )

    derivative = np.zeros(layout.size)

    for channel in range(2):
        derivative[layout.first[channel]] = first_gradient[channel]

    derivative[layout.second] = second_gradient
    derivative[layout.combine] = combine_gradient
    derivative[layout.bias] = bias_gradient
    derivative[layout.blocks] = block_gradient

    require(
        np.all(np.isfinite(derivative)),
        "Nekonacan analiticki gradijent.",
    )

    return losses, derivative


@dataclass
class Model:
    layout: Layout
    parameters: np.ndarray
    regularization: float


def initial_parameters(layout, warm):
    result = np.zeros(layout.size)

    if warm is None:
        return result

    require(
        warm.layout.n == layout.n
        and warm.layout.scales == layout.scales,
        "Nesaglasan pocetni model.",
    )

    result[:layout.blocks.start] = (
        warm.parameters[:warm.layout.blocks.start]
    )

    old_blocks = warm.parameters[warm.layout.blocks].reshape(-1, 9)
    new_blocks = result[layout.blocks].reshape(-1, 9)

    old_index = {
        tuple(group): index
        for index, group in enumerate(warm.layout.groups)
    }

    for index, group in enumerate(layout.groups):
        key = tuple(group)

        if key in old_index:
            new_blocks[index] = old_blocks[old_index[key]]

    return result








def fit_model(data, layout, regularization, warm=None):
    parameters = initial_parameters(layout, warm)
    dimension = layout.size
    tolerance = GRADIENT_TOLERANCE

    radius = 0.10
    maximum_radius = 1.0
    minimum_radius = 1e-12
    maximum_attempts = 3000

    iterations = 0
    accepted_steps = 0
    consecutive_rejections = 0
    last_report = time.monotonic()

    # Aproksimacija Hessove matrice, ne njene inverzije.
    hessian = np.eye(dimension)

    if warm is not None:
        previous = getattr(warm, "_cnn_trust_hessian", None)

        if (
            previous is not None
            and previous.shape == (dimension, dimension)
            and np.array_equal(warm.layout.groups, layout.groups)
            and warm.layout.scales == layout.scales
            and np.all(np.isfinite(previous))
        ):
            hessian = previous.copy()
            hessian += (
                (regularization - warm.regularization)
                / dimension
            ) * np.eye(dimension)

    def repair_hessian(matrix):
        matrix = 0.5 * (matrix + matrix.T)

        require(
            np.all(np.isfinite(matrix)),
            "Nekonacna procena Hessove matrice.",
        )

        values, vectors = np.linalg.eigh(matrix)

        # Stabilizacija preuslovljavanja.
        # Ne menja model, NLL ili prag konvergencije.
        values = np.clip(values, 1e-7, 1e8)
        repaired = (vectors * values[None, :]) @ vectors.T

        return 0.5 * (repaired + repaired.T)

    hessian = repair_hessian(hessian)

    def evaluate(point):
        losses, gradient = objective_parts(
            point, data, layout, gradient=True
        )

        gradient = (
            gradient + regularization * point / dimension
        )

        require(
            np.all(np.isfinite(losses))
            and np.all(np.isfinite(gradient)),
            "Nekonacan cilj ili gradijent.",
        )

        return losses, gradient

    def objective_change(old_point, displacement, old_losses, new_losses):
        # Razlika se racuna pre sabiranja, bez oduzimanja
        # dva gotovo jednaka ukupna NLL-a.
        likelihood = np.mean(
            new_losses - old_losses,
            dtype=np.longdouble,
        )

        penalty = (
            np.longdouble(regularization)
            / np.longdouble(dimension)
            * (
                np.sum(
                    old_point * displacement,
                    dtype=np.longdouble,
                )
                + 0.5 * np.sum(
                    displacement * displacement,
                    dtype=np.longdouble,
                )
            )
        )

        return float(likelihood + penalty)

    def audit_gradient(point, analytic):
        # Kontrolisi najosetljiviji parametar i najosetljiviji
        # parametar spajanja vremenskih skala.
        indices = {int(np.argmax(np.abs(analytic)))}

        combine_indices = np.arange(
            layout.combine.start,
            layout.combine.stop,
        )
        indices.add(
            int(
                combine_indices[
                    np.argmax(np.abs(analytic[combine_indices]))
                ]
            )
        )

        for index in sorted(indices):
            step = 1e-4 * max(1.0, abs(point[index]))
            previous_estimate = None
            confirmed = False

            for _ in range(24):
                plus = point.copy()
                minus = point.copy()
                plus[index] += step
                minus[index] -= step

                if (
                    plus[index] == point[index]
                    or minus[index] == point[index]
                ):
                    break

                lp, _ = objective_parts(
                    plus, data, layout, gradient=False
                )
                lm, _ = objective_parts(
                    minus, data, layout, gradient=False
                )

                estimate = float(
                    np.mean(lp - lm, dtype=np.longdouble)
                    / (2.0 * step)
                    + regularization * point[index] / dimension
                )

                if previous_estimate is not None:
                    numerical_tolerance = (
                        5e-7 + 1e-4 * abs(estimate)
                    )

                    if (
                        abs(estimate - previous_estimate)
                        <= numerical_tolerance
                    ):
                        comparison_tolerance = (
                            5e-7
                            + 1e-3
                            * max(abs(estimate), abs(analytic[index]))
                        )

                        require(
                            abs(estimate - analytic[index])
                            <= comparison_tolerance,
                            "Stabilizovana provera gradijenta nije prosla: "
                            f"parametar={index}; "
                            f"analiticki={analytic[index]:.9g}; "
                            f"numericki={estimate:.9g}; "
                            f"korak={step:.3g}.",
                        )

                        confirmed = True
                        break

                previous_estimate = estimate
                step *= 0.5

            require(
                confirmed,
                "Numericka provera gradijenta nije se stabilizovala: "
                f"parametar={index}.",
            )

    def dogleg_step(matrix, gradient, trust_radius):
        gradient_norm = float(np.linalg.norm(gradient))

        if gradient_norm == 0.0:
            return np.zeros_like(gradient)

        newton = -np.linalg.solve(matrix, gradient)

        if np.linalg.norm(newton) <= trust_radius:
            return newton

        bg = matrix @ gradient
        curvature = float(gradient @ bg)

        if curvature <= 0.0:
            return -trust_radius * gradient / gradient_norm

        cauchy = -(
            float(gradient @ gradient) / curvature
        ) * gradient

        if np.linalg.norm(cauchy) >= trust_radius:
            return -trust_radius * gradient / gradient_norm

        direction = newton - cauchy

        a = float(direction @ direction)
        b = 2.0 * float(cauchy @ direction)
        c = float(cauchy @ cauchy) - trust_radius ** 2

        if a <= 0.0:
            return cauchy

        discriminant = max(b * b - 4.0 * a * c, 0.0)
        fraction = (
            -b + math.sqrt(discriminant)
        ) / (2.0 * a)

        fraction = min(1.0, max(0.0, fraction))
        return cauchy + fraction * direction

    losses, gradient = evaluate(parameters)
    audit_gradient(parameters, gradient)

    for attempt in range(maximum_attempts):
        iterations = attempt + 1
        gradient_norm = float(np.max(np.abs(gradient)))

        if gradient_norm <= tolerance:
            # Ne oslanjaj se samo na status optimizatora.
            final_losses, final_gradient = evaluate(parameters)
            final_norm = float(np.max(np.abs(final_gradient)))

            require(
                final_norm <= tolerance,
                "Zavrsna provera konvergencije nije prosla: "
                f"gradijent={final_norm:.6g}.",
            )

            audit_gradient(parameters, final_gradient)

            model = Model(
                layout,
                parameters.copy(),
                regularization,
            )
            model._cnn_trust_hessian = hessian.copy()
            return model

        try:
            step = dogleg_step(hessian, gradient, radius)
        except np.linalg.LinAlgError:
            hessian = repair_hessian(hessian)
            step = dogleg_step(hessian, gradient, radius)

        predicted_reduction = float(
            -gradient @ step
            - 0.5 * step @ (hessian @ step)
        )

        if (
            not math.isfinite(predicted_reduction)
            or predicted_reduction <= 0.0
        ):
            hessian = repair_hessian(hessian)
            radius *= 0.25

            require(
                radius >= minimum_radius,
                "Oblast poverenja je postala numericki premala.",
            )
            continue

        candidate = parameters + step

        try:
            candidate_losses, candidate_gradient = evaluate(candidate)

            change = objective_change(
                parameters,
                step,
                losses,
                candidate_losses,
            )
            actual_reduction = -change
            ratio = actual_reduction / predicted_reduction

        except (RuntimeError, FloatingPointError) as exc:
            # Odbaci iskljucivo numericki neizvodljiv probni korak.
            # Ostale greske ne prikrivaj.
            message = str(exc).lower()
            numerical_failure = any(
                text in message
                for text in (
                    "nulta ili nekonacna",
                    "nekonacan",
                    "nije normalizovan",
                )
            )

            if not numerical_failure:
                raise

            candidate_losses = None
            candidate_gradient = None
            actual_reduction = -math.inf
            ratio = -math.inf

        step_norm = float(np.linalg.norm(step))

        if ratio < 0.25:
            radius *= 0.25
        elif ratio > 0.75 and step_norm >= 0.8 * radius:
            radius = min(2.0 * radius, maximum_radius)

        if (
            ratio >= 0.10
            and actual_reduction > 0.0
            and candidate_gradient is not None
        ):
            difference = candidate_gradient - gradient
            bs = hessian @ step

            sbs = float(step @ bs)
            sy = float(step @ difference)

            # Powell prigusenje cuva pozitivnu definitnost.
            if sbs > 1e-24:
                if sy < 0.2 * sbs:
                    damping = 0.8 * sbs / (sbs - sy)
                    difference = (
                        damping * difference
                        + (1.0 - damping) * bs
                    )
                    sy = float(step @ difference)

                if sy > 1e-24:
                    hessian += (
                        np.outer(difference, difference) / sy
                        - np.outer(bs, bs) / sbs
                    )
                    hessian = 0.5 * (hessian + hessian.T)

            parameters = candidate
            losses = candidate_losses
            gradient = candidate_gradient

            accepted_steps += 1
            consecutive_rejections = 0

            if accepted_steps % 40 == 0:
                hessian = repair_hessian(hessian)

        else:
            consecutive_rejections += 1

            if consecutive_rejections == 8:
                audit_gradient(parameters, gradient)
                hessian = repair_hessian(hessian)

        require(
            radius >= minimum_radius,
            "Optimizacija je dostigla numericku granicu koraka; "
            f"gradijent={np.max(np.abs(gradient)):.6g}.",
        )

        now = time.monotonic()

        if now - last_report >= 60.0:
            print(
                f"Ucenje: korak {iterations} | "
                f"gradijent: {np.max(np.abs(gradient)):.6g} | "
                f"oblast: {radius:.3g}",
                flush=True,
            )
            last_report = now

    raise RuntimeError(
        "Optimizacija nije dostigla zadati prag; "
        f"gradijent={np.max(np.abs(gradient)):.6g}; "
        f"prihvaceni koraci={accepted_steps}; "
        f"pokusaji={iterations}."
    )






def evaluate_model(model, data):
    losses, _ = objective_parts(
        model.parameters, data, model.layout, gradient=False
    )
    return float(losses.mean())


# ------------------------------------------------------------
# TACAN DEKODER PO BLOKOVIMA
# ------------------------------------------------------------


@njit(cache=False)
def forbidden_run(mask, maximum):
    intersection = mask

    for shift in range(1, maximum + 1):
        intersection &= mask >> np.uint64(shift)

    return intersection != np.uint64(0)


@njit(cache=False)
def suffix_maximum(log_probability, state_masks, counts, k):
    groups = len(log_probability)

    best = np.full((groups + 1, k + 1), -np.inf)
    masks = np.zeros((groups + 1, k + 1), dtype=np.uint64)
    best[groups, 0] = 0.0

    for group in range(groups - 1, -1, -1):
        for remaining in range(k + 1):
            for state in range(8):
                count = counts[state]

                if count > remaining:
                    continue

                tail = best[group + 1, remaining - count]

                if not math.isfinite(tail):
                    continue

                score = log_probability[group, state] + tail
                mask = (
                    state_masks[group, state]
                    | masks[group + 1, remaining - count]
                )

                if (
                    score > best[group, remaining]
                    or (
                        score == best[group, remaining]
                        and mask < masks[group, remaining]
                    )
                ):
                    best[group, remaining] = score
                    masks[group, remaining] = mask

    return best, masks


@njit(cache=False)
def constrained_search(
    group,
    remaining,
    mask,
    score,
    log_probability,
    state_masks,
    counts,
    suffix,
    maximum_run,
    global_upper,
    best_score,
    best_mask,
):
    tolerance = 1e-12 * (1.0 + abs(global_upper))

    if best_score[0] >= global_upper - tolerance:
        return

    if group == len(log_probability):
        if remaining == 0:
            if (
                score > best_score[0]
                or (
                    score == best_score[0]
                    and mask < best_mask[0]
                )
            ):
                best_score[0] = score
                best_mask[0] = mask
        return

    upper = score + suffix[group, remaining]

    if upper < best_score[0] - 1e-11 * (1.0 + abs(upper)):
        return

    priorities = np.full(8, -np.inf)

    for state in range(8):
        count = counts[state]

        if count <= remaining:
            tail = suffix[group + 1, remaining - count]

            if math.isfinite(tail):
                priorities[state] = log_probability[group, state] + tail

    order = np.argsort(priorities)

    for index in range(7, -1, -1):
        state = order[index]

        if not math.isfinite(priorities[state]):
            continue

        candidate = mask | state_masks[group, state]

        if forbidden_run(candidate, maximum_run):
            continue

        constrained_search(
            group + 1,
            remaining - counts[state],
            candidate,
            score + log_probability[group, state],
            log_probability,
            state_masks,
            counts,
            suffix,
            maximum_run,
            global_upper,
            best_score,
            best_mask,
        )


@dataclass
class Prediction:
    groups: np.ndarray
    log_probability: np.ndarray
    log_z: float
    n: int
    k: int

    def score(self, mask):
        value = -self.log_z

        for index, group in enumerate(self.groups):
            state = sum(
                ((int(mask) >> int(number)) & 1) << bit
                for bit, number in enumerate(group)
            )
            value += self.log_probability[index, state]

        return float(value)


def make_prediction(model, data):
    require(data.size == 1, "NEXT zahteva jedan kontekst.")

    states, _ = model_forward(
        model.parameters, data, model.layout
    )
    probabilities = np.abs(states) ** 2

    _, full_prefix = count_prefix(
        probabilities, model.layout.n
    )
    cardinality = full_prefix[0, -1]

    require(
        abs(cardinality.sum() - 1.0) < 1e-9,
        "Puna raspodela broja ukljucenih elemenata nije normalizovana.",
    )

    z = float(cardinality[model.layout.k])
    require(z > 0 and math.isfinite(z), "Neispravna NEXT normalizacija.")

    log_probability = np.full_like(probabilities[0], -np.inf)
    positive = probabilities[0] > 0
    log_probability[positive] = np.log(probabilities[0][positive])

    return Prediction(
        model.layout.groups.copy(),
        log_probability,
        math.log(z),
        model.layout.n,
        model.layout.k,
    )


def decode(prediction, maximum_run=MAX_CONSECUTIVE):
    state_masks = np.zeros(
        (len(prediction.groups), 8), dtype=np.uint64
    )

    for group_index, group in enumerate(prediction.groups):
        for state in range(8):
            mask = sum(
                1 << int(number)
                for bit, number in enumerate(group)
                if (state >> bit) & 1
            )
            state_masks[group_index, state] = np.uint64(mask)

    suffix, masks = suffix_maximum(
        prediction.log_probability,
        state_masks,
        COUNTS,
        prediction.k,
    )

    mask = masks[0, prediction.k]
    score = suffix[0, prediction.k]

    require(math.isfinite(score), "Ne postoji dozvoljen sektor kardinalnosti.")

    if forbidden_run(mask, maximum_run):
        best_score = np.array([-math.inf])
        best_mask = np.array([0], dtype=np.uint64)

        constrained_search(
            0,
            prediction.k,
            np.uint64(0),
            0.0,
            prediction.log_probability,
            state_masks,
            COUNTS,
            suffix,
            maximum_run,
            float(score),
            best_score,
            best_mask,
        )

        mask = best_mask[0]
        score = best_score[0]

    integer_mask = int(mask)

    require(
        math.isfinite(score)
        and integer_mask.bit_count() == prediction.k
        and not forbidden_run(mask, maximum_run),
        "NEXT dekoder nije dobio ispravan maksimum.",
    )

    conditional_score = float(score - prediction.log_z)

    require(
        abs(prediction.score(integer_mask) - conditional_score) < 1e-9,
        "Skor dekodera nije saglasan sa raspodelom.",
    )

    numbers = tuple(
        j + 1
        for j in range(prediction.n)
        if (integer_mask >> j) & 1
    )

    return numbers, conditional_score


# ------------------------------------------------------------
# INTERNE PROVERE
# ------------------------------------------------------------


def add_block_circuit(circuit, group, angles):
    for angle, specification in zip(angles, BLOCK_GATES):
        kind, target, control_mask = specification

        if kind == "z":
            circuit.rz(float(angle), int(group[target]))
            continue

        controls = [
            int(group[bit])
            for bit in range(3)
            if (control_mask >> bit) & 1
        ]

        circuit.append(
            RYGate(float(angle)).control(len(controls)),
            controls + [int(group[target])],
        )


def check_qiskit(parameters, data, layout):
    states, cache = model_forward(parameters, data, layout)
    _, combine_cache, _, lookup, local_states = cache

    # Deljeni vremenski filteri.
    test_inputs = data.patterns.reshape(-1, 3)[:3]

    for channel in range(2):
        actual = (
            layout.first_base[channel]
            + parameters[layout.first[channel]]
        )

        expected, _ = kernel_forward(test_inputs, actual)

        for row, inputs in enumerate(test_inputs):
            circuit = QuantumCircuit(1)
            circuit.ry(float(actual[0]), 0)

            for j, value in enumerate(inputs):
                a, phase, rotation = actual[1 + 3 * j:4 + 3 * j]
                circuit.rz(float(a * value + phase), 0)
                circuit.ry(float(rotation), 0)

            actual_state = Statevector.from_instruction(circuit).data

            require(
                np.max(np.abs(actual_state - expected[row])) < 1e-10,
                "Qiskit provera vremenskog filtera nije prosla.",
            )

    # Spajanje vremenskih skala.
    bias, combine, _ = combine_cache
    row = data.size - 1
    number = 0

    circuit = QuantumCircuit(1)
    circuit.ry(float(bias[number]), 0)

    for scale_index in range(len(layout.scales)):
        features = lookup[data.ids[row, number, scale_index]]
        a, b, offset = combine[scale_index]

        circuit.rz(float(a * features[0]), 0)
        circuit.ry(float(b * features[1] + offset), 0)

    actual_state = Statevector.from_instruction(circuit).data

    require(
        np.max(
            np.abs(actual_state - local_states[row, number])
        ) < 1e-10,
        "Qiskit provera spajanja skala nije prosla.",
    )

    # Ceo mali model: sest kanala, dva trokubITna bloka.
    circuit = QuantumCircuit(layout.n)

    for number in range(layout.n):
        circuit.append(
            StatePreparation(local_states[row, number]),
            [number],
        )

    angles = (
        BLOCK_BASE[None, :]
        + parameters[layout.blocks].reshape(-1, 9)
    )

    for group, angle in zip(layout.groups, angles):
        add_block_circuit(circuit, group, angle)

    full_state = Statevector.from_instruction(circuit).data
    probabilities = np.abs(states[row:row + 1]) ** 2
    _, prefix = count_prefix(probabilities, layout.n)

    actual_cardinality = np.zeros(layout.n + 1)

    for mask in range(1 << layout.n):
        expected_amplitude = 1.0 + 0.0j

        for group_index, group in enumerate(layout.groups):
            local_mask = sum(
                ((mask >> int(number)) & 1) << bit
                for bit, number in enumerate(group)
            )
            expected_amplitude *= states[row, group_index, local_mask]

        require(
            abs(full_state[mask] - expected_amplitude) < 1e-9,
            "Qiskit provera zajednickih amplituda nije prosla.",
        )

        actual_cardinality[mask.bit_count()] += abs(full_state[mask]) ** 2

    require(
        np.max(
            np.abs(actual_cardinality - prefix[0, -1])
        ) < 1e-9,
        "Qiskit provera normalizacije sektora nije prosla.",
    )


def internal_checks():
    n, k = 6, 2
    groups = np.array([[0, 2, 4], [1, 3, 5]], dtype=np.int64)
    layout = Layout(groups, n, k, scales=(1, 2, 4))

    combinations = list(itertools.combinations(range(n), k))
    draws = np.asarray(
        [
            combinations[(7 * t + t // 3) % len(combinations)]
            for t in range(14)
        ],
        dtype=np.int64,
    )
    history = binary_draws(draws, n)

    data = Dataset(
        history, np.arange(len(history) - 1), layout
    )

    parameters = 0.025 * np.sin(
        0.37 * np.arange(layout.size) + 0.2
    )

    _, analytic = objective_parts(
        parameters, data, layout, gradient=True
    )
    numerical = np.zeros_like(parameters)

    for index in range(layout.size):
        step = 1e-5
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += step
        minus[index] -= step

        fp, _ = objective_parts(
            plus, data, layout, gradient=False
        )
        fm, _ = objective_parts(
            minus, data, layout, gradient=False
        )

        numerical[index] = (fp.mean() - fm.mean()) / (2.0 * step)

    error = float(np.max(np.abs(analytic - numerical)))

    require(
        error < 3e-6,
        f"Analiticki gradijent nije prosao proveru: {error:.6g}.",
    )

    check_qiskit(parameters, data, layout)

    model = Model(layout, parameters, 0.1)
    next_data = Dataset(
        history, [len(history) - 1], layout, observed=False
    )
    prediction = make_prediction(model, next_data)

    probability_sum = 0.0
    brute_score = -math.inf

    for combination in combinations:
        mask = sum(1 << j for j in combination)
        score = prediction.score(mask)
        probability_sum += math.exp(score)
        brute_score = max(brute_score, score)

    require(
        abs(probability_sum - 1.0) < 1e-10,
        "Raspodela cele kombinacije nije normalizovana.",
    )

    _, decoded_score = decode(prediction)

    require(
        abs(decoded_score - brute_score) < 1e-10,
        "Dekoder nije pronasao globalni maksimum.",
    )

    # Poseban primer u kojem nekontrolisani maksimum sadrzi pet u nizu.
    inclusion = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.1])
    block_probability = np.ones((1, 2, 8))

    for group_index, group in enumerate(groups):
        for state in range(8):
            for bit, number in enumerate(group):
                p = inclusion[number]
                block_probability[0, group_index, state] *= (
                    p if (state >> bit) & 1 else 1.0 - p
                )

    _, prefix = count_prefix(block_probability, 5)

    controlled = Prediction(
        groups,
        np.log(block_probability[0]),
        math.log(prefix[0, -1, 5]),
        n,
        5,
    )

    _, controlled_score = decode(controlled)
    brute_score = -math.inf

    for combination in itertools.combinations(range(n), 5):
        mask = sum(1 << j for j in combination)

        if not forbidden_run(np.uint64(mask), MAX_CONSECUTIVE):
            brute_score = max(brute_score, controlled.score(mask))

    require(
        abs(controlled_score - brute_score) < 1e-10,
        "Ograniceni dekoder nije prosao proveru.",
    )

    # Buduci redovi ne smeju menjati ranije ulaze ili ciljeve.
    queries = np.arange(5)
    original = Dataset(history, queries, layout)

    changed = history.copy()
    changed[7:] = np.roll(changed[7:], 1, axis=1)
    altered = Dataset(changed, queries, layout)

    require(
        np.array_equal(original.ids, altered.ids)
        and np.array_equal(original.patterns, altered.patterns)
        and np.array_equal(original.observed, altered.observed),
        "Otkriven pristup buducim podacima.",
    )

    # Promena oznaka mora cuvati rezultat uz odgovarajuce
    # prekodiranje grupa i lokalnih parametara.
    permutation = np.array([4, 1, 5, 0, 3, 2])
    inverse = np.argsort(permutation)

    permuted_groups = inverse[groups]
    permuted_layout = Layout(
        permuted_groups, n, k, scales=layout.scales
    )

    permuted_parameters = parameters.copy()
    permuted_parameters[permuted_layout.bias] = (
        parameters[layout.bias][permutation]
    )

    permuted_data = Dataset(
        history[:, permutation],
        np.arange(len(history) - 1),
        permuted_layout,
    )

    original_loss, _ = objective_parts(
        parameters, data, layout, gradient=False
    )
    permuted_loss, _ = objective_parts(
        permuted_parameters,
        permuted_data,
        permuted_layout,
        gradient=False,
    )

    require(
        np.max(np.abs(original_loss - permuted_loss)) < 1e-9,
        "Model uvodi zavisnost od numerickog susedstva oznaka.",
    )


# ------------------------------------------------------------
# HRONOLOSKA VALIDACIJA, TEST I ZAVRSNO UCENJE
# ------------------------------------------------------------


def prepare_stage(history, end):
    groups = learn_groups(history[1:end + 1])
    layout = Layout(groups, N, K, SCALES)

    training = Dataset(
        history[:end + 1], np.arange(end), layout
    )

    return layout, training


def process_csv(path):
    draws = read_draws(path)
    history = binary_draws(draws, N)
    transitions = len(history) - 1

    train_end = int(TRAIN_FRACTION * transitions)
    validation_end = int(VALIDATION_END * transitions)

    require(
        0 < train_end < validation_end < transitions,
        "Neispravna hronoloska podela.",
    )

    layout, training = prepare_stage(history, train_end)

    validation = Dataset(
        history[:validation_end + 1],
        np.arange(train_end, validation_end),
        layout,
    )

    best_model = None
    best_validation = math.inf
    warm = None

    for regularization in REGULARIZATIONS:
        print(
            f"Ucenje vremenskih filtera: reg={regularization:g}.",
            flush=True,
        )

        try:
            candidate = fit_model(
                training, layout, regularization, warm
            )
            value = evaluate_model(candidate, validation)
            warm = candidate

        except (RuntimeError, ValueError, FloatingPointError) as exc:
            print(
                f"PROVERA: reg={regularization:g} odbacen: {exc}"
            )
            continue

        if value < best_validation:
            best_validation = value
            best_model = candidate

    if best_model is None:
        raise RuntimeError(
            "Nijedan validacioni kandidat nije prosao provere."
        )

    print(
        f"Validacija NLL: {best_validation:.6f} | "
        f"reg: {best_model.regularization:g}"
    )

    del training, validation

    print("Ucenje modela za test.", flush=True)

    test_layout, test_training = prepare_stage(
        history, validation_end
    )
    test_model = fit_model(
        test_training,
        test_layout,
        best_model.regularization,
        warm=best_model,
    )

    del test_training

    test_data = Dataset(
        history,
        np.arange(validation_end, transitions),
        test_layout,
    )

    print(f"Test NLL: {evaluate_model(test_model, test_data):.6f}")
    del test_data

    print("Zavrsno ucenje: svi istorijski prelazi.", flush=True)

    final_layout, full_training = prepare_stage(
        history, transitions
    )
    final_model = fit_model(
        full_training,
        final_layout,
        best_model.regularization,
        warm=test_model,
    )

    del full_training

    next_data = Dataset(
        history,
        [len(history) - 1],
        final_layout,
        observed=False,
    )

    prediction = make_prediction(final_model, next_data)
    numbers, score = decode(prediction)

    require(
        len(numbers) == K
        and len(set(numbers)) == K
        and min(numbers) >= 1
        and max(numbers) <= N
        and score <= 1e-9,
        "Neispravna NEXT kombinacija ili verovatnoca.",
    )

    print(f"NEXT: {list(numbers)}")

    return {
        "name": path.name,
        "next": numbers,
        "prediction": prediction,
        "fingerprint": hashlib.sha256(
            np.ascontiguousarray(draws).tobytes()
        ).hexdigest(),
    }


def diagnostic_difference(first, second):
    largest = 0.0

    for step in (1, 2, 4, 5, 7, 8, 10, 11, 14, 16, 17, 19):
        for start in range(N):
            numbers = (start + step * np.arange(K)) % N
            mask = sum(1 << int(j) for j in numbers)

            largest = max(
                largest,
                abs(first.score(mask) - second.score(mask)),
            )

    return largest


def main():
    print(
        "Quantum Temporal CNN V1: deljeni vremenski filtri "
        "na tri skale i trokubITne amplitude."
    )
    print(
        "Lokalna simulacija; hibridno prosledjivanje odziva "
        "i klasicna optimizacija."
    )
    print(
        "Tacan NLL cele sedmorke; "
        "dekodiranje maksimuma do numericke tolerancije."
    )
    print("NEXT: najvise cetiri uzastopna broja.")
    print("Pokretanje internih provera.", flush=True)

    internal_checks()

    print("Interne provere: prosle.")

    results = []
    failures = 0

    for path in FILES:
        print()
        print(f"CSV: {path.name}", flush=True)

        try:
            current = process_csv(path)

            for previous in results:
                if current["next"] != previous["next"]:
                    continue

                same_input = (
                    current["fingerprint"] == previous["fingerprint"]
                )

                difference = diagnostic_difference(
                    current["prediction"],
                    previous["prediction"],
                )

                print(
                    f"PROVERA: isti NEXT za "
                    f"{previous['name']} i {current['name']}."
                )

                input_status = "isti" if same_input else "razliciti"

                print(
                    f"CSV sadrzaji: {input_status}; "
                    "maksimalna razlika logP na provernim kombinacijama: "
                    f"{difference:.6g}."
                )

            results.append(current)

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            failures += 1
            print(f"STOP za ovaj CSV: {exc}")

            if DIAGNOSTICS:
                import traceback
                traceback.print_exc()

    if failures:
        print()
        print(
            "Obrada nije kompletna: neki CSV-ovi nisu prosli provere."
        )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print("Obrada prekinuta korisnickim zahtevom.")

    except Exception as exc:
        print()
        print(f"STOP: {exc}")

        if DIAGNOSTICS:
            import traceback
            traceback.print_exc()



"""
Quantum Temporal CNN V1: deljeni vremenski filtri na tri skale i trokubITne amplitude.
Lokalna simulacija; hibridno prosledjivanje odziva i klasicna optimizacija.
Tacan NLL cele sedmorke; dekodiranje maksimuma do numericke tolerancije.
NEXT: najvise cetiri uzastopna broja.
Pokretanje internih provera.
Interne provere: prosle.

CSV: loto7_4724_k93.csv
Ucenje vremenskih filtera: reg=1.
Ucenje: korak 737 | gradijent: 0.0944789 | oblast: 2.44e-05
Ucenje: korak 1462 | gradijent: 0.530059 | oblast: 0.00313
Ucenje: korak 2182 | gradijent: 2.10193 | oblast: 3.05e-06
Ucenje: korak 2838 | gradijent: 0.782824 | oblast: 3.05e-06
PROVERA: reg=1 odbacen: Optimizacija nije dostigla zadati prag; gradijent=17.4005; prihvaceni koraci=2320; pokusaji=3000.
Ucenje vremenskih filtera: reg=0.1.
Ucenje: korak 749 | gradijent: 0.0296816 | oblast: 0.0125
Ucenje: korak 1499 | gradijent: 0.038016 | oblast: 9.77e-05
Ucenje: korak 2188 | gradijent: 5.50892 | oblast: 0.000781
Ucenje: korak 2809 | gradijent: 2.97891 | oblast: 1.91e-07
PROVERA: reg=0.1 odbacen: Optimizacija nije dostigla zadati prag; gradijent=2.47565; prihvaceni koraci=2510; pokusaji=3000.
Ucenje vremenskih filtera: reg=0.01.
Ucenje: korak 746 | gradijent: 0.00201433 | oblast: 0.4
Ucenje: korak 1491 | gradijent: 5.72874e-06 | oblast: 0.1
Validacija NLL: 16.616067 | reg: 0.01
Ucenje modela za test.
Ucenje: korak 596 | gradijent: 0.00268421 | oblast: 0.2
Ucenje: korak 1196 | gradijent: 0.000210453 | oblast: 0.2
Ucenje: korak 1796 | gradijent: 0.000593114 | oblast: 0.05
Test NLL: 16.620077
Zavrsno ucenje: svi istorijski prelazi.
Ucenje: korak 495 | gradijent: 0.00274756 | oblast: 0.2
Ucenje: korak 990 | gradijent: 0.000314876 | oblast: 0.2
Ucenje: korak 1495 | gradijent: 0.00129088 | oblast: 0.2
Ucenje: korak 1998 | gradijent: 0.00680026 | oblast: 0.2
NEXT: [8, x, 24, y, 31, z, 34]

CSV: loto7_4724_k93_loto_2984.csv
Ucenje vremenskih filtera: reg=1.
Ucenje: korak 1180 | gradijent: 0.112649 | oblast: 9.77e-05
Ucenje: korak 2227 | gradijent: 0.297801 | oblast: 3.81e-07
PROVERA: reg=1 odbacen: Optimizacija nije dostigla zadati prag; gradijent=12.5616; prihvaceni koraci=2280; pokusaji=3000.
Ucenje vremenskih filtera: reg=0.1.
Ucenje: korak 1181 | gradijent: 5.3055 | oblast: 6.1e-06
Ucenje: korak 2188 | gradijent: 1.76051 | oblast: 2.38e-08
PROVERA: reg=0.1 odbacen: Optimizacija nije dostigla zadati prag; gradijent=29.929; prihvaceni koraci=2385;pokusaji=3000.
Ucenje vremenskih filtera: reg=0.01.
Ucenje: korak 1174 | gradijent: 0.00064133 | oblast: 0.05
Validacija NLL: 16.650568 | reg: 0.01
Ucenje modela za test.
Ucenje: korak 959 | gradijent: 0.000945345 | oblast: 0.4
Test NLL: 16.606966
Zavrsno ucenje: svi istorijski prelazi.
Ucenje: korak 791 | gradijent: 0.00310903 | oblast: 0.2
NEXT: [5, x, 16, y, 33, z, 39]

CSV: loto7_4724_k93_loto_plus_1740.csv
Ucenje vremenskih filtera: reg=1.
Ucenje vremenskih filtera: reg=0.1.
Ucenje: korak 1612 | gradijent: 0.538871 | oblast: 5.96e-09
PROVERA: reg=0.1 odbacen: Optimizacija nije dostigla zadati prag; gradijent=19.7968; prihvaceni koraci=2240; pokusaji=3000.
Ucenje vremenskih filtera: reg=0.01.
Ucenje: korak 1740 | gradijent: 0.938552 | oblast: 2.98e-09
PROVERA: reg=0.01 odbacen: Optimizacija nije dostigla zadati prag; gradijent=12.539; prihvaceni koraci=2318; pokusaji=3000.
Validacija NLL: 16.746839 | reg: 1
Ucenje modela za test.
Ucenje: korak 1607 | gradijent: 1.07646 | oblast: 0.00313
STOP za ovaj CSV: Optimizacija nije dostigla zadati prag; gradijent=0.0788353; prihvaceni koraci=2456; pokusaji=3000.

Obrada nije kompletna: neki CSV-ovi nisu prosli provere.
"""



"""
CNN — vremenska konvolucija ima konkretan, nov potencijal za NEXT predikciju.

Isti učeni filter primenjen kroz celu istoriju. 
Tako model može da prepoznaje ponovljene obrasce kroz nekoliko kola, bez posebnog učenja za svako mesto njihovog pojavljivanja. 
Konvolucija se uspešno koristi i za vremenske nizove; NIJE ograničena samo na slike. 

- Konvoluciju duž vremena: 
    brojevi predstavljaju kanale. Blizina oznaka, poput 12 i 13, ne postaje unapred zadato susedstvo.
- Više vremenskih razmera: 
    razmaknuti filteri mogu povezivati bliža i udaljenija kola, uz relativno malo parametara.
- Hijerarhiju obrazaca: 
    naredni slojevi kombinuju jednostavnije vremenske odnose u složenije.
- Isključivo prošlost: 
    vremenski filteri ne smeju koristiti buduća kola.

Padding, stride i pooling služe organizovanju računanja; sami nisu novi prediktivni signal. 
Za početnu razradu zadržavam svako kolo, bez agresivnog smanjivanja vremenske rezolucije.

Učenje ponovljivih vremenskih obrazaca sa deljenim parametrima. 
Kvantna realizacija mora da očuva upravo taj mehanizam konvolucije.

Uči ponovljive vremenske obrasce na više razmera, koristeći iste filtere kroz istoriju.
Učenje ponovljivih obrazaca kroz više vremenskih razmera. 

V1 sa kauzalnim vremenskim konvolucijama: isti učeni filteri kroz istoriju, više vremenskih razmera i zajednička amplitudna raspodela sedmorke.
"""



"""
Convolutional Neural Network (CNN)

Data Type:
CNN: Best suited for spatial data (e.g., images).

Architecture Complexity:
CNN: Convolutional and pooling layers, followed by fully connected layers.

Learning Capabilities:
CNN: Excels at detecting spatial hierarchies.

Training Challenges:
CNN: High computational requirements.
"""
