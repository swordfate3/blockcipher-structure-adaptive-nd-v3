"""Post-hoc audit helpers for learned constant assignments.

These helpers only *compare* an already-fitted assignment with the true round
keys and probe functional equivalence. They never initialise, constrain or
retrain a model, and they are not imported by the training or model packages.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..representations.simon_cone import SimonCone

KeyBits = dict[tuple[int, int], int]


def evaluate_cone_bit(cone: SimonCone, plaintext: int, key_bits: KeyBits) -> int:
    """Evaluate one public Boolean cone for a plaintext under a constant vector."""
    if type(plaintext) is not int or not 0 <= plaintext < (1 << 64):
        raise ValueError("plaintext must be an unsigned 64-bit integer")
    values: list[int] = []
    for node in cone.nodes:
        if node.kind == "plaintext":
            (coordinate,) = node.coordinate
            values.append((plaintext >> (63 - coordinate)) & 1)
        elif node.kind == "round_key":
            values.append(key_bits[(node.coordinate[0], node.coordinate[1])])
        else:
            left, right = node.args
            left_value, right_value = values[left], values[right]
            values.append(left_value & right_value if node.kind == "and"
                          else left_value ^ right_value)
    return values[cone.output]


def assignment_bits(assignment: int, coordinates: tuple[tuple[int, int], ...]) -> KeyBits:
    """Decode a fitted ``assignment`` integer into per-coordinate bit values."""
    if type(assignment) is not int or not 0 <= assignment < 2 ** len(coordinates):
        raise ValueError("assignment is outside the coordinate domain")
    return {coordinate: (assignment >> index) & 1
            for index, coordinate in enumerate(coordinates)}


def schedule_bits(round_key_words: tuple[int, ...],
                  coordinates: tuple[tuple[int, int], ...]) -> KeyBits:
    """Map MSB-first round-key bits onto symbolic coordinates.

    Convention verified against the scalar cipher: coordinate ``(round, bit)``
    is bit ``31 - bit`` of the round-key word ``round``.
    """
    return {(round_index, bit): (round_key_words[round_index] >> (31 - bit)) & 1
            for round_index, bit in coordinates}


def differing_coordinates(learned: KeyBits, reference: KeyBits,
                          coordinates: tuple[tuple[int, int], ...]
                          ) -> tuple[tuple[int, int], ...]:
    """Coordinates where two assignments disagree, in deterministic order."""
    return tuple(coordinate for coordinate in coordinates
                 if learned[coordinate] != reference[coordinate])


def disagreement_count(cone: SimonCone, plaintexts: tuple[int, ...],
                       left: KeyBits, right: KeyBits) -> int:
    """Count plaintexts whose cone output differs between two assignments."""
    return sum(evaluate_cone_bit(cone, plaintext, left)
               != evaluate_cone_bit(cone, plaintext, right)
               for plaintext in plaintexts)


@dataclass(frozen=True)
class AmbiguityWitness:
    """Measured relation between a fitted assignment and the true schedule."""

    differing: tuple[tuple[int, int], ...]
    differing_set_flip_preserves_function: bool
    individually_sensitive: tuple[tuple[int, int], ...]
    individually_inert: tuple[tuple[int, int], ...]
    probes: int
    output_mismatches_vs_reference: int


def audit_assignment(cone: SimonCone, plaintexts: tuple[int, ...],
                     learned: KeyBits, reference: KeyBits) -> AmbiguityWitness:
    """Measure whether learned constants are unique or lie in an equivalence class.

    A positive ``differing_set_flip_preserves_function`` with a nonempty
    ``differing`` tuple proves the target function does not identify a unique
    constant vector: prediction can be exact while the parameters differ from the
    true schedule.
    """
    if not plaintexts:
        raise ValueError("audit requires at least one plaintext probe")
    coordinates = cone.round_key_coordinates
    missing = [coordinate for coordinate in coordinates
               if coordinate not in learned or coordinate not in reference]
    if missing:
        raise ValueError(f"assignments are missing coordinates: {missing}")

    differing = differing_coordinates(learned, reference, coordinates)
    flipped = dict(learned)
    for coordinate in differing:
        flipped[coordinate] ^= 1
    preserved = bool(differing) and disagreement_count(cone, plaintexts, flipped, learned) == 0

    sensitive: list[tuple[int, int]] = []
    inert: list[tuple[int, int]] = []
    for coordinate in coordinates:
        probe = dict(learned)
        probe[coordinate] ^= 1
        (inert if disagreement_count(cone, plaintexts, probe, learned) == 0
         else sensitive).append(coordinate)

    return AmbiguityWitness(
        differing=differing,
        differing_set_flip_preserves_function=preserved,
        individually_sensitive=tuple(sensitive),
        individually_inert=tuple(inert),
        probes=len(plaintexts),
        output_mismatches_vs_reference=disagreement_count(
            cone, plaintexts, learned, reference),
    )
