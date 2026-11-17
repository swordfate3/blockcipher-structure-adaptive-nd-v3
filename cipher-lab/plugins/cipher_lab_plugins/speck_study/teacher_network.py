# ruff: noqa: UP007, UP045
# Keep runtime-evaluable annotations Python 3.9 safe (no PEP604).
"""Pinned make_resnet_ghor(depth=1), inference only; Python 3.9 safe.

Source read-only: FourierNeuralDistinguishers/main/core/net.py SHA256
5971229cbb618b0b278a8b0db599d8f2ed82269336d966e1b75dd7cdb6954ccb.
No import of official code, custom deserializers, optimizer or shared config.
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Callable, Optional  # noqa: UP035 - provider Python 3.9

MODEL_SPEC = {'architecture': 'make_resnet_ghor', 'depth': 1, 'num_blocks': 2,
              'num_filters': 32, 'num_outputs': 1, 'd1': 64, 'd2': 64,
              'word_size': 16, 'ks': 3, 'reg_param': .0001,
              'final_activation': 'sigmoid'}
ALLOCATOR_MIB = 8192
GpuAttestor = Callable[[Any], dict[str, Any]]


def prepare_environment(gpu_uuid: str) -> None:
    """Must run before TF import (and before any CUDA initialization)."""
    if 'tensorflow' in sys.modules:
        raise RuntimeError('TF already imported before GPU binding')
    if not gpu_uuid.startswith('GPU-') or not gpu_uuid.isascii():
        raise ValueError('invalid GPU UUID')
    inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
    if inherited not in (None, gpu_uuid):
        raise RuntimeError('inherited GPU binding mismatch')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def configure_tensorflow(tf: Any, gpu_uuid: str, pci_bus_id: str,
                         attestor: Optional[GpuAttestor]) -> dict[str, Any]:
    """Configure allocator before logical-device initialization, then attest.

    Task3 must supply a runtime attestor (CUDA/OS device evidence, not environment
    echoes), and parent independently verifies its evidence. Missing hook stops
    BEFORE model creation. The returned mapping is evidence, never proof by itself.
    """
    if attestor is None:
        raise RuntimeError('GPU_UNVERIFIED: runtime attestor required')
    physical = tf.config.list_physical_devices('GPU')
    if len(physical) != 1:
        raise RuntimeError('GPU_UNVERIFIED: expected one visible GPU')
    tf.config.set_visible_devices(physical[0], 'GPU')
    tf.config.set_logical_device_configuration(
        physical[0], [tf.config.LogicalDeviceConfiguration(memory_limit=ALLOCATOR_MIB)])
    logical = tf.config.list_logical_devices('GPU')
    if len(logical) != 1:
        raise RuntimeError('GPU_UNVERIFIED: expected logical GPU0')
    evidence = attestor(tf)
    if (evidence.get('gpu_uuid') != gpu_uuid or evidence.get('pci_bus_id') != pci_bus_id
            or evidence.get('pid') != os.getpid() or evidence.get('logical_device') != 0
            or not evidence.get('evidence')):
        raise RuntimeError('GPU_UNVERIFIED: runtime identity mismatch')
    return evidence


def make_official_teacher(tf: Any = None) -> Any:
    """Exact official graph defaults except depth=1. No compile/load_model."""
    if tf is None:
        tf = importlib.import_module('tensorflow')
    layers = tf.keras.layers
    regularizer = tf.keras.regularizers.l2(.0001)
    inp = layers.Input(shape=(64,))
    x = layers.Reshape((4, 16))(inp)
    x = layers.Permute((2, 1))(x)
    x = layers.Conv1D(32, kernel_size=1, padding='same', kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    shortcut = x
    for _ in range(1):
        x = layers.Conv1D(32, kernel_size=3, padding='same',
                          kernel_regularizer=regularizer)(shortcut)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(32, kernel_size=3, padding='same', kernel_regularizer=regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        shortcut = layers.Add()([shortcut, x])
    x = layers.Flatten()(shortcut)
    x = layers.Dense(64, kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(64, kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    out = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizer)(x)
    return tf.keras.models.Model(inputs=inp, outputs=out)


def points_to_bits(payload: bytes) -> Any:
    """Packed big-endian uint64 == four 16-bit words, each MSB first."""
    from .teacher_protocol import unpack_points
    unpack_points(payload)  # strict count/length check before NumPy allocation
    import numpy as np
    return np.unpackbits(np.frombuffer(payload, dtype=np.uint8).reshape(-1, 8), axis=1)
