from sqlalchemy.orm import relationship
from . import make_table
from .synapse import Synapse


__all__ = ['Conductance']


Conductance = make_table(
    name='conductance',
    comment="reversal potential, conductance, and adjusted psp amplitude of connections",
    columns=[
        ('synapse_id', 'synapse.id', 'The ID of the entry in the synapse table to which these results apply', {'index': True}),
        ('effective_conductance', 'float', 'Effective conductance measured in IC from VC reversal potential'),
        ('adj_psp_amplitude', 'float', 'Resting state psp amplitude adjusted to ideal_holding_potential'),
        ('reversal_potential', 'float', 'Reversal potential calculated from VC'),
        ('ideal_holding_potential', 'float', 'The holding potential used to calculate adj_psp_amplitude such that measurements of \
                                                of strength could be compared across connections at different measured baseline potentials. \
                                                For inhibitory connections this value is -55 mV, for excitatory connections -70 mV.'),
        ('avg_baseline_potential', 'float', 'Average measured baseline potential in IC from qc-pass resting state pulses')
    ]
)

Conductance.synapse = relationship(Synapse, back_populates="conductance", cascade="delete", single_parent=True, uselist=False)
Synapse.conductance = relationship(Conductance, back_populates="synapse", single_parent=True)