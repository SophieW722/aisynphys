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
        ('adj_psp_amplitude', 'float', 'Resting state psp amplitude adjusted to the target holding potential'),
        ('reversal_potential', 'float', 'Reversal potential calculated from VC'),
        ('target_holding_potential', 'float', 'Target holding potential; -55 mV for inhibitory, -70 mV for excitatory')
    ]
)

Conductance.synapse = relationship(Synapse, back_populates="conductance", cascade="delete", single_parent=True, uselist=False)
Synapse.conductance = relationship(Conductance, back_populates="synapse", single_parent=True)