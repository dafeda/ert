// Shared parameter definitions - simulates a database table
// Both Parameters and Observations tabs import from this single source

export const availableZones = [
  { id: 'zone1', label: 'Zone 1' },
  { id: 'zone2', label: 'Zone 2' },
  { id: 'zone3', label: 'Zone 3' },
  { id: 'zone4', label: 'Zone 4' },
  { id: 'zone5', label: 'Zone 5' },
  { id: 'zone6', label: 'Zone 6' },
  { id: 'zone7', label: 'Zone 7' },
  { id: 'zone8', label: 'Zone 8' },
  { id: 'zone9', label: 'Zone 9' }
];

export function getZoneLabel(zoneId) {
  const zone = availableZones.find(z => z.id === zoneId);
  return zone ? zone.label : zoneId;
}

// This is the single source of truth for parameter definitions
// Simulates a "parameters" table in a database
export const parameters = [
  {
    id: 'permeability',
    name: 'PermeabilityXY',
    type: 'grid',
    typeLabel: 'Field',
    dimensionality: 3,
    zones: ['zone1', 'zone2'],
    gridInfo: { nx: 100, ny: 80, nz: 25, xinc: 50, yinc: 50 }
  },
  {
    id: 'porosity',
    name: 'Porosity',
    type: 'grid',
    typeLabel: 'Field',
    dimensionality: 3,
    zones: ['zone2', 'zone3'],
    gridInfo: { nx: 100, ny: 80, nz: 25, xinc: 50, yinc: 50 }
  },
  {
    id: 'gaussian1',
    name: 'Gaussian1',
    type: 'grid',
    typeLabel: 'Field',
    dimensionality: 3,
    zones: ['zone1', 'zone3', 'zone5'],
    gridInfo: { nx: 100, ny: 80, nz: 25, xinc: 50, yinc: 50 }
  },
  {
    id: 'gaussian2',
    name: 'Gaussian2',
    type: 'grid',
    typeLabel: 'Field',
    dimensionality: 3,
    zones: ['zone6', 'zone7'],
    gridInfo: { nx: 100, ny: 80, nz: 25, xinc: 50, yinc: 50 }
  },
  {
    id: 'faultseal',
    name: 'FaultSeal',
    type: 'surface',
    typeLabel: 'Surface',
    dimensionality: 2,
    zones: ['zone1', 'zone2', 'zone5'],
    gridInfo: { ncol: 200, nrow: 150, xinc: 25, yinc: 25 }
  },
  {
    id: 'topreservoir',
    name: 'TopReservoir',
    type: 'surface',
    typeLabel: 'Surface',
    dimensionality: 2,
    zones: ['zone3', 'zone4'],
    gridInfo: { ncol: 200, nrow: 150, xinc: 25, yinc: 25 }
  },
  {
    id: 'multiplier1',
    name: 'Multiplier_1',
    type: 'scalar',
    typeLabel: 'GenKw',
    dimensionality: 1,
    zones: null
  },
  {
    id: 'multiplier2',
    name: 'Multiplier_2',
    type: 'scalar',
    typeLabel: 'GenKw',
    dimensionality: 1,
    zones: null
  },
  {
    id: 'everestA',
    name: 'EverestParam_A',
    type: 'scalar',
    typeLabel: 'GenKw',
    dimensionality: 1,
    zones: null
  }
];

export function getTypeLabel(dim) {
  if (dim === 3) return 'Grid 3D';
  if (dim === 2) return 'Surface 2D';
  return 'Scalar 1D';
}
