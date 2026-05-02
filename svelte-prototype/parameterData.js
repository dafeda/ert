// Shared parameter data for both Parameters and Observations tabs
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

// Canonical parameter list used by both Parameters and Observations tabs
export let parameters = [
  {
    name: 'PERMX',
    type: 'Field',
    dimensionality: 3,
    zones: ['zone1', 'zone2'],
    update: true,
    algorithm: 'ES-MDA',
    settings: { truncation: 1.0, localization: 'distance', correlationThreshold: null },
    gridInfo: { nx: 100, ny: 80, nz: 25, xinc: 50, yinc: 50 },
    expanded: false
  },
  {
    name: 'PORO',
    type: 'Field',
    dimensionality: 3,
    zones: ['zone2', 'zone3'],
    update: true,
    algorithm: 'ES-MDA',
    settings: { truncation: 1.0, localization: 'distance', correlationThreshold: null },
    gridInfo: { nx: 100, ny: 80, nz: 25, xinc: 50, yinc: 50 },
    expanded: false
  },
  {
    name: 'Gaussian1',
    type: 'Field',
    dimensionality: 3,
    zones: ['zone1', 'zone3', 'zone5'],
    update: true,
    algorithm: 'ES',
    settings: { truncation: 0.98, localization: 'none', correlationThreshold: null },
    gridInfo: { nx: 100, ny: 80, nz: 25, xinc: 50, yinc: 50 },
    expanded: false
  },
  {
    name: 'Gaussian2',
    type: 'Field',
    dimensionality: 3,
    zones: ['zone6', 'zone7'],
    update: false,
    algorithm: 'ES-MDA',
    settings: { truncation: 1.0, localization: 'distance', correlationThreshold: null },
    gridInfo: { nx: 100, ny: 80, nz: 25, xinc: 50, yinc: 50 },
    expanded: false
  },
  {
    name: 'TopSurface',
    type: 'Surface',
    dimensionality: 2,
    zones: ['zone1', 'zone3', 'zone5'],
    update: true,
    algorithm: 'ES-MDA',
    settings: { truncation: 1.0, localization: 'distance', correlationThreshold: null },
    gridInfo: { ncol: 200, nrow: 150, xinc: 25, yinc: 25 },
    expanded: false
  },
  {
    name: 'BaseSurface',
    type: 'Surface',
    dimensionality: 2,
    zones: ['zone6', 'zone7'],
    update: false,
    algorithm: 'ES',
    settings: { truncation: 0.99, localization: 'none', correlationThreshold: null },
    gridInfo: { ncol: 200, nrow: 150, xinc: 25, yinc: 25 },
    expanded: false
  },
  {
    name: 'MULT_X',
    type: 'GenKw',
    dimensionality: 1,
    update: true,
    algorithm: 'ES-MDA',
    settings: { truncation: 1.0, localization: 'adaptive', correlationThreshold: 0.5 },
    expanded: false
  },
  {
    name: 'MULT_Y',
    type: 'GenKw',
    dimensionality: 1,
    update: true,
    algorithm: 'EnIF',
    settings: { truncation: 1.0, localization: 'graph', correlationThreshold: null },
    expanded: false
  }
];
