<script>
  import AlgorithmSelection from './AlgorithmSelection.svelte';

  let parameters = [
    { name: 'PermeabilityXY', type: 'grid', dimensionality: 3, east: 452000, north: 6802000, zones: ['zone1', 'zone2', 'zone3'] },
    { name: 'Porosity', type: 'grid', dimensionality: 3, east: 452000, north: 6802000, zones: ['zone1', 'zone2'] },
    { name: 'Gaussian1', type: 'grid', dimensionality: 3, east: 453000, north: 6803000, zones: ['zone2', 'zone3'] },
    { name: 'Gaussian2', type: 'grid', dimensionality: 3, east: 455000, north: 6805000, zones: ['zone4', 'zone5'] },
    { name: 'FaultSeal', type: 'surface', dimensionality: 2, east: 451000, north: 6801000, zones: ['zone1'] },
    { name: 'TopReservoir', type: 'surface', dimensionality: 2, east: 454000, north: 6804000, zones: ['zone3', 'zone4'] },
    { name: 'Multiplier_1', type: 'scalar', dimensionality: 1, east: null, north: null, zones: [] },
    { name: 'Multiplier_2', type: 'scalar', dimensionality: 1, east: null, north: null, zones: [] },
    { name: 'EverestParam_A', type: 'scalar', dimensionality: 1, east: null, north: null, zones: [] }
  ];

  let observations = [
    { name: 'WELL-1_PRESSURE', type: 'Summary', east: 450000, north: 6800000, mainRange: 1500, zones: ['zone1', 'zone2'] },
    { name: 'WELL-1_WATERCUT', type: 'Summary', east: 450000, north: 6800000, mainRange: 1500, zones: ['zone1', 'zone2'] },
    { name: 'WELL-2_PRESSURE', type: 'Summary', east: 455000, north: 6805000, mainRange: 1200, zones: ['zone2', 'zone3'] },
    { name: 'WELL-2_WATERCUT', type: 'Summary', east: 455000, north: 6805000, mainRange: 1200, zones: ['zone2', 'zone3'] },
    { name: 'RFT_WELL-3', type: 'RFT', east: 460000, north: 6810000, mainRange: 2000, zones: ['zone4', 'zone5'] },
    { name: 'GENOBS_SEISMIC_1', type: 'General', east: 452000, north: 6802000, mainRange: 3000, zones: ['zone1', 'zone3', 'zone5'] },
    { name: 'GENOBS_SEISMIC_2', type: 'General', east: 458000, north: 6808000, mainRange: 2500, zones: ['zone6', 'zone7'] }
  ];

  function getLocalizationForCell(param, obs) {
    if (param.east === null || param.north === null) {
      return { status: 'no-location', distance: null, zonesMatch: false };
    }

    const distance = Math.sqrt(Math.pow(param.east - obs.east, 2) + Math.pow(param.north - obs.north, 2));
    const mainRange = obs.mainRange || 1500;

    const zonesMatch = param.zones.some(z => obs.zones.includes(z));

    if (distance <= mainRange && zonesMatch) {
      return { status: 'active', distance: Math.round(distance), zonesMatch: true };
    } else if (!zonesMatch) {
      return { status: 'zone-mismatch', distance: Math.round(distance), zonesMatch: false };
    } else {
      return { status: 'out-of-range', distance: Math.round(distance), zonesMatch: false };
    }
  }

  function getTypeBadgeClass(dim) {
    if (dim === 3) return 'grid';
    if (dim === 2) return 'surface';
    return 'scalar';
  }

  function getTypeLabel(dim) {
    if (dim === 3) return 'Grid 3D';
    if (dim === 2) return 'Surface 2D';
    return 'Scalar 1D';
  }
</script>

<div class="matrix-view">
  <div class="table-container">
    <table>
      <thead>
        <tr>
          <th class="sticky-col">Parameter</th>
          {#each observations as obs}
            <th class="obs-col" title="East: {obs.east}, North: {obs.north}">
              <div class="obs-name">{obs.name}</div>
              <div class="obs-meta">{obs.type}</div>
            </th>
          {/each}
        </tr>
      </thead>
      <tbody>
        {#each parameters as param}
          <tr>
            <td class="sticky-col param-name">
              <div class="param-name-text">{param.name}</div>
              <div class="param-meta">
                <span class="type-badge {getTypeBadgeClass(param.dimensionality)}">
                  {getTypeLabel(param.dimensionality)}
                </span>
              </div>
            </td>
            {#each observations as obs}
              {@const cell = getLocalizationForCell(param, obs)}
              <td class="matrix-cell"
                  class:active={cell.status === 'active'}
                  class:out-of-range={cell.status === 'out-of-range'}
                  class:zone-mismatch={cell.status === 'zone-mismatch'}
                  class:no-location={cell.status === 'no-location'}
                  title={cell.distance ? `Distance: ${cell.distance}m, Zones match: ${cell.zonesMatch}` : "No location data"}>
                {#if cell.status === 'active'}
                  <span class="checkmark">✓</span>
                {:else if cell.status === 'zone-mismatch'}
                  <span class="dash" title="Zone mismatch">Z</span>
                {:else if cell.status === 'no-location'}
                  <span class="dash">N/A</span>
                {:else}
                  <span class="dash">-</span>
                {/if}
              </td>
            {/each}
          </tr>
        {/each}
      </tbody>
    </table>
  </div>

  <div class="legend">
    <div class="legend-item">
      <span class="legend-box active"></span>
      <span>Active (within range + zone match)</span>
    </div>
    <div class="legend-item">
      <span class="legend-box zone-mismatch"></span>
      <span>Zone mismatch</span>
    </div>
    <div class="legend-item">
      <span class="legend-box out-of-range"></span>
      <span>Out of range</span>
    </div>
    <div class="legend-item">
      <span class="legend-box no-location"></span>
      <span>No location (scalar)</span>
    </div>
  </div>
</div>

<style>
  .matrix-view {
    font-size: 14px;
  }

  .table-container {
    overflow-x: auto;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }

  th {
    background: #f5f5f5;
    padding: 10px 8px;
    text-align: center;
    border-bottom: 2px solid #e0e0e0;
    border-right: 1px solid #e0e0e0;
    font-weight: 600;
    font-size: 11px;
    color: #555;
    min-width: 80px;
  }

  .sticky-col {
    position: sticky;
    left: 0;
    background: #f5f5f5;
    z-index: 1;
  }

  td {
    padding: 8px;
    border-bottom: 1px solid #e0e0e0;
    border-right: 1px solid #e0e0e0;
    text-align: center;
  }

  .param-name {
    font-weight: 600;
    text-align: left;
    min-width: 150px;
  }

  .obs-col {
    min-width: 100px;
  }

  .obs-name {
    font-weight: 600;
    font-size: 10px;
    margin-bottom: 2px;
    writing-mode: horizontal-tb;
    white-space: nowrap;
  }

  .obs-meta {
    font-size: 9px;
    color: #999;
  }

  .matrix-cell {
    width: 50px;
    height: 40px;
    transition: background 0.2s;
  }

  .matrix-cell.active {
    background: #c8e6c9;
  }

  .matrix-cell.out-of-range {
    background: #f5f5f5;
  }

  .matrix-cell.no-location {
    background: #fafafa;
  }

  .matrix-cell.zone-mismatch {
    background: #fff3e0;
  }

  .checkmark {
    color: #4caf50;
    font-weight: bold;
    font-size: 14px;
  }

  .dash {
    color: #ccc;
    font-size: 14px;
  }

  .no-location .dash {
    color: #999;
    font-size: 10px;
  }

  .type-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 500;
  }

  .type-badge.grid {
    background: #e3f2fd;
    color: #1976d2;
  }

  .type-badge.surface {
    background: #fff3e0;
    color: #f57c00;
  }

  .type-badge.scalar {
    background: #e8f5e9;
    color: #388e3c;
  }

  .legend {
    display: flex;
    gap: 20px;
    margin-top: 15px;
    font-size: 12px;
    color: #666;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .legend-box {
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 1px solid #e0e0e0;
  }

  .legend-box.active {
    background: #c8e6c9;
  }

  .legend-box.out-of-range {
    background: #f5f5f5;
  }

  tr:hover .sticky-col {
    background: #e8e8e8;
  }

  tr:hover .matrix-cell.active {
    background: #a5d6a7;
  }
</style>
