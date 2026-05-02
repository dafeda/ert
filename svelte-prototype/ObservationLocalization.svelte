<script>
  import { parameters as availableParameters, availableZones, getZoneLabel } from './parameterStore.js';

  // Helper to get type label from parameter
  function getParamTypeLabel(param) {
    return param.typeLabel;
  }

  let observations = [
    {
      name: 'WELL-1_PRESSURE',
      type: 'Summary',
      east: 450000,
      north: 6800000,
      enabled: true,
      zones: ['zone1', 'zone2'],
      expanded: false,
      parameterRadii: [
        { paramName: 'PermeabilityXY', paramType: 'Field', radius: 1500, enabled: true },
        { paramName: 'Porosity', paramType: 'Field', radius: 1500, enabled: true },
        { paramName: 'FaultSeal', paramType: 'Surface', radius: 1000, enabled: true }
      ]
    },
    {
      name: 'WELL-1_WATERCUT',
      type: 'Summary',
      east: 450000,
      north: 6800000,
      enabled: true,
      zones: ['zone1', 'zone2'],
      expanded: false,
      parameterRadii: [
        { paramName: 'PermeabilityXY', paramType: 'Field', radius: 1500, enabled: true },
        { paramName: 'FaultSeal', paramType: 'Surface', radius: 1000, enabled: true }
      ]
    },
    {
      name: 'WELL-2_PRESSURE',
      type: 'Summary',
      east: 455000,
      north: 6805000,
      enabled: true,
      zones: ['zone2', 'zone3'],
      expanded: false,
      parameterRadii: [
        { paramName: 'PermeabilityXY', paramType: 'Field', radius: 1200, enabled: true },
        { paramName: 'Porosity', paramType: 'Field', radius: 1200, enabled: true },
        { paramName: 'TopReservoir', paramType: 'Surface', radius: 900, enabled: true }
      ]
    },
    {
      name: 'WELL-2_WATERCUT',
      type: 'Summary',
      east: 455000,
      north: 6805000,
      enabled: false,
      zones: ['zone2', 'zone3'],
      expanded: false,
      parameterRadii: [
        { paramName: 'PermeabilityXY', paramType: 'Field', radius: 1200, enabled: false },
        { paramName: 'Porosity', paramType: 'Field', radius: 1200, enabled: false }
      ]
    },
    {
      name: 'RFT_WELL-3',
      type: 'RFT',
      east: 460000,
      north: 6810000,
      enabled: true,
      zones: ['zone4', 'zone5'],
      expanded: false,
      parameterRadii: [
        { paramName: 'PermeabilityXY', paramType: 'Field', radius: 2000, enabled: true },
        { paramName: 'Porosity', paramType: 'Field', radius: 2000, enabled: true },
        { paramName: 'FaultSeal', paramType: 'Surface', radius: 1500, enabled: true },
        { paramName: 'TopReservoir', paramType: 'Surface', radius: 1500, enabled: true }
      ]
    },
    {
      name: 'GENOBS_SEISMIC_1',
      type: 'General',
      east: 452000,
      north: 6802000,
      enabled: true,
      zones: ['zone1', 'zone3', 'zone5'],
      expanded: false,
      parameterRadii: [
        { paramName: 'PermeabilityXY', paramType: 'Field', radius: 3000, enabled: true },
        { paramName: 'Porosity', paramType: 'Field', radius: 3000, enabled: true },
        { paramName: 'FaultSeal', paramType: 'Surface', radius: 2500, enabled: true },
        { paramName: 'Multiplier_1', paramType: 'GenKw', radius: null, enabled: true }
      ]
    },
    {
      name: 'GENOBS_SEISMIC_2',
      type: 'General',
      east: 458000,
      north: 6808000,
      enabled: true,
      zones: ['zone6', 'zone7'],
      expanded: false,
      parameterRadii: [
        { paramName: 'Gaussian2', paramType: 'Field', radius: 2500, enabled: true },
        { paramName: 'TopReservoir', paramType: 'Surface', radius: 2000, enabled: true },
        { paramName: 'Multiplier_2', paramType: 'GenKw', radius: null, enabled: true }
      ]
    }
  ];

  let selectedParamToAdd = '';

  function toggleExpand(index) {
    observations[index].expanded = !observations[index].expanded;
    observations = observations;
  }

  function addParameterToObservation(obs, paramName) {
    if (!paramName) return;
    const param = availableParameters.find(p => p.name === paramName);
    if (param && !obs.parameterRadii.some(p => p.paramName === param.name)) {
      const defaultRadius = param.dimensionality === 3 ? 1500 : param.dimensionality === 2 ? 1000 : null;
      obs.parameterRadii = [...obs.parameterRadii, {
        paramName: param.name,
        paramType: param.typeLabel,
        radius: defaultRadius,
        enabled: true
      }];
      selectedParamToAdd = '';
      observations = observations;
    }
  }

  function removeParameterFromObservation(obs, idx) {
    obs.parameterRadii = obs.parameterRadii.filter((_, i) => i !== idx);
    observations = observations;
  }

  function getParamInfo(paramName) {
    return availableParameters.find(p => p.name === paramName);
  }

  function getAvailableParamsForObs(obs) {
    return availableParameters.filter(p => {
      if (obs.parameterRadii.some(pr => pr.paramName === p.name)) return false;
      if (!p.zones) return true; // GenKw has no zones, always available
      return p.zones.some(z => obs.zones.includes(z)); // zone overlap check
    });
  }
</script>

  <div class="observation-localization">
    <div class="table-container">
      <table>
        <thead>
          <tr>
            <th class="expand-col"></th>
            <th>Enabled</th>
            <th>Observation</th>
            <th>Type</th>
            <th>Parameters</th>
          </tr>
        </thead>
        <tbody>
          {#each observations as obs, i}
            <tr class="main-row" on:click={() => toggleExpand(i)} class:disabled={!obs.enabled}>
              <td class="expand-col">
                <span class="chevron" class:expanded={obs.expanded}>›</span>
              </td>
              <td style="text-align: center">
                <input type="checkbox" bind:checked={obs.enabled} on:click|stopPropagation />
              </td>
              <td>{obs.name}</td>
              <td><span class="type-badge" class:summary={obs.type === 'Summary'} class:rft={obs.type === 'RFT'} class:general={obs.type === 'General'}>{obs.type}</span></td>
              <td class="param-count">{obs.parameterRadii.length} parameter{obs.parameterRadii.length !== 1 ? 's' : ''}</td>
            </tr>
            {#if obs.expanded}
              <tr class="details-row">
                <td colspan="5">
                  <div class="details-panel">
                    <div class="details-section">
                      <h4 class="section-title">Location</h4>
                      <div class="details-grid">
                        <div class="detail-item">
                          <label>East (m):</label>
                          <span class="mono">{obs.east.toLocaleString()}</span>
                        </div>
                        <div class="detail-item">
                          <label>North (m):</label>
                          <span class="mono">{obs.north.toLocaleString()}</span>
                        </div>
                      </div>
                    </div>

                    <div class="details-section">
                      <h4 class="section-title">Zones</h4>
                      <div class="zone-tags">
                        {#each obs.zones as zoneId}
                          <span class="zone-badge active">{getZoneLabel(zoneId)}</span>
                        {/each}
                      </div>
                    </div>

                    <div class="details-section">
                      <h4 class="section-title">Parameter Radii</h4>
                      <table class="param-table">
                        <thead>
                          <tr>
                            <th>Param Name</th>
                            <th>Type</th>
                            <th>Radius (m)</th>
                            <th>Zones</th>
                            <th>Enabled</th>
                            <th></th>
                          </tr>
                        </thead>
                        <tbody>
                          {#each obs.parameterRadii as param, idx}
                            <tr>
                              <td>{param.paramName}</td>
                              <td><span class="type-badge small" class:field={param.paramType === 'Field'} class:surface={param.paramType === 'Surface'} class:genkw={param.paramType === 'GenKw'}>{param.paramType}</span></td>
                              <td>
                                {#if getParamInfo(param.paramName)?.dimensionality === 1}
                                  <span class="na-text">N/A (scalar)</span>
                                {:else}
                                  <input type="number" bind:value={param.radius} class="range-input" disabled={!param.enabled || !obs.enabled} on:click|stopPropagation />
                                {/if}
                              </td>
                              <td class="zone-cell">
                                {#if getParamInfo(param.paramName)?.zones}
                                  <div class="zone-tags small">
                                    {#each getParamInfo(param.paramName).zones as zone}
                                      <span class="zone-badge" class:active={obs.zones.includes(zone)}>{getZoneLabel(zone)}</span>
                                    {/each}
                                  </div>
                                {:else}
                                  <span class="na-text">All zones</span>
                                {/if}
                              </td>
                              <td style="text-align: center">
                                <input type="checkbox" bind:checked={param.enabled} disabled={!obs.enabled} on:click|stopPropagation />
                              </td>
                              <td>
                                <button class="remove-btn" on:click|stopPropagation={() => removeParameterFromObservation(obs, idx)} disabled={!obs.enabled}>×</button>
                              </td>
                            </tr>
                          {/each}
                        </tbody>
                      </table>
                      <div class="add-param-row">
                        <select bind:value={selectedParamToAdd} on:change={() => addParameterToObservation(obs, selectedParamToAdd)} on:click|stopPropagation disabled={!obs.enabled}>
                          <option value="">+ Add parameter...</option>
                          {#each getAvailableParamsForObs(obs) as param}
                            <option value={param.name}>
                              {param.name} ({param.type})
                              {#if param.zones}
                                - Zones: {param.zones.map(z => getZoneLabel(z)).join(', ')}
                              {/if}
                            </option>
                          {/each}
                        </select>
                      </div>
                    </div>
                  </div>
                </td>
              </tr>
            {/if}
          {/each}
        </tbody>
      </table>
    </div>
  </div>

<style>
  .observation-localization {
    font-size: 14px;
  }

  .table-container {
    overflow-x: auto;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 20px;
    font-size: 13px;
  }

  th {
    background: #f5f5f5;
    padding: 10px 12px;
    text-align: left;
    border-bottom: 2px solid #e0e0e0;
    font-weight: 600;
    font-size: 12px;
    color: #555;
    white-space: nowrap;
  }

  td {
    padding: 10px 12px;
    border-bottom: 1px solid #e0e0e0;
  }

  tr:hover {
    background: #f9f9f9;
  }

  tr.disabled {
    opacity: 0.5;
  }

  .mono {
    font-family: 'Courier New', monospace;
    font-size: 12px;
    color: #555;
  }

  .range-input {
    padding: 6px 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 12px;
    width: 100px;
  }

  .range-input:disabled {
    background: #f5f5f5;
  }

  .type-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 500;
  }

  .type-badge.summary {
    background: #e8eaf6;
    color: #3f51b5;
  }

  .type-badge.rft {
    background: #fce4ec;
    color: #c2185b;
  }

  .type-badge.general {
    background: #e0f2f1;
    color: #00796b;
  }

  .zone-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }

  .expand-col {
    width: 30px;
    text-align: center;
  }

  .main-row {
    cursor: pointer;
  }

  .main-row:hover {
    background: #f9f9f9;
  }

  .chevron {
    display: inline-block;
    transition: transform 0.2s;
    font-size: 18px;
    color: #999;
  }

  .chevron.expanded {
    transform: rotate(90deg);
  }

  .details-row td {
    padding: 0;
    border-bottom: 2px solid #e0e0e0;
  }

  .details-panel {
    background: #fafafa;
    padding: 20px;
    border-top: 1px solid #e0e0e0;
  }

  .details-section {
    margin-bottom: 20px;
  }

  .details-section:last-child {
    margin-bottom: 0;
  }

  .section-title {
    font-size: 13px;
    font-weight: 600;
    color: #e67e22;
    margin: 0 0 12px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #e0e0e0;
  }

  .details-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 15px;
  }

  .detail-item {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }

  .detail-item label {
    font-size: 12px;
    font-weight: 600;
    color: #555;
  }

  .param-count {
    font-size: 12px;
    color: #666;
  }

  .param-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    margin-top: 10px;
  }

  .param-table th {
    background: #f0f0f0;
    padding: 8px 10px;
    font-size: 11px;
    color: #555;
    border-bottom: 1px solid #e0e0e0;
    font-weight: 600;
  }

  .param-table td {
    padding: 8px 10px;
    border-bottom: 1px solid #f0f0f0;
  }

  .type-badge.small {
    padding: 2px 6px;
    font-size: 10px;
  }

  .type-badge.small.field {
    background: #e3f2fd;
    color: #1976d2;
  }

  .type-badge.small.surface {
    background: #e8f5e9;
    color: #2e7d32;
  }

  .type-badge.small.genkw {
    background: #f3e5f5;
    color: #7b1fa2;
  }

  .na-text {
    font-size: 12px;
    color: #999;
    font-style: italic;
  }

  .zone-cell {
    min-width: 200px;
  }

  .zone-tags.small {
    display: flex;
    flex-wrap: wrap;
    gap: 2px;
  }

  .zone-badge {
    display: inline-block;
    padding: 1px 5px;
    border-radius: 8px;
    font-size: 9px;
    border: 1px solid #e0e0e0;
    background: #f5f5f5;
    color: #999;
  }

  .zone-badge.active {
    background: #e3f2fd;
    border-color: #1976d2;
    color: #1976d2;
    font-weight: 600;
  }

  .remove-btn {
    background: none;
    border: none;
    color: #e74c3c;
    font-size: 16px;
    cursor: pointer;
    padding: 0 5px;
    line-height: 1;
  }

  .remove-btn:disabled {
    color: #ccc;
    cursor: not-allowed;
  }

  .add-param-row {
    margin-top: 10px;
  }

  .add-param-row select {
    padding: 6px 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 12px;
    width: 100%;
    max-width: 300px;
  }

  .add-param-row select:disabled {
    background: #f5f5f5;
  }
</style>
