<script>
  import { parameters as paramDefinitions, availableZones, getZoneLabel, getTypeLabel } from './parameterStore.js';

  // Merge parameter definitions with algorithm-specific UI state
  let parameters = paramDefinitions.map(p => ({
    ...p,
    update: p.type !== 'surface' || p.name === 'FaultSeal',
    algorithm: p.dimensionality === 1 ? (p.name === 'Multiplier_2' ? 'EnIF' : p.name === 'EverestParam_A' ? 'ES' : 'ES-MDA') : 'ES-MDA',
    settings: {
      truncation: p.name === 'Gaussian1' ? 0.98 : p.name === 'TopReservoir' ? 0.99 : p.name === 'EverestParam_A' ? 0.95 : 1.0,
      localization: p.dimensionality === 1 ? (p.name === 'Multiplier_1' ? 'adaptive' : p.name === 'EverestParam_A' ? 'adaptive' : 'graph') : 'distance',
      correlationThreshold: p.name === 'Multiplier_1' ? 0.5 : p.name === 'EverestParam_A' ? 0.3 : null
    },
    expanded: false
  }));

  const algorithms = ['ES', 'ES-MDA', 'EnIF'];
  let expandedRow = null;

  function toggleExpand(index) {
    parameters[index].expanded = !parameters[index].expanded;
    parameters = parameters;
  }

  function getLocalizationOptions(algo, dim) {
    if (algo === 'EnIF') return ['graph'];
    if (dim === 1) return ['none', 'adaptive'];
    return ['none', 'adaptive', 'distance'];
  }
</script>

<div class="algorithm-selection">
  <table>
    <thead>
      <tr>
        <th class="expand-col"></th>
        <th>Update</th>
        <th>Parameter Name</th>
        <th>Type</th>
        <th>Zones</th>
        <th>Algorithm</th>
      </tr>
    </thead>
    <tbody>
      {#each parameters as param, i}
      <tr class="main-row" on:click={() => param.update && toggleExpand(i)} class:clickable={param.update}>
        <td class="expand-col">
          {#if param.update}
            <span class="chevron" class:expanded={param.expanded}>›</span>
          {/if}
        </td>
        <td>
          <input type="checkbox" bind:checked={param.update} on:click|stopPropagation />
        </td>
        <td>{param.name}</td>
        <td><span class="type-badge" class:grid={param.dimensionality === 3} class:surface={param.dimensionality === 2} class:scalar={param.dimensionality === 1}>{getTypeLabel(param.dimensionality)}</span></td>
        <td class="zone-cell">
          {#if param.zones}
            <div class="zone-tags small">
              {#each param.zones as zone}
                <span class="zone-badge">{getZoneLabel(zone)}</span>
              {/each}
            </div>
          {:else}
            <span class="na-text">All zones</span>
          {/if}
        </td>
        <td>
          <select bind:value={param.algorithm} class="setting-input mono" disabled={!param.update} on:click|stopPropagation>
            {#each algorithms as algo}
              <option value={algo}>{algo}</option>
            {/each}
          </select>
        </td>
      </tr>
        {#if param.expanded}
           <tr class="settings-row">
              <td colspan="6">
              <div class="settings-panel">
                <div class="settings-section">
                  <h4 class="section-title">General Information</h4>
                  <div class="settings-grid">
                    <div class="setting-item">
                      <label>Parameter Type:</label>
                      <span class="info-value">{getTypeLabel(param.dimensionality)}</span>
                    </div>
                    {#if param.zones}
                      <div class="setting-item">
                        <label>Zones:</label>
                        <div class="zone-tags small">
                          {#each param.zones as zone}
                            <span class="zone-badge">{getZoneLabel(zone)}</span>
                          {/each}
                        </div>
                      </div>
                    {/if}
                    {#if param.gridInfo}
                      <div class="setting-item">
                        <label>{param.dimensionality === 3 ? 'Grid Size:' : 'Surface Size:'}</label>
                        <span class="info-value">{param.dimensionality === 3 ? `${param.gridInfo.nx} × ${param.gridInfo.ny} × ${param.gridInfo.nz}` : `${param.gridInfo.ncol} × ${param.gridInfo.nrow}`}</span>
                      </div>
                      <div class="setting-item">
                        <label>{param.dimensionality === 3 ? 'Grid Spacing:' : 'Cell Size:'}</label>
                        <span class="info-value">{param.gridInfo.xinc}m × {param.gridInfo.yinc}m</span>
                      </div>
                    {/if}
                  </div>
                </div>

                <div class="settings-section">
                  <h4 class="section-title">Algorithm Settings</h4>
                  <div class="settings-grid">
                    <div class="setting-item">
                      <label>Algorithm:</label>
                      <select bind:value={param.algorithm} class="setting-input" on:click|stopPropagation>
                        {#each algorithms as algo}
                          <option value={algo}>{algo}</option>
                        {/each}
                      </select>
                    </div>
                    {#if param.algorithm !== 'EnIF'}
                    <div class="setting-item">
                      <label>Truncation:</label>
                      <input type="number" bind:value={param.settings.truncation} step="0.01" min="0" max="1" class="setting-input" />
                    </div>
                    {/if}
                  </div>
                </div>

{#if param.algorithm !== 'EnIF'}
                <div class="settings-section">
                  <h4 class="section-title">Localization Setup</h4>
                  <div class="settings-grid">
                    <div class="setting-item">
                      <label>Localization Method:</label>
                      <select bind:value={param.settings.localization} class="setting-input">
                        {#each getLocalizationOptions(param.algorithm, param.dimensionality) as loc}
                          <option value={loc}>{loc}</option>
                        {/each}
                      </select>
                    </div>

                    {#if param.settings.localization === 'adaptive'}
                      <div class="setting-item">
                        <label>Correlation Threshold:</label>
                        <input type="number" bind:value={param.settings.correlationThreshold} step="0.1" min="0" max="1" class="setting-input" placeholder="3/√N" />
                      </div>
                    {/if}
                  </div>
                </div>
                {/if}
              </div>
            </td>
          </tr>
        {/if}
      {/each}
    </tbody>
  </table>
</div>

<style>
  .algorithm-selection {
    font-size: 14px;
  }

  .description {
    color: #666;
    margin-bottom: 20px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 20px;
  }

  th {
    background: #f5f5f5;
    padding: 12px;
    text-align: left;
    border-bottom: 2px solid #e0e0e0;
    font-weight: 600;
    font-size: 13px;
    color: #555;
  }

  .expand-col {
    width: 30px;
    text-align: center;
  }

  td {
    padding: 12px;
    border-bottom: 1px solid #e0e0e0;
  }

  .main-row:hover {
    background: #f9f9f9;
  }

  .main-row.clickable {
    cursor: pointer;
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

  select {
    padding: 6px 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 13px;
    background: white;
    cursor: pointer;
  }

  select:disabled {
    background: #f5f5f5;
    cursor: not-allowed;
  }

  .type-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
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

  .mono {
    font-family: 'Courier New', monospace;
    font-size: 13px;
    color: #555;
  }

  .settings-row td {
    padding: 0;
    border-bottom: 2px solid #e0e0e0;
    background: #fafafa;
  }

  .settings-panel {
    padding: 20px;
    border-top: 1px solid #e0e0e0;
  }

  .settings-section {
    margin-bottom: 20px;
  }

  .settings-section:last-child {
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

  .settings-panel h4 {
    margin: 0 0 15px 0;
    font-size: 14px;
    color: #2c3e50;
  }

  .settings-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 15px;
  }

  .setting-item {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }

  .setting-item label {
    font-size: 12px;
    font-weight: 600;
    color: #555;
  }

  .setting-input {
    padding: 6px 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 13px;
  }

  .info-text {
    color: #666;
    font-size: 13px;
    font-style: italic;
    padding: 10px 0;
  }

  .info-value {
    font-family: 'Courier New', monospace;
    font-size: 13px;
    color: #555;
    padding: 6px 0;
    display: inline-block;
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
    border: 1px solid #1976d2;
    background: #e3f2fd;
    color: #1976d2;
    font-weight: 600;
  }

  .na-text {
    font-size: 12px;
    color: #999;
    font-style: italic;
  }
</style>
