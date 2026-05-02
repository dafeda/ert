<script>
  let activeSubTab = 'grid';

  const subTabs = [
    { id: 'grid', label: 'Grid Parameters (3D)' },
    { id: 'surface', label: 'Surface Parameters (2D)' },
    { id: 'scalar', label: 'Scalar Parameters (1D)' }
  ];

  let gridParams = [
    { name: 'PermeabilityXY', nx: 100, ny: 80, nz: 25, xinc: 50, yinc: 50, useDistanceLocalization: true, radiusOverride: null },
    { name: 'Porosity', nx: 100, ny: 80, nz: 25, xinc: 50, yinc: 50, useDistanceLocalization: true, radiusOverride: 1500 },
    { name: 'Gaussian1', nx: 100, ny: 80, nz: 25, xinc: 50, yinc: 50, useDistanceLocalization: false, radiusOverride: null },
    { name: 'Gaussian2', nx: 100, ny: 80, nz: 25, xinc: 50, yinc: 50, useDistanceLocalization: true, radiusOverride: null }
  ];

  let surfaceParams = [
    { name: 'FaultSeal', ncol: 200, nrow: 150, xinc: 25, yinc: 25, useDistanceLocalization: true, radiusOverride: 800 },
    { name: 'TopReservoir', ncol: 200, nrow: 150, xinc: 25, yinc: 25, useDistanceLocalization: false, radiusOverride: null }
  ];

  let scalarParams = [
    { name: 'Multiplier_1', localizationType: 'adaptive', correlationThreshold: 0.5 },
    { name: 'Multiplier_2', localizationType: 'enif', correlationThreshold: null },
    { name: 'EverestParam_A', localizationType: 'adaptive', correlationThreshold: 0.3 }
  ];

  const localizationOptions = [
    { value: 'none', label: 'None' },
    { value: 'adaptive', label: 'Adaptive Localization' },
    { value: 'enif', label: 'EnIF' }
  ];

  function getDefaultRadius() {
    return 1000;
  }
</script>

<div class="localization-setup">
  <p class="description">
    Configure localization settings per parameter type. Inspired by ResX's "Define localization setup" tab.
  </p>

  <div class="sub-tabs">
    {#each subTabs as tab}
      <button
        class="sub-tab"
        class:active={activeSubTab === tab.id}
        on:click={() => activeSubTab = tab.id}
      >
        {tab.label}
      </button>
    {/each}
  </div>

  <div class="sub-content">
    {#if activeSubTab === 'grid'}
      <div class="grid-section">
        <h3>Grid Parameters (3D - Field)</h3>
        <table>
          <thead>
            <tr>
              <th>Parameter</th>
              <th>Grid Size</th>
              <th>Grid Spacing</th>
              <th>Use Distance Localization</th>
              <th>Radius Override (m)</th>
            </tr>
          </thead>
          <tbody>
            {#each gridParams as param}
              <tr>
                <td>{param.name}</td>
                <td class="mono">{param.nx} × {param.ny} × {param.nz}</td>
                <td class="mono">{param.xinc}m × {param.yinc}m</td>
                <td style="text-align: center">
                  <input type="checkbox" bind:checked={param.useDistanceLocalization} />
                </td>
                <td>
                  {#if param.useDistanceLocalization}
                    <input type="number" bind:value={param.radiusOverride} placeholder={getDefaultRadius()} class="radius-input" />
                  {:else}
                    <span class="disabled-text">--</span>
                  {/if}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>

    {:else if activeSubTab === 'surface'}
      <div class="surface-section">
        <h3>Surface Parameters (2D - SurfaceConfig)</h3>
        <table>
          <thead>
            <tr>
              <th>Parameter</th>
              <th>Surface Size</th>
              <th>Cell Size</th>
              <th>Use Distance Localization</th>
              <th>Radius Override (m)</th>
            </tr>
          </thead>
          <tbody>
            {#each surfaceParams as param}
              <tr>
                <td>{param.name}</td>
                <td class="mono">{param.ncol} × {param.nrow}</td>
                <td class="mono">{param.xinc}m × {param.yinc}m</td>
                <td style="text-align: center">
                  <input type="checkbox" bind:checked={param.useDistanceLocalization} />
                </td>
                <td>
                  {#if param.useDistanceLocalization}
                    <input type="number" bind:value={param.radiusOverride} placeholder={getDefaultRadius()} class="radius-input" />
                  {:else}
                    <span class="disabled-text">--</span>
                  {/if}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>

    {:else if activeSubTab === 'scalar'}
      <div class="scalar-section">
        <h3>Scalar Parameters (1D - GenKw, EverestControl)</h3>
        <table>
          <thead>
            <tr>
              <th>Parameter</th>
              <th>Localization Method</th>
              <th>Correlation Threshold</th>
            </tr>
          </thead>
          <tbody>
            {#each scalarParams as param}
              <tr>
                <td>{param.name}</td>
                <td>
                  <select bind:value={param.localizationType} class="localization-select">
                    {#each localizationOptions as option}
                      <option value={option.value}>{option.label}</option>
                    {/each}
                  </select>
                </td>
                <td>
                  {#if param.localizationType === 'adaptive'}
                    <input type="number" bind:value={param.correlationThreshold} step="0.1" min="0" max="1" class="threshold-input" />
                  {:else}
                    <span class="disabled-text">--</span>
                  {/if}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
        <div class="info-box">
          <strong>Note:</strong> EnIF uses graph-based precision matrices for localization and does not require adaptive localization settings.
        </div>
      </div>
    {/if}
  </div>
</div>

<style>
  .localization-setup {
    font-size: 14px;
  }

  .description {
    color: #666;
    margin-bottom: 20px;
  }

  .sub-tabs {
    display: flex;
    gap: 0;
    border-bottom: 2px solid #e0e0e0;
    margin-bottom: 20px;
  }

  .sub-tab {
    padding: 10px 20px;
    border: none;
    background: none;
    cursor: pointer;
    font-size: 13px;
    color: #666;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    transition: all 0.2s;
  }

  .sub-tab:hover {
    color: #2c3e50;
    background: #f5f5f5;
  }

  .sub-tab.active {
    color: #e67e22;
    border-bottom-color: #e67e22;
    font-weight: 600;
  }

  h3 {
    font-size: 16px;
    color: #2c3e50;
    margin-bottom: 15px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 20px;
  }

  th {
    background: #f5f5f5;
    padding: 10px 12px;
    text-align: left;
    border-bottom: 2px solid #e0e0e0;
    font-weight: 600;
    font-size: 12px;
    color: #555;
  }

  td {
    padding: 10px 12px;
    border-bottom: 1px solid #e0e0e0;
  }

  tr:hover {
    background: #f9f9f9;
  }

  .mono {
    font-family: 'Courier New', monospace;
    font-size: 13px;
    color: #555;
  }

  .radius-input,
  .threshold-input {
    padding: 6px 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 13px;
    width: 120px;
  }

  .localization-select {
    padding: 6px 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 13px;
    background: white;
    cursor: pointer;
    width: 180px;
  }

  .localization-select:focus {
    outline: none;
    border-color: #e67e22;
  }

  .disabled-text {
    color: #ccc;
  }

  .info-box {
    padding: 12px 16px;
    background: #e8f4f8;
    border-left: 4px solid #3498db;
    border-radius: 4px;
    font-size: 13px;
    color: #2c3e50;
    margin-top: 10px;
  }
</style>
