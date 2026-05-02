<script>
  import { onMount } from 'svelte';

  let parameters = [
    { name: 'PermeabilityXY', algorithm: 'ES-MDA', update: true, localization: 'Distance (1500m)' },
    { name: 'Porosity', algorithm: 'ES-MDA', update: true, localization: 'Distance (1500m)' },
    { name: 'Gaussian1', algorithm: 'ES', update: true, localization: 'None' },
    { name: 'Gaussian2', algorithm: 'ES-MDA', update: false, localization: 'Distance (default)' },
    { name: 'FaultSeal', algorithm: 'ES-MDA', update: true, localization: 'Distance (800m)' },
    { name: 'TopReservoir', algorithm: 'ES', update: false, localization: 'None' },
    { name: 'Multiplier_1', algorithm: 'ES-MDA', update: true, localization: 'Adaptive (0.5)' },
    { name: 'Multiplier_2', algorithm: 'EnIF', update: true, localization: 'Graph-based' },
    { name: 'EverestParam_A', algorithm: 'ES', update: true, localization: 'Adaptive (0.3)' }
  ];

  let observations = [
    { name: 'WELL-1_PRESSURE', enabled: true, mainRange: 1500 },
    { name: 'WELL-1_WATERCUT', enabled: true, mainRange: 1500 },
    { name: 'WELL-2_PRESSURE', enabled: true, mainRange: 1200 },
    { name: 'WELL-2_WATERCUT', enabled: false, mainRange: 1200 },
    { name: 'RFT_WELL-3', enabled: true, mainRange: 2000 },
    { name: 'GENOBS_SEISMIC_1', enabled: true, mainRange: 3000 },
    { name: 'GENOBS_SEISMIC_2', enabled: true, mainRange: 2500 }
  ];

  let validationWarnings = [
    { type: 'warning', message: 'RFT_WELL-3 has main_range of 2000m - consider reducing for better localization' },
    { type: 'info', message: 'Gaussian2 is disabled for update' },
    { type: 'info', message: 'TopReservoir is disabled for update' },
    { type: 'warning', message: 'WELL-2_WATERCUT is disabled - it will not be used in the update' }
  ];

  function handleRun() {
    alert('Starting Update Run...\n\nThis would trigger the ensemble update with the configured strategies.');
  }
</script>

<div class="summary-validation">
  <div class="section">
    <h3>Parameter Configuration Summary</h3>
    <table>
      <thead>
        <tr>
          <th>Parameter</th>
          <th>Algorithm</th>
          <th>Update</th>
          <th>Localization</th>
        </tr>
      </thead>
      <tbody>
        {#each parameters as param}
          <tr>
            <td>{param.name}</td>
            <td><span class="algo-badge" class:es={param.algorithm === 'ES'} class:esmda={param.algorithm === 'ES-MDA'} class:enif={param.algorithm === 'EnIF'}>{param.algorithm}</span></td>
            <td>
              {#if param.update}
                <span class="status enabled">Yes</span>
              {:else}
                <span class="status disabled">No</span>
              {/if}
            </td>
            <td>{param.localization}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h3>Observation Configuration Summary</h3>
    <table>
      <thead>
        <tr>
          <th>Observation</th>
          <th>Enabled</th>
          <th>Main Range (m)</th>
        </tr>
      </thead>
      <tbody>
        {#each observations as obs}
          <tr>
            <td>{obs.name}</td>
            <td>
              {#if obs.enabled}
                <span class="status enabled">Yes</span>
              {:else}
                <span class="status disabled">No</span>
              {/if}
            </td>
            <td class="mono">{obs.mainRange.toLocaleString()}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h3>Validation</h3>
    {#each validationWarnings as warning}
      <div class="validation-item" class:warning={warning.type === 'warning'} class:info={warning.type === 'info'}>
        <span class="icon">{warning.type === 'warning' ? '⚠' : 'ℹ'}</span>
        <span>{warning.message}</span>
      </div>
    {/each}
  </div>

  <div class="actions">
    <button class="run-btn" on:click={handleRun}>
      Run Update
    </button>
  </div>
</div>

<style>
  .summary-validation {
    font-size: 14px;
  }

  .section {
    margin-bottom: 30px;
  }

  h3 {
    font-size: 16px;
    color: #2c3e50;
    margin-bottom: 15px;
    padding-bottom: 8px;
    border-bottom: 1px solid #e0e0e0;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 10px;
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
    font-size: 12px;
    color: #555;
  }

  .algo-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 600;
  }

  .algo-badge.es {
    background: #e3f2fd;
    color: #1976d2;
  }

  .algo-badge.esmda {
    background: #fff3e0;
    color: #f57c00;
  }

  .algo-badge.enif {
    background: #f3e5f5;
    color: #7b1fa2;
  }

  .status {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 500;
  }

  .status.enabled {
    background: #e8f5e9;
    color: #2e7d32;
  }

  .status.disabled {
    background: #fafafa;
    color: #999;
  }

  .validation-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 4px;
    margin-bottom: 8px;
    font-size: 13px;
  }

  .validation-item.warning {
    background: #fff3cd;
    color: #856404;
    border-left: 4px solid #ffc107;
  }

  .validation-item.info {
    background: #d1ecf1;
    color: #0c5460;
    border-left: 4px solid #17a2b8;
  }

  .icon {
    font-size: 16px;
  }

  .actions {
    margin-top: 30px;
    padding-top: 20px;
    border-top: 2px solid #e0e0e0;
    text-align: right;
  }

  .run-btn {
    padding: 12px 32px;
    background: #3498db;
    color: white;
    border: none;
    border-radius: 6px;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
  }

  .run-btn:hover {
    background: #2980b9;
  }
</style>
