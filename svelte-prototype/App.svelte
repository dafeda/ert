<script>
  import { onMount } from 'svelte';
  import AlgorithmSelection from './AlgorithmSelection.svelte';
  import ObservationLocalization from './ObservationLocalization.svelte';
  import ParameterObservationMatrix from './ParameterObservationMatrix.svelte';
  import SummaryValidation from './SummaryValidation.svelte';

  let activeTab = 'parameters';

  const tabs = [
    { id: 'parameters', label: 'Parameters' },
    { id: 'observations', label: 'Observations' },
    { id: 'matrix', label: 'Parameters vs. Observations' },
    { id: 'summary', label: 'Summary/Validation' }
  ];
</script>

<main>
  <div class="tabs">
    {#each tabs as tab}
      <button
        class="tab"
        class:active={activeTab === tab.id}
        on:click={() => activeTab = tab.id}
      >
        {tab.label}
      </button>
    {/each}
  </div>

  <div class="content">
    {#if activeTab === 'parameters'}
      <AlgorithmSelection />
    {:else if activeTab === 'observations'}
      <ObservationLocalization />
    {:else if activeTab === 'matrix'}
      <ParameterObservationMatrix />
    {:else if activeTab === 'summary'}
      <SummaryValidation />
    {/if}
  </div>
</main>

<style>
  main {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    color: #333;
  }

  h1 {
    font-size: 24px;
    margin-bottom: 20px;
    color: #2c3e50;
  }

  .tabs {
    display: flex;
    gap: 0;
    border-bottom: 2px solid #e0e0e0;
    margin-bottom: 20px;
  }

  .tab {
    padding: 12px 24px;
    border: none;
    background: none;
    cursor: pointer;
    font-size: 14px;
    color: #666;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    transition: all 0.2s;
  }

  .tab:hover {
    color: #2c3e50;
    background: #f5f5f5;
  }

  .tab.active {
    color: #3498db;
    border-bottom-color: #3498db;
    font-weight: 600;
  }

  .content {
    background: white;
    border-radius: 8px;
    padding: 20px;
  }
</style>
