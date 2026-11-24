"""InferBench HTML Report Generator.

Generates a professional benchmark report from InferBench JSON results.

Usage:
    python -m inferbench.report results.json --output report.html
"""

import argparse
import json
from datetime import datetime


def generate_report(results, output_path="inferbench_report.html"):
    """Generate an HTML report from InferBench results."""
    engines = results.get("engines", {})
    timestamp = results.get("timestamp", datetime.now().isoformat())

    # Collect engine names
    engine_names = []
    engine_data = {}
    for key, data in engines.items():
        if "error" not in data:
            name = data.get("engine", key)
            engine_names.append(name)
            engine_data[name] = data

    if not engine_names:
        print("No valid results to report.")
        return

    # Build comparison data
    def get_val(data, *path):
        try:
            v = data
            for p in path:
                v = v[p]
            return v
        except (KeyError, TypeError, IndexError):
            return None

    # Generate chart data
    throughput_data = []
    for name in engine_names:
        d = engine_data[name]
        val = get_val(d, "dimensions", "throughput", "results",
                      "batch_1", "requests_per_second")
        throughput_data.append(val or 0)

    latency_p50 = []
    latency_p99 = []
    for name in engine_names:
        d = engine_data[name]
        p50 = get_val(d, "dimensions", "latency", "results",
                      "concurrency_1", "p50_ms")
        p99 = get_val(d, "dimensions", "latency", "results",
                      "concurrency_1", "p99_ms")
        latency_p50.append(round(p50, 4) if p50 else 0)
        latency_p99.append(round(p99, 4) if p99 else 0)

    cold_start = []
    for name in engine_names:
        d = engine_data[name]
        val = get_val(d, "dimensions", "cold_start", "results", "mean_ms")
        cold_start.append(round(val, 1) if val else 0)

    obs_scores = []
    agent_scores = []
    for name in engine_names:
        d = engine_data[name]
        obs = get_val(d, "dimensions", "observability", "results", "score")
        agent = get_val(d, "dimensions", "agent_readiness", "results",
                        "score")
        obs_scores.append(obs or "0/8")
        agent_scores.append(agent or "0/6")

    # Build HTML
    colors = ["#e8709a", "#b48ed2", "#6cc4a1", "#e4b04e", "#e8926a",
              "#7ab8d4"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>InferBench Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root {{
    --bg: #fef5f7;
    --surface: #ffffff;
    --border: #f5d0d8;
    --text: #3d2a33;
    --text-dim: #9e7f8a;
    --accent: #e8709a;
    --accent2: #b48ed2;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 24px; max-width: 1200px; margin: 0 auto; }}
h1 {{ font-size: 2rem; margin-bottom: 4px; }}
.subtitle {{ color: var(--text-dim); margin-bottom: 32px; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
@media (max-width: 768px) {{ .grid {{ grid-template-columns: 1fr; }} }}
.card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }}
.card h2 {{ font-size: 1rem; color: var(--accent); margin-bottom: 16px; text-transform: uppercase; letter-spacing: 1px; }}
.card canvas {{ max-height: 300px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
th {{ color: var(--accent); font-weight: 600; font-size: 0.85rem; text-transform: uppercase; }}
td {{ font-size: 0.95rem; }}
.score {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-weight: 600; font-size: 0.85rem; }}
.score-high {{ background: rgba(108,196,161,0.15); color: #4aab84; }}
.score-mid {{ background: rgba(228,176,78,0.15); color: #c89a30; }}
.score-low {{ background: rgba(224,107,122,0.15); color: #c0505e; }}
.full-width {{ grid-column: 1 / -1; }}
.badge {{ display: inline-block; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; margin-right: 8px; margin-bottom: 4px; }}
.badge-yes {{ background: rgba(108,196,161,0.15); color: #4aab84; }}
.badge-no {{ background: rgba(224,107,122,0.08); color: #bba0a7; }}
footer {{ text-align: center; color: var(--text-dim); padding: 32px 0; font-size: 0.85rem; }}
</style>
</head>
<body>

<h1>InferBench Report</h1>
<p class="subtitle">Multi-Dimensional Inference Engine Benchmark &middot; {timestamp[:10]}</p>

<div class="grid">
  <div class="card">
    <h2>Throughput (requests/sec)</h2>
    <canvas id="throughputChart"></canvas>
  </div>
  <div class="card">
    <h2>Latency (ms)</h2>
    <canvas id="latencyChart"></canvas>
  </div>
  <div class="card">
    <h2>Cold Start (ms)</h2>
    <canvas id="coldStartChart"></canvas>
  </div>
  <div class="card">
    <h2>Capability Scores</h2>
    <canvas id="capabilityChart"></canvas>
  </div>
</div>

<div class="grid">
  <div class="card full-width">
    <h2>Full Comparison</h2>
    <table>
      <tr>
        <th>Dimension</th>
        {"".join(f'<th>{n}</th>' for n in engine_names)}
      </tr>
      <tr>
        <td>Throughput (req/s, batch=1)</td>
        {"".join(f'<td><strong>{t:.1f}</strong></td>' for t in throughput_data)}
      </tr>
      <tr>
        <td>Latency p50 (ms)</td>
        {"".join(f'<td>{p:.4f}</td>' for p in latency_p50)}
      </tr>
      <tr>
        <td>Latency p99 (ms)</td>
        {"".join(f'<td>{p:.4f}</td>' for p in latency_p99)}
      </tr>
      <tr>
        <td>Cold Start (ms)</td>
        {"".join(f'<td>{c:.1f}</td>' for c in cold_start)}
      </tr>
      <tr>
        <td>Observability</td>
        {"".join(f'<td><span class="score {"score-high" if int(s.split("/")[0]) >= 5 else "score-low"}">{s}</span></td>' for s in obs_scores)}
      </tr>
      <tr>
        <td>Agent Readiness</td>
        {"".join(f'<td><span class="score {"score-high" if int(s.split("/")[0]) >= 4 else "score-low"}">{s}</span></td>' for s in agent_scores)}
      </tr>
    </table>
  </div>
</div>
"""

    # Add observability detail cards
    for name in engine_names:
        d = engine_data[name]
        obs_features = get_val(d, "dimensions", "observability", "results",
                               "features") or {}
        agent_features = get_val(d, "dimensions", "agent_readiness",
                                 "results", "features") or {}
        html += f"""
<div class="grid" style="margin-top: 0;">
  <div class="card">
    <h2>{name} — Observability</h2>
    <div>
      {"".join(f'<span class="badge {"badge-yes" if v else "badge-no"}">{k.replace("_", " ").title()}</span>' for k, v in obs_features.items())}
    </div>
  </div>
  <div class="card">
    <h2>{name} — Agent Readiness</h2>
    <div>
      {"".join(f'<span class="badge {"badge-yes" if v else "badge-no"}">{k.replace("_", " ").title()}</span>' for k, v in agent_features.items())}
    </div>
  </div>
</div>
"""

    html += f"""
<footer>
  Generated by InferBench v0.1.0 &middot; ApicalAI Project
</footer>

<script>
const labels = {json.dumps(engine_names)};
const colors = {json.dumps(colors[:len(engine_names)])};
const chartDefaults = {{
  responsive: true,
  plugins: {{ legend: {{ display: false }} }},
}};

new Chart(document.getElementById('throughputChart'), {{
  type: 'bar',
  data: {{
    labels: labels,
    datasets: [{{ data: {json.dumps(throughput_data)}, backgroundColor: colors }}]
  }},
  options: {{ ...chartDefaults, scales: {{ y: {{ beginAtZero: true, ticks: {{ color: '#9e7f8a' }}, grid: {{ color: '#f5d0d8' }} }}, x: {{ ticks: {{ color: '#9e7f8a' }}, grid: {{ display: false }} }} }} }}
}});

new Chart(document.getElementById('latencyChart'), {{
  type: 'bar',
  data: {{
    labels: labels,
    datasets: [
      {{ label: 'p50', data: {json.dumps(latency_p50)}, backgroundColor: colors.map(c => c + '99') }},
      {{ label: 'p99', data: {json.dumps(latency_p99)}, backgroundColor: colors }}
    ]
  }},
  options: {{ ...chartDefaults, plugins: {{ legend: {{ display: true, labels: {{ color: '#9e7f8a' }} }} }}, scales: {{ y: {{ beginAtZero: true, ticks: {{ color: '#9e7f8a' }}, grid: {{ color: '#f5d0d8' }} }}, x: {{ ticks: {{ color: '#9e7f8a' }}, grid: {{ display: false }} }} }} }}
}});

new Chart(document.getElementById('coldStartChart'), {{
  type: 'bar',
  data: {{
    labels: labels,
    datasets: [{{ data: {json.dumps(cold_start)}, backgroundColor: colors }}]
  }},
  options: {{ ...chartDefaults, scales: {{ y: {{ beginAtZero: true, ticks: {{ color: '#9e7f8a' }}, grid: {{ color: '#f5d0d8' }} }}, x: {{ ticks: {{ color: '#9e7f8a' }}, grid: {{ display: false }} }} }} }}
}});

const obsNums = {json.dumps([int(s.split('/')[0]) for s in obs_scores])};
const agentNums = {json.dumps([int(s.split('/')[0]) for s in agent_scores])};
new Chart(document.getElementById('capabilityChart'), {{
  type: 'bar',
  data: {{
    labels: labels,
    datasets: [
      {{ label: 'Observability (/8)', data: obsNums, backgroundColor: '#e8709a99' }},
      {{ label: 'Agent Readiness (/6)', data: agentNums, backgroundColor: '#b48ed299' }}
    ]
  }},
  options: {{ ...chartDefaults, plugins: {{ legend: {{ display: true, labels: {{ color: '#9e7f8a' }} }} }}, scales: {{ y: {{ beginAtZero: true, max: 8, ticks: {{ color: '#9e7f8a' }}, grid: {{ color: '#f5d0d8' }} }}, x: {{ ticks: {{ color: '#9e7f8a' }}, grid: {{ display: false }} }} }} }}
}});
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Report generated: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate InferBench HTML report from JSON results"
    )
    parser.add_argument("results", help="Path to InferBench JSON results")
    parser.add_argument(
        "--output", "-o", default="inferbench_report.html",
        help="Output HTML file path"
    )
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    generate_report(results, args.output)


if __name__ == "__main__":
    main()
