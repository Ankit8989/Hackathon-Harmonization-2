"""
Report generation utilities for the Agentic AI Data Harmonization System.
Generates HTML, JSON, and visualization reports.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from jinja2 import Environment, BaseLoader

from utils.logger import get_logger

logger = get_logger("ReportGenerator")


# HTML Report Template
AUDIT_REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --bg: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #334155;
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --border: #475569;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        header {
            background: linear-gradient(135deg, var(--primary) 0%, #8b5cf6 100%);
            padding: 3rem 2rem;
            margin-bottom: 2rem;
            border-radius: 1rem;
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);
        }
        
        header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        header .subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .meta-info {
            display: flex;
            gap: 2rem;
            margin-top: 1.5rem;
            flex-wrap: wrap;
        }
        
        .meta-item {
            background: rgba(255,255,255,0.15);
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-size: 0.9rem;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .card {
            background: var(--bg-secondary);
            border-radius: 1rem;
            padding: 1.5rem;
            border: 1px solid var(--border);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }
        
        .card h3 {
            color: var(--primary);
            margin-bottom: 1rem;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .score-card {
            text-align: center;
            padding: 2rem;
        }
        
        .score {
            font-size: 4rem;
            font-weight: 700;
            line-height: 1;
        }
        
        .score.high { color: var(--success); }
        .score.medium { color: var(--warning); }
        .score.low { color: var(--error); }
        
        .score-label {
            color: var(--text-muted);
            margin-top: 0.5rem;
        }
        
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 2rem;
            font-size: 0.9rem;
            font-weight: 600;
        }
        
        .status-success {
            background: rgba(16, 185, 129, 0.2);
            color: var(--success);
            border: 1px solid var(--success);
        }
        
        .status-warning {
            background: rgba(245, 158, 11, 0.2);
            color: var(--warning);
            border: 1px solid var(--warning);
        }
        
        .status-error {
            background: rgba(239, 68, 68, 0.2);
            color: var(--error);
            border: 1px solid var(--error);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        
        th, td {
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        
        th {
            background: var(--bg-card);
            color: var(--primary);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.05em;
        }
        
        tr:hover td {
            background: rgba(99, 102, 241, 0.1);
        }
        
        .section {
            margin-bottom: 2rem;
        }
        
        .section-title {
            font-size: 1.5rem;
            color: var(--text);
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--primary);
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .timeline {
            position: relative;
            padding-left: 2rem;
        }
        
        .timeline::before {
            content: '';
            position: absolute;
            left: 0.5rem;
            top: 0;
            bottom: 0;
            width: 2px;
            background: var(--border);
        }
        
        .timeline-item {
            position: relative;
            padding: 1rem 0;
            padding-left: 1.5rem;
        }
        
        .timeline-item::before {
            content: '';
            position: absolute;
            left: -1.5rem;
            top: 1.5rem;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--primary);
            border: 2px solid var(--bg);
        }
        
        .timeline-item.success::before { background: var(--success); }
        .timeline-item.warning::before { background: var(--warning); }
        .timeline-item.error::before { background: var(--error); }
        
        .timeline-time {
            color: var(--text-muted);
            font-size: 0.85rem;
        }
        
        .timeline-content {
            background: var(--bg-card);
            padding: 1rem;
            border-radius: 0.5rem;
            margin-top: 0.5rem;
        }
        
        .issue-list {
            list-style: none;
        }
        
        .issue-item {
            padding: 1rem;
            background: var(--bg-card);
            border-radius: 0.5rem;
            margin-bottom: 0.75rem;
            border-left: 4px solid var(--border);
        }
        
        .issue-item.blocking { border-left-color: var(--error); }
        .issue-item.fixable { border-left-color: var(--warning); }
        .issue-item.ignorable { border-left-color: var(--success); }
        
        .issue-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        
        .issue-type {
            font-weight: 600;
            color: var(--text);
        }
        
        .issue-description {
            color: var(--text-muted);
            font-size: 0.9rem;
        }
        
        .llm-analysis {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
            border: 1px solid var(--primary);
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .llm-analysis h4 {
            color: var(--primary);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .llm-analysis p {
            color: var(--text);
            line-height: 1.8;
        }
        
        .progress-bar {
            height: 8px;
            background: var(--bg-card);
            border-radius: 4px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        
        .progress-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        
        .progress-fill.high { background: var(--success); }
        .progress-fill.medium { background: var(--warning); }
        .progress-fill.low { background: var(--error); }
        
        footer {
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            font-size: 0.9rem;
            border-top: 1px solid var(--border);
            margin-top: 3rem;
        }
        
        @media (max-width: 768px) {
            .container { padding: 1rem; }
            header h1 { font-size: 1.8rem; }
            .grid { grid-template-columns: 1fr; }
            .meta-info { flex-direction: column; gap: 0.5rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 {{ title }}</h1>
            <p class="subtitle">Agentic AI Data Harmonization System - Final Audit Report</p>
            <div class="meta-info">
                <span class="meta-item">📁 {{ input_file }}</span>
                <span class="meta-item">🕐 {{ timestamp }}</span>
                <span class="meta-item">⏱️ {{ duration }}s</span>
                <span class="meta-item">🔄 Pipeline ID: {{ pipeline_id }}</span>
            </div>
        </header>
        
        <!-- Summary Cards -->
        <div class="grid">
            <div class="card score-card">
                <div class="score {{ 'high' if overall_status == 'success' else 'medium' if overall_status == 'warning' else 'low' }}">
                    {{ quality_score }}%
                </div>
                <div class="score-label">Overall Quality Score</div>
            </div>
            
            <div class="card score-card">
                <div class="score {{ 'high' if confidence_score >= 0.9 else 'medium' if confidence_score >= 0.7 else 'low' }}">
                    {{ "%.0f"|format(confidence_score * 100) }}%
                </div>
                <div class="score-label">Confidence Score</div>
            </div>
            
            <div class="card score-card">
                <span class="status-badge status-{{ overall_status }}">
                    {{ '✅' if overall_status == 'success' else '⚠️' if overall_status == 'warning' else '❌' }}
                    {{ overall_status|upper }}
                </span>
                <div class="score-label" style="margin-top: 1rem;">Pipeline Status</div>
            </div>
            
            <div class="card">
                <h3>📊 Processing Summary</h3>
                <table>
                    <tr><td>Input Records</td><td>{{ input_records }}</td></tr>
                    <tr><td>Output Records</td><td>{{ output_records }}</td></tr>
                    <tr><td>LLM Calls</td><td>{{ llm_calls }}</td></tr>
                    <tr><td>Tokens Used</td><td>{{ tokens_used }}</td></tr>
                </table>
            </div>
        </div>
        
        <!-- Agent Results -->
        <div class="section">
            <h2 class="section-title">🤖 Agent Execution Results</h2>
            
            {% for agent in agents %}
            <div class="card" style="margin-bottom: 1rem;">
                <h3>
                    {{ agent.emoji }} {{ agent.name }}
                    <span class="status-badge status-{{ agent.status }}">{{ agent.status }}</span>
                </h3>
                
                <div style="display: flex; gap: 2rem; margin: 1rem 0;">
                    <div>
                        <span style="color: var(--text-muted);">Confidence:</span>
                        <strong>{{ "%.1f"|format(agent.confidence * 100) }}%</strong>
                    </div>
                    <div>
                        <span style="color: var(--text-muted);">Duration:</span>
                        <strong>{{ "%.2f"|format(agent.duration) }}s</strong>
                    </div>
                    <div>
                        <span style="color: var(--text-muted);">Issues:</span>
                        <strong>{{ agent.issues_count }}</strong>
                    </div>
                </div>
                
                <div class="progress-bar">
                    <div class="progress-fill {{ 'high' if agent.confidence >= 0.9 else 'medium' if agent.confidence >= 0.7 else 'low' }}" 
                         style="width: {{ agent.confidence * 100 }}%"></div>
                </div>
                
                {% if agent.summary %}
                <p style="color: var(--text-muted); margin-top: 1rem;">{{ agent.summary }}</p>
                {% endif %}
            </div>
            {% endfor %}
        </div>
        
        <!-- Data Quality Issues -->
        {% if issues %}
        <div class="section">
            <h2 class="section-title">⚠️ Data Quality Issues</h2>
            <ul class="issue-list">
                {% for issue in issues %}
                <li class="issue-item {{ issue.severity }}">
                    <div class="issue-header">
                        <span class="issue-type">{{ issue.type }}</span>
                        <span class="status-badge status-{{ 'error' if issue.severity == 'blocking' else 'warning' if issue.severity == 'fixable' else 'success' }}">
                            {{ issue.severity }}
                        </span>
                    </div>
                    <p class="issue-description">{{ issue.description }}</p>
                    {% if issue.column %}
                    <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 0.5rem;">
                        Column: <code>{{ issue.column }}</code> | Affected: {{ issue.affected_rows }} rows ({{ issue.affected_pct }}%)
                    </p>
                    {% endif %}
                </li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}
        
        <!-- LLM Analysis -->
        {% if llm_analysis %}
        <div class="section">
            <h2 class="section-title">🧠 LLM Analysis & Recommendations</h2>
            <div class="llm-analysis">
                <h4>🤖 AI-Generated Insights</h4>
                <p>{{ llm_analysis }}</p>
            </div>
            
            {% if recommendations %}
            <div class="card" style="margin-top: 1rem;">
                <h3>💡 Recommendations</h3>
                <ul style="padding-left: 1.5rem;">
                    {% for rec in recommendations %}
                    <li style="margin: 0.5rem 0; color: var(--text-muted);">{{ rec }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        <!-- Audit Trail -->
        <div class="section">
            <h2 class="section-title">📜 Audit Trail</h2>
            <div class="timeline">
                {% for entry in audit_trail %}
                <div class="timeline-item {{ entry.status }}">
                    <div class="timeline-time">{{ entry.timestamp }}</div>
                    <div class="timeline-content">
                        <strong>{{ entry.agent }}</strong> - {{ entry.action }}
                        {% if entry.confidence %}
                        <span style="color: var(--text-muted);"> ({{ "%.1f"|format(entry.confidence * 100) }}% confidence)</span>
                        {% endif %}
                        {% if entry.details %}
                        <p style="color: var(--text-muted); font-size: 0.9rem; margin-top: 0.5rem;">{{ entry.details }}</p>
                        {% endif %}
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <!-- Output Files -->
        <div class="section">
            <h2 class="section-title">📄 Generated Files</h2>
            <div class="card">
                <table>
                    <thead>
                        <tr>
                            <th>File</th>
                            <th>Type</th>
                            <th>Path</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for file in output_files %}
                        <tr>
                            <td>{{ file.name }}</td>
                            <td>{{ file.type }}</td>
                            <td><code>{{ file.path }}</code></td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        
        <footer>
            <p>Generated by Agentic AI Data Harmonization System</p>
            <p>Powered by Azure OpenAI GPT-5.2</p>
            <p>{{ timestamp }}</p>
        </footer>
    </div>
</body>
</html>
"""


class ReportGenerator:
    """
    Generate various reports for the harmonization pipeline.
    Supports HTML, JSON, and comparison reports.
    """
    
    def __init__(self, reports_dir: Union[str, Path]):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.jinja_env = Environment(loader=BaseLoader())
    
    def generate_audit_report(
        self,
        pipeline_result: Dict[str, Any],
        output_name: str = "final_audit.html"
    ) -> Path:
        """
        Generate the final HTML audit report.
        
        Args:
            pipeline_result: Complete pipeline result dictionary
            output_name: Output file name
            
        Returns:
            Path to generated report
        """
        logger.info("Generating final audit report...")
        
        # Prepare template data
        template_data = self._prepare_audit_data(pipeline_result)
        
        # Render template
        template = self.jinja_env.from_string(AUDIT_REPORT_TEMPLATE)
        html_content = template.render(**template_data)
        
        # Write file
        output_path = self.reports_dir / output_name
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Audit report generated: {output_path}")
        return output_path
    
    def _prepare_audit_data(self, pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for the audit report template"""
        
        # Determine overall status
        success = pipeline_result.get('success', False)
        confidence = pipeline_result.get('final_confidence_score', 0)
        
        if success and confidence >= 0.9:
            overall_status = 'success'
        elif success and confidence >= 0.7:
            overall_status = 'warning'
        else:
            overall_status = 'error'
        
        # Prepare agent results
        agents = []
        
        # Structural Validation
        sv = pipeline_result.get('structural_validation', {})
        if sv:
            agents.append({
                'emoji': '🔍',
                'name': 'Structural Validation Agent',
                'status': 'success' if sv.get('success') else 'error',
                'confidence': sv.get('confidence_score', 0),
                'duration': sv.get('execution_time_seconds', 0),
                'issues_count': len(sv.get('errors', [])),
                'summary': sv.get('result', {}).get('llm_analysis', '')[:200] if sv.get('result') else ''
            })
        
        # Data Quality
        dq = pipeline_result.get('data_quality', {})
        if dq:
            dq_result = dq.get('result', {})
            agents.append({
                'emoji': '📊',
                'name': 'Data Quality Agent',
                'status': 'success' if dq.get('success') else 'error',
                'confidence': dq.get('confidence_score', 0),
                'duration': dq.get('execution_time_seconds', 0),
                'issues_count': dq_result.get('total_issues', 0) if dq_result else 0,
                'summary': dq_result.get('llm_summary', '')[:200] if dq_result else ''
            })
        
        # Harmonization
        harm = pipeline_result.get('harmonization', {})
        if harm:
            agents.append({
                'emoji': '🔄',
                'name': 'Harmonization Agent',
                'status': 'success' if harm.get('success') else 'error',
                'confidence': harm.get('confidence_score', 0),
                'duration': harm.get('execution_time_seconds', 0),
                'issues_count': len(harm.get('result', {}).get('harmonization_issues', [])) if harm.get('result') else 0,
                'summary': f"Processed {harm.get('result', {}).get('input_records', 0)} records" if harm.get('result') else ''
            })
        
        # Prepare issues list
        issues = []
        if dq and dq.get('result'):
            dq_result = dq['result']
            for issue in dq_result.get('blocking_issues', []):
                issues.append({
                    'type': issue.get('issue_type', 'Unknown'),
                    'severity': 'blocking',
                    'description': issue.get('description', ''),
                    'column': issue.get('column_name'),
                    'affected_rows': issue.get('affected_rows', 0),
                    'affected_pct': issue.get('affected_percentage', 0)
                })
            for issue in dq_result.get('fixable_issues', []):
                issues.append({
                    'type': issue.get('issue_type', 'Unknown'),
                    'severity': 'fixable',
                    'description': issue.get('description', ''),
                    'column': issue.get('column_name'),
                    'affected_rows': issue.get('affected_rows', 0),
                    'affected_pct': issue.get('affected_percentage', 0)
                })
        
        # Prepare audit trail
        audit_trail = []
        for entry in pipeline_result.get('audit_trail', []):
            audit_trail.append({
                'timestamp': entry.get('timestamp', ''),
                'agent': entry.get('agent_name', ''),
                'action': entry.get('action', ''),
                'status': 'success' if entry.get('status') == 'completed' else 'warning',
                'confidence': entry.get('confidence_score'),
                'details': entry.get('details', '')
            })
        
        # Prepare output files list
        output_files = [
            {'name': 'Harmonized Data', 'type': 'CSV', 'path': pipeline_result.get('output_file', 'N/A')},
            {'name': 'Validation Report', 'type': 'JSON', 'path': str(self.reports_dir / 'validation_report.json')},
            {'name': 'Data Quality Report', 'type': 'JSON', 'path': str(self.reports_dir / 'dq_report.json')},
            {'name': 'Harmonization Report', 'type': 'JSON', 'path': str(self.reports_dir / 'harmonization_report.json')},
            {'name': 'Final Audit', 'type': 'HTML', 'path': str(self.reports_dir / 'final_audit.html')}
        ]
        
        # Get LLM analysis and recommendations
        llm_analysis = ""
        recommendations = []
        
        if dq and dq.get('result'):
            llm_analysis = dq['result'].get('llm_summary', '')
            recommendations = dq['result'].get('recommendations', [])
        
        # Build template data
        return {
            'title': 'Data Harmonization Audit Report',
            'input_file': pipeline_result.get('input_file', 'Unknown'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'duration': f"{pipeline_result.get('total_processing_time_seconds', 0):.2f}",
            'pipeline_id': pipeline_result.get('pipeline_id', 'N/A'),
            'quality_score': int(pipeline_result.get('final_quality_score', 0)),
            'confidence_score': pipeline_result.get('final_confidence_score', 0),
            'overall_status': overall_status,
            'input_records': harm.get('result', {}).get('input_records', 0) if harm else 0,
            'output_records': harm.get('result', {}).get('output_records', 0) if harm else 0,
            'llm_calls': pipeline_result.get('total_llm_calls', 0),
            'tokens_used': pipeline_result.get('total_tokens_used', 0),
            'agents': agents,
            'issues': issues[:10],  # Limit to first 10 issues
            'llm_analysis': llm_analysis,
            'recommendations': recommendations[:5],
            'audit_trail': audit_trail,
            'output_files': output_files
        }
    
    def generate_json_report(
        self,
        data: Dict[str, Any],
        report_name: str
    ) -> Path:
        """
        Generate a JSON report.
        
        Args:
            data: Data to write
            report_name: Output file name (without extension)
            
        Returns:
            Path to generated report
        """
        output_path = self.reports_dir / f"{report_name}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"JSON report generated: {output_path}")
        return output_path
    
    def generate_validation_report(
        self,
        validation_result: Dict[str, Any]
    ) -> Path:
        """Generate the structural validation report"""
        return self.generate_json_report(validation_result, "validation_report")
    
    def generate_dq_report(
        self,
        dq_result: Dict[str, Any]
    ) -> Path:
        """Generate the data quality report"""
        return self.generate_json_report(dq_result, "dq_report")
    
    def generate_harmonization_report(
        self,
        harmonization_result: Dict[str, Any]
    ) -> Path:
        """Generate the harmonization report"""
        return self.generate_json_report(harmonization_result, "harmonization_report")
    
    def generate_comparison_report(
        self,
        comparison_data: Dict[str, Any]
    ) -> Path:
        """Generate the before/after comparison report"""
        return self.generate_json_report(comparison_data, "comparison_report")

