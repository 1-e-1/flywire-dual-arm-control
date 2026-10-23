"""Evidence manifest: software passes cannot override scientific claim gaps."""
import argparse,hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET
from importlib.metadata import version,PackageNotFoundError
import platform,sys
from .scene import ROOT


def build(output,tests):
    files={}
    def read(relative):
        p=ROOT/relative;files[relative]=hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text(encoding='utf-8'))
    run=read('reports/research_async_integrated_01/result.json')
    audit=read('reports/research_async_integrated_01/independent_audit.json')
    surface=read('reports/research_async_integrated_01/surface_accuracy_audit.json')
    planning=read('reports/research_async_integrated_01/planning_snapshot_audit.json')
    formal=read('reports/formal_paired_50_v1/statistics.json')
    protocol=read('reports/formal_paired_50_v1/protocol.json')
    rows=read('reports/formal_paired_50_v1/results.json')
    fault=read('reports/sensor_stress_01/blank_permanent/result.json')
    transient=read('reports/sensor_stress_01/blank_transient/independent_audit.json')
    residual=read('reports/reaction_feasibility_actual_01.json')
    test_path=ROOT/tests;files[tests]=hashlib.sha256(test_path.read_bytes()).hexdigest()
    xml=ET.parse(test_path).getroot();suites=[xml] if xml.tag=='testsuite' else list(xml)
    totals={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','errors','failures','skipped')}
    unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in protocol['source_sha256'].items())
    checks=dict(tests=totals['tests']>0 and totals['errors']==totals['failures']==0,
        integrated_delivery=run['status']=='PASS' and len(run['completed'])==3 and audit['status']=='PASS',
        streaming_budget_this_run=run['cycle_deadline_misses']==0,
        measured_complex_surface=surface['measured_vertex_precision']['count']>0,
        frozen_experiment_accounted=bool(formal['final']) and len(rows)==len(protocol['seeds']),
        frozen_source_identity=unchanged,
        observation_fault_stops=fault['status']=='STOPPED' and len(fault['completed'])==0,
        transient_fault_recovery=transient['status']=='PASS')
    gaps={
        'measured_map_only_full_arm_planning':'NOT_PASSED: full mission uses known CAD; saved free-space utility is insufficient and no measured-only full-arm mission is certified.',
        'continuous_observed_visual_quality':'NOT_PROVEN: interval certificate covers geometry; arbitrary occlusion and probabilistic measurement error are not certified.',
        'unconditional_recursive_feasibility':'NOT_PROVEN: bounded reacquisition and conditional stopping are not arbitrary-environment invariance.',
        'universal_zero_reaction':'NOT_ACHIEVABLE_FOR_ALL_TESTED_FIXED_RIGHT_COMMANDS: retain bounded residual diagnostics rather than rename low reaction as zero.',
        'arbitrary_unknown_object_count_and_interfaces':'NOT_EVALUATED: frozen study has three known standard magnetic interfaces.',
        'complete_task_multimetric_superiority':'DO_NOT_ASSUME: use final paired effects, confidence intervals and failures; local objective improvement is insufficient.',
        'entire_stack_hard_realtime':'NOT_CERTIFIED: stream timings exclude between-segment planning and do not bound OS scheduling.',
        'publication_ready_novelty_and_acceptance':'NOT_ESTABLISHED: manuscript and verified related work are delivered; scientific/peer-review conclusions cannot be asserted by software.'}
    packages={}
    for name in ('numpy','scipy','torch','warp-lang','osqp','usd-core','opencv-python','matplotlib','pytest'):
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]='No distribution metadata; may be supplied by simulation environment'
    manifest=dict(status='IMPLEMENTATION_AND_EVALUATION_DELIVERED_WITH_RESEARCH_GAPS' if all(checks.values()) else 'EVIDENCE_CHECK_INCOMPLETE',
        physical_validation_in_scope=False,complete_nonhardware_research_project=False,
        checks=checks,tests=totals,claim_gaps=gaps,
        actual_completion_blockers=[
            'Environment collision authority is unresolved: complete missions use CAD, not a measured-map-only full-arm planning loop.',
            'The latest post-freeze integrated variant has nominal/regression evidence, not its own independent multi-scene comparative acceptance.',
            'The intended paper-level new-method benefit must be reconciled with mixed primary effects and remaining perception limitations.'],
        not_additional_user_requirements=[
            'Universal exact zero is NOT imposed: the user explicitly permits lowest feasible reaction.',
            'Unconditional safety in every arbitrary dynamic environment and certified OS hard-real-time scheduling are NOT new mandatory requirements; they are unsupported claims to avoid.',
            'Journal acceptance is an external outcome, not an executable software acceptance test.'],
        integrated_run=dict(simulation_time_s=run['simulation_time_s'],capture_gaps_m=audit['capture_gaps_m'],
            cycle_p99_ms=run['cycle_p99_ms'],surface_p95_m=surface['measured_vertex_precision']['p95_m'],
            complete_parts_5mm_coverage=surface['coverage_by_group']['parts']['within_5mm'],planning_audit=planning),
        formal= formal,
        independent_residual_report_scope=residual.get('scope','See original report'),
        inspected_environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages=packages,
            caveat='Recorded at release inspection; not an independently frozen binary-environment attestation for every primary trial.'),
        evidence_sha256=files,
        interpretation='Hardware is excluded, but remaining simulation/research gaps are not waived. Original failures are retained.')
    p=ROOT/output;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(manifest,indent=2),encoding='utf-8');return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='reports/nonhardware_acceptance_01.json')
    p.add_argument('--tests',default='reports/nonhardware_all_tests_04.xml');args=p.parse_args()
    result=build(args.output,args.tests);print(json.dumps({k:result[k] for k in ('status','complete_nonhardware_research_project','checks','tests')},indent=2))


if __name__=='__main__':main()
