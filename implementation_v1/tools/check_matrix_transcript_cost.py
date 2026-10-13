"""Reconcile measured public matrix frame counts; no additional protocol runs."""
import hashlib
import json
from pathlib import Path
from tools.build_matrix_cost import INPUT, RETURN, TARGET, GC_INPUT_OT

ROOT=Path(__file__).resolve().parents[1]


def main():
    paths=[f'records/matrix107_small_verified_{s}.json'
           for s in ('3x17x7_m0','3x17x7_m1','16x17x33_m0')]
    paths += [f'records/matrix107_shapes_{s}_m0.json'
              for s in ('1x2048x2048','1x2048x6144','1x6144x2048','1x2048x151936')]
    cases=[]
    for path in paths:
        r=json.loads((ROOT/path).read_text())
        t,h,c=r['output_coordinates'],r['input_ciphertexts'],r['returned_ciphertext_groups']
        expected={'application_bytes':t*(TARGET+GC_INPUT_OT)+h*INPUT+c*RETURN+1893217,
                  'frames_sent':128*t+h+259,'frames_received':39*t+c+129,
                  'transcript_direction_runs':2*t+6}
        assert all(r[key]==value for key,value in expected.items()),path
        cases.append({'record':path,'shape':r['shape_B_K_O'],'measured_counts_match':True,**expected})
    report={'kind':'current_serial_transcript_counts_reconciliation',
            'cases':cases,'result':'all_measured_application_counts_match',
            'per_target_bytes_with_GC_input_OT':TARGET+GC_INPUT_OT,
            'per_target_GC_input_OT_bytes':GC_INPUT_OT,
            'one_fresh_protocol_setup_and_descriptor_bytes':1893217,
            'setup_constant_scope':'Observed fixed fresh-session overhead, reconciled across all listed shapes; not a full session or TLS overhead bound',
            'direction_run_expression':'2*B*O+6 for these fresh single-matrix sessions',
            'direction_run_meaning':'Maximal consecutive same-direction application frame runs in the current serial transcript; not measured RTT or cryptographic circuit depth',
            'GC_division_rounds':60,'GC_AND_gates_per_target':67625,
            'GC_rounds_are_local_circuit_stages_not_network_RTT':True,
            'excludes_TLS_TCP_IP_and_diagnostic_reconstruction':True,
            'full_model_R2':False,
            'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths+[
                'tools/check_matrix_transcript_cost.py','tools/build_matrix_cost.py',
                'native/src/bfv.cpp','third_party/EzPC/SCI/src/OT/iknp.h']}}
    (ROOT/'records/matrix107_transcript_cost_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'result':report['result'],'cases':len(cases)}))


if __name__=='__main__':main()
