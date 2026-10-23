import json,sys,numpy as np
d=json.load(open(sys.argv[1]))["checkpoints"]
print("ckpt steps | mu_shot std_x | shot hook | advstd_shot | gS_head gH_head gS_tr gH_tr | cosSH_head cosSH_tr cosAimRest | xS-S_h xH-H_h xS-H_h xS-H_tr | KLrun_end off_at aimoff_end nostop_end | mb@1ep")
for c,v in d.items():
    b=v["batches"]; m=lambda k: np.mean([r[k] for r in b if r.get(k) is not None])
    kl=[r for r in b if "kl_run" in r]
    km=lambda mode,k: np.mean([r[f"kl_{mode}"][k] for r in kl if r[f"kl_{mode}"][k] is not None])
    off=[r["kl_run"]["off_at_minibatch"] for r in kl]
    ep1=np.mean([np.mean(r["kl_run"]["per_minibatch"][1:8]) for r in kl])
    x=v["cross_batch_cos"]
    print(f'{c[5:9]} {v["agent_steps"]/1e6:.2f} | {m("mu_len_shot"):.2f} {np.log(np.mean([r["std"][0] for r in b])):.2f} | {m("shot_decisions"):.0f} {m("hook_only_decisions"):.0f} | {m("adv_std_shot"):.4f} | {m("g_shot_head"):.3f} {m("g_hook_head"):.3f} {m("g_shot_trunk"):.3f} {m("g_hook_trunk"):.3f} | {m("cos_shot_hook_head"):.2f} {m("cos_shot_hook_trunk"):.2f} {m("cos_aim_rest_trunk"):.2f} | {x["shot_shot_head"]:.2f} {x["hook_hook_head"]:.2f} {x["shot_hook_head"]:.2f} {x["shot_hook_trunk"]:.2f} | {km("run","end_masked"):.4f} {off} {km("aim_off","end_masked"):.4f} {km("no_stop","end_masked"):.4f} | {ep1:.4f}')
