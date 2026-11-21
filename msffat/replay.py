"""Frozen-reference detector replay. No fitting or oracle-based tuning."""
from .deployment import detector_step, oracle_requires_update


MODES = ("accuracy_only", "js_only", "joint_fixed", "joint_dynamic")


def replay_trajectories(trajectories, *, q=.04, w_pp=2., e=.005, boost_cap=.02, decay=.005, oracle_pp=2.):
    if not trajectories:
        raise ValueError("No measured trajectories")
    result = {}
    for mode in MODES:
        counts = dict(tp=0,tn=0,fp=0,fn=0)
        traces, episodes = [], []
        for trajectory in trajectories:
            threshold = q  # Each independent trajectory starts from q0.
            events = trajectory["events"]
            if not events:
                raise ValueError("Empty replay trajectory")
            pending = None
            for index,event in enumerate(events):
                outcome = detector_step(jsd=event["jsd"], probe_accuracy=lambda event=event: event["probe_accuracy"],
                                        reference_accuracy=event["reference_probe_accuracy"],
                                        q=threshold,q_base=q,boost_cap=boost_cap,decay=decay,w_pp=w_pp,e=e,mode=mode)
                oracle = oracle_requires_update(event["remainder_drop_pp"],oracle_pp)
                key = "tp" if oracle and outcome.trigger else "fn" if oracle else "fp" if outcome.trigger else "tn"
                counts[key] += 1
                traces.append({"trajectory":trajectory["id"],"event":index,"checkpoint":event["checkpoint"],
                               "q_before":threshold,"q_after":outcome.q_after,"oracle":oracle,
                               "boost_after":outcome.boost_after,"accuracy_evaluated":outcome.accuracy_evaluated,
                               "trigger":outcome.trigger,"classification":key})
                threshold = outcome.q_after
                if oracle and pending is None:
                    pending = index
                if pending is not None and pending >= 0 and oracle and outcome.trigger:
                    episodes.append({"trajectory":trajectory["id"],"onset_event":pending,
                                     "checkpoint_delay":index-pending,"censored":False})
                    # Do not start another episode until the oracle becomes false.
                    pending = -1
                if not oracle and pending is not None:
                    if pending >= 0:
                        episodes.append({"trajectory":trajectory["id"],"onset_event":pending,
                                         "checkpoint_delay":None,"censored":True})
                    pending = None
            if pending is not None and pending >= 0:
                episodes.append({"trajectory":trajectory["id"],"onset_event":pending,
                                 "checkpoint_delay":None,"censored":True})
        pos,neg = counts["tp"]+counts["fn"],counts["tn"]+counts["fp"]
        result[mode] = {**counts,"positive_events":pos,"negative_events":neg,
                        "false_alarm_rate":counts["fp"]/neg if neg else None,
                        "miss_rate":counts["fn"]/pos if pos else None,
                        "probe_accuracy_checks":sum(event["accuracy_evaluated"] for event in traces),
                        "events":traces,"episodes":episodes}
    return result
