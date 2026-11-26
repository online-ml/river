def collect_alarms(detector, values):
    alarms = []

    for index, value in enumerate(values):
        detector.update(value)
        if detector.drift_detected:
            alarms.append(index)

    return alarms

def score_events(changes, alarms, tolerance = 30, policy="oldest"):
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    if policy not in ("oldest", "nearest"):
        raise ValueError("unknown matching policy")

    remaining = set(changes)
    matches = []
    unmatched_alarms = []

    for alarm in sorted(alarms):
        eligible = [
            cp for cp in remaining
            if 0 <= alarm - cp <= tolerance
        ]

        if eligible:
            cp = min(eligible) if policy == "oldest" else max(eligible)
            matches.append((cp, alarm))
            remaining.remove(cp)
        else:
            unmatched_alarms.append(alarm)

    matched = len(matches)

    mean_delay = (
        sum(alarm - cp for cp, alarm in matches) / matched
        if matched else None
    )

    return {
        "matches": matches,
        "matched": matched,
        "missed": len(remaining),
        "false_alarms": len(unmatched_alarms),
        "mean_delay": mean_delay
    }