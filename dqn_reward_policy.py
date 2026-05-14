PM25_SAFE_THRESHOLD = 75.0
TSP_SAFE_THRESHOLD = 200.0
SPRAY_WATER_COST = {
    0: 0.0,
    1: 1.0,
    2: 1.8,
}


def classify_pollution(pm25, tsp, predicted_pm25=None):
    if pm25 > PM25_SAFE_THRESHOLD or tsp > TSP_SAFE_THRESHOLD:
        return "high"
    if predicted_pm25 is not None and predicted_pm25 > PM25_SAFE_THRESHOLD:
        return "predicted_high"
    if pm25 < PM25_SAFE_THRESHOLD * 0.6 and tsp < TSP_SAFE_THRESHOLD * 0.6:
        return "clean"
    return "safe"


def pollution_load(pm25, tsp):
    return float(max(pm25 - PM25_SAFE_THRESHOLD, 0.0) + max(tsp - TSP_SAFE_THRESHOLD, 0.0))


def calculate_reward(
    action,
    pm25,
    tsp,
    predicted_pm25=None,
    previous_pm25=None,
    previous_tsp=None,
    previous_action=None,
):
    action = int(action)
    pm25_excess = float(max(pm25 - PM25_SAFE_THRESHOLD, 0.0))
    tsp_excess = float(max(tsp - TSP_SAFE_THRESHOLD, 0.0))
    state = classify_pollution(pm25, tsp, predicted_pm25)
    predicted_high = predicted_pm25 is not None and predicted_pm25 > PM25_SAFE_THRESHOLD
    water_cost = SPRAY_WATER_COST.get(action, SPRAY_WATER_COST[2])
    switched = previous_action is not None and int(previous_action) != action

    if state == "high":
        reward = 5.0 - pm25_excess * 6.0 - tsp_excess * 4.0
        if action == 2:
            reward += 65.0
            action_reason = "spray_high_high_pollution"
        elif action == 1:
            reward += 45.0
            action_reason = "spray_low_high_pollution"
        else:
            reward -= 90.0
            action_reason = "spray_off_high_pollution"
    elif predicted_high:
        reward = 8.0
        if action == 2:
            reward += 28.0
            action_reason = "spray_high_predicted_high"
        elif action == 1:
            reward += 35.0
            action_reason = "spray_low_predicted_high"
        else:
            reward -= 35.0
            action_reason = "spray_off_predicted_high"
    else:
        reward = 25.0
        if action == 0:
            reward += 18.0
            action_reason = "spray_off_safe"
        else:
            reward -= 8.0 + water_cost * 8.0
            action_reason = "spray_on_safe_water_cost"
            if state == "clean":
                reward -= 12.0

    reward -= water_cost * 4.0
    if switched:
        reward -= 10.0

    trend = 0.0
    if previous_pm25 is not None and previous_tsp is not None:
        previous_load = pollution_load(previous_pm25, previous_tsp)
        current_load = pollution_load(pm25, tsp)
        trend = previous_load - current_load
        if action > 0 and trend > 0:
            reward += min(35.0, trend * (0.35 + 0.08 * action))
        elif action > 0 and trend < 0 and state == "high":
            reward -= min(25.0, abs(trend) * 0.3)
        elif action == 0 and trend < 0 and state in {"high", "predicted_high"}:
            reward -= min(20.0, abs(trend) * 0.3)

    return {
        "reward": float(max(min(reward, 120.0), -500.0)),
        "pollution_state": state,
        "action_reason": action_reason,
        "pm25_excess": pm25_excess,
        "tsp_excess": tsp_excess,
        "pollution_trend": float(trend),
        "water_cost": float(water_cost),
        "switch_penalty_applied": int(switched),
    }
