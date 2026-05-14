import unittest

from dqn_reward_policy import calculate_reward


class DqnRewardPolicyTest(unittest.TestCase):
    def test_high_pollution_prefers_spray_on(self):
        spray_on = calculate_reward(1, pm25=90.0, tsp=240.0, predicted_pm25=88.0)
        spray_off = calculate_reward(0, pm25=90.0, tsp=240.0, predicted_pm25=88.0)

        self.assertGreater(spray_on["reward"], spray_off["reward"])
        self.assertEqual(spray_on["pollution_state"], "high")
        self.assertEqual(spray_off["action_reason"], "spray_off_high_pollution")

    def test_predicted_high_prefers_spray_on(self):
        spray_on = calculate_reward(1, pm25=60.0, tsp=160.0, predicted_pm25=82.0)
        spray_off = calculate_reward(0, pm25=60.0, tsp=160.0, predicted_pm25=82.0)

        self.assertGreater(spray_on["reward"], spray_off["reward"])
        self.assertEqual(spray_on["pollution_state"], "predicted_high")

    def test_safe_state_prefers_spray_off(self):
        spray_off = calculate_reward(0, pm25=40.0, tsp=100.0, predicted_pm25=42.0)
        spray_on = calculate_reward(1, pm25=40.0, tsp=100.0, predicted_pm25=42.0)

        self.assertGreater(spray_off["reward"], spray_on["reward"])
        self.assertEqual(spray_off["action_reason"], "spray_off_safe")

    def test_decreasing_pollution_improves_spray_on_reward(self):
        decreasing = calculate_reward(
            1,
            pm25=80.0,
            tsp=210.0,
            predicted_pm25=78.0,
            previous_pm25=100.0,
            previous_tsp=260.0,
        )
        increasing = calculate_reward(
            1,
            pm25=100.0,
            tsp=260.0,
            predicted_pm25=98.0,
            previous_pm25=80.0,
            previous_tsp=210.0,
        )

        self.assertGreater(decreasing["reward"], increasing["reward"])
        self.assertGreater(decreasing["pollution_trend"], 0)


if __name__ == "__main__":
    unittest.main()
